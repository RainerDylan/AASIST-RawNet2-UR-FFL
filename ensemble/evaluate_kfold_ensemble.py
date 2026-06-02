import sys
import os
import glob
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, auc

warnings.filterwarnings("ignore", message="No positive samples in y_true")

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(CURRENT_DIR) == 'ensemble':
    ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
else:
    ROOT_DIR = CURRENT_DIR
sys.path.append(ROOT_DIR)

RESULTS_DIR = os.path.join(ROOT_DIR, "results")
MODELS_DIR = os.path.join(ROOT_DIR, "saved_models")
os.makedirs(RESULTS_DIR, exist_ok=True)

# We import your original ASVspoofDataset which successfully loaded the 7000 files previously
from src.data.dataset import ASVspoofDataset
from src.models.aasist import AASIST
from src.models.resnet_simam import resnet18_simam

# =====================================================================
# PATH CONFIGURATIONS 
# =====================================================================
BASE_DATASET_DIR = r"D:\SAMPOERNA\Semester 8\Capstone\Dataset"

PREPROCESSED_LA_DIR = os.path.join(BASE_DATASET_DIR, "preprocessed_la")
PREPROCESSED_DF_DIR = os.path.join(BASE_DATASET_DIR, "preprocessed_df")

PROT_LA = r"D:\SAMPOERNA\Semester 8\Capstone\Dataset\2019\LA\ASVspoof2019_LA_cm_protocols\ASVspoof2019.LA.cm.eval.trl.txt"
PROT_DF = r"D:\SAMPOERNA\Semester 8\Capstone\Dataset\2021\ASVspoof2021_DF_eval_part00\ASVspoof2021_DF_eval\trial_metadata.txt"

# =====================================================================
# PROTOCOL STANDARDIZER
# =====================================================================
def create_unified_protocol(original_protocol_path, dataset_name):
    print(f"\n[!] Verifying and standardizing protocol labels...")
    
    metadata_path = original_protocol_path
    has_labels = False
    
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            if 'bonafide' in f.read(50000).lower():
                has_labels = True
                
    if not has_labels:
        print("[!] No 'bonafide' labels found in the provided path. Auto-searching...")
        search_dir = os.path.join(BASE_DATASET_DIR, "2021" if dataset_name == "DF" else "2019")
        if not os.path.exists(search_dir): 
            search_dir = BASE_DATASET_DIR
        
        found = False
        for root, dirs, files in os.walk(search_dir):
            for file in files:
                if file.endswith(".txt"):
                    test_path = os.path.join(root, file)
                    try:
                        with open(test_path, 'r', encoding='utf-8') as f:
                            if 'bonafide' in f.read(50000).lower():
                                metadata_path = test_path
                                found = True
                                break
                    except:
                        continue
            if found: break
            
        if found:
            print(f"[+] Found label metadata at: {metadata_path}")
        else:
            raise FileNotFoundError("Could not find any protocol file containing 'bonafide' labels.")

    temp_path = os.path.join(ROOT_DIR, f"temp_fixed_protocol_{dataset_name}.txt")
    lines_to_write = []
    
    with open(metadata_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split()
            
            fname = parts[1] if len(parts) >= 2 else parts[0]
            label_str = "bonafide" if "bonafide" in line.lower() else "spoof"
            
            # Format exactly as 5 columns to satisfy your original dataset.py
            lines_to_write.append(f"DUMMY {fname} - - {label_str}\n")
            
    with open(temp_path, 'w') as f:
        f.writelines(lines_to_write)
        
    print("[+] Standardized protocol generated successfully.")
    return temp_path

# =====================================================================
# UNIFIED MODEL CLASSES 
# =====================================================================
class MetaLearner(nn.Module):
    def __init__(self, input_dim=616, hidden_dim=256, dropout=0.5):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, 2)
        
    def forward(self, feat_a, feat_r):
        x = torch.cat([feat_a, feat_r], dim=1)
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.drop(x)
        return self.fc2(x)

class BackboneWrapper(nn.Module):
    def __init__(self, backbone, fc_attr="fc"):
        super().__init__()
        self.backbone = backbone
        self.fc_attr = fc_attr
        self.fc = getattr(backbone, fc_attr)
        setattr(backbone, fc_attr, nn.Identity())
        
    def forward(self, x):
        feat = self.backbone(x)
        logits = self.fc(feat)
        return logits, feat

class CrossAttentionFuser_Baseline(nn.Module):
    def __init__(self, dim_a=104, dim_r=512, embed_dim=256, num_heads=8, num_classes=2, dropout=0.30):
        super().__init__()
        self.proj_a = nn.Sequential(nn.Linear(dim_a, embed_dim), nn.LayerNorm(embed_dim))
        self.proj_r = nn.Sequential(nn.Linear(dim_r, embed_dim), nn.LayerNorm(embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_emb = nn.Parameter(torch.zeros(1, 3, embed_dim))
        
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

        enc = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=embed_dim * 4, dropout=dropout,
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc, num_layers=2)
        
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, ea, er):
        ea = ea.float()
        er = er.float()
        B = ea.size(0)
        seq = torch.cat([
            self.cls_token.expand(B, -1, -1),
            self.proj_a(ea).unsqueeze(1),
            self.proj_r(er).unsqueeze(1),
        ], dim=1) + self.pos_emb
        return self.head(self.transformer(seq)[:, 0, :])

class CrossAttentionFuser_URFFL(nn.Module):
    def __init__(self, dim_a=104, dim_r=512, embed_dim=128, num_heads=4, num_classes=2, dropout=0.45, emb_dropout=0.20):
        super().__init__()
        self.emb_drop_a = nn.Dropout(emb_dropout)
        self.emb_drop_r = nn.Dropout(emb_dropout)
        
        self.proj_a = nn.Sequential(nn.Linear(dim_a, embed_dim), nn.LayerNorm(embed_dim))
        self.proj_r = nn.Sequential(nn.Linear(dim_r, embed_dim), nn.LayerNorm(embed_dim))
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_emb = nn.Parameter(torch.zeros(1, 3, embed_dim))
        
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=embed_dim * 4, dropout=dropout,
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=1)

        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, ea, er):
        ea = ea.float()
        er = er.float()
        ea = self.emb_drop_a(ea)
        er = self.emb_drop_r(er)
        B = ea.size(0)
        seq = torch.cat([
            self.cls_token.expand(B, -1, -1),
            self.proj_a(ea).unsqueeze(1),
            self.proj_r(er).unsqueeze(1),
        ], dim=1) + self.pos_emb
        out = self.transformer(seq)
        return self.head(out[:, 0])

class EndToEndEnsemble(nn.Module):
    def __init__(self, aasist, resnet, fuser, mode="baseline"):
        super().__init__()
        self.mode = mode
        if self.mode in ["urffl_crossattention", "baseline_crossattention"]:
            self.aasist_w = BackboneWrapper(aasist, fc_attr="fc")
            self.resnet_w = BackboneWrapper(resnet, fc_attr="fc")
        else:
            self.aasist = aasist
            self.resnet = resnet
            self.emb_a = [None]
            self.emb_r = [None]
            def hook_a(m, i, o): self.emb_a[0] = i[0]
            def hook_r(m, i, o): self.emb_r[0] = i[0]
            self.aasist.fc.register_forward_hook(hook_a)
            self.resnet.fc.register_forward_hook(hook_r)
            
        self.fusion_head = fuser

    def forward(self, wav, mel):
        if self.mode in ["urffl_crossattention", "baseline_crossattention"]:
            logits_a, feat_a = self.aasist_w(wav)
            logits_r, feat_r = self.resnet_w(mel)
            return self.fusion_head(feat_a, feat_r)
        else:
            out_a = self.aasist(wav)
            out_r = self.resnet(mel)
            return self.fusion_head(self.emb_a[0], self.emb_r[0])

# =====================================================================
# METRICS AND LOADER
# =====================================================================
def compute_min_dcf(fpr, fnr, thresholds, p_target=0.05, c_miss=1.0, c_fa=1.0):
    dcf_values = c_miss * fnr * p_target + c_fa * fpr * (1.0 - p_target)
    min_dcf_idx = np.argmin(dcf_values)
    min_dcf = dcf_values[min_dcf_idx]
    default_dcf = min(c_miss * p_target, c_fa * (1.0 - p_target))
    return min_dcf / default_dcf

def calculate_metrics(y_true, y_score):
    unique_classes = np.unique(y_true)
    y_pred = (np.array(y_score) >= 0.5).astype(int)
    accuracy = np.mean(np.array(y_true) == y_pred) * 100

    if len(unique_classes) < 2:
        return float('nan'), float('nan'), float('nan'), accuracy

    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    fnr = 1 - tpr
    
    eer_idx = np.nanargmin(np.absolute(fnr - fpr))
    eer = fpr[eer_idx] * 100
    
    auc_score = auc(fpr, tpr)
    min_dcf = compute_min_dcf(fpr, fnr, thresholds)
    
    return eer, auc_score, min_dcf, accuracy

def find_fold_path(fold_num):
    pattern = os.path.join(MODELS_DIR, f"*fold_{fold_num}.pth")
    matches = glob.glob(pattern)
    if not matches:
        return None
    for m in matches:
        if "crossattention" in m.lower():
            return m
    return matches[0]

def load_ensemble_fold(device, fold_num):
    selected_weights_path = find_fold_path(fold_num)
    if not selected_weights_path:
        raise FileNotFoundError(f"Could not find any weights for Fold {fold_num} in {MODELS_DIR}")

    aasist_model = AASIST(stft_window=698, stft_hop=398, freq_bins=116, gat_layers=2, heads=5, head_dim=104, hidden_dim=455, dropout=0.33).to(device)
    resnet_model = resnet18_simam(num_classes=2, dropout_rate=0.22).to(device)
    
    filename = os.path.basename(selected_weights_path).lower()
    
    if "urffl" in filename and "crossattention" in filename:
        mode = "urffl_crossattention"
        fusion_head = CrossAttentionFuser_URFFL().to(device)
    elif "crossattention" in filename:
        mode = "baseline_crossattention"
        fusion_head = CrossAttentionFuser_Baseline().to(device)
    else:
        mode = "meta"
        fusion_head = MetaLearner(input_dim=616).to(device)
    
    wrapper = EndToEndEnsemble(aasist_model, resnet_model, fusion_head, mode).to(device)
    checkpoint = torch.load(selected_weights_path, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint)

    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k
        if 'meta_learner' in new_key:
            new_key = new_key.replace('meta_learner', 'fusion_head')
            
        if mode in ["urffl_crossattention", "baseline_crossattention"]:
            if new_key.startswith('aasist.'):
                new_key = new_key.replace('aasist.', 'aasist_w.backbone.')
            if new_key.startswith('resnet.'):
                new_key = new_key.replace('resnet.', 'resnet_w.backbone.')
            
            if 'aasist_w.backbone.fc.' in new_key:
                new_key = new_key.replace('aasist_w.backbone.fc.', 'aasist_w.fc.')
            if 'resnet_w.backbone.fc.' in new_key:
                new_key = new_key.replace('resnet_w.backbone.fc.', 'resnet_w.fc.')
                
        new_state_dict[new_key] = v

    if mode in ["urffl_crossattention", "baseline_crossattention"]:
        head_0_weight = new_state_dict.get('fusion_head.head.0.weight')
        if head_0_weight is not None and len(head_0_weight.shape) == 2:
            embed_dim = head_0_weight.shape[1] 
            wrapper.fusion_head.head = torch.nn.Sequential(
                torch.nn.Linear(embed_dim, embed_dim),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.30),
                torch.nn.Linear(embed_dim, 2)
            ).to(device)
            
        if 'fusion_head.pos_emb' not in new_state_dict and hasattr(wrapper.fusion_head, 'pos_emb'):
            new_state_dict['fusion_head.pos_emb'] = wrapper.fusion_head.pos_emb.data

    wrapper.load_state_dict(new_state_dict)
    wrapper.eval()
    
    mel_transform = T.MelSpectrogram(sample_rate=16000, n_fft=512, hop_length=160, n_mels=80).to(device)
    amp_to_db = T.AmplitudeToDB(stype='power', top_db=80).to(device)
    
    return wrapper, mel_transform, amp_to_db, filename

# =====================================================================
# MAIN EVALUATION LOOP
# =====================================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*50}\n K-FOLD CROSS VALIDATION EVALUATION \n{'='*50}")
    print("[1] ASVspoof 2019 LA Evaluation")
    print("[2] ASVspoof 2021 DF Evaluation")
    
    choice = input("Select dataset to evaluate on (1 or 2): ").strip()
    
    if choice == '1':
        dataset_name = "LA"
        dataset_dir = PREPROCESSED_LA_DIR
        protocol_path = PROT_LA
    elif choice == '2':
        dataset_name = "DF"
        dataset_dir = PREPROCESSED_DF_DIR
        protocol_path = PROT_DF
    else:
        print("Invalid choice. Exiting.")
        return

    if not os.path.exists(dataset_dir):
        print(f"\n[!] ERROR: Directory does not exist: {dataset_dir}")
        return

    # Generate the standardized 5-column protocol file
    temp_protocol_path = create_unified_protocol(protocol_path, dataset_name)
    
    # Load dataset using your exact original ASVspoofDataset class
    dataset = ASVspoofDataset(dataset_dir, temp_protocol_path)

    if len(dataset) == 0:
        print(f"\n[!] ERROR: 0 audio files were loaded!")
        print(f"Check directory: {dataset_dir}")
        return
        
    print(f"\nSuccessfully loaded {len(dataset)} audio samples.")
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
    
    fold_eers = []
    fold_aucs = []
    fold_min_dcfs = []
    fold_accs = []
    
    print(f"\n🚀 Evaluating across 5 Folds on {dataset_name}...")
    
    for fold in range(1, 6):
        try:
            model, mel_t, a2db, fname = load_ensemble_fold(device, fold)
        except FileNotFoundError as e:
            print(f"Skipping Fold {fold}: {e}")
            continue
            
        y_true = []
        y_score = []
        
        with torch.no_grad():
            for wav, lbl in tqdm(loader, desc=f"Testing Fold {fold}/5", dynamic_ncols=True, leave=False):
                wav = wav.squeeze(1).to(device)
                mel = a2db(mel_t(wav)).unsqueeze(1)
                
                logits = model(wav, mel)
                probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
                
                y_true.extend(lbl.cpu().numpy().tolist())
                y_score.extend(probs.tolist())
                
        eer, auc_score, min_dcf, acc = calculate_metrics(y_true, y_score)
        
        if not np.isnan(eer): fold_eers.append(eer)
        if not np.isnan(auc_score): fold_aucs.append(auc_score)
        if not np.isnan(min_dcf): fold_min_dcfs.append(min_dcf)
        fold_accs.append(acc)
        
        print(f" -> Fold {fold} completed.")

    if len(fold_accs) > 0:
        print(f"\n{'='*50}")
        print(f" 📊 FINAL STATISTICAL STABILITY REPORT ({dataset_name})")
        print(f"{'='*50}")
        
        print("\n[ ACCURACY ]")
        for i, val in enumerate(fold_accs):
            print(f"Fold {i+1} = {val:.4f}%")
        print(f"\nAverage = {np.mean(fold_accs):.4f}%")
        print(f"Stdev   = {np.std(fold_accs):.4f}%")
        
        if len(fold_eers) > 0:
            print("\n[ EER - Equal Error Rate ]")
            for i, val in enumerate(fold_eers):
                print(f"Fold {i+1} = {val:.4f}%")
            print(f"\nAverage = {np.mean(fold_eers):.4f}%")
            print(f"Stdev   = {np.std(fold_eers):.4f}%")

            print("\n[ AUC - Area Under Curve ]")
            for i, val in enumerate(fold_aucs):
                print(f"Fold {i+1} = {val:.4f}")
            print(f"\nAverage = {np.mean(fold_aucs):.4f}")
            print(f"Stdev   = {np.std(fold_aucs):.4f}")

            print("\n[ MINDCF - Minimum Detection Cost Function ]")
            for i, val in enumerate(fold_min_dcfs):
                print(f"Fold {i+1} = {val:.4f}")
            print(f"\nAverage = {np.mean(fold_min_dcfs):.4f}")
            print(f"Stdev   = {np.std(fold_min_dcfs):.4f}")
            
            plt.figure(figsize=(8, 5))
            plt.boxplot(fold_eers, patch_artist=True, boxprops=dict(facecolor="steelblue"))
            plt.title(f"K-Fold Stability - {dataset_name} Dataset")
            plt.ylabel("Equal Error Rate (%)")
            plt.grid(True, linestyle=":", alpha=0.6)
            plt.xticks([1], [f"Mean EER: {np.mean(fold_eers):.2f}%"])
            plt.tight_layout()
            plt.savefig(os.path.join(RESULTS_DIR, f"kfold_stability_boxplot_{dataset_name.lower()}.png"), dpi=300)
            plt.close()
        else:
            print("\n [Note: EER, AUC, and minDCF could not be calculated because the dataset lacked 'bonafide' labels]")
            
        print(f"\n{'='*50}\n")
    else:
        print("\nEvaluation failed. No fold models were successfully tested.")

if __name__ == "__main__":
    main()