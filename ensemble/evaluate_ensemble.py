import sys
import os
import random
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, auc, DetCurveDisplay, confusion_matrix, ConfusionMatrixDisplay
from scipy.interpolate import interp1d
from scipy.optimize import brentq

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..')) if 'ensemble' in CURRENT_DIR else CURRENT_DIR
sys.path.append(ROOT_DIR)

RESULTS_DIR = os.path.join(ROOT_DIR, "results")
MODELS_DIR = os.path.join(ROOT_DIR, "saved_models")
os.makedirs(RESULTS_DIR, exist_ok=True)

from src.data.dataset import ASVspoofDataset
from src.models.aasist import AASIST
from src.models.resnet_simam import resnet18_simam

BASE_DATASET_DIR = r"D:\SAMPOERNA\Semester 8\Capstone\Dataset"

PREPROCESSED_LA_DIR = os.path.join(BASE_DATASET_DIR, "preprocessed_la")
PREPROCESSED_DF_DIR = os.path.join(BASE_DATASET_DIR, "preprocessed_df")

RAW_LA_EVAL_DIR = r"D:\SAMPOERNA\Semester 8\Capstone\Dataset\2019\LA\ASVspoof2019_LA_eval\flac"
PROTOCOL_LA_EVAL = r"D:\SAMPOERNA\Semester 8\Capstone\Dataset\2019\LA\ASVspoof2019_LA_cm_protocols\ASVspoof2019.LA.cm.eval.trl.txt"

RAW_DF_EVAL_DIR = r"D:\SAMPOERNA\Semester 8\Capstone\Dataset\2021\ASVspoof2021_DF_eval_part00\ASVspoof2021_DF_eval\flac"
PROTOCOL_DF_EVAL = r"D:\SAMPOERNA\Semester 8\Capstone\Dataset\2021\ASVspoof2021_DF_eval_part00\ASVspoof2021_DF_eval\trial_metadata.txt" 

# ── ARCHITECTURE DEFINITIONS ──────────────────────────────────────────────────

class MetaLearner(nn.Module):
    def __init__(self, input_dim=616, hidden_dim=256, num_classes=2, dropout=0.3):
        super(MetaLearner, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, emb_aasist, emb_resnet):
        if emb_aasist.dtype != emb_resnet.dtype:
            emb_aasist = emb_aasist.to(emb_resnet.dtype)
        x = torch.cat([emb_aasist, emb_resnet], dim=1)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        return self.fc2(x)

class BackboneWrapper(nn.Module):
    def __init__(self, backbone: nn.Module, fc_attr: str = "fc"):
        super().__init__()
        self.backbone = backbone
        self._emb     = None
        fc = getattr(backbone, fc_attr, None)
        self._handle = fc.register_forward_hook(self._capture)

    def _capture(self, module, inp, out):
        self._emb = inp[0] 

    def forward(self, x):
        self._emb = None
        logit     = self.backbone(x.float())
        emb       = self._emb
        self._emb = None
        return logit, emb


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

# Baseline Fuser (No PosEmb, No Initial LayerNorm in Head)
class CrossAttentionFuser_URFFL(nn.Module):
    def __init__(self, dim_a=104, dim_r=512, embed_dim=256, num_heads=8, num_classes=2, dropout=0.30):
        super().__init__()
        self.proj_a    = nn.Sequential(nn.Linear(dim_a, embed_dim), nn.LayerNorm(embed_dim))
        self.proj_r    = nn.Sequential(nn.Linear(dim_r, embed_dim), nn.LayerNorm(embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_emb   = nn.Parameter(torch.zeros(1, 3, embed_dim))
        
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
        ea = ea.float(); er = er.float()
        B  = ea.size(0)
        seq = torch.cat([
            self.cls_token.expand(B, -1, -1),
            self.proj_a(ea).unsqueeze(1),
            self.proj_r(er).unsqueeze(1),
        ], dim=1) + self.pos_emb
        return self.head(self.transformer(seq)[:, 0, :])

class EndToEndEnsemble(nn.Module):
    def __init__(self, aasist, resnet, fuser, mode="baseline"):
        super().__init__()
        self.mode = mode
        if self.mode == "urffl_crossattention":
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
        if self.mode == "urffl_crossattention":
            _, ea = self.aasist_w(wav.float())
            _, er = self.resnet_w(mel.float())
            return self.fusion_head(ea, er)
        else:
            _ = self.aasist(wav)
            with torch.amp.autocast('cuda'):
                _ = self.resnet(mel)
                return self.fusion_head(self.emb_a[0], self.emb_r[0])

# ── METRICS & DATA ────────────────────────────────────────────────────────────

def compute_eer(y_true, y_scores):
    try:
        fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
        fnr = 1. - tpr
        eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
        thresh = interp1d(fpr, thresholds)(eer)
    except Exception:
        fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
        fnr = 1. - tpr
        idx = np.nanargmin(np.abs(fnr - fpr))
        eer = float((fpr[idx] + fnr[idx]) / 2.)
        thresh = thresholds[idx]
    return float(eer * 100.), thresh

def compute_min_dcf(fpr, fnr, thresholds, p_target=0.05, c_miss=1.0, c_fa=1.0):
    dcf = c_miss * fnr * p_target + c_fa * fpr * (1.0 - p_target)
    min_dcf_idx = np.argmin(dcf)
    min_dcf = dcf[min_dcf_idx]
    default_dcf = min(c_miss * p_target, c_fa * (1.0 - p_target))
    min_dcf_norm = min_dcf / default_dcf
    return min_dcf_norm, thresholds[min_dcf_idx]

def create_shuffled_protocol(original_protocol, target_protocol):
    bonafide_lines = []
    spoof_lines = []
    with open(original_protocol, 'r') as f:
        for line in f:
            line_str = line.strip()
            if not line_str: continue
            parts = line_str.split()
            if len(parts) < 2: continue
            speaker = parts[0]
            fname = parts[1]
            is_bonafide = 'bonafide' in line_str.lower()
            label_str = 'bonafide' if is_bonafide else 'spoof'
            std_line = f"{speaker} {fname} - - {label_str}"
            if is_bonafide: bonafide_lines.append(std_line)
            else: spoof_lines.append(std_line)
                
    random.seed(42)
    random.shuffle(bonafide_lines)
    random.shuffle(spoof_lines)
    
    interleaved = []
    max_len = max(len(bonafide_lines), len(spoof_lines))
    for i in range(max_len):
        if i < len(bonafide_lines): interleaved.append(bonafide_lines[i])
        if i < len(spoof_lines): interleaved.append(spoof_lines[i])
            
    with open(target_protocol, 'w') as f:
        for line in interleaved: f.write(line + '\n')

def apply_vad_and_norm(waveform):
    original_waveform = waveform.clone()
    try:
        waveform = torchaudio.functional.vad(waveform, sample_rate=16000)
        if waveform.shape[-1] <= 1: waveform = original_waveform
    except Exception:
        waveform = original_waveform
    mean = waveform.mean()
    std = waveform.std()
    waveform = (waveform - mean) / (std + 1e-8)
    return waveform

def apply_preemphasis(waveform):
    alpha = 0.97
    return torch.cat([waveform[:, :1], waveform[:, 1:] - alpha * waveform[:, :-1]], dim=1)

def preprocess_evaluation(eval_dir, protocol_file, target_dir, target_total=7000, target_length=64600):
    with open(protocol_file, 'r') as f:
        lines = f.readlines()
                
    valid_lines = []
    success_bonafide = 0
    success_spoof = 0
    pbar = tqdm(total=target_total, desc="Preprocessing Eval Dataset")
    
    for line in lines:
        if (success_bonafide + success_spoof) >= target_total: break
        parts = line.strip().split()
        if len(parts) < 5: continue
            
        fname = parts[1]
        is_bonafide = (parts[4] == 'bonafide')
        file_path = os.path.join(eval_dir, f"{fname}.flac")
        if not os.path.exists(file_path): continue
            
        try:
            waveform, _ = torchaudio.load(file_path)
        except Exception:
            continue
            
        waveform = apply_vad_and_norm(waveform)
        seq_len = waveform.shape[-1]
        
        if seq_len > target_length:
            start = (seq_len - target_length) // 2
            waveform = waveform[:, start:start + target_length]
        elif seq_len < target_length:
            if seq_len <= 1:
                waveform = F.pad(waveform, (0, target_length - seq_len), mode='constant', value=0)
            else:
                while waveform.shape[-1] < target_length:
                    current_len = waveform.shape[-1]
                    pad_amount = min(target_length - current_len, current_len - 1)
                    waveform = F.pad(waveform, (0, pad_amount), mode='reflect')
                    
        pre_emphasized = apply_preemphasis(waveform)
        torch.save(pre_emphasized, os.path.join(target_dir, f"{fname}.pt"))
        valid_lines.append(line)
        
        if is_bonafide: success_bonafide += 1
        else: success_spoof += 1
        pbar.update(1)
        
    pbar.close()
    with open(protocol_file, 'w') as f:
        for line in valid_lines: f.write(line)

def load_ensemble(device, selected_weights_path):
    print(f"\nLoading End-to-End Ensemble from {selected_weights_path}...")
    
    # Base Models with MAIN branch dimensions
    aasist_model = AASIST(stft_window=698, stft_hop=398, freq_bins=116, gat_layers=2, heads=5, head_dim=104, hidden_dim=455, dropout=0.33).to(device)
    resnet_model = resnet18_simam(num_classes=2, dropout_rate=0.22).to(device)
    
    filename = os.path.basename(selected_weights_path).lower()
    
    if "urffl" in filename and "crossattention" in filename:
        print("-> Detected UR-FFL Cross-Attention Architecture.")
        mode = "urffl_crossattention"
        fusion_head = CrossAttentionFuser_URFFL().to(device)
    elif "crossattention" in filename:
        print("-> Detected Baseline Cross-Attention Architecture.")
        mode = "baseline_crossattention"
        fusion_head = CrossAttentionFuser_Baseline().to(device)
    else:
        print("-> Detected Standard MLP Meta-Learner head.")
        mode = "meta"
        fusion_head = MetaLearner(input_dim=616).to(device)
    
    wrapper = EndToEndEnsemble(aasist_model, resnet_model, fusion_head, mode).to(device)
    
    checkpoint = torch.load(selected_weights_path, map_location=device)
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint

    # 1. Map legacy weight prefixes dynamically
    new_state_dict = {}
    for k, v in state_dict.items():
        if 'meta_learner' in k:
            k = k.replace('meta_learner', 'fusion_head')
            
        # Handle legacy weights without BackboneWrapper (aasist -> aasist_w.backbone)
        if mode == "urffl_crossattention":
            if k.startswith('aasist.'):
                k = k.replace('aasist.', 'aasist_w.backbone.')
            if k.startswith('resnet.'):
                k = k.replace('resnet.', 'resnet_w.backbone.')
                
        new_state_dict[k] = v

    # 2. Patch Fusion Head architecture mismatches for legacy weights
    if mode in ["urffl_crossattention", "baseline_crossattention"]:
        head_0_weight = new_state_dict.get('fusion_head.head.0.weight')
        
        # If the checkpoint's head.0 is a 2D Linear layer instead of a 1D LayerNorm
        if head_0_weight is not None and len(head_0_weight.shape) == 2:
            print("-> [Legacy Checkpoint] Rebuilding Cross-Attention head to match old architecture.")
            embed_dim = head_0_weight.shape[1] # Auto-detect legacy dimension (e.g., 256)
            
            # Rebuild the head without LayerNorm to match the old saved weights perfectly
            wrapper.fusion_head.head = torch.nn.Sequential(
                torch.nn.Linear(embed_dim, embed_dim),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.30),
                torch.nn.Linear(embed_dim, 2)
            ).to(device)
            
        # If checkpoint is missing the newly added Positional Embeddings, inject dummy zeroes
        if 'fusion_head.pos_emb' not in new_state_dict and hasattr(wrapper.fusion_head, 'pos_emb'):
            print("-> [Legacy Checkpoint] Injecting missing positional embeddings.")
            new_state_dict['fusion_head.pos_emb'] = wrapper.fusion_head.pos_emb.data

    # Load the patched dictionary into the dynamically adjusted model
    wrapper.load_state_dict(new_state_dict)
    wrapper.eval()
    
    mel_transform = T.MelSpectrogram(sample_rate=16000, n_fft=512, hop_length=160, n_mels=80).to(device)
    amp_to_db = T.AmplitudeToDB(stype='power', top_db=80).to(device)
    
    return wrapper, mel_transform, amp_to_db

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    pth_files = glob.glob(os.path.join(MODELS_DIR, '*ensemble*.pth'))
    if not pth_files:
        print(f"No Ensemble .pth files found in {MODELS_DIR}.")
        return
        
    print("========================================")
    print("AVAILABLE ENSEMBLE MODELS")
    print("========================================")
    for i, file_path in enumerate(pth_files):
        print(f"[{i+1}] {os.path.basename(file_path)}")
        
    while True:
        try:
            model_choice = int(input("\nSelect the model to evaluate (number): ").strip())
            if 1 <= model_choice <= len(pth_files):
                selected_weights = pth_files[model_choice - 1]
                break
            else:
                print("Invalid selection.")
        except ValueError:
            print("Invalid input.")

    print("\n========================================")
    print("ENSEMBLE EVALUATION SCRIPT")
    print("========================================")
    print("[1] Official ASVspoof 2019 LA Evaluation (Balanced Subset)")
    print("[2] Official ASVspoof 2021 DF Evaluation (Balanced Subset)")
    
    choice = input("Select mode (1 or 2): ").strip()
    
    os.makedirs(PREPROCESSED_LA_DIR, exist_ok=True)
    os.makedirs(PREPROCESSED_DF_DIR, exist_ok=True)

    wrapper, mel_transform, amp_to_db = load_ensemble(device, selected_weights)

    if choice == '1':
        print("\n--- OFFICIAL ASVSPOOF 2019 LA EVALUATION ---")
        raw_dir = RAW_LA_EVAL_DIR
        orig_protocol = PROTOCOL_LA_EVAL
        dataset_name = "LA"
        target_dir = PREPROCESSED_LA_DIR
    elif choice == '2':
        print("\n--- OFFICIAL ASVSPOOF 2021 DF EVALUATION ---")
        raw_dir = RAW_DF_EVAL_DIR
        orig_protocol = PROTOCOL_DF_EVAL
        dataset_name = "DF"
        target_dir = PREPROCESSED_DF_DIR
    else:
        print("Invalid selection.")
        return
        
    subset_protocol = os.path.join(target_dir, "subset_protocol.txt")
    existing_files = glob.glob(os.path.join(target_dir, '*.pt'))
    
    if len(existing_files) >= 7000 and os.path.exists(subset_protocol):
        print(f"Found existing evaluation dataset for {dataset_name}.")
    else:
        print(f"Clearing folder and preprocessing evaluation dataset subset for {dataset_name}...")
        for f in existing_files:
            os.remove(f)
        create_shuffled_protocol(orig_protocol, subset_protocol)
        preprocess_evaluation(raw_dir, subset_protocol, target_dir, target_total=7000)
        
    print("Executing Inference Engine...")
    eval_dataset = ASVspoofDataset(target_dir, subset_protocol)
    eval_loader = DataLoader(eval_dataset, batch_size=64, shuffle=False, num_workers=4)
    
    all_labels = []
    all_probs = []
    all_preds = []
    
    with torch.no_grad():
        for waveforms, labels in tqdm(eval_loader, desc="Evaluating Benchmarks"):
            waveforms = waveforms.squeeze(1).to(device)
            labels = labels.to(device)
            
            mel = mel_transform(waveforms)
            mel_db = amp_to_db(mel).unsqueeze(1)
            
            outputs = wrapper(waveforms, mel_db)
                
            probs = torch.softmax(outputs, dim=1)[:, 1]
            _, predicted_classes = torch.max(outputs.data, 1)
            
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(predicted_classes.cpu().numpy())
            
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)
    
    total_samples = len(all_labels)
    correct_predictions = (all_preds == all_labels).sum()
    accuracy = (correct_predictions / total_samples) * 100
    
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs, pos_label=1)
    fnr = 1 - tpr
    auc_score = auc(fpr, tpr)
    
    eer, eer_threshold = compute_eer(all_labels, all_probs)
    min_dcf_norm, dcf_threshold = compute_min_dcf(fpr, fnr, thresholds)
    
    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    deepfake_scores = all_probs[all_labels == 0]
    bonafide_scores = all_probs[all_labels == 1]
    
    print("\n========================================")
    print("AI DIAGNOSTIC REPORT")
    print("========================================")
    print("--- CONFUSION MATRIX (At Default 0.5 Threshold) ---")
    print(f"True Positives (Correct Spoof):  {tp}")
    print(f"True Negatives (Correct Bonafide): {tn}")
    print(f"False Positives (False Spoof):   {fp}")
    print(f"False Negatives (Missed Spoof):  {fn}")
    print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1-Score: {f1:.4f}")
    
    print("\n--- PERFORMANCE THRESHOLDS ---")
    print(f"Equal Error Rate (EER) Threshold: {eer_threshold:.4f}")
    print(f"Min DCF Threshold:                {dcf_threshold:.4f}")
    
    print("\n--- SCORE DISTRIBUTIONS ---")
    print(f"Mean Bonafide Score: {bonafide_scores.mean():.4f} (std: {bonafide_scores.std():.4f})")
    print(f"Mean Deepfake Score: {deepfake_scores.mean():.4f} (std: {deepfake_scores.std():.4f})")
    print(f"Separation Margin:   {abs(bonafide_scores.mean() - deepfake_scores.mean()):.4f}")
    
    print("\n========================================")
    print("FINAL EVALUATION METRICS")
    print("========================================")
    print(f"Average Accuracy: {accuracy:.4f}%")
    print(f"Equal Error Rate (EER): {eer:.4f}%")
    print(f"Area Under the Curve (AUC): {auc_score:.4f}")
    print(f"Normalized Minimum DCF (t-DCF Proxy): {min_dcf_norm:.4f}")
    print("========================================\n")
    
    # ── PLOTTING ─────────────────────────────────────────────────────────────────
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FAR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title(f'ROC Curve ({dataset_name} Evaluation)')
    plt.legend(loc="lower right")
    plt.grid(True, linestyle=':', alpha=0.6)
    roc_path = os.path.join(RESULTS_DIR, f"crossattention_roc_curve_{dataset_name.lower()}.png")
    plt.savefig(roc_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    fig, ax = plt.subplots(figsize=(7, 6))
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Deepfake', 'Bonafide'])
    cm_display.plot(cmap=plt.cm.Blues, ax=ax, values_format='d')
    plt.title(f'Confusion Matrix ({dataset_name})', fontsize=12, pad=15)
    cm_path = os.path.join(RESULTS_DIR, f"crossattention_confusion_matrix_{dataset_name.lower()}.png")
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(10, 6))
    plt.hist(bonafide_scores, bins=50, alpha=0.6, color='blue', density=True, label='Bonafide (Genuine)')
    plt.hist(deepfake_scores, bins=50, alpha=0.6, color='red', density=True, label='Spoof (Deepfake)')
    plt.axvline(eer_threshold, color='black', linestyle='dashed', linewidth=2, label=f'EER Threshold ({eer_threshold:.2f})')
    plt.title(f'Model Confidence Score Distribution ({dataset_name})')
    plt.xlabel('Predicted Probability of being Bonafide')
    plt.ylabel('Density')
    plt.legend(loc='upper center')
    plt.grid(True, linestyle=':', alpha=0.6)
    dist_path = os.path.join(RESULTS_DIR, f"crossattention_score_distribution_{dataset_name.lower()}.png")
    plt.savefig(dist_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    fig, ax = plt.subplots(figsize=(8, 8))
    DetCurveDisplay(fpr=fpr, fnr=fnr, estimator_name=f"Ensemble (EER={eer:.2f}%)").plot(ax=ax)
    plt.title(f'Detection Error Tradeoff (DET) Curve - {dataset_name}')
    plt.grid(True, linestyle=':', alpha=0.6)
    det_path = os.path.join(RESULTS_DIR, f"crossattention_det_curve_{dataset_name.lower()}.png")
    plt.savefig(det_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Graph 5: Min DCF Curve
    p_target = 0.05
    c_miss = 1.0
    c_fa = 1.0
    
    # Recalculate DCF curve points for plotting
    dcf_values = c_miss * fnr * p_target + c_fa * fpr * (1.0 - p_target)
    default_dcf = min(c_miss * p_target, c_fa * (1.0 - p_target))
    norm_dcf_values = dcf_values / default_dcf
    
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, norm_dcf_values, color='purple', lw=2, label='Normalized DCF')
    plt.axvline(dcf_threshold, color='red', linestyle='--', label=f'Min DCF Threshold ({dcf_threshold:.4f})')
    plt.plot(dcf_threshold, min_dcf_norm, 'ro')
    
    # Add text label slightly offset from the minimum point
    plt.text(dcf_threshold + 0.02, min_dcf_norm + 0.02, f'Min t-DCF = {min_dcf_norm:.4f}', color='red', fontweight='bold')
    
    plt.xlim([0.0, 1.0])
    # Dynamically scale Y-axis so the curve fits nicely without zooming out too far
    plt.ylim([0.0, max(1.2, np.max(norm_dcf_values[(thresholds >= 0.0) & (thresholds <= 1.0)]) * 1.1)])
    plt.xlabel('Detection Threshold')
    plt.ylabel('Normalized DCF (Lower is Better)')
    plt.title(f'Detection Cost Function (DCF) Curve - {dataset_name}')
    plt.legend(loc="upper right")
    plt.grid(True, linestyle=':', alpha=0.6)
    
    dcf_path = os.path.join(RESULTS_DIR, f"crossattention_dcf_curve_{dataset_name.lower()}.png")
    plt.savefig(dcf_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Successfully generated and saved 4 Diagnostic Graphics to {RESULTS_DIR}")

if __name__ == "__main__":
    main()