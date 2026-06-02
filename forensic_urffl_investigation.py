import sys
import os
import torch
import torch.nn as nn
import numpy as np
import torchaudio.transforms as T
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import roc_curve
from collections import Counter
import random
from tqdm import tqdm
import warnings
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(CURRENT_DIR) in ['ensemble', 'src']:
    ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
else:
    ROOT_DIR = CURRENT_DIR
sys.path.append(ROOT_DIR)

from src.data.dataset import ASVspoofDataset
from src.models.aasist import AASIST
from src.models.resnet_simam import resnet18_simam
from src.ur_ffl.selector import DegradationSelector

# ── EVALUATION PATHS ─────────────
PREPROCESSED_EVAL_DIR = r"D:\SAMPOERNA\Semester 8\Capstone\Dataset\preprocessed_la"
PROTOCOL_EVAL = r"D:\SAMPOERNA\Semester 8\Capstone\Dataset\preprocessed_la\subset_protocol.txt"

RESULTS_DIR = os.path.join(ROOT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

MODELS_DIR = os.path.join(ROOT_DIR, "saved_models")
AASIST_WEIGHTS = os.path.join(MODELS_DIR, "aasist_unified_best.pth")
RESNET_WEIGHTS = os.path.join(MODELS_DIR, "resnet_unified_best.pth")

def get_subset(dataset, size=1500):
    size = min(size, len(dataset))
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    return Subset(dataset, indices[:size])

def simulate_historical_mc_sensor(model, inputs, passes=10):
    """Recreates the exact MC Dropout conditions present during training."""
    model.eval()
    for m in model.modules():
        if m.__class__.__name__.startswith("Dropout"):
            m.train()

    with torch.no_grad():
        probs_list = []
        for _ in range(passes):
            logits = model(inputs)
            p = torch.softmax(logits, dim=1)[:, 1]
            probs_list.append(p.unsqueeze(0))

    probs = torch.cat(probs_list, dim=0)
    mu = probs.mean(dim=0)
    
    eps = 1e-8
    mu_clamped = torch.clamp(mu, eps, 1.0 - eps)
    H = -mu_clamped * torch.log(mu_clamped) - (1.0 - mu_clamped) * torch.log(1.0 - mu_clamped)
    return H

def plot_forensic_graphs(model_name, z_u_array, all_selections, bonafide_probs, spoof_probs, opt_thresh):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    axes[0].hist(z_u_array, bins=30, color='purple', alpha=0.7, edgecolor='black')
    axes[0].set_title(f"{model_name.upper()} Historical Uncertainty ($z_u$)")
    axes[0].set_xlabel("Predictive Entropy (nats)")
    axes[0].axvline(0.10, color='red', linestyle='--', label="Smear (<0.1)")
    axes[0].axvline(0.30, color='orange', linestyle='--', label="Codec (<0.3)")
    axes[0].axvline(0.50, color='green', linestyle='--', label="Flatten (<0.5)")
    axes[0].axvline(0.60, color='blue', linestyle='--', label="Noise (<0.6)")
    axes[0].legend()

    sel_counts = Counter(all_selections)
    labels = ['Clean', 'Smear (LnL)', 'Codec (ISD)', 'Flatten (SSI)', 'Noise (Mild SSI)']
    keys = ['clean', 'smear', 'codec', 'flatten', 'noise']
    sizes = [sel_counts.get(k, 0) for k in keys]
    colors = ['#2ca02c', '#d62728', '#ff7f0e', '#1f77b4', '#9467bd']
    
    filtered_sizes = [s for s in sizes if s > 0]
    filtered_labels = [l for s, l in zip(sizes, labels) if s > 0]
    filtered_colors = [c for s, c in zip(sizes, colors) if s > 0]
    
    if filtered_sizes:
        axes[1].pie(filtered_sizes, labels=filtered_labels, colors=filtered_colors, autopct='%1.1f%%', startangle=140)
        axes[1].set_title(f"Historical Training Degradations")
    else:
        axes[1].text(0.5, 0.5, 'No Selections', ha='center', va='center')

    axes[2].hist(bonafide_probs, bins=40, alpha=0.6, color='blue', density=True, label='Bonafide')
    axes[2].hist(spoof_probs, bins=40, alpha=0.6, color='red', density=True, label='Spoof')
    axes[2].axvline(0.50, color='black', linestyle='-', linewidth=2, label="0.50 Thresh")
    axes[2].axvline(opt_thresh, color='gold', linestyle='--', linewidth=2, label=f"Opt Thresh ({opt_thresh:.2f})")
    axes[2].set_title(f"Score Hedging & Compression Effect")
    axes[2].set_xlabel("Predicted Probability")
    axes[2].legend()

    plt.tight_layout()
    save_path = os.path.join(RESULTS_DIR, f"{model_name}_forensic_investigation.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    return save_path

def investigate_model(model_name, model, dataloader, device):
    print(f"\n{'='*60}")
    print(f" FORENSIC ANALYSIS: {model_name.upper()} ON LA EVAL DATASET")
    print(f"{'='*60}")
    
    selector = DegradationSelector()
    mel_transform = T.MelSpectrogram(sample_rate=16000, n_fft=512, hop_length=160, n_mels=80).to(device)
    amp_to_db = T.AmplitudeToDB(stype="power", top_db=80).to(device)
    
    all_zu, all_selections, all_probs, all_labels = [], [], [], []
    
    for waveforms, labels in tqdm(dataloader, desc=f"Profiling {model_name}"):
        waveforms = waveforms.squeeze(1).to(device)
        labels = labels.to(device)
        
        # 1. Format Inputs and FIX NaN by clamping mel
        with torch.no_grad():
            if model_name == 'resnet':
                mel = mel_transform(waveforms)
                mel = torch.clamp(mel, min=1e-8) 
                inputs = amp_to_db(mel).unsqueeze(1)
            else:
                inputs = waveforms
        
        # 2. Simulate historical MC Dropout Sensor
        z_u = simulate_historical_mc_sensor(model, inputs)
        all_zu.extend(z_u.cpu().numpy())
        
        # 3. Simulate Selector
        selections = selector.select(z_u)
        all_selections.extend(selections if isinstance(selections, list) else selections.cpu().numpy())
        
        # 4. Get standard eval probabilities
        model.eval()
        with torch.no_grad():
            logits = model(inputs)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy())

    fpr, tpr, thresholds = roc_curve(all_labels, all_probs, pos_label=1)
    idx = np.nanargmin(np.abs((1 - tpr) - fpr))
    eer = (fpr[idx] + (1 - tpr)[idx]) / 2.0 * 100
    opt_thresh = thresholds[idx]
    
    acc_def = (np.array(all_probs) > 0.5).astype(int) == np.array(all_labels)
    acc_opt = (np.array(all_probs) > opt_thresh).astype(int) == np.array(all_labels)
    
    print(f"\n[1] EER vs ACCURACY PARADOX")
    print(f"Equal Error Rate:      {eer:.2f}% (Ranking capability remains strong)")
    print(f"Accuracy @ 0.50 Thresh:{acc_def.mean()*100:.2f}% (Ruined by score compression)")
    print(f"Accuracy @ Opt Thresh: {acc_opt.mean()*100:.2f}% (Recovered capability @ {opt_thresh:.4f})")
    
    graph_path = plot_forensic_graphs(model_name, np.array(all_zu), all_selections, 
                                      np.array(all_probs)[np.array(all_labels)==1], 
                                      np.array(all_probs)[np.array(all_labels)==0], opt_thresh)
    print(f"\n=> Forensic graphs generated: {graph_path}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading EVALUATION Dataset subset...")
    dataset = ASVspoofDataset(PREPROCESSED_EVAL_DIR, PROTOCOL_EVAL)
    dataloader = DataLoader(get_subset(dataset, 1500), batch_size=32, shuffle=False, num_workers=2)

    aasist = AASIST(stft_window=698, stft_hop=398, freq_bins=116, gat_layers=2, heads=5, head_dim=104, hidden_dim=455, dropout=0.33).to(device)
    if os.path.exists(AASIST_WEIGHTS):
        ckpt_a = torch.load(AASIST_WEIGHTS, map_location=device)
        aasist.load_state_dict(ckpt_a['model_state_dict'] if 'model_state_dict' in ckpt_a else ckpt_a)
    investigate_model("aasist", aasist, dataloader, device)
    
    resnet = resnet18_simam(num_classes=2, dropout_rate=0.22).to(device)
    if os.path.exists(RESNET_WEIGHTS):
        ckpt_r = torch.load(RESNET_WEIGHTS, map_location=device)
        resnet.load_state_dict(ckpt_r['model_state_dict'] if 'model_state_dict' in ckpt_r else ckpt_r)
    investigate_model("resnet", resnet, dataloader, device)

if __name__ == "__main__":
    main()