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

warnings.filterwarnings("ignore", category=UserWarning)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(CURRENT_DIR) in ['ensemble', 'src']:
    ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
else:
    ROOT_DIR = CURRENT_DIR
sys.path.append(ROOT_DIR)

from src.data.dataset import ASVspoofDataset
from src.models.aasist import AASIST
from src.models.resnet_simam import resnet18_simam

# ── PATHS ─────────────
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
            # Safe catch for NaN logits (prevents math errors from corrupted weights)
            if torch.isnan(logits).any():
                logits = torch.nan_to_num(logits, nan=0.0)
            p = torch.softmax(logits, dim=1)[:, 1]
            probs_list.append(p.unsqueeze(0))

    probs = torch.cat(probs_list, dim=0)
    mu = probs.mean(dim=0)
    
    eps = 1e-8
    mu_clamped = torch.clamp(mu, eps, 1.0 - eps)
    H = -mu_clamped * torch.log(mu_clamped) - (1.0 - mu_clamped) * torch.log(1.0 - mu_clamped)
    return H

def map_zscores_to_selections(z_scores):
    selections = []
    for z in z_scores:
        if z < -1.5: selections.append("smear")
        elif z < -0.5: selections.append("codec")
        elif z < 0.5: selections.append("flatten")
        elif z < 1.5: selections.append("noise")
        else: selections.append("clean")
    return selections

def plot_zscore_investigation(model_name, z_u_array, z_scores, all_selections):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. Z-Score Histogram
    axes[0].hist(z_scores, bins=30, color='teal', alpha=0.7, edgecolor='black')
    axes[0].set_title(f"{model_name.upper()} Z-Score Entropy Distribution")
    axes[0].set_xlabel("Z-Score (Standardized Uncertainty)")
    axes[0].set_ylabel("Frequency")
    axes[0].axvline(-1.5, color='red', linestyle='--', label="Smear (< -1.5)")
    axes[0].axvline(-0.5, color='orange', linestyle='--', label="Codec (< -0.5)")
    axes[0].axvline(0.5, color='green', linestyle='--', label="Flatten (< 0.5)")
    axes[0].axvline(1.5, color='blue', linestyle='--', label="Noise (< 1.5)")
    axes[0].legend()

    # 2. Selection Pie Chart
    sel_counts = Counter(all_selections)
    labels = ['Clean', 'Smear (LnL)', 'Codec (ISD)', 'Flatten (SSI)', 'Noise (SSI)']
    keys = ['clean', 'smear', 'codec', 'flatten', 'noise']
    sizes = [sel_counts.get(k, 0) for k in keys]
    colors = ['#2ca02c', '#d62728', '#ff7f0e', '#1f77b4', '#9467bd']
    
    filtered_sizes = [s for s in sizes if s > 0]
    filtered_labels = [l for s, l in zip(sizes, labels) if s > 0]
    filtered_colors = [c for s, c in zip(sizes, colors) if s > 0]
    
    if filtered_sizes:
        axes[1].pie(filtered_sizes, labels=filtered_labels, colors=filtered_colors, autopct='%1.1f%%', startangle=140)
        axes[1].set_title(f"Dynamic Degradation Assignment")
    else:
        axes[1].text(0.5, 0.5, 'No Selections', ha='center', va='center')

    plt.tight_layout()
    save_path = os.path.join(RESULTS_DIR, f"{model_name}_zscore_investigation.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    return save_path

def investigate_model(model_name, model, dataloader, device):
    print(f"\n{'='*60}")
    print(f" Z-SCORE SENSITIVITY ANALYSIS: {model_name.upper()}")
    print(f"{'='*60}")
    
    mel_transform = T.MelSpectrogram(sample_rate=16000, n_fft=512, hop_length=160, n_mels=80).to(device)
    amp_to_db = T.AmplitudeToDB(stype="power", top_db=80).to(device)
    
    all_zu = []
    
    for waveforms, labels in tqdm(dataloader, desc=f"Profiling MC Entropy"):
        waveforms = waveforms.squeeze(1).to(device)
        
        with torch.no_grad():
            if model_name == 'resnet':
                mel = mel_transform(waveforms)
                mel = torch.clamp(mel, min=1e-8) 
                inputs = amp_to_db(mel).unsqueeze(1)
            else:
                inputs = waveforms
        
        # Simulate MC Dropout Sensor
        z_u = simulate_historical_mc_sensor(model, inputs)
        all_zu.extend(z_u.cpu().numpy())
        
    # Clean up NaNs in the array
    z_u_array = np.array(all_zu)
    if np.isnan(z_u_array).any():
        print("  [WARNING] NaN values detected in entropy outputs. Cleaning array...")
        valid_mean = np.nanmean(z_u_array)
        z_u_array = np.nan_to_num(z_u_array, nan=valid_mean if not np.isnan(valid_mean) else 0.0)

    std_zu = np.std(z_u_array)
    mean_zu = np.mean(z_u_array)
    
    # Calculate Z-Scores safely
    if std_zu < 1e-6 or np.isnan(std_zu):
        z_scores = np.zeros_like(z_u_array)
    else:
        z_scores = (z_u_array - mean_zu) / std_zu
        
    all_selections = map_zscores_to_selections(z_scores)
    sel_counts = Counter(all_selections)
    total = len(all_selections)
    
    print("\n[1] ARCHITECTURAL ENTROPY BASELINE")
    print(f"  Mean Entropy (nats): {mean_zu:.4f}")
    print(f"  Std Deviation:       {std_zu:.4f}")
    if std_zu < 1e-6 or np.isnan(std_zu):
        print("  WARNING: This model exhibits zero variance (likely due to corrupted weights).")
        print("  Z-Scoring cannot be performed safely. Defaulting to 'Flatten'.")
    
    print("\n[2] ADAPTIVE Z-SCORE DEGRADATION DISTRIBUTION")
    print("This shows the fair, relative degradation assignment:")
    print(f"  Smear (LnL)     [Z < -1.5]: {sel_counts.get('smear', 0) / total * 100:.1f}%")
    print(f"  Codec (ISD)     [Z < -0.5]: {sel_counts.get('codec', 0) / total * 100:.1f}%")
    print(f"  Flatten (SSI)   [Z < 0.5] : {sel_counts.get('flatten', 0) / total * 100:.1f}%")
    print(f"  Noise (SSI)     [Z < 1.5] : {sel_counts.get('noise', 0) / total * 100:.1f}%")
    print(f"  Clean Audio     [Z > 1.5] : {sel_counts.get('clean', 0) / total * 100:.1f}%")
    
    print("\n=> METHODOLOGY JUSTIFICATION:")
    print(f"By standardizing the Epistemic Uncertainty (MC Dropout) into Z-Scores, the")
    print(f"Degradation Selector normalizes the architectural biases of {model_name.upper()}.")
    print(f"The model is now punished or protected relative to its own baseline variance,")
    print(f"ensuring fair curriculum learning across completely different architectures.")

    # Generate Plots
    graph_path = plot_zscore_investigation(model_name, z_u_array, z_scores, all_selections)
    print(f"\n=> Diagnostic graphs generated: {graph_path}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading EVALUATION Dataset subset...")
    try:
        dataset = ASVspoofDataset(PREPROCESSED_EVAL_DIR, PROTOCOL_EVAL)
        dataloader = DataLoader(get_subset(dataset, 1500), batch_size=32, shuffle=False, num_workers=2)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

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