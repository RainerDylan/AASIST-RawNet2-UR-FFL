import sys
import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import random

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(CURRENT_DIR) in ['ensemble', 'src']:
    ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
else:
    ROOT_DIR = CURRENT_DIR
sys.path.append(ROOT_DIR)

from src.data.dataset import ASVspoofDataset
from src.ur_ffl.actuator import DegradationActuator
from src.models.aasist import AASIST

PREPROCESSED_DEV_DIR = r"D:\SAMPOERNA\Semester 8\Capstone\Dataset\ASVspoof2019_LA_dev_preprocessed"
PROTOCOL_DEV = r"D:\SAMPOERNA\Semester 8\Capstone\Dataset\2019\LA\ASVspoof2019_LA_cm_protocols\ASVspoof2019.LA.cm.dev.trl.txt"

RESULTS_DIR = os.path.join(ROOT_DIR, "results")
MODELS_DIR = os.path.join(ROOT_DIR, "saved_models")
os.makedirs(RESULTS_DIR, exist_ok=True)

class FeatureExtractor:
    def __init__(self, model):
        self.model = model
        self.features = None
        # Intercept the Graph Embedding before the final FC classification layer
        self.hook = self.model.pool.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.features = output.squeeze(-1) # Shape: (B, hidden_dim)

    def extract(self, x):
        # Forward pass returns the final classification logits (for KL Divergence)
        logits = self.model(x)
        # self.features holds the deep graph embeddings (for Cosine Similarity)
        return self.features.clone(), logits

    def remove(self):
        self.hook.remove()

def compute_cosine_similarity(f_clean, f_deg):
    """Equation 12: Cosine Similarity between feature vectors."""
    return F.cosine_similarity(f_clean + 1e-8, f_deg + 1e-8, dim=1)

def compute_kl_divergence(logits_clean, logits_deg):
    """
    Equation 14: KL Divergence 
    Calculated on the final prediction distribution (Logits) to accurately 
    measure how much the degradation confused the network's decision.
    """
    p_clean = F.softmax(logits_clean, dim=1)
    p_deg = F.softmax(logits_deg, dim=1)
    kl_div = F.kl_div(p_deg.log(), p_clean, reduction='none').sum(dim=1)
    return kl_div

def get_subset(dataset, size=1500):
    size = min(size, len(dataset))
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    return Subset(dataset, indices[:size])

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initiating Advanced Forensic Quantification Analysis on {device}...")

    try:
        val_dataset = ASVspoofDataset(PREPROCESSED_DEV_DIR, PROTOCOL_DEV)
        # Using a subset for statistical analysis to prevent excessive processing time
        val_loader = DataLoader(get_subset(val_dataset, 1000), batch_size=32, shuffle=False, num_workers=2)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    print("Loading AASIST Feature Extractor (Sinc-Convolutional Front-end)...")
    
    # Hardcoded to perfectly match your pretrained weights
    model = AASIST(
        stft_window=698,
        stft_hop=398,
        freq_bins=116,
        gat_layers=2,
        heads=5,
        head_dim=104,
        hidden_dim=455,
        dropout=0.3311
    ).to(device)
    
    model_path = os.path.join(MODELS_DIR, "aasist_unified_best.pth")
    if os.path.exists(model_path):
        ckpt = torch.load(model_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)
        print("=> Successfully loaded pre-trained AASIST weights.")
    else:
        print("=> WARNING: Pre-trained weights not found. Math will be unstable.")
    
    model.eval()
    extractor = FeatureExtractor(model)
    actuator = DegradationActuator(device)

    # Divided into 4 specific ranks based on your updated methodology
    profiles = {
        "Smear (LnL)": "smear",
        "Codec (ISD)": "codec",
        "Flatten (SSI)": "flatten",
        "Noise (SSI)": "noise"
    }

    metrics = {p: {"cosine": [], "kl": []} for p in profiles.keys()}
    alpha_test_level = 0.8 # Fixed high severity to clearly separate the degradations

    with torch.no_grad():
        for waveforms, labels in tqdm(val_loader, desc="Extracting & Comparing Features"):
            waveforms = waveforms.squeeze(1).to(device)
            labels = labels.to(device)

            # 1. Clean Feature Extraction
            f_clean, logits_clean = extractor.extract(waveforms)

            # 2. Degraded Feature Extraction & Comparison
            for profile_name, actuator_key in profiles.items():
                selections = [actuator_key] * waveforms.size(0)
                aug_wav = actuator.apply(waveforms, labels, selections, alpha_test_level)
                
                f_deg, logits_deg = extractor.extract(aug_wav)
                
                # Compute similarities and divergences
                cos_sims = compute_cosine_similarity(f_clean, f_deg)
                kl_divs = compute_kl_divergence(logits_clean, logits_deg)
                
                metrics[profile_name]["cosine"].extend(cos_sims.cpu().numpy())
                metrics[profile_name]["kl"].extend(kl_divs.cpu().numpy())

    # Compile Final Statistics
    print("\n" + "="*80)
    print(" 🎯 FORENSIC HIERARCHY QUANTIFICATION RESULTS")
    print("="*80)
    print(f"{'Profile':<22} | {'MFS (Cosine) ↑':<18} | {'KL Divergence (nats) ↓':<20}")
    print("-" * 80)
    
    plot_data = []
    for name in profiles.keys():
        mfs = np.mean(metrics[name]["cosine"])
        mfs_std = np.std(metrics[name]["cosine"])
        
        kl = np.mean(metrics[name]["kl"])
        kl_std = np.std(metrics[name]["kl"])
        
        plot_data.append((name, mfs, mfs_std, kl, kl_std))
        
        print(f"{name:<22} | {mfs:>6.4f} ± {mfs_std:<6.4f} | {kl:>6.4f} ± {kl_std:<6.4f}")

    # Generate Simple, Separated Graphs
    names = [x[0] for x in plot_data]
    mfs_means = [x[1] for x in plot_data]
    mfs_stds = [x[2] for x in plot_data]
    kl_means = [x[3] for x in plot_data]
    kl_stds = [x[4] for x in plot_data]

    # Bar colors
    bar_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    # Graph 1: Mean Feature Similarity
    plt.figure(figsize=(10, 6))
    x_pos = np.arange(len(names))
    plt.bar(x_pos, mfs_means, yerr=mfs_stds, capsize=10, color=bar_colors, edgecolor='black', alpha=0.8)
    plt.title('Mean Feature Similarity (MFS)', fontsize=14, fontweight='bold')
    plt.ylabel('Cosine Similarity (Lower = Harder)', fontsize=12)
    plt.xticks(x_pos, names, fontsize=11, fontweight='bold')
    plt.grid(axis='y', linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "hierarchy_MFS_chart.png"), dpi=300)
    plt.close()

    # Graph 2: KL Divergence
    plt.figure(figsize=(10, 6))
    plt.bar(x_pos, kl_means, yerr=kl_stds, capsize=10, color=bar_colors, edgecolor='black', alpha=0.8)
    plt.title('KL Divergence (Information Loss)', fontsize=14, fontweight='bold')
    plt.ylabel('KL Divergence in Nats (Higher = Harder)', fontsize=12)
    plt.xticks(x_pos, names, fontsize=11, fontweight='bold')
    plt.grid(axis='y', linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "hierarchy_KLD_chart.png"), dpi=300)
    plt.close()
    
    print("\n=> EXPECTED METHODOLOGY RANKING:")
    print(" 1. Smear (LnL)   : Lowest MFS, Highest KL (Hardest)")
    print(" 2. Codec (ISD)   : Moderate MFS, Moderate KL")
    print(" 3. Flatten (SSI) : Highest MFS, Lowest KL (Easiest)")
    print(" 4. Noise (SSI)   : Highest MFS, Lowest KL (Easiest)")
    print(f"\n=> Clean graphs successfully saved to: {RESULTS_DIR}")

    extractor.remove()

if __name__ == "__main__":
    main()