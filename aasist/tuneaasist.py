import sys
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from sklearn.metrics import roc_curve
from scipy.interpolate import interp1d
from scipy.optimize import brentq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ray Tune & BOHB
import ray
from ray import train, tune
from ray.tune.search.bohb import TuneBOHB
from ray.tune.schedulers import HyperBandForBOHB

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.append(ROOT_DIR)

RESULTS_DIR = os.path.join(ROOT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

from src.data.dataset import ASVspoofDataset
from src.models.aasist import AASIST

# Train Set
PREPROCESSED_TRAIN_DIR = r"D:\SAMPOERNA\Semester 8\Capstone\Dataset\ASVspoof2019_LA_train_preprocessed"
PROTOCOL_TRAIN = r"D:\SAMPOERNA\Semester 8\Capstone\Dataset\2019\LA\ASVspoof2019_LA_cm_protocols\ASVspoof2019.LA.cm.train.trn.txt"

# Dev Set (Crucial for preventing data leakage)
PREPROCESSED_DEV_DIR = r"D:\SAMPOERNA\Semester 8\Capstone\Dataset\ASVspoof2019_LA_dev_preprocessed"
PROTOCOL_DEV = r"D:\SAMPOERNA\Semester 8\Capstone\Dataset\2019\LA\ASVspoof2019_LA_cm_protocols\ASVspoof2019.LA.cm.dev.trl.txt"

def get_balanced_subset(dataset, size=1000):
    bonafide_idx = [i for i, label in enumerate(dataset.labels) if label == 1]
    spoof_idx = [i for i, label in enumerate(dataset.labels) if label == 0]
    random.shuffle(bonafide_idx)
    random.shuffle(spoof_idx)
    half = size // 2
    indices = bonafide_idx[:half] + spoof_idx[:half]
    return Subset(dataset, indices)

def compute_eer(y_true, y_scores):
    try:
        fpr, tpr, _ = roc_curve(y_true, y_scores, pos_label=1)
        fnr = 1. - tpr
        eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    except Exception:
        fpr, tpr, _ = roc_curve(y_true, y_scores, pos_label=1)
        fnr = 1. - tpr
        idx = np.nanargmin(np.abs(fnr - fpr))
        eer = float((fpr[idx] + fnr[idx]) / 2.)
    return float(eer * 100.)

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, label_smoothing=0.05):
        super().__init__()
        self.gamma = gamma
        self.ls = label_smoothing

    def forward(self, logits, targets):
        n_cls = logits.shape[1]
        with torch.no_grad():
            smooth = torch.zeros_like(logits).fill_(self.ls / (n_cls - 1))
            smooth.scatter_(1, targets.unsqueeze(1), 1.0 - self.ls)
        log_p = F.log_softmax(logits, dim=1)
        p = log_p.exp()
        pt = (p * smooth).sum(dim=1)
        weight = (1.0 - pt).pow(self.gamma)
        ce = -(smooth * log_p).sum(dim=1)
        return (weight * ce).mean()

# BOHB Training Function
def train_aasist(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = AASIST(
        stft_window=config["stft_window"], 
        stft_hop=config["stft_hop"], 
        freq_bins=config["freq_bins"],
        gat_layers=config["gat_layers"], 
        heads=config["heads"], 
        head_dim=config["head_dim"], 
        hidden_dim=config["hidden_dim"], 
        dropout=config["dropout"]
    ).to(device)
    
    train_dataset = ASVspoofDataset(PREPROCESSED_TRAIN_DIR, PROTOCOL_TRAIN)
    val_dataset = ASVspoofDataset(PREPROCESSED_DEV_DIR, PROTOCOL_DEV)
    
    train_subset = get_balanced_subset(train_dataset, size=1000)
    val_subset = get_balanced_subset(val_dataset, size=400)
    
    train_loader = DataLoader(train_subset, batch_size=config["batch_size"], shuffle=True, num_workers=2)
    val_loader = DataLoader(val_subset, batch_size=config["batch_size"], shuffle=False, num_workers=2)
    
    optimizer = optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    criterion = FocalLoss(gamma=2.0, label_smoothing=0.05)
    
    # Ray Tune handles the epochs dynamically for BOHB
    while True:
        model.train()
        for waveforms, labels in train_loader:
            waveforms = waveforms.squeeze(1).to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(waveforms)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
        model.eval()
        val_labels = []
        val_probs = []
        
        with torch.no_grad():
            for wv, lv in val_loader:
                wv = wv.squeeze(1).to(device)
                lv = lv.to(device)
                outputs = model(wv)
                val_labels.extend(lv.cpu().numpy())
                val_probs.extend(torch.softmax(outputs, dim=1)[:, 1].cpu().numpy())
                
        val_eer = compute_eer(val_labels, val_probs)
        
        # Report metric to BOHB Scheduler
        train.report({"eer": val_eer})

def plot_bohb_graphs(results_df, prefix="aasist"):
    # 1. Optimization History
    plt.figure(figsize=(10, 6))
    valid_df = results_df.dropna(subset=['eer'])
    valid_df = valid_df.sort_values(by='training_iteration')
    
    best_eer_so_far = []
    current_best = float('inf')
    for eer in valid_df['eer']:
        if eer < current_best:
            current_best = eer
        best_eer_so_far.append(current_best)
        
    plt.plot(range(len(valid_df)), valid_df['eer'], 'o', alpha=0.3, label='Objective Value (EER)')
    plt.plot(range(len(valid_df)), best_eer_so_far, color='red', lw=2, label='Best Value')
    plt.title(f"{prefix.upper()} BOHB Optimization History")
    plt.xlabel("Trial Number")
    plt.ylabel("Validation EER (%)")
    plt.legend()
    plt.grid(True, ls=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"{prefix}_bohb_opt_history.png"), dpi=300)
    plt.close()

if __name__ == "__main__":
    print("Initiating BOHB (Bayesian Optimization + HyperBand) Tuning for AASIST...")
    
    # Ray Tune Search Space
    search_space = {
        "stft_window": tune.randint(256, 1024),
        "stft_hop": tune.randint(64, 512),
        "freq_bins": tune.randint(64, 256),
        "gat_layers": tune.randint(2, 4),
        "heads": tune.randint(2, 8),
        "head_dim": tune.randint(32, 128),
        "hidden_dim": tune.randint(64, 512),
        "dropout": tune.uniform(0.1, 0.5),
        "lr": tune.loguniform(1e-6, 5e-3),
        "batch_size": tune.choice([16, 32, 64]),
        "weight_decay": tune.loguniform(1e-6, 1e-3)
    }

    # BOHB Configuration
    algo = TuneBOHB(metric="eer", mode="min")
    scheduler = HyperBandForBOHB(
        time_attr="training_iteration",
        max_t=15, # Max epochs per trial
        reduction_factor=3
    )

    tuner = tune.Tuner(
        train_aasist,
        tune_config=tune.TuneConfig(
            metric="eer",
            mode="min",
            search_alg=algo,
            scheduler=scheduler,
            num_samples=50, # Number of trials
        ),
        param_space=search_space,
    )

    results = tuner.fit()
    
    # Extract Best Parameters
    best_result = results.get_best_result("eer", "min")
    best_config = best_result.config
    best_eer = best_result.metrics["eer"]
    
    print("\n=================================================")
    print("BOHB Optimization Completed!")
    print(f"Best Validation EER: {best_eer:.4f}%")
    
    txt_path = os.path.join(RESULTS_DIR, "aasist_best_params_bohb.txt")
    with open(txt_path, "w") as f:
        f.write("=========================================\n")
        f.write("AASIST OPTIMAL HYPERPARAMETERS (BOHB)\n")
        f.write("=========================================\n")
        f.write(f"Best Validation EER: {best_eer:.4f}%\n\n")
        for key, value in best_config.items():
            print(f"  {key}: {value}")
            f.write(f"{key}: {value}\n")
    print(f"\nSaved optimal parameters to {txt_path}")

    # Generate Graphs
    print("\nGenerating Diagnostic BOHB Graphs...")
    results_df = results.get_dataframe()
    plot_bohb_graphs(results_df, prefix="aasist")
    print(f"Graphs successfully saved to {RESULTS_DIR}")