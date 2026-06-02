import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchaudio.transforms as T
import numpy as np
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import KFold
from tqdm import tqdm

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(CURRENT_DIR) == 'ensemble':
    ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
else:
    ROOT_DIR = CURRENT_DIR
sys.path.append(ROOT_DIR)

from src.data.dataset import ASVspoofDataset
from src.models.aasist import AASIST
from src.models.resnet_simam import resnet18_simam
from train_urffl_crossattention_ensemble import (
    CrossAttentionFuser, EndToEndEnsemble, FocalLoss,
    _PDController, _init_aasist, _init_resnet
)

# Import UR-FFL components directly from your src folder
from src.ur_ffl.sensor import UncertaintySensor
from src.ur_ffl.selector import DegradationSelector
from src.ur_ffl.actuator import DegradationActuator

# Provide a dummy init since the fuser class handles its own weights internally
def _init_fuser(m):
    pass

# --- CONFIGURATION ---
PREPROCESSED_TRAIN_DIR = r"D:\SAMPOERNA\Semester 8\Capstone\Dataset\ASVspoof2019_LA_train_preprocessed"
PROTOCOL_TRAIN = r"D:\SAMPOERNA\Semester 8\Capstone\Dataset\2019\LA\ASVspoof2019_LA_cm_protocols\ASVspoof2019.LA.cm.train.trn.txt"
MODELS_DIR = os.path.join(ROOT_DIR, "saved_models")
os.makedirs(MODELS_DIR, exist_ok=True)

BATCH_SIZE = 24
TOTAL_EPOCHS = 15
K_FOLDS = 5
LR_BASE = 1e-4
LR_FUSER = 1e-5
WEIGHT_DECAY = 1e-4

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initiating {K_FOLDS}-Fold UR-FFL Cross-Attention Ensemble Training on {device}")
    
    # Load full training dataset
    full_dataset = ASVspoofDataset(PREPROCESSED_TRAIN_DIR, PROTOCOL_TRAIN)
    kfold = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
    
    mel_t = T.MelSpectrogram(sample_rate=16000, n_fft=512, hop_length=160, n_mels=80).to(device)
    a2db = T.AmplitudeToDB(stype="power", top_db=80).to(device)
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(full_dataset)):
        print(f"\n{'='*50}\n🚀 STARTING FOLD {fold + 1}/{K_FOLDS}\n{'='*50}")
        
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)
        
        train_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=4)
        val_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, sampler=val_sampler, num_workers=4)

        # Initialize Models from Scratch for each fold
        aasist = AASIST(stft_window=698, stft_hop=398, freq_bins=116, gat_layers=2, heads=5, head_dim=104, hidden_dim=455, dropout=0.33)
        resnet = resnet18_simam(num_classes=2, dropout_rate=0.22)
        fuser = CrossAttentionFuser()
        
        _init_aasist(aasist)
        _init_resnet(resnet)
        _init_fuser(fuser)
        
        model = EndToEndEnsemble(aasist, resnet, fuser).to(device)
        
        # UR-FFL Components
        sensor = UncertaintySensor(mc_passes=5)
        controller = _PDController() 
        selector = DegradationSelector()
        actuator = DegradationActuator(device)
        
        optimizer = optim.AdamW([
            {'params': model.aasist_w.parameters(), 'lr': LR_BASE, 'weight_decay': WEIGHT_DECAY},
            {'params': model.resnet_w.parameters(), 'lr': LR_BASE, 'weight_decay': WEIGHT_DECAY},
            {'params': model.fusion_head.parameters(), 'lr': LR_FUSER, 'weight_decay': WEIGHT_DECAY},
        ])
        
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TOTAL_EPOCHS, eta_min=1e-7)
        criterion = FocalLoss()
        scaler = torch.amp.GradScaler('cuda')
        
        best_val_loss = float('inf')
        save_path = os.path.join(MODELS_DIR, f"crossattention_ensemble_urffl_fold_{fold + 1}.pth")

        for epoch in range(TOTAL_EPOCHS):
            model.train()
            train_loss = 0.0
            
            # Using dynamic_ncols and leave=False to prevent multi-line terminal spam and show clean ETA
            pbar = tqdm(train_loader, desc=f"Fold {fold+1} | Ep {epoch+1}/{TOTAL_EPOCHS} [Tr]", dynamic_ncols=True, leave=False)
            for wav, lbl in pbar:
                wav = wav.squeeze(1).to(device)
                lbl = lbl.to(device)
                
                with torch.no_grad():
                    mel = a2db(mel_t(wav)).unsqueeze(1)
                    z_u, _ = sensor.measure(model.aasist_w.backbone, wav)
                    
                sel = selector.select(z_u)
                alpha = controller.alpha
                aug = actuator.apply(wav, lbl, sel, alpha)
                
                optimizer.zero_grad(set_to_none=True)
                
                # Forward Pass
                om, oa, or_ = model(wav, mel, return_base=True)
                om_aug, oa_aug, or_aug = model(aug, a2db(mel_t(aug)).unsqueeze(1), return_base=True)
                
                lm = criterion(om.float(), lbl) + criterion(om_aug.float(), lbl)
                la = criterion(oa.float(), lbl) + criterion(or_.float(), lbl)
                la_aug = criterion(oa_aug.float(), lbl) + criterion(or_aug.float(), lbl)
                
                lt = lm + 0.10 * (la + la_aug)
                
                scaler.scale(lt).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                
                train_loss += lt.item()
                pbar.set_postfix({'Loss': f"{lt.item():.4f}", 'Alpha': f"{alpha:.2f}"})
                    
                controller.update(z_u.mean().item())

            scheduler.step()
            train_loss /= len(train_loader)

            # Validation Phase
            model.eval()
            val_loss = 0.0
            
            val_pbar = tqdm(val_loader, desc=f"Fold {fold+1} | Ep {epoch+1} [Val]", dynamic_ncols=True, leave=False)
            with torch.no_grad():
                for wav, lbl in val_pbar:
                    wav = wav.squeeze(1).to(device)
                    lbl = lbl.to(device)
                    mel = a2db(mel_t(wav)).unsqueeze(1)
                    out = model(wav, mel)
                    val_loss += criterion(out.float(), lbl).item()
            
            val_loss /= len(val_loader)
            print(f"  -> Fold {fold+1} | Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            
            # Save Best Model for this Fold
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), save_path)
                print(f"  [+] New best model saved for Fold {fold+1}!")

        print(f"  -> Fold {fold+1} completed. Best Val Loss: {best_val_loss:.4f}. Weights saved.")

if __name__ == "__main__":
    main()