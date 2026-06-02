"""
train_urffl_crossattention_ensemble_resnet_sensor.py
═══════════════════════════════════════════════════════════════════════════════
UR-FFL Cross-Attention Ensemble — ResNet-SimAM as UR-FFL Sensor Head.

This script is MATHEMATICALLY IDENTICAL to the reference single-branch
UR-FFL script (AASIST as sensor head) in every dimension:
  • Same loss weights  : CLEAN_W=0.35, DEG_W=0.35, AUX_W=0.20, CONS_W=0.10
  • Same PD controller : SETPOINT=3.0pp, alpha_max=0.55, Kp=0.005, Kd=0.002
  • Same gap tracking  : FUSER output (omc vs omd)
  • Same checkpoint    : CKPT_CLEAN_W=0.65, CKPT_AUG_W=0.35
  • Same LR / schedule : LR_BASE=3e-4, LR_FUSER=5e-4, warmup=5, cosine
  • Same FocalLoss     : γ=2.0, label_smoothing=0.10
  • Same SpecAugment   : freq_mask=15, time_mask=25
  • Same patience      : 20 epochs

ONLY CHANGE vs. reference:
  sensor.measure(model.resnet_w.backbone, mel_c_raw)   [was: aasist_w / wav]

ResNet-SimAM processes Mel spectrograms, so the clean Mel (mel_c_raw, before
SpecAugment) is computed first and passed to the sensor.  The RawBoost
actuator still operates on waveforms (it is a waveform-domain algorithm) and
the resulting aug waveform is converted to Mel for ResNet, exactly mirroring
how the AASIST version converts aug waveforms to feed AASIST.

This ensures 100% methodological fairness for a paired comparison between
  (a) AASIST-sensor train_urffl_crossattention_ensemble.py
  (b) ResNet-sensor train_urffl_crossattention_ensemble_resnet_sensor.py

References
──────────
Jung  J. et al. (2022): AASIST
Tak   H. et al. (2022): RawBoost
Gal & Ghahramani (2016): MC-Dropout
Ogata K. (2010): PD control
"""

import sys
import os
import time
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchaudio.transforms as T
from torch.utils.data import DataLoader, WeightedRandomSampler, SubsetRandomSampler
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from scipy.interpolate import interp1d
from scipy.optimize import brentq

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR    = (os.path.abspath(os.path.join(CURRENT_DIR, ".."))
               if "ensemble" in CURRENT_DIR else CURRENT_DIR)
sys.path.append(ROOT_DIR)

RESULTS_DIR = os.path.join(ROOT_DIR, "results")
MODELS_DIR  = os.path.join(ROOT_DIR, "saved_models")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR,  exist_ok=True)

from src.data.dataset        import ASVspoofDataset
from src.models.aasist       import AASIST
from src.models.resnet_simam import resnet18_simam
from src.ur_ffl.sensor       import UncertaintySensor
from src.ur_ffl.selector     import DegradationSelector
from src.ur_ffl.actuator     import DegradationActuator

# ── Paths ──────────────────────────────────────────────────────────────────────
PREPROCESSED_TRAIN_DIR = r"D:\SAMPOERNA\Semester 8\Capstone\Dataset\ASVspoof2019_LA_train_preprocessed"
PREPROCESSED_DEV_DIR   = r"D:\SAMPOERNA\Semester 8\Capstone\Dataset\ASVspoof2019_LA_dev_preprocessed"
PROTOCOL_TRAIN = (r"D:\SAMPOERNA\Semester 8\Capstone\Dataset\2019\LA"
                  r"\ASVspoof2019_LA_cm_protocols\ASVspoof2019.LA.cm.train.trn.txt")
PROTOCOL_DEV   = (r"D:\SAMPOERNA\Semester 8\Capstone\Dataset\2019\LA"
                  r"\ASVspoof2019_LA_cm_protocols\ASVspoof2019.LA.cm.dev.trl.txt")

# Separate output files so both runs can coexist in saved_models/
OUTPUT_WEIGHTS       = os.path.join(MODELS_DIR,
                           "crossattention_ensemble_urffl_resnet_sensor_best.pth")
OUTPUT_WEIGHTS_CLEAN = os.path.join(MODELS_DIR,
                           "crossattention_ensemble_urffl_resnet_sensor_best_clean.pth")

# ── Hyperparameters — identical to reference AASIST-sensor script ─────────────
TOTAL_EPOCHS  = 100
BATCH_SIZE    = 16

LR_BASE       = 3e-4    # AASIST + ResNet backbones
LR_FUSER      = 5e-4    # CrossAttentionFuser (leads backbone)

WEIGHT_DECAY  = 3e-4
WARMUP_EPOCHS = 5
PATIENCE      = 20

# ── Loss weights — IDENTICAL to reference ─────────────────────────────────────
CLEAN_W = 0.35   # fuser on clean pair
DEG_W   = 0.35   # fuser on degraded pair
AUX_W   = 0.20   # backbone clean auxiliaries, 0.10 per model
CONS_W  = 0.10   # fuser clean↔degraded consistency

# ── Checkpoint weights — IDENTICAL to reference ────────────────────────────────
CKPT_CLEAN_W = 0.65
CKPT_AUG_W   = 0.35


# ══════════════════════════════════════════════════════════════════════════════
# PD Controller — IDENTICAL to reference (same class, same defaults)
# ══════════════════════════════════════════════════════════════════════════════
class _PDController:
    """
    Exact copy of the reference PD controller.
    SETPOINT=3.0pp, alpha_max=0.55, Kp=0.005, Kd=0.002, MAX_STEP=0.015.
    """
    SETPOINT   = 3.0
    MAX_STEP   = 0.015
    EMA_BETA   = 0.35
    SAT_THRESH = 5

    def __init__(self):
        self.Kp        = 0.005;  self.Kd        = 0.002
        self.alpha_min = 0.0;    self.alpha_max = 0.55
        self.alpha     = 0.0
        self.setpoint  = self.SETPOINT
        self._sp_min   = 2.0;    self._sp_max   = 6.0
        self._ema      = 0.0;    self._prev_err = 0.0
        self._warmup   = True;   self._sat_hi   = 0;  self._sat_lo = 0

    def reset(self):
        self.alpha     = 0.0;  self.setpoint  = self.SETPOINT
        self._ema      = 0.0;  self._prev_err = 0.0
        self._warmup   = True; self._sat_hi   = 0;  self._sat_lo = 0

    def update(self, gap: float) -> float:
        if self._warmup:
            self._ema    = gap
            self._warmup = False
            print(f"  [PD] Warmup: EMA={gap:.1f}pp  SP={self.setpoint:.1f}pp  "
                  f"α={self.alpha:.4f}")
            return self.alpha

        self._ema = self.EMA_BETA * self._ema + (1. - self.EMA_BETA) * gap
        err       = self._ema - self.setpoint
        delta     = float(np.clip(
            self.Kp * err + self.Kd * (err - self._prev_err),
            -self.MAX_STEP, self.MAX_STEP
        ))

        at_max = self.alpha >= self.alpha_max - 1e-4
        at_min = self.alpha <= self.alpha_min + 1e-4

        if at_max:
            self._sat_hi += 1;  self._sat_lo = 0
            if delta < 0: delta = 0.
            if self._sat_hi >= self.SAT_THRESH:
                old = self.setpoint
                self.setpoint = min(self.setpoint + 0.5, self._sp_max)
                self._sat_hi  = 0
                if self.setpoint != old:
                    print(f"  [PD] Anti-windup↑: SP {old:.1f}→{self.setpoint:.1f}pp")
        elif at_min:
            self._sat_lo += 1;  self._sat_hi = 0
            if delta > 0: delta = 0.
            if self._sat_lo >= self.SAT_THRESH:
                old = self.setpoint
                self.setpoint = max(self.setpoint - 0.5, self._sp_min)
                self._sat_lo  = 0
                if self.setpoint != old:
                    print(f"  [PD] Anti-windup↓: SP {old:.1f}→{self.setpoint:.1f}pp")
        else:
            self._sat_hi = 0;  self._sat_lo = 0

        prev       = self.alpha
        self.alpha = float(np.clip(self.alpha - delta,
                                   self.alpha_min, self.alpha_max))
        self._prev_err = err
        d = "↑" if self.alpha > prev + 1e-4 else ("↓" if self.alpha < prev - 1e-4 else "–")
        print(f"  [PD] gap={gap:.1f}pp EMA={self._ema:.1f}pp SP={self.setpoint:.1f}pp "
              f"err={err:+.2f} δ={-delta:+.4f} α={self.alpha:.4f}{d}")
        return self.alpha


# ══════════════════════════════════════════════════════════════════════════════
# FocalLoss — IDENTICAL to reference (γ=2.0, ls=0.10)
# ══════════════════════════════════════════════════════════════════════════════
class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, label_smoothing: float = 0.10):
        super().__init__()
        self.gamma = gamma
        self.ls    = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        n = logits.shape[1]
        with torch.no_grad():
            s = torch.zeros_like(logits).fill_(self.ls / (n - 1))
            s.scatter_(1, targets.unsqueeze(1), 1. - self.ls)
        lp = F.log_softmax(logits, dim=1)
        w  = (1. - (lp.exp() * s).sum(1)).pow(self.gamma)
        return (w * -(s * lp).sum(1)).mean()


# ══════════════════════════════════════════════════════════════════════════════
# BackboneWrapper — IDENTICAL to reference
# ══════════════════════════════════════════════════════════════════════════════
class BackboneWrapper(nn.Module):
    """Pre-FC embedding hook; gradient tape intact; pure fp32."""

    def __init__(self, backbone: nn.Module, fc_attr: str = "fc"):
        super().__init__()
        self.backbone = backbone
        self._emb     = None
        fc = getattr(backbone, fc_attr, None)
        if not isinstance(fc, nn.Linear):
            raise AttributeError(
                f"BackboneWrapper: no nn.Linear '{fc_attr}' on backbone")
        self._handle = fc.register_forward_hook(self._capture)

    def _capture(self, module, inp, out):
        self._emb = inp[0]   # gradient tape intact

    def forward(self, x: torch.Tensor):
        self._emb = None
        logit     = self.backbone(x.float())
        emb       = self._emb
        if emb is None:
            raise RuntimeError("BackboneWrapper hook did not fire")
        self._emb = None
        return logit, emb

    def remove_hook(self):
        self._handle.remove()


# ══════════════════════════════════════════════════════════════════════════════
# CrossAttentionFuser — IDENTICAL to reference
# (256-dim, 2-layer, 8-head, dropout=0.30, Pre-LN)
# ══════════════════════════════════════════════════════════════════════════════
class CrossAttentionFuser(nn.Module):
    def __init__(self, dim_a: int = 104, dim_r: int = 512,
                 embed_dim: int = 256, num_heads: int = 8,
                 num_classes: int = 2, dropout: float = 0.30):
        super().__init__()
        self.proj_a    = nn.Sequential(nn.Linear(dim_a, embed_dim),
                                       nn.LayerNorm(embed_dim))
        self.proj_r    = nn.Sequential(nn.Linear(dim_r, embed_dim),
                                       nn.LayerNorm(embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_emb   = nn.Parameter(torch.zeros(1, 3, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_emb,   std=0.02)
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
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, ea: torch.Tensor, er: torch.Tensor) -> torch.Tensor:
        ea = ea.float();  er = er.float()
        B  = ea.size(0)
        seq = torch.cat([
            self.cls_token.expand(B, -1, -1),
            self.proj_a(ea).unsqueeze(1),
            self.proj_r(er).unsqueeze(1),
        ], dim=1) + self.pos_emb
        return self.head(self.transformer(seq)[:, 0, :])


# ══════════════════════════════════════════════════════════════════════════════
# EndToEndEnsemble — IDENTICAL to reference
# ══════════════════════════════════════════════════════════════════════════════
class EndToEndEnsemble(nn.Module):
    def __init__(self, aasist: nn.Module, resnet: nn.Module,
                 fuser: CrossAttentionFuser):
        super().__init__()
        self.aasist_w    = BackboneWrapper(aasist, fc_attr="fc")
        self.resnet_w    = BackboneWrapper(resnet,  fc_attr="fc")
        self.fusion_head = fuser

    def forward(self, wav: torch.Tensor, mel: torch.Tensor,
                return_base: bool = False):
        oa, ea  = self.aasist_w(wav.float())
        or_, er = self.resnet_w(mel.float())
        om      = self.fusion_head(ea, er)
        return (om, oa, or_) if return_base else om

    def remove_hooks(self):
        self.aasist_w.remove_hook()
        self.resnet_w.remove_hook()


# ══════════════════════════════════════════════════════════════════════════════
# Weight initialisation — IDENTICAL to reference
# ══════════════════════════════════════════════════════════════════════════════
def _init_aasist(m: nn.Module):
    attn = ("attn", "gat", "attention", "query", "key", "value")
    for n, mod in m.named_modules():
        ia = any(k in n.lower() for k in attn)
        if isinstance(mod, (nn.Conv1d, nn.Conv2d)):
            (nn.init.xavier_uniform_ if ia else nn.init.kaiming_normal_)(
                mod.weight,
                **({} if ia else {"mode": "fan_out", "nonlinearity": "relu"}))
            if mod.bias is not None: nn.init.zeros_(mod.bias)
        elif isinstance(mod, nn.Linear):
            (nn.init.xavier_uniform_ if ia else nn.init.kaiming_normal_)(
                mod.weight,
                **({} if ia else {"mode": "fan_in", "nonlinearity": "relu"}))
            if mod.bias is not None: nn.init.zeros_(mod.bias)
        elif isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d)):
            nn.init.ones_(mod.weight); nn.init.zeros_(mod.bias)


def _init_resnet(m: nn.Module):
    for mod in m.modules():
        if isinstance(mod, (nn.Conv2d, nn.Conv1d)):
            nn.init.kaiming_normal_(mod.weight, mode="fan_out",
                                    nonlinearity="relu")
            if mod.bias is not None: nn.init.zeros_(mod.bias)
        elif isinstance(mod, (nn.BatchNorm2d, nn.BatchNorm1d)):
            nn.init.ones_(mod.weight); nn.init.zeros_(mod.bias)
        elif isinstance(mod, nn.Linear):
            nn.init.normal_(mod.weight, 0., 0.01)
            if mod.bias is not None: nn.init.zeros_(mod.bias)


# ══════════════════════════════════════════════════════════════════════════════
# Data utilities — IDENTICAL to reference
# ══════════════════════════════════════════════════════════════════════════════
def create_train_sampler(dataset: ASVspoofDataset) -> WeightedRandomSampler:
    labels = dataset.labels
    counts = torch.bincount(torch.tensor(labels)).float()
    w      = len(labels) / counts
    return WeightedRandomSampler([w[l] for l in labels], len(labels),
                                 replacement=True)


def create_balanced_val_indices(dataset: ASVspoofDataset) -> list:
    labels    = np.array(dataset.labels)
    gen_idx   = np.where(labels == 0)[0]
    spoof_idx = np.where(labels == 1)[0]
    n   = min(len(gen_idx), len(spoof_idx))
    rng = np.random.RandomState(42)
    idx = np.concatenate([rng.choice(gen_idx,   n, replace=False),
                          rng.choice(spoof_idx, n, replace=False)])
    rng.shuffle(idx)
    print(f"  Balanced val subset: {n} genuine + {n} spoof = {2 * n} total")
    return idx.tolist()


def compute_eer(y_true, y_scores) -> float:
    try:
        fpr, tpr, _ = roc_curve(y_true, y_scores, pos_label=1)
        eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    except Exception:
        fpr, tpr, _ = roc_curve(y_true, y_scores, pos_label=1)
        fnr = 1. - tpr
        idx = np.nanargmin(np.abs(fnr - fpr))
        eer = float((fpr[idx] + fnr[idx]) / 2.)
    return float(eer * 100.)


def compute_min_dcf(y_true, y_scores,
                    p: float = 0.05, cm: float = 1., cf: float = 1.) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1. - tpr
    return float(
        np.min(cm * fnr * p + cf * fpr * (1. - p)) / min(cm * p, cf * (1. - p)))


# ── SpecAugment — IDENTICAL to reference (freq=15, time=25) ───────────────────
def spec_augment(mel_db: torch.Tensor,
                 freq_mask: int = 15, time_mask: int = 25) -> torch.Tensor:
    """mel_db: (B, 1, n_mels, T)"""
    B, C, F, TT = mel_db.shape
    out = mel_db.clone()
    for b in range(B):
        f0 = torch.randint(0, max(1, F  - freq_mask), (1,)).item()
        out[b, :, f0:f0 + freq_mask, :] = 0.
        t0 = torch.randint(0, max(1, TT - time_mask), (1,)).item()
        out[b, :, :, t0:t0 + time_mask] = 0.
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"UR-FFL Cross-Attention Ensemble (ResNet-SimAM Sensor) — {device}")
    print("=" * 70)

    # ── Build models (cold-start, no pre-trained weights) ─────────────────────
    aasist = AASIST(stft_window=698, stft_hop=398, freq_bins=116,
                    gat_layers=2, heads=5, head_dim=104, hidden_dim=455,
                    dropout=0.33)
    resnet = resnet18_simam(num_classes=2, dropout_rate=0.22)
    fuser  = CrossAttentionFuser()

    _init_aasist(aasist)
    _init_resnet(resnet)
    print("  AASIST : Kaiming(Conv/Linear) + Xavier(GAT/attn)")
    print("  ResNet : Kaiming(Conv) + small-normal(Linear)")
    print("  Fuser  : trunc-normal(std=0.02) — CrossAttentionFuser._init_weights")

    model = EndToEndEnsemble(aasist, resnet, fuser).to(device)
    print(f"  Params : {sum(p.numel() for p in model.parameters()):,}")
    print("=" * 70)

    # ── Datasets & loaders ────────────────────────────────────────────────────
    train_ds      = ASVspoofDataset(PREPROCESSED_TRAIN_DIR, PROTOCOL_TRAIN)
    val_ds        = ASVspoofDataset(PREPROCESSED_DEV_DIR,   PROTOCOL_DEV)
    train_sampler = create_train_sampler(train_ds)
    val_indices   = create_balanced_val_indices(val_ds)
    val_sampler   = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              sampler=train_sampler,
                              num_workers=4, pin_memory=False)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE,
                              sampler=val_sampler,
                              num_workers=4, pin_memory=False)

    mel_t = T.MelSpectrogram(sample_rate=16000, n_fft=512,
                              hop_length=160, n_mels=80).to(device)
    a2db  = T.AmplitudeToDB(stype="power", top_db=80).to(device)

    # ── UR-FFL components — ResNet-SimAM as sensor head ───────────────────────
    # THE ONLY STRUCTURAL DIFFERENCE FROM THE REFERENCE SCRIPT:
    # sensor.measure() is called with model.resnet_w.backbone and mel_c_raw.
    # Everything else (selector, actuator, PD controller) is identical.
    sensor     = UncertaintySensor(mc_passes=5)
    controller = _PDController()
    selector   = DegradationSelector()
    actuator   = DegradationActuator(device)

    print(f"UR-FFL sensor : ResNet-SimAM (Mel spectrogram domain)")
    print(f"UR-FFL: SP={controller.SETPOINT}pp | Kp={controller.Kp} | "
          f"Kd={controller.Kd} | α_max={controller.alpha_max}")
    print("=" * 70)

    # ── Optimiser & LR schedule — IDENTICAL to reference ─────────────────────
    optimizer = optim.AdamW([
        {"params": model.aasist_w.parameters(),
         "lr": LR_BASE,  "weight_decay": WEIGHT_DECAY},
        {"params": model.resnet_w.parameters(),
         "lr": LR_BASE,  "weight_decay": WEIGHT_DECAY},
        {"params": model.fusion_head.parameters(),
         "lr": LR_FUSER, "weight_decay": WEIGHT_DECAY},
    ])
    warmup = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0,
        total_iters=WARMUP_EPOCHS)
    cosine = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, TOTAL_EPOCHS - WARMUP_EPOCHS), eta_min=1e-7)
    sched     = optim.lr_scheduler.SequentialLR(
        optimizer, [warmup, cosine], [WARMUP_EPOCHS])
    criterion = FocalLoss()   # γ=2.0, ls=0.10

    # ── History ───────────────────────────────────────────────────────────────
    H = {k: [] for k in [
        "tr_loss", "vl_loss",
        "l_clean", "l_deg", "l_aux", "l_cons",
        "tr_eer",
        "vl_eer_c", "vl_eer_a",
        "vl_auc", "vl_mdc", "composite", "alpha",
    ]}

    best_comp      = float("inf")
    best_clean_eer = float("inf")
    no_imp         = 0
    t0             = time.time()

    for ep in range(TOTAL_EPOCHS):
        alpha = controller.alpha

        # ── Training epoch ────────────────────────────────────────────────────
        model.train()
        st = sc = sd = sa = ss = 0.
        tl: list = []
        tp: list = []
        gaps: list = []
        nan_n = 0

        bar = tqdm(train_loader, desc=f"Ep {ep + 1}/{TOTAL_EPOCHS} [Tr]")
        for wav, lbl in bar:
            wav = wav.squeeze(1).to(device)   # (B, L)
            lbl = lbl.to(device)

            # ── Step 1: Compute clean Mel (no SpecAugment) for sensor ─────────
            # ResNet-SimAM processes Mel spectrograms, so the uncertainty sensor
            # requires the clean Mel as input.  This is analogous to the
            # reference script passing clean waveforms to the AASIST sensor.
            with torch.no_grad():
                mel_c_raw = a2db(mel_t(wav)).unsqueeze(1)   # (B, 1, n_mels, T)

            # ── Step 2: UR-FFL — ResNet-SimAM measures uncertainty on clean Mel
            # THE ONLY CHANGE FROM REFERENCE: sensor uses resnet_w.backbone
            # and receives mel_c_raw instead of wav.
            with torch.no_grad():
                z_u, _ = sensor.measure(model.resnet_w.backbone, mel_c_raw)
                sel     = selector.select(z_u)
                # RawBoost actuator always operates on waveforms (waveform-domain)
                aug     = actuator.apply(wav, lbl, sel, alpha)   # (B, L)

            # ── Step 3: Compute aug Mel + apply SpecAugment to both Mels ─────
            with torch.no_grad():
                mel_a_raw = a2db(mel_t(aug)).unsqueeze(1)   # (B, 1, n_mels, T)
                mel_c     = spec_augment(mel_c_raw)
                mel_a     = spec_augment(mel_a_raw)

            # ── Step 4: Batched forward passes ───────────────────────────────
            # Structure IDENTICAL to reference:
            # AASIST sees: [wav_clean | aug_wav]
            # ResNet sees: [mel_c_specaugmented | mel_a_specaugmented]
            # Fuser  sees: embeddings from the above
            comb     = torch.cat([wav, aug], 0)          # (2B, L)
            mel_comb = torch.cat([mel_c, mel_a], 0)      # (2B, 1, F, T)

            optimizer.zero_grad(set_to_none=True)

            om, oa, or_ = model(comb, mel_comb, return_base=True)
            B_ = wav.size(0)

            # Fuser outputs
            omc = om[:B_].float()    # fuser on clean pair
            omd = om[B_:].float()    # fuser on degraded pair
            # Backbone auxiliary (clean only — identical to reference)
            oac = oa[:B_].float()    # AASIST clean
            orc = or_[:B_].float()   # ResNet  clean

            # ── Step 5: Loss — IDENTICAL to reference ─────────────────────────
            lc = criterion(omc, lbl)                                    # 0.35
            ld = criterion(omd, lbl)                                    # 0.35
            ls = F.mse_loss(F.softmax(omc, 1), F.softmax(omd, 1))     # 0.10
            la = criterion(oac, lbl) + criterion(orc, lbl)             # 0.20
            lt = CLEAN_W * lc + DEG_W * ld + AUX_W * la + CONS_W * ls

            if torch.isnan(lt) or torch.isinf(lt):
                nan_n += 1
                optimizer.zero_grad(set_to_none=True)
                if nan_n <= 3:
                    print(f"\n  [NaN] ep{ep + 1} — skipping")
                continue

            lt.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # ── Step 6: Gap tracking — IDENTICAL to reference ─────────────────
            # Gap is measured on the FUSER output (omc vs omd), NOT on any
            # individual backbone. This is unchanged from the reference script.
            with torch.no_grad():
                pc = torch.softmax(omc, 1)
                pd = torch.softmax(omd, 1)
                ac = pc.gather(1, lbl.view(-1, 1)).mean().item() * 100.
                ad = pd.gather(1, lbl.view(-1, 1)).mean().item() * 100.
                gaps.append(ac - ad)
                tl.extend(lbl.cpu().numpy())
                tp.extend(pc[:, 1].cpu().numpy())

            st += lt.item();  sc += lc.item();  sd += ld.item()
            sa += la.item();  ss += ls.item()
            bar.set_postfix({
                "L":   f"{lt.item():.4f}",
                "α":   f"{alpha:.3f}",
                "gap": f"{ac - ad:.1f}pp",
            })

        # ── End-of-epoch PD update — IDENTICAL to reference ──────────────────
        mean_gap = float(np.mean(gaps)) if gaps else 0.
        alpha    = controller.update(mean_gap)
        nb       = max(1, len(train_loader) - nan_n)

        H["tr_loss"].append(st / nb);  H["l_clean"].append(sc / nb)
        H["l_deg"].append(sd / nb);    H["l_aux"].append(sa / nb)
        H["l_cons"].append(ss / nb);   H["tr_eer"].append(compute_eer(tl, tp))
        H["alpha"].append(alpha)

        # ── Validation — IDENTICAL to reference ──────────────────────────────
        model.eval()
        sv = 0.
        vl_c: list = [];  vp_c: list = []
        vl_a: list = [];  vp_a: list = []

        with torch.no_grad():
            for wav_v, lbl_v in tqdm(val_loader,
                                     desc=f"Ep {ep + 1} [Val]", leave=False):
                wav_v = wav_v.squeeze(1).to(device)
                lbl_v = lbl_v.to(device)

                # Clean inference
                mel_v = a2db(mel_t(wav_v)).unsqueeze(1)
                out   = model(wav_v, mel_v)
                sv   += criterion(out.float(), lbl_v).item()
                vl_c.extend(lbl_v.cpu().numpy())
                vp_c.extend(torch.softmax(out.float(), 1)[:, 1].cpu().numpy())

                # Augmented probe — identical to reference
                aug_v   = actuator._ssi(wav_v, alpha=max(0.30, controller.alpha))
                mel_aug = a2db(mel_t(aug_v)).unsqueeze(1)
                out_av  = model(aug_v, mel_aug)
                vl_a.extend(lbl_v.cpu().numpy())
                vp_a.extend(torch.softmax(out_av.float(), 1)[:, 1].cpu().numpy())

        eer_c = compute_eer(vl_c, vp_c)
        eer_a = compute_eer(vl_a, vp_a)
        comp  = CKPT_CLEAN_W * eer_c + CKPT_AUG_W * eer_a
        vauc  = roc_auc_score(vl_c, vp_c)
        vmdc  = compute_min_dcf(vl_c, vp_c)

        H["vl_loss"].append(sv / len(val_loader))
        H["vl_eer_c"].append(eer_c);   H["vl_eer_a"].append(eer_a)
        H["vl_auc"].append(vauc);      H["vl_mdc"].append(vmdc)
        H["composite"].append(comp)

        arr_lbl  = np.array(vl_c);  arr_prb = np.array(vp_c)
        mean_gen = (arr_prb[arr_lbl == 0].mean()
                    if (arr_lbl == 0).any() else float("nan"))
        mean_spf = (arr_prb[arr_lbl == 1].mean()
                    if (arr_lbl == 1).any() else float("nan"))

        sched.step()
        lr0 = optimizer.param_groups[0]["lr"]
        lr2 = optimizer.param_groups[2]["lr"]
        eta = str(datetime.timedelta(seconds=int(
            (time.time() - t0) / (ep + 1) * (TOTAL_EPOCHS - ep - 1))))

        print(f"Ep {ep + 1:3d} | LR_base {lr0:.1e} LR_fuse {lr2:.1e} | "
              f"α {alpha:.3f} gap {mean_gap:.1f}pp | "
              f"Tr {st / nb:.4f} | Val {sv / len(val_loader):.4f} | "
              f"EER_c {eer_c:.3f}% | EER_a {eer_a:.3f}% | "
              f"Score {comp:.3f}% | ETA {eta}")
        print(f"       P(spoof)|genuine={mean_gen:.3f}  "
              f"P(spoof)|spoof={mean_spf:.3f}  "
              f"[gap={mean_spf - mean_gen:.3f}]")

        # ── Checkpointing — IDENTICAL to reference ────────────────────────────
        ckpt = {
            "epoch":            ep + 1,
            "model_state_dict": model.state_dict(),
            "eer_clean":        eer_c,
            "eer_aug":          eer_a,
            "composite":        comp,
            "val_auc":          vauc,
            "val_min_dcf":      vmdc,
            "alpha":            alpha,
        }

        if comp < best_comp:
            best_comp, no_imp = comp, 0
            torch.save(ckpt, OUTPUT_WEIGHTS)
            print(f"  -> Composite {best_comp:.4f}% "
                  f"(EER_c={eer_c:.4f}%, EER_a={eer_a:.4f}%) — saved ✓")
        else:
            no_imp += 1

        if eer_c < best_clean_eer:
            best_clean_eer = eer_c
            torch.save(ckpt, OUTPUT_WEIGHTS_CLEAN)
            print(f"  -> Clean-EER best: {best_clean_eer:.4f}% "
                  f"— saved to _best_clean.pth ✓")
        else:
            print(f"  -> No composite improvement ({no_imp}/{PATIENCE})")

        if no_imp >= PATIENCE:
            print("\nEarly stopping.")
            break

    model.remove_hooks()
    elapsed = str(datetime.timedelta(seconds=int(time.time() - t0)))
    print(f"\nDone: {elapsed} | "
          f"Best composite: {best_comp:.4f}% | "
          f"Best clean EER: {best_clean_eer:.4f}%")
    print(f"  Composite checkpoint : {OUTPUT_WEIGHTS}")
    print(f"  Clean-EER checkpoint : {OUTPUT_WEIGHTS_CLEAN}")

    # ── Plots — IDENTICAL to reference ────────────────────────────────────────
    E = range(1, len(H["tr_loss"]) + 1)
    fig, axes = plt.subplots(2, 3, figsize=(21, 12))
    ax = axes.flatten()

    ax[0].plot(E, H["tr_loss"], label="Train Loss",        color="blue")
    ax[0].plot(E, H["vl_loss"], label="Val Loss (clean)",  color="red",        ls="--")
    ax[0].plot(E, H["l_clean"], label="L_clean",           color="steelblue",  ls=":")
    ax[0].plot(E, H["l_deg"],   label="L_deg",             color="darkorange", ls=":")
    ax[0].plot(E, H["l_aux"],   label="L_aux (backbones)", color="green",      ls=":")
    ax[0].plot(E, H["l_cons"],  label="L_cons",            color="purple",     ls=":")
    ax[0].set_title("Loss Trajectory"); ax[0].set_xlabel("Epoch")
    ax[0].legend(fontsize=7); ax[0].grid(True, ls=":", alpha=0.6)

    ax[1].plot(E, H["vl_eer_c"], label="EER_clean %", color="green")
    ax[1].plot(E, H["vl_eer_a"], label="EER_aug %",   color="purple", ls="--")
    ax[1].plot(E, H["tr_eer"],   label="Train EER %", color="teal",   ls=":")
    ax[1].set_title("Equal Error Rate — Balanced Val (↓ Better)")
    ax[1].set_xlabel("Epoch"); ax[1].legend(); ax[1].grid(True, ls=":", alpha=0.6)

    ax[2].plot(E, H["composite"], label="Composite score", color="navy")
    ax[2].set_title(f"Checkpoint Metric (↓ Better)\n"
                    f"= {CKPT_CLEAN_W}·EER_c + {CKPT_AUG_W}·EER_a")
    ax[2].set_xlabel("Epoch"); ax[2].legend(); ax[2].grid(True, ls=":", alpha=0.6)

    ax[3].plot(E, H["vl_auc"], label="ROC-AUC (balanced)", color="darkgreen")
    ax[3].axhline(0.5, ls=":", color="gray", alpha=0.5, label="Random")
    ax[3].set_ylim(0.4, 1.02)
    ax[3].set_title("Validation AUC — Balanced Val (↑ Better)")
    ax[3].set_xlabel("Epoch"); ax[3].legend(); ax[3].grid(True, ls=":", alpha=0.6)

    ax[4].plot(E, H["vl_mdc"], label="minDCF", color="darkred")
    ax[4].axhline(1., ls=":", color="gray", alpha=0.5, label="Chance")
    ax[4].set_title("Validation minDCF — Balanced Val (↓ Better)")
    ax[4].set_xlabel("Epoch"); ax[4].legend(); ax[4].grid(True, ls=":", alpha=0.6)

    ax5b = ax[5].twinx()
    ax[5].plot(E, H["alpha"], label="α (PD-controlled)",
               color="orange", lw=2)
    ax[5].axhline(controller.alpha_max, ls=":", color="red", alpha=0.5,
                  label=f"α_max={controller.alpha_max}")
    ax[5].axhline(controller.alpha_min, ls=":", color="gray", alpha=0.5,
                  label=f"α_min={controller.alpha_min}")
    ax5b.plot(E, H["vl_eer_a"], label="EER_aug %",
              color="purple", ls="--", alpha=0.5)
    ax[5].set_title("UR-FFL Alpha Trajectory (ResNet-SimAM sensor)")
    ax[5].set_xlabel("Epoch")
    ax[5].set_ylabel("α", color="orange")
    ax5b.set_ylabel("EER_aug (%)", color="purple")
    ax[5].set_ylim(-0.05, 1.05)
    l1, b1 = ax[5].get_legend_handles_labels()
    l2, b2 = ax5b.get_legend_handles_labels()
    ax[5].legend(l1 + l2, b1 + b2, fontsize=7)
    ax[5].grid(True, ls=":", alpha=0.6)

    cfg_text = (
        f"Mode: Cold-start, end-to-end\n"
        f"UR-FFL sensor: ResNet-SimAM (Mel domain)\n"
        f"PD: SP={controller.SETPOINT}pp, α_max={controller.alpha_max}\n"
        f"Kp={controller.Kp}, Kd={controller.Kd}, step≤{controller.MAX_STEP}\n"
        f"Loss: clean={CLEAN_W} deg={DEG_W} aux={AUX_W} cons={CONS_W}\n"
        f"Ckpt: {CKPT_CLEAN_W}·EER_c + {CKPT_AUG_W}·EER_a\n"
        f"FocalLoss: γ=2.0, ls=0.10\n"
        f"LR_base={LR_BASE:.0e}, LR_fuser={LR_FUSER:.0e}\n"
        f"SpecAugment: freq=15, time=25\n"
        f"Best composite: {best_comp:.4f}%\n"
        f"Best clean EER: {best_clean_eer:.4f}%"
    )
    fig.text(0.72, 0.08, cfg_text, fontsize=7.5, family="monospace",
             bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    fig.suptitle(
        "UR-FFL Cross-Attention Ensemble — ResNet-SimAM Sensor\n"
        "(Calibrated PD · fp32 · Corrected LR · Dual Checkpoint · SpecAugment)",
        fontsize=13,
    )
    fig.tight_layout()
    gp = os.path.join(RESULTS_DIR,
                      "crossattention_ensemble_urffl_resnet_sensor_metrics.png")
    fig.savefig(gp, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Plots → {gp}")


if __name__ == "__main__":
    main()