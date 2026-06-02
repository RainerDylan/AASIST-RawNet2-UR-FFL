import os
import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T

# Force Matplotlib to use the headless 'Agg' backend
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

import librosa
import librosa.display

# =====================================================================
# PATH CONFIGURATION
# =====================================================================
# Pointing directly to the raw FLAC directory you specified
RAW_DATA_DIR = r"D:\SAMPOERNA\Semester 8\Capstone\Dataset\2019\LA\ASVspoof2019_LA_train\flac"
TARGET_FILE = "LA_T_1000406"

# =====================================================================
# HYPERPARAMETERS (Must match your dataset & model pipeline exactly)
# =====================================================================
SAMPLE_RATE = 16000
MAX_LEN = 64000  # Exact padding/truncation length from your dataset.py
N_FFT = 512
HOP_LENGTH = 160
N_MELS = 80
TOP_DB = 80

def process_and_visualize(target_filename, search_dir):
    print(f"Looking for raw audio file: '{target_filename}' in {search_dir}...")
    
    found_path = None
    
    # Check directly first
    direct_path = os.path.join(search_dir, f"{target_filename}.flac")
    if os.path.exists(direct_path):
        found_path = direct_path
    else:
        # Fallback: Recursive search just in case it's in a subfolder
        for root, dirs, files in os.walk(search_dir):
            for file in files:
                if file.startswith(target_filename) and file.endswith('.flac'):
                    found_path = os.path.join(root, file)
                    break
            if found_path:
                break
            
    if not found_path:
        print(f"\n[!] ERROR: Could not find '{target_filename}.flac' anywhere in {search_dir}")
        return

    print(f"[+] Found raw audio file at: {found_path}")
    print("Applying exact Dataset preprocessing (64,000 samples)...")
    
    # 1. Load the raw audio
    wav, sr = torchaudio.load(found_path)
    
    # Resample if necessary (ASVspoof is usually natively 16kHz, but just to be safe)
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
        
    # 2. Mimic the ASVspoofDataset __getitem__ padding/truncating
    if wav.shape[1] > MAX_LEN:
        wav = wav[:, :MAX_LEN]
    elif wav.shape[1] < MAX_LEN:
        pad_amount = MAX_LEN - wav.shape[1]
        wav = F.pad(wav, (0, pad_amount), "constant", 0)
        
    print(f"[+] Audio shape after dataset preprocessing: {wav.shape}")
    print("Applying ResNet Mel Spectrogram transformation...")
    
    # 3. Initialize ResNet transformations
    mel_transform = T.MelSpectrogram(
        sample_rate=SAMPLE_RATE, 
        n_fft=N_FFT, 
        hop_length=HOP_LENGTH, 
        n_mels=N_MELS
    )
    amp_to_db = T.AmplitudeToDB(stype='power', top_db=TOP_DB)

    # 4. Apply transformation
    # Wav is currently [1, 64000]. Mel transform expects this and outputs [1, N_MELS, Frames]
    mel = amp_to_db(mel_transform(wav))
    
    # Squeeze out the channel dimension for plotting -> [N_MELS, Frames]
    mel_np = mel.squeeze(0).numpy()

    # 5. Plotting
    fig, ax = plt.subplots(figsize=(10, 4))
    
    img = librosa.display.specshow(
        mel_np, 
        sr=SAMPLE_RATE, 
        hop_length=HOP_LENGTH, 
        x_axis='time', 
        y_axis='mel', 
        ax=ax,
        cmap='magma' 
    )
    
    actual_filename = os.path.basename(found_path)
    ax.set_title(f"ResNet Mel Spectrogram Input (Preprocessed to {MAX_LEN} samples)\nTarget: {actual_filename}")
    fig.colorbar(img, ax=ax, format="%+2.f dB")

    plt.tight_layout()
    output_path = f"resnet_spectrogram_{target_filename}.png"
    plt.savefig(output_path, dpi=300)
    print(f"\n[+] SUCCESS: Visualization perfectly matches ResNet input.")
    print(f"[+] Saved to: {output_path}")

if __name__ == "__main__":
    process_and_visualize(TARGET_FILE, RAW_DATA_DIR)