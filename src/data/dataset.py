import os
import torch
import torch.nn.functional as F
import soundfile as sf
import librosa
import numpy as np
from torch.utils.data import Dataset

class ASVspoofDataset(Dataset):
    """
    Dataset implementing exact preprocessing from Table 6:
    - Silence Threshold: -30 dB
    - Pre-emphasis: 0.97
    - Normalization: Z-score
    - Padding Mode: Reflection
    - Target Length: 64,600
    """
    def __init__(self, flac_dir, protocol_file, max_frames=64600):
        self.flac_dir = flac_dir
        self.max_frames = max_frames
        self.file_paths = []
        self.labels = []
        
        with open(protocol_file, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            parts = line.strip().split()
            file_name = parts[1]
            label = 0 if parts[4] == "bonafide" else 1
            file_path = os.path.join(self.flac_dir, f"{file_name}.flac")
            
            if os.path.exists(file_path):
                self.file_paths.append(file_path)
                self.labels.append(label)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]
        
        # Load audio
        audio_data, _ = sf.read(file_path)
        
        # 1. Silence Threshold (-30 dB)
        audio_data, _ = librosa.effects.trim(audio_data, top_db=30)
        
        # 2. Pre-emphasis (alpha = 0.97)
        if len(audio_data) > 1:
            audio_data = np.append(audio_data[0], audio_data[1:] - 0.97 * audio_data[:-1])
        
        # 3. Z-score Normalization (μ=0, σ=1)
        mean = np.mean(audio_data)
        std = np.std(audio_data)
        audio_data = (audio_data - mean) / (std + 1e-8)
        
        waveform = torch.tensor(audio_data, dtype=torch.float32).unsqueeze(0).unsqueeze(0) # (1, 1, L)
        num_frames = waveform.shape[-1]
        
        # 4. Target Length & Reflection Padding
        if num_frames > self.max_frames:
            waveform = waveform[:, :, :self.max_frames]
        elif num_frames < self.max_frames:
            # Fallback for extremely short files after silence trimming
            if waveform.shape[-1] <= 1:
                pad_amount = self.max_frames - waveform.shape[-1]
                waveform = F.pad(waveform, (0, pad_amount), mode='constant', value=0)
            else:
                # Loop the reflection padding so it never exceeds the current length constraint
                while waveform.shape[-1] < self.max_frames:
                    current_length = waveform.shape[-1]
                    pad_amount = min(current_length - 1, self.max_frames - current_length)
                    waveform = F.pad(waveform, (0, pad_amount), mode='reflect')
            
        waveform = waveform.squeeze(0) # Output shape (1, 64600)
            
        return waveform, torch.tensor(label, dtype=torch.long)