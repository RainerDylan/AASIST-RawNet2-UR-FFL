import torch
import numpy as np

class DegradationActuator:
    def __init__(self, device):
        self.device = device

    def apply_noise(self, waveform, severity):
        # Adjusted for Z-score standard deviation
        noise = torch.randn_like(waveform) * (0.5 * severity)
        return waveform + noise

    def apply_quantize(self, waveform, severity):
        # Adjusted quantization bins for floating-point tensors
        bins = max(4, int(256 - (severity * 252)))
        return torch.round(waveform * bins) / bins

    def apply_smear(self, waveform, severity):
        n_fft = 512
        hop_length = 256
        window = torch.hann_window(n_fft).to(self.device)
        
        stft = torch.stft(waveform, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True)
        mag = torch.abs(stft)
        phase = torch.angle(stft)
        
        noise = (torch.rand_like(phase) - 0.5) * 2 * np.pi * severity
        phase = phase + noise
        
        stft = torch.polar(mag, phase)
        return torch.istft(stft, n_fft=n_fft, hop_length=hop_length, window=window, length=waveform.size(1))

    def apply_ripple(self, waveform, severity):
        n_fft = 512
        hop_length = 256
        window = torch.hann_window(n_fft).to(self.device)
        
        stft = torch.stft(waveform, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True)
        mag = torch.abs(stft)
        phase = torch.angle(stft)
        
        freqs = torch.linspace(0, 1, mag.size(1)).to(self.device)
        ripple = 1.0 + severity * torch.sin(2 * np.pi * 10 * freqs)
        ripple = ripple.unsqueeze(0).unsqueeze(-1)
        
        mag = mag * ripple
        stft = torch.polar(mag, phase)
        return torch.istft(stft, n_fft=n_fft, hop_length=hop_length, window=window, length=waveform.size(1))

    def apply(self, waveforms, selections, severity):
        if severity <= 0.01:
            return waveforms
            
        aug_waveforms = waveforms.clone()
        for i, choice in enumerate(selections):
            if choice == 'noise':
                aug_waveforms[i] = self.apply_noise(aug_waveforms[i], severity)
            elif choice == 'quantize':
                aug_waveforms[i] = self.apply_quantize(aug_waveforms[i], severity)
            elif choice == 'smear':
                aug_waveforms[i] = self.apply_smear(aug_waveforms[i], severity)
            elif choice == 'ripple':
                aug_waveforms[i] = self.apply_ripple(aug_waveforms[i], severity)
        return aug_waveforms