import torch

class UncertaintySensor:
    def __init__(self, mc_passes=5):
        self.mc_passes = mc_passes

    def measure(self, model, waveforms):
        # Lock BatchNorm to prevent memory scrambling, but enable Dropout
        model.eval()
        for m in model.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()
                
        with torch.no_grad():
            outputs = []
            for _ in range(self.mc_passes):
                logits = model(waveforms)
                probs = torch.softmax(logits, dim=1)
                outputs.append(probs.unsqueeze(0))
                
            outputs = torch.cat(outputs, dim=0) 
            mean_probs = outputs.mean(dim=0)    
            
            # Shannon Entropy calculation
            entropy = -torch.sum(mean_probs * torch.log(mean_probs + 1e-8), dim=1)
            batch_uncertainty = entropy.mean().item()
            
        return batch_uncertainty