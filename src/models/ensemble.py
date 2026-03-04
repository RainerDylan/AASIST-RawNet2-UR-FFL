import torch
import torch.nn as nn
from src.models.rawnet2 import RawNet2
from src.models.aasist import AASIST

class DeepfakeEnsemble(nn.Module):
    """
    Stacked Generalization (Stacking) Meta-Learner
    Implements Section 3.4.3 Equation 5: y_final = σ(w_1·P_Raw(x) + w_2·P_AAS(x) + b)
    """
    def __init__(self, raw_config={}, aasist_config={}):
        super(DeepfakeEnsemble, self).__init__()
        # Base models
        self.rawnet = RawNet2(**raw_config)
        self.aasist = AASIST(**aasist_config)
        
        # Meta-learner (Logistic Regression Classifier)
        # Takes the 2-dimensional concatenated vector V as input
        self.meta_learner = nn.Sequential(
            nn.Linear(2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Ensure base models are treated as frozen during deployment
        out_raw = self.rawnet(x)
        out_aasist = self.aasist(x)
        
        # Convert logits to probabilities (P_Raw and P_AAS)
        p_raw = torch.softmax(out_raw, dim=1)[:, 1].unsqueeze(1) # Prob of fake
        p_aasist = torch.softmax(out_aasist, dim=1)[:, 1].unsqueeze(1)
        
        # Meta-Feature Vector V = [P_Raw(x), P_AAS(x)]
        v = torch.cat((p_raw, p_aasist), dim=1)
        
        # Meta-Learner Prediction Equation (5)
        y_final = self.meta_learner(v)
        
        # Return shaped for BCE Loss compatibility in pipeline
        return torch.cat((1 - y_final, y_final), dim=1)