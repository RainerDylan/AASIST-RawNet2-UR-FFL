"""
DegradationSelector utilizing Dynamic Z-Score Thresholding.

This maps epistemic uncertainty from MC Dropout entropy to degradations
relative to the model's own batch statistics. This normalizes architectural
biases between high-variance models like AASIST and low-variance models like ResNet.
"""

import torch

class DegradationSelector:

    def select(self, entropy_scores: torch.Tensor) -> list:
        """
        Parameters
        ----------
        entropy_scores : (B,) tensor of per-sample predictive entropy values H.

        Returns
        -------
        list[str] : length B, one profile label per sample.
        """
        std = entropy_scores.std()
        mean = entropy_scores.mean()

        # Safety catch for zero-variance outputs typical of CNNs with low dropout
        if std < 1e-6:
            return ["flatten"] * len(entropy_scores)

        z_scores = (entropy_scores - mean) / std

        selections = []
        for z in z_scores.tolist():
            if z < -1.5:
                selections.append("smear")
            elif z < -0.5:
                selections.append("codec")
            elif z < 0.5:
                selections.append("flatten")
            elif z < 1.5:
                selections.append("noise")
            else:
                selections.append("clean")
                
        return selections