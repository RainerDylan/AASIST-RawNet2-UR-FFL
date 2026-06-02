

import torch


class UncertaintySensor:
    """
    MC-Dropout epistemic uncertainty via predictive entropy.

    The sensor is called on CLEAN audio so the Selector can identify each
    sample's inherent difficulty before augmentation is applied.
    """

    def __init__(self, mc_passes: int = 50):
        """
        Parameters
        ----------
        mc_passes : int
            Number of stochastic forward passes T.
            Gal & Ghahramani (2016) recommend T ∈ [10, 100]; we default
            to 10 for a good speed/reliability trade-off in Phase 2.
        """
        self.mc_passes = mc_passes

    def measure(
        self,
        model: torch.nn.Module,
        waveforms: torch.Tensor,
    ):
        """
        Compute per-sample predictive entropy on the given waveforms.

        BatchNorm layers are kept in eval() (stable running stats) while
        Dropout layers are forced into train() (stochastic) to generate
        T different predictions per sample.

        Parameters
        ----------
        model     : AASIST model with Dropout layers.
        waveforms : (B, L) float32 tensor on model's device.

        Returns
        -------
        H_scores : (B,) tensor — per-sample entropy in nats ∈ [0, ln 2].
                   Higher = more confused, Lower = more confident.
                   Used by the Selector for within-class z-scoring.
        mean_H   : float — batch-mean entropy.
                   NOT used by the controller (which uses val_codec_loss).
        """
        model.eval()
        for m in model.modules():
            if m.__class__.__name__.startswith("Dropout"):
                m.train()

        with torch.no_grad():
            probs_list = []
            for _ in range(self.mc_passes):
                logits = model(waveforms)
                p = torch.softmax(logits, dim=1)[:, 1]   # p(bonafide), (B,)
                probs_list.append(p.unsqueeze(0))         # (1, B)

        probs = torch.cat(probs_list, dim=0)   # (T, B)
        mu    = probs.mean(dim=0)              # (B,) predictive mean

        eps = 1e-8
        H   = -(mu * torch.log(mu + eps) +
                (1.0 - mu) * torch.log(1.0 - mu + eps))   # (B,) nats

        return H, H.mean().item()
