"""D2 noise simulator: corrupt clean tracking to look like Metrica's automatic tracking (ATD).

    corrupt(m_clean, noise_cfg, seed) -> Match        noisy copy, labels unchanged
    target_stats(m) -> dict[str, float]               label-free noise statistics
    fit_noise_cfg(m_target, m_clean=None) -> dict     moment-matched config
    severity_grid(cfg, k) -> list[dict]               mild-to-harsh configs around cfg
    domain_gap(real, sim) -> dict                     real-vs-sim window classifier AUC
"""
from .sim import ANCHORED, DEFAULT_CFG, corrupt, resolve_cfg
from .fit import crop_match, fit_noise_cfg, severity_grid, stats_loss
from .gap import domain_gap, window_features
from .stats import target_stats

__all__ = ["corrupt", "target_stats", "fit_noise_cfg", "severity_grid", "domain_gap",
           "window_features", "stats_loss", "crop_match", "resolve_cfg", "DEFAULT_CFG", "ANCHORED"]
