from dataclasses import dataclass

@dataclass(frozen=True)
class ModeConfig:
    use_uncertainty: bool
    use_neg_weight: bool
    use_pos_weighting: bool
    use_only_uncertainty: bool | None = None


def get_loss_mode_config(mode: str):
    MODES = {
    "full": ModeConfig(True, True, True),
    "no_uncertainty": ModeConfig(False, True, False),
    "uncertainity_curriculum_lr": ModeConfig(True, False, True),
    "no_weighting": ModeConfig(False, False, False),
    "uncertainty_only": ModeConfig(False, False, False, True),
}
    if mode not in MODES:
        raise ValueError(f"Unknown mode: {mode}")
    return MODES[mode]




