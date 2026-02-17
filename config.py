from dataclasses import dataclass
from typing import Optional
@dataclass
class Config:
    image_size: int = 224
    mask_size: int = 224
    dropout_rate: float = 0.6
    backbone: str = "efficientnetv2-b3"
    threshold: float = 0.5
    segmentation_weight: float = 8
    classification_weight: float = 0.5
    fracture_weight: float = 1.0
    body_part_weight: float = 0.3
    use_segmentation: bool = True
    use_cross_validation: bool = False
    use_balanced_sampling: bool = True
    use_adaptive_loss_weighting: bool = False
    adaptive_loss_initial_log_vars: float = 0.0
    use_clahe: bool = True
    use_cutmix: bool = False
    cutmix_prob: float = 0.0
    hard_negative_ratio: float = 0.3
    cv_folds: int = 5
    model_path: Optional[str] = None
    model_name: Optional[str] = None
    batch_size: int = 32
    epochs: int = 150
    learning_rate: float = 1e-4
    warmup_epochs: int = 10
    early_stop_patience: int = 15
    copy_to_local: bool = True
    

    @classmethod
    def from_dict(cls, config_dict: dict) -> "Config":
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__dataclass_fields__})
    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in self.__dataclass_fields__.values()}