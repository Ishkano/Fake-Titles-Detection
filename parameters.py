import torch
from dataclasses import dataclass


@dataclass(frozen=False)
class Parameters:
    """NetworkMaster parameters"""

    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    network_model_type: str = 'default'
    preproc_model_type: str = 'default'
    embed_model_type: str = 'default'
    epochs: int = 8
    batch_size: int = 1
    learning_rate: float = 1e-3
