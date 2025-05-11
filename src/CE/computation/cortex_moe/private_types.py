import torch
from typing import List, Tuple
Kernels = List[torch.Tensor]
Prefetch = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor]]
