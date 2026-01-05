from typing import Dict

import numpy as np
import torch

from ..models import ModelName

W0_PATH = "../output/{}.npz"
QUANT_DIR = "../output/quant/{}_{}/"


class FP16DeltaLoader:
    """
    fp16-only loader
    Each pack contains {'delta_fp16': Tensor[out,in] (cpu fp16), 'shape': int32[2]}.
    """

    def __init__(self, device: torch.device, model_name: ModelName):
        self.device = device
        self.model_name = model_name
        w0_path = W0_PATH.format(model_name)
        W0 = torch.from_numpy(np.load(w0_path)["arr"])
        self.W0_cpu = W0.to(torch.float32)
        self.W0 = self.W0_cpu.to(device, non_blocking=True)

    def _load_pack_cpu(self, case_id: str, rel_id: str) -> Dict[str, torch.Tensor]:
        p = QUANT_DIR.format(self.model_name, rel_id) + f"{case_id}.pt"
        pack = torch.load(p, map_location="cpu")
        # Pin for async H2D
        if torch.cuda.is_available():
            pack["delta_fp16"] = pack["delta_fp16"].pin_memory()
            pack["shape"] = pack["shape"].pin_memory()
        else:
            pack["delta_fp16"] = pack["delta_fp16"].contiguous()
            pack["shape"] = pack["shape"]
        return pack

    def reconstruct(self, case_id: str, rel_id: str, device: torch.device = None) -> torch.Tensor:
        if device is None:
            device = self.device
            
        pack = self._load_pack_cpu(case_id, rel_id)
        
        if device.type == "cpu":
            delta = pack["delta_fp16"].to(torch.float32)
            return self.W0_cpu + delta
        else:
            delta = pack["delta_fp16"].to(device, non_blocking=True).to(torch.float32)
            if device == self.device:
                return self.W0 + delta
            return self.W0_cpu.to(device) + delta

    def reconstruct_on_device(self, case_id: str, rel_id: str) -> torch.Tensor:
        return self.reconstruct(case_id, rel_id, self.device)
