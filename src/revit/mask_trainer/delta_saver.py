from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import torch


class FP16DeltaSaver:
    """
    Convert edited weights
    into fp16 deltas ΔW = (W - W0), saving:
      - (dict: {'delta_fp16': Tensor[out,in] fp16,
                'shape': int32[2]}).
    """

    def __init__(self, w0_path: Path, edited_dir: Path, out_dir: Path):
        self.w0_path = Path(w0_path)
        self.edited_dir = Path(edited_dir)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _load_w(w_path: Path) -> torch.Tensor:
        w_path = Path(w_path)
        if w_path.suffix == ".pt":
            W0 = torch.load(str(w_path), map_location="cpu")
            return W0.to(torch.float32).contiguous()
        if w_path.suffix == ".npz":
            arr = np.load(str(w_path))["arr"]
            return torch.from_numpy(arr).to(torch.float32).contiguous()
        raise ValueError(f"Unsupported W0 file: {w_path}")

    def convert(self, pattern: str = "*", overwrite: bool = True) -> Dict:
        """Convert all W files and return a summary dict."""
        W0 = self._load_w(self.w0_path)  # CPU fp32

        files: Iterable[Path] = sorted(
            p for p in self.edited_dir.glob(pattern) if p.suffix in (".pt", ".npz")
        )
        saved = []
        for p in files:
            hat = self._load_w(p)  # CPU fp32
            if hat.shape != W0.shape:
                raise ValueError(
                    f"Shape mismatch: {p.name} {tuple(hat.shape)} vs W0 {tuple(W0.shape)}"
                )
            delta_fp16 = (hat - W0).to(torch.float16).contiguous()
            out_path = self.out_dir / f"{p.stem}.pt"
            if out_path.exists() and not overwrite:
                continue
            torch.save(
                {
                    "delta_fp16": delta_fp16,
                    "shape": torch.tensor(hat.shape, dtype=torch.int32),
                },
                out_path,
            )
            saved.append(out_path.name)

        return {"n_files": len(saved), "out_dir": str(self.out_dir), "saved": saved}
