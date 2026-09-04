"""CNN arms for the privacy × deep-learning dermoscopy benchmark.

Track: `experiment/image` (design contract: `research/privacy-dl-image/README.md`).
- **CNN-S** (~28.6k params, ≈0.11 MB fp32 wire): the reference arm — sized so record-level
  DP-SGD's $\\sqrt{p}$ noise penalty stays in the same order as the track's logreg baseline,
  isolating "deep model per se" from "parameter count per se".
- **CNN-M** (ResNet-lite, ≈1.5M params, ≈5.9 MB): the scaling probe that shows what record-level
  DP actually costs once $p\\to 10^6$ on images.

Input contract: frames from the mapper are uint8 pixels flattened row-major RGB pixel-major
(`f0000..f2351` for 28×28). The runner rescales to [0, 1] and reshapes to (N, 3, S, S).

Same param-vector flattening contract as the TS tree (`ParamVectorMixin` semantics) so the
Flower strategies see one dense float32 vector, mirroring how `models/logreg.py` presents
`coef_`/`intercept_` to FedAvg.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

CROP = {28: 28, 64: 64}


class ParamVectorMixin:
    """Expose parameters as a single dense float32 vector (the privacy-machinery contract)."""

    def param_vector(self) -> torch.Tensor:
        parts = [t.detach().reshape(-1) for t in self.state_dict().values()
                 if t.dtype.is_floating_point]
        return torch.cat(parts) if parts else torch.zeros(0)

    def load_param_vector(self, vec: torch.Tensor) -> None:
        ptr = 0
        with torch.no_grad():
            for t in self.state_dict().values():
                if not t.dtype.is_floating_point:
                    continue
                n = t.numel()
                t.copy_(vec[ptr : ptr + n].view_as(t).type_as(t))
                ptr += n
        assert ptr == vec.numel(), f"vector has {vec.numel()} scalars, model consumed {ptr}"

    def num_params(self) -> int:
        return sum(t.numel() for t in self.parameters())


class CNNSmall(ParamVectorMixin, nn.Module):
    """conv(3→16,k5)+pool × conv(16→32,k3)+pool × conv(32→64,k3)+pool → GAP → fc → logit."""

    def __init__(self, size: int = 28):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 5), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.conv(x)).squeeze(-1)


class _BasicBlock(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(cout)
        self.short = (
            nn.Sequential(nn.Conv2d(cin, cout, 1, stride=stride, bias=False),
                          nn.BatchNorm2d(cout))
            if (stride != 1 or cin != cout) else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.b1(self.c1(x)))
        out = self.b2(self.c2(out))
        return F.relu(out + self.short(x))


class ResNetLite(ParamVectorMixin, nn.Module):
    """Channel-widths (32, 64, 128, 256) × 2 basic blocks — CNN-M scaling probe arm."""

    def __init__(self, size: int = 28, widths: tuple = (32, 64, 128, 256)):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, widths[0], 3, padding=1, bias=False),
                                  nn.BatchNorm2d(widths[0]), nn.ReLU())
        stages = []
        cin = widths[0]
        for i, w in enumerate(widths[1:]):
            stride = 1 if i == 0 else 2
            stages.append(_BasicBlock(cin, w, stride))
            stages.append(_BasicBlock(w, w, 1))
            cin = w
        self.stages = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(widths[-1], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.pool(self.stages(self.stem(x))).squeeze(-1).squeeze(-1)
        return self.head(z).squeeze(-1)


def to_images(X: torch.Tensor, size: int = 28) -> torch.Tensor:
    """(N, 3*S*S) flattened row-major RGB pixel-major uint8-valued floats → (N, 3, S, S) [0,1]."""
    return (X.reshape(-1, size, size, 3).permute(0, 3, 1, 2) / 255.0).contiguous()


def temperature_check() -> dict:
    """Param counts + forward-shape unit smoke."""
    torch.manual_seed(42)
    s, m = CNNSmall(), ResNetLite()
    x = torch.randn(4, 3, 28, 28)
    out = {"CNN-S": {"params": s.num_params(), "out": tuple(s(x).shape)},
           "CNN-M": {"params": m.num_params(), "out": tuple(m(x).shape)}}
    v = s.param_vector()
    clone = CNNSmall(); clone.load_param_vector(v + 1.0)
    assert torch.allclose(clone.param_vector(), v + 1.0), "param vector round-trip broken"
    out["vector_roundtrip"] = True
    return out


if __name__ == "__main__":
    import pprint

    pprint.pprint(temperature_check())
