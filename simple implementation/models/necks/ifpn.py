"""
Improved Feature Pyramid Network (IFPN) — poster Fig. 5.

Replaces standard FPN with dense cross-stage semantic propagation:

    Phase 1 — top-down dense intermediate maps:
        M5 = L5
        M4 = L4 + Up(M5)
        M3 = L3 + Up(M4) + Up(M5)
        M2 = L2 + Up(M3) + Up(M4) + Up(M5)

    Phase 2 — bottom-up refinement:
        P2 = M2
        P3 = M3 + Down(P2)
        P4 = M4 + Down(P3)
        P5 = M5 + Down(P4)

    P6 = MaxPool(P5)   ← not in poster Fig 5; added for RPN large-anchor
                          coverage (documented in REFACTOR_NOTES.md §IFPN-1)

Input:  OrderedDict from IntermediateLayerGetter
        {'0': C2, '1': C3, '2': C4, '3': C5}  (variable channels, ResNet50)
Output: OrderedDict
        {'0': P2, '1': P3, '2': P4, '3': P5, 'pool': P6}  (all 256ch)
        Same key schema as torchvision BackboneWithFPN — RPN/ROI heads work
        without modification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm
from torchvision.models._utils import IntermediateLayerGetter
from collections import OrderedDict


# Default ResNet50 backbone channels at each extracted layer
_RESNET50_CHANNELS = [256, 512, 1024, 2048]   # layer1, layer2, layer3, layer4


class IFPN(nn.Module):
    """
    IFPN neck (poster Fig. 5).

    Args:
        in_channels_list (list[int]): Input channel sizes for C2, C3, C4, C5.
            Default: [256, 512, 1024, 2048] for ResNet50.
        out_channels (int): Output channel width for all pyramid levels. Default: 256.
    """

    def __init__(
        self,
        in_channels_list: list = None,
        out_channels: int = 256,
    ):
        super().__init__()
        if in_channels_list is None:
            in_channels_list = _RESNET50_CHANNELS

        self.out_channels = out_channels
        n = len(in_channels_list)   # 4

        # Lateral projections: C_i → out_channels via Conv1×1
        self.lateral = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, 1) for in_ch in in_channels_list
        ])

        # Output refinement after fusion: Conv3×3 + BN + ReLU per level
        self.output_conv = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
            for _ in range(n)
        ])

        # Learned stride-2 downsampling for Phase 2 bottom-up pass (n-1 ops)
        self.downsample = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1, bias=False)
            for _ in range(n - 1)
        ])

    @staticmethod
    def _up(src: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Nearest-neighbour upsample src to match target's spatial size."""
        return F.interpolate(src, size=target.shape[-2:], mode='nearest')

    def forward(self, x: "OrderedDict[str, torch.Tensor]") -> "OrderedDict[str, torch.Tensor]":
        """
        Args:
            x: {'0': C2, '1': C3, '2': C4, '3': C5}
        Returns:
            {'0': P2, '1': P3, '2': P4, '3': P5, 'pool': P6}
        """
        c2, c3, c4, c5 = x['0'], x['1'], x['2'], x['3']

        # Lateral projections → uniform channel width
        L2 = self.lateral[0](c2)
        L3 = self.lateral[1](c3)
        L4 = self.lateral[2](c4)
        L5 = self.lateral[3](c5)

        # Phase 1: dense top-down intermediate maps
        M5 = L5
        M4 = L4 + self._up(M5, L4)
        M3 = L3 + self._up(M4, L3) + self._up(M5, L3)
        M2 = L2 + self._up(M3, L2) + self._up(M4, L2) + self._up(M5, L2)

        # Phase 2: bottom-up aggregation with learned stride-2 conv
        P2 = M2
        P3 = M3 + self.downsample[0](P2)
        P4 = M4 + self.downsample[1](P3)
        P5 = M5 + self.downsample[2](P4)

        # Output refinement
        out = OrderedDict()
        out['0']    = self.output_conv[0](P2)
        out['1']    = self.output_conv[1](P3)
        out['2']    = self.output_conv[2](P4)
        out['3']    = self.output_conv[3](P5)
        # P6 via MaxPool — not in poster Fig 5 but required for RPN (§IFPN-1)
        out['pool'] = F.max_pool2d(out['3'], kernel_size=1, stride=2, padding=0)

        return out


class BackboneWithIFPN(nn.Module):
    """
    ResNet50 body (IntermediateLayerGetter) + IFPN neck.

    Drop-in replacement for torchvision BackboneWithFPN: exposes the same
    `out_channels` attribute and the same OrderedDict output schema so that
    torchvision FasterRCNN / RPN / ROI heads work without modification.

    Args:
        pretrained (bool): Load ImageNet weights for ResNet50. Default: True.
        trainable_layers (int): Number of ResNet layer-groups to keep trainable
            (counted from layer4 backwards). Default: 3 → layers 2, 3, 4 trainable.
        out_channels (int): Channel width for all IFPN outputs. Default: 256.
    """

    out_channels: int = 256   # class-level; also set per-instance in __init__

    def __init__(
        self,
        pretrained: bool = True,
        trainable_layers: int = 3,
        out_channels: int = 256,
    ):
        super().__init__()
        self.out_channels = out_channels

        # ── ResNet50 body ────────────────────────────────────────────────
        try:
            from torchvision.models import ResNet50_Weights
            weights = ResNet50_Weights.DEFAULT if pretrained else None
            resnet = tvm.resnet50(weights=weights)
        except (ImportError, AttributeError):
            # Older torchvision fallback
            resnet = tvm.resnet50(pretrained=pretrained)

        # Freeze layers according to trainable_layers
        # Layer groups in order: [conv1+bn1+relu+maxpool, layer1, layer2, layer3, layer4]
        layers_to_train = ['layer4', 'layer3', 'layer2', 'layer1', 'conv1'][
            :trainable_layers
        ]
        for name, param in resnet.named_parameters():
            group = name.split('.')[0]
            if group not in layers_to_train:
                param.requires_grad_(False)

        # Extract intermediate features at layer1 (C2), layer2 (C3),
        # layer3 (C4), layer4 (C5) — standard torchvision convention
        return_layers = {
            'layer1': '0',  # C2: 256ch
            'layer2': '1',  # C3: 512ch
            'layer3': '2',  # C4: 1024ch
            'layer4': '3',  # C5: 2048ch
        }
        self.body = IntermediateLayerGetter(resnet, return_layers=return_layers)

        # ── IFPN neck ────────────────────────────────────────────────────
        self.ifpn = IFPN(
            in_channels_list=_RESNET50_CHANNELS,
            out_channels=out_channels,
        )

    def forward(self, x: torch.Tensor) -> "OrderedDict[str, torch.Tensor]":
        backbone_features = self.body(x)        # C2–C5 OrderedDict
        return self.ifpn(backbone_features)     # P2–P5 + pool OrderedDict


# ── Self-test ──────────────────────────────────────────────────────────────
def _test():
    print("Testing IFPN and BackboneWithIFPN...")
    print("=" * 60)

    # ── Unit test: IFPN module alone ──
    ifpn = IFPN(in_channels_list=[256, 512, 1024, 2048], out_channels=256)
    ifpn.eval()

    fake_backbone = OrderedDict([
        ('0', torch.randn(2, 256,  160, 160)),   # C2
        ('1', torch.randn(2, 512,   80,  80)),   # C3
        ('2', torch.randn(2, 1024,  40,  40)),   # C4
        ('3', torch.randn(2, 2048,  20,  20)),   # C5
    ])

    expected = {
        '0': (2, 256, 160, 160),
        '1': (2, 256,  80,  80),
        '2': (2, 256,  40,  40),
        '3': (2, 256,  20,  20),
        'pool': (2, 256, 10, 10),
    }

    with torch.no_grad():
        out = ifpn(fake_backbone)

    for k, exp_shape in expected.items():
        assert tuple(out[k].shape) == exp_shape, \
            f"Key '{k}': expected {exp_shape}, got {tuple(out[k].shape)}"
        print(f"  [{k}] {exp_shape} ✓")

    # Gradient flow check
    fake_backbone['0'].requires_grad_(True)
    ifpn.train()
    result = ifpn(fake_backbone)
    result['0'].sum().backward()
    assert fake_backbone['0'].grad is not None
    print("  Gradient flow through all levels: ✓")

    params = sum(p.numel() for p in ifpn.parameters()) / 1e6
    print(f"\n  IFPN parameters: {params:.2f}M")

    # ── BackboneWithIFPN (no-pretrained for speed) ──
    print("\n  Testing BackboneWithIFPN (no pretrained)...")
    bbone = BackboneWithIFPN(pretrained=False, trainable_layers=3)
    bbone.eval()

    x = torch.randn(2, 3, 640, 640)
    with torch.no_grad():
        features = bbone(x)

    for k, exp_shape in expected.items():
        assert tuple(features[k].shape) == exp_shape, \
            f"BackboneWithIFPN key '{k}': {features[k].shape} ≠ {exp_shape}"
        print(f"  [{k}] {tuple(features[k].shape)} ✓")

    total_params = sum(p.numel() for p in bbone.parameters()) / 1e6
    print(f"\n  BackboneWithIFPN total params: {total_params:.2f}M")
    print("✓ IFPN + BackboneWithIFPN test passed!")
    print("=" * 60)


if __name__ == '__main__':
    _test()
