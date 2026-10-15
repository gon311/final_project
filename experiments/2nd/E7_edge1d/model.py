"""E7 1D CNN: 파라미터 50k 이하, bias 없는 conv + BatchNorm(MKD 논문의 상수 출력 붕괴
경고 반영, 지시서 "모델" 절). 입력은 길이 81(half_window=40) 원본 밝기(0~255) 프로파일
1개다.

정규화: conv에 넣기 전에 프로파일마다 (x - median(x)) / (std(x)+eps)로 z-score
정규화한다 — 밝기·대비 증강이 절대 레벨을 바꿔도 엣지의 상대적 모양은 거의 그대로이므로
(E2에서 이 증강들의 Δ상단·하단이 0.03px 판정선 안쪽이었던 것과 같은 이유), 모델이
절대 레벨이 아니라 모양에서 엣지 위치를 읽게 유도한다.

헤드 두 종류:
- RegressionHead: 전역 평균 풀링 -> Linear(C,1). 서브픽셀 오프셋을 직접 회귀.
- HeatmapHead: 1x1 conv로 길이 L 로짓을 만들고 softmax + soft-argmax(기대값)로
  서브픽셀 위치를 얻는다.
"""
import torch
import torch.nn as nn


class Backbone(nn.Module):
    """4개 conv층, 기본 채널로 약 34k 파라미터(헤드 포함해도 50k 이하) — teacher 좌표에
    기대지 않는 coarse 중심에서 실제 엣지를 찾아야 하는 더 어려운 과제(Codex
    adversarial-review [high] 반영 이후, `train_eval.py`·`build_cache.py` 참고)에 맞춰
    처음 버전(c1=16,c2=24,c3=24, 3층, 5065 파라미터)보다 용량을 키웠다 — 작은 모델은
    300 epoch까지 늘려도 0.141px(0.05px 기준 못 미침)에서 수렴이 느려지는 정체를
    보였다(README 한계 참고)."""

    def __init__(self, c1: int = 32, c2: int = 48, c3: int = 48, c4: int = 48):
        super().__init__()
        self.conv1 = nn.Conv1d(1, c1, kernel_size=9, padding=4, bias=False)
        self.bn1 = nn.BatchNorm1d(c1)
        self.conv2 = nn.Conv1d(c1, c2, kernel_size=7, padding=3, bias=False)
        self.bn2 = nn.BatchNorm1d(c2)
        self.conv3 = nn.Conv1d(c2, c3, kernel_size=5, padding=2, bias=False)
        self.bn3 = nn.BatchNorm1d(c3)
        self.conv4 = nn.Conv1d(c3, c4, kernel_size=5, padding=2, bias=False)
        self.bn4 = nn.BatchNorm1d(c4)
        self.act = nn.ReLU(inplace=True)
        self.out_channels = c4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        x = self.act(self.bn3(self.conv3(x)))
        x = self.act(self.bn4(self.conv4(x)))
        return x  # (B, c4, L)


class RegressionHead(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(in_channels, 1)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        x = self.pool(feat).squeeze(-1)
        return self.fc(x).squeeze(-1)


class HeatmapHead(nn.Module):
    def __init__(self, in_channels: int, half_window: int):
        super().__init__()
        self.half_window = half_window
        self.logits = nn.Conv1d(in_channels, 1, kernel_size=1, bias=False)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        logit = self.logits(feat).squeeze(1)  # (B, L)
        prob = torch.softmax(logit, dim=-1)
        positions = torch.arange(logit.shape[-1], device=logit.device, dtype=logit.dtype) - self.half_window
        return (prob * positions).sum(dim=-1)


class EdgeCNN1D(nn.Module):
    def __init__(self, head: str = "reg", half_window: int = 40):
        super().__init__()
        self.backbone = Backbone()
        self.half_window = half_window
        if head == "reg":
            self.head = RegressionHead(self.backbone.out_channels)
        elif head == "heatmap":
            self.head = HeatmapHead(self.backbone.out_channels, half_window)
        else:
            raise ValueError(f"unknown head: {head!r}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L) 원본 밝기 프로파일. 반환: (B,) 중심 기준 서브픽셀 오프셋 예측."""
        med = x.median(dim=-1, keepdim=True).values
        std = x.std(dim=-1, keepdim=True) + 1e-6
        xn = (x - med) / std
        feat = self.backbone(xn.unsqueeze(1))
        return self.head(feat)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    for head in ("reg", "heatmap"):
        m = EdgeCNN1D(head=head)
        n = count_params(m)
        print(head, n, "OK" if n <= 50_000 else "FAIL >50k")
