"""ConvLSTM forecaster (Shi et al., 2015, without peepholes) with the handout's 9-step rollout."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLSTMCell(nn.Module):
    def __init__(self, cin, ch, k=3, dilation=1):
        super().__init__()
        self.ch = ch
        # One convolution produces all four gates [i, f, o, g].
        self.conv = nn.Conv2d(cin + ch, 4 * ch, k, padding=dilation * (k // 2), dilation=dilation, bias=True)

    def forward(self, u, h, c):
        i, f, o, g = self.conv(torch.cat([u, h], 1)).chunk(4, 1)
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h = torch.sigmoid(o) * torch.tanh(c)
        return h, c


class ConvLSTMForecaster(nn.Module):
    """Model A with the defaults (patch 4, two layers of 64 channels, 3x3 kernels)."""

    def __init__(self, ch=64, patch=4, k=3, dilation=1):
        super().__init__()
        self.ch, self.patch = ch, patch
        cin = patch * patch
        self.l1 = ConvLSTMCell(cin, ch, k, dilation)
        self.l2 = ConvLSTMCell(ch, ch, k, dilation)
        self.out = nn.Conv2d(ch, cin, 1, bias=True)

    def forward(self, X, n_in=5, n_out=5):
        """X (B, 5, 1, 64, 64) -> Y_hat (B, 5, 1, 64, 64).

        Steps 1..5 take the input frames; steps 6..9 take the network's own previous
        prediction (no teacher forcing, gradients flow through it). Outputs of
        steps 5..9 are the forecast.
        """
        B, _, _, H, W = X.shape
        s = (H // self.patch, W // self.patch)
        h1 = c1 = h2 = c2 = X.new_zeros(B, self.ch, *s)
        outs, x_hat = [], None
        for step in range(n_in + n_out - 1):
            u = X[:, step] if step < n_in else x_hat
            u = F.pixel_unshuffle(u, self.patch)
            h1, c1 = self.l1(u, h1, c1)
            h2, c2 = self.l2(h1, h2, c2)
            x_hat = torch.sigmoid(F.pixel_shuffle(self.out(h2), self.patch))
            if step >= n_in - 1:
                outs.append(x_hat)
        return torch.stack(outs, 1)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    m = ConvLSTMForecaster()
    n = count_params(m)
    print(f"model A trainable parameters: {n:,} (expected 480,784) -> {'OK' if n == 480_784 else 'MISMATCH'}")
    y = m(torch.rand(2, 5, 1, 64, 64))
    print("output shape:", tuple(y.shape))
