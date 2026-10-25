"""Moving MNIST loading, symmetry transforms and speed windows.

Conventions (handout): frames are numbered 1..20, so a start frame t is 1-based;
Python indices are 0-based. Every function here is used both by the Table 1
checks and by the experiments.
"""
import torch
import torchvision

N_FRAMES = 20
N_IN, N_OUT = 5, 5

# Allowed start frames t (1-based, inclusive upper bound).
MAX_T_NORMAL = 11                   # models A, B: all five targets exist (1x only)
MAX_T_SPEED = {1: 14, 2: 8, 3: 2}   # models C, D: 5 inputs + at least 2 targets


def load_moving_mnist(root="data"):
    """Return (train, val, test) uint8 tensors of shape (N, 20, 1, 64, 64), split by video index."""
    ds = torchvision.datasets.MovingMNIST(root=root, split=None, download=True)
    v = ds.data
    assert v.shape == (10000, 20, 1, 64, 64) and v.dtype == torch.uint8, (v.shape, v.dtype)
    return v[:8000], v[8000:9000], v[9000:]


def to_float(x):
    """uint8 -> float32 in [0, 1]."""
    return x.float() / 255.0


# ---------------------------------------------------------------- transforms
# All act on the last two (row, column) axes, so they work for any leading shape.

def M(x):
    """Mirror: (Mx)[i, j] = x[i, 63 - j]."""
    return x.flip(-1)


def R(x, a=1):
    """Rotate by a * 90 degrees anticlockwise."""
    return torch.rot90(x, a, dims=(-2, -1))


# The 8 symmetries of the square as (a, m): g = M^m R^a ("rotate first, then mirror").
G = [(a, m) for m in (0, 1) for a in range(4)]


def g_apply(x, a, m):
    y = R(x, a)
    return M(y) if m else y


def g_inverse(x, a, m):
    # R^a has inverse R^-a; M R^a is its own inverse.
    return g_apply(x, a, m) if m else R(x, -a)


# ---------------------------------------------------------------- windows

def window_indices(t, k):
    """0-based frame indices of the input and (existing) target frames for start t (1-based), speed k."""
    idx_in = [t - 1 + k * i for i in range(N_IN)]
    idx_tg = [t - 1 + k * i for i in range(N_IN, N_IN + N_OUT)]
    idx_tg = [i for i in idx_tg if i < N_FRAMES]
    return idx_in, idx_tg


def window(video, t, k=1):
    """Cut a window from video (20, C, H, W).

    Returns X (5, C, H, W), Y (5, C, H, W) zero-padded where the target frame does
    not exist, and mask (5,) with 1 for existing target frames.
    """
    idx_in, idx_tg = window_indices(t, k)
    assert len(idx_in) == N_IN and idx_in[-1] < N_FRAMES, f"inputs out of range for t={t}, k={k}"
    X = video[idx_in]
    Y = torch.zeros_like(X)
    Y[: len(idx_tg)] = video[idx_tg]
    mask = torch.zeros(N_OUT)
    mask[: len(idx_tg)] = 1
    return X, Y, mask


def batch_window(videos, t, k=1):
    """Same window for a batch of videos (B, 20, C, H, W) -> X, Y (B, 5, ...), mask (5,)."""
    idx_in, idx_tg = window_indices(t, k)
    X = videos[:, idx_in]
    Y = torch.zeros_like(X)
    Y[:, : len(idx_tg)] = videos[:, idx_tg]
    mask = torch.zeros(N_OUT)
    mask[: len(idx_tg)] = 1
    return X, Y, mask
