"""
Minimal in-process shim for ``pytorch3d.renderer.implicit.harmonic_embedding``.

Meta's ``locate-3d/models/encoder_3djepa.py`` imports a single class from
pytorch3d::

    from pytorch3d.renderer.implicit.harmonic_embedding import HarmonicEmbedding

That's the only pytorch3d touchpoint in the entire Locate-3D model code.
Installing pytorch3d from source requires a CUDA build toolchain matched
to the runtime PyTorch wheel, which is non-trivial on the cluster. Since
``HarmonicEmbedding`` is a 10-line Fourier-feature module, we provide an
exact-output drop-in here and register it under the same import path so
Meta's source can stay vanilla.

Behavior matches pytorch3d 0.7.8 (the version pinned in
``locate-3d/environment.yml``):

    HarmonicEmbedding(n_harmonic_functions=6, omega_0=1.0,
                      logspace=True, append_input=True)

    forward(x, diag_cov=None, eps=1e-6) -> tensor of shape
        (..., D * 2 * n_harmonic_functions [+ D if append_input])

Usage::

    from tools._pytorch3d_shim import install_shim
    install_shim()  # idempotent; no-op if real pytorch3d is importable
    # ... now safe to import models.encoder_3djepa, models.locate_3d, etc.
"""

import sys
import types

import torch


class HarmonicEmbedding(torch.nn.Module):
    """Drop-in for ``pytorch3d.renderer.implicit.harmonic_embedding.HarmonicEmbedding``.

    Sin/cos Fourier-feature encoding of the input tensor's last dim::

        embed[..., 2k    , d] = sin(omega_0 * 2^k * x[..., d])     # k in [0, N)
        embed[..., 2k + 1, d] = cos(omega_0 * 2^k * x[..., d])

    flattened to ``(..., D * 2 * N)``, optionally with ``x`` itself
    appended (``append_input=True``).
    """

    def __init__(
        self,
        n_harmonic_functions: int = 6,
        omega_0: float = 1.0,
        logspace: bool = True,
        append_input: bool = True,
    ):
        super().__init__()
        if logspace:
            frequencies = 2.0 ** torch.arange(
                n_harmonic_functions, dtype=torch.float32
            )
        else:
            frequencies = torch.linspace(
                1.0,
                2.0 ** (n_harmonic_functions - 1),
                n_harmonic_functions,
                dtype=torch.float32,
            )
        self.register_buffer(
            "_frequencies", frequencies * omega_0, persistent=False
        )
        self.append_input = bool(append_input)

    def forward(self, x, diag_cov=None, eps: float = 1e-6):
        # diag_cov is a pytorch3d feature for stochastic harmonic embedding
        # (used by some NeRF-style mip-NeRF code). The Locate-3D encoder
        # never passes it, so we ignore the argument entirely.
        embed = (x[..., None] * self._frequencies).reshape(*x.shape[:-1], -1)
        embed = torch.cat((embed.sin(), embed.cos()), dim=-1)
        if self.append_input:
            embed = torch.cat((embed, x), dim=-1)
        return embed


def install_shim() -> bool:
    """Register the shim under ``pytorch3d.renderer.implicit.harmonic_embedding``.

    Returns True if the shim was actually installed (real pytorch3d was
    not importable), False if the real package is already available.
    Idempotent -- safe to call multiple times.
    """
    try:
        import pytorch3d.renderer.implicit.harmonic_embedding  # noqa: F401
        return False
    except ImportError:
        pass

    pkg = types.ModuleType("pytorch3d")
    pkg.__path__ = []  # implicit namespace
    renderer = types.ModuleType("pytorch3d.renderer")
    renderer.__path__ = []
    implicit = types.ModuleType("pytorch3d.renderer.implicit")
    implicit.__path__ = []
    he_mod = types.ModuleType("pytorch3d.renderer.implicit.harmonic_embedding")
    he_mod.HarmonicEmbedding = HarmonicEmbedding

    sys.modules.setdefault("pytorch3d", pkg)
    sys.modules.setdefault("pytorch3d.renderer", renderer)
    sys.modules.setdefault("pytorch3d.renderer.implicit", implicit)
    sys.modules.setdefault("pytorch3d.renderer.implicit.harmonic_embedding", he_mod)
    return True


if __name__ == "__main__":
    # Quick sanity: shim's HarmonicEmbedding produces the same output
    # shape / dtype as pytorch3d 0.7.8.
    he = HarmonicEmbedding(n_harmonic_functions=4, omega_0=1.0,
                           logspace=True, append_input=True)
    x = torch.randn(2, 3, 5)
    y = he(x)
    expected = (2, 3, 5 * 2 * 4 + 5)  # D * 2N + D
    assert tuple(y.shape) == expected, (y.shape, expected)
    print("OK shim produces", y.shape, "as expected")
