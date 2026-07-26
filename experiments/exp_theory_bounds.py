"""B-5 theory artifacts: explicit GELU second-order bound + general-affine ViT bound.

Reviewer B-5 asked for two things the rebuttal promised but did not deliver:
  (a) an explicit second-order (Lagrange-remainder) bound on the GELU linearization
      error at eps=8/255, with a numerical value — not an assertion of "negligible";
  (b) a bound on the ViT patch-embedding activation perturbation that does NOT assume
      Toeplitz / shift-invariance (the patch embedding is a general linear projection).

Both are computed here. (a) is closed-form + numeric; (b) uses the induced infinity-norm
(max absolute row sum) of the real vit_base patch-embedding weight, which bounds the
output perturbation for ANY linear map: ||W_p delta||_inf <= ||W_p||_inf * ||delta||_inf.

    python experiments/exp_theory_bounds.py
"""
from __future__ import annotations

import numpy as np


def gelu_second_deriv_max() -> tuple[float, float]:
    """max |g''(x)| for GELU g(x)=x*Phi(x). g''(x)=phi(x)*(2 - x^2)."""
    x = np.linspace(-6, 6, 200001)
    phi = np.exp(-(x**2) / 2) / np.sqrt(2 * np.pi)
    g2 = phi * (2 - x**2)
    i = np.argmax(np.abs(g2))
    return float(np.abs(g2[i])), float(x[i])


def main() -> None:
    eps = 8.0 / 255.0
    print("=== B-5(a): GELU linearization error (Lagrange remainder) ===")
    M, xstar = gelu_second_deriv_max()
    rem = 0.5 * M * eps**2
    print(f"  GELU g(x)=x*Phi(x),  g''(x)=phi(x)*(2 - x^2)")
    print(f"  max|g''(x)| = {M:.4f}  at x={xstar:+.4f}  (closed form: 2*phi(0)=0.7979 at x=0)")
    print(f"  eps = 8/255 = {eps:.5f}")
    print(f"  |g(x+d) - g(x) - g'(x)d| <= (1/2) max|g''| eps^2 = {rem:.3e}  per coordinate")
    print(f"  => the second-order term is <= {rem:.2e}, i.e. ~{rem:.0e} — negligible vs unit-scale")
    print(f"     pre-activations; the linear (first-order) analysis is valid to O(eps^2)={eps**2:.1e}.\n")

    print("=== B-5(b): general-affine ViT patch-embedding bound (no Toeplitz needed) ===")
    try:
        import timm
        import torch

        m = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=100)
        # patch embed projection: Conv2d(3, D, kernel=16, stride=16) == linear per patch
        proj = m.patch_embed.proj
        W = proj.weight.detach().reshape(proj.out_channels, -1).numpy()  # (D, 3*16*16)
        row_abs_sum = np.abs(W).sum(axis=1)
        inf_norm = float(row_abs_sum.max())  # induced inf-norm = max abs row sum
        l1_max = float(np.abs(W).sum(axis=0).max())
        print(f"  W_p shape = {W.shape} (D x (3*16*16));  general linear map, NOT Toeplitz/shift-invariant")
        print(f"  Induced infinity-norm ||W_p||_inf (max abs row sum) = {inf_norm:.4f}")
        print(f"  Bound holds for ANY affine map:  ||W_p delta||_inf <= ||W_p||_inf * ||delta||_inf")
        print(f"  At ||delta||_inf = eps = {eps:.5f}:  ||W_p delta||_inf <= {inf_norm * eps:.4f}")
        print(f"  (Per-coordinate column L1 max ||W_p||_1 = {l1_max:.4f} gives the dual bound.)")
        print(f"  => the Prop-1 operator-norm control extends to the patch embedding via the")
        print(f"     standard induced-norm inequality, WITHOUT requiring Toeplitz structure.")
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] could not load vit_base ({type(e).__name__}: {e}); (a) is independent of this.")


if __name__ == "__main__":
    main()
