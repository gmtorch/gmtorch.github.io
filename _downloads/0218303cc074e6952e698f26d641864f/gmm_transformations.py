# ruff: noqa: D205, D400, INP001
"""
Transforming a 2D GMM
=====================

This example applies geometric transformations to a two-dimensional Gaussian mixture.
The same operations are available as in-place methods on ``GaussianMixture`` and as
non-mutating helper functions in ``gmtorch.arithmetic``.
"""

import torch
from _plotting import plot_2d_gmm_density_panels

from gmtorch import GaussianMixture
from gmtorch.arithmetic import (
    affine_transform_gmm,
    rotate_2d_gmm,
    scale_gmm,
    shift_gmm,
)
from gmtorch.propagation import Rotate2D, Scale, Shift, closed_form_propagation

N_COMPONENTS = 3
DIM = 2
DTYPE = torch.float64

# %%
# 1. Build a reference mixture
# ----------------------------
#
# The mixture is asymmetric and has full covariance matrices. That makes translation,
# scaling, rotation, and shearing visually distinct.

weights = torch.tensor([0.30, 0.45, 0.25], dtype=DTYPE)
means = torch.tensor(
    [
        [-2.5, -1.0],
        [0.5, 2.0],
        [2.8, -0.8],
    ],
    dtype=DTYPE,
)
covariances = torch.tensor(
    [
        [[0.70, 0.25], [0.25, 0.45]],
        [[0.55, -0.20], [-0.20, 0.90]],
        [[0.40, 0.08], [0.08, 0.35]],
    ],
    dtype=DTYPE,
)

reference_gmm = GaussianMixture(
    n_components=N_COMPONENTS,
    dim=DIM,
    weights=weights,
    means=means,
    covariances=covariances,
)

plot_2d_gmm_density_panels([reference_gmm], ["Reference GMM"])

# %%
# 2. Scale, shift, and rotate
# ---------------------------
#
# The functional helpers return transformed copies and leave the input mixture unchanged:
#
# - ``scale_gmm(gmm, s)`` maps ``x`` to ``s * x``.
# - ``shift_gmm(gmm, b)`` maps ``x`` to ``x + b``.
# - ``rotate_2d_gmm(gmm, angle, center)`` rotates around a point.
#
# The underlying in-place methods follow the PyTorch naming convention and end in an
# underscore, for example ``scale_``.

scale = 1.4
offset = torch.tensor([2.5, -1.5], dtype=DTYPE)
angle_degrees = 45.0
angle = torch.deg2rad(torch.tensor(angle_degrees, dtype=DTYPE))
rotation_center = torch.tensor([0.5, -0.5], dtype=DTYPE)

scaled_gmm = scale_gmm(reference_gmm, scale)
shifted_gmm = shift_gmm(reference_gmm, offset)
rotated_gmm = rotate_2d_gmm(reference_gmm, angle, center=rotation_center)

in_place_scaled_gmm = reference_gmm.clone()
returned_gmm = in_place_scaled_gmm.scale_(scale)
assert returned_gmm is in_place_scaled_gmm
torch.testing.assert_close(in_place_scaled_gmm.means, scaled_gmm.means)
torch.testing.assert_close(reference_gmm.means, means)

_, axes = plot_2d_gmm_density_panels(
    [reference_gmm, scaled_gmm, shifted_gmm, rotated_gmm],
    [
        "Reference",
        f"Scaled by {scale}",
        f"Shifted by {offset.tolist()}",
        f"Rotated {angle_degrees:.0f} deg",
    ],
    n_rows=2,
    n_cols=2,
)

axes[-1].scatter(
    rotation_center[0].item(),
    rotation_center[1].item(),
    marker="o",
    s=55,
    color="black",
    label="rotation center",
)
axes[-1].legend(loc="best")

# %%
# 3. Apply a general affine transformation
# ----------------------------------------
#
# A full-row-rank matrix ``A`` and offset ``b`` map the random variable as
# ``x' = A x + b``. For each component, the transformed parameters are
# ``mu' = A mu + b`` and ``Sigma' = A Sigma A.T``.

matrix = torch.tensor(
    [
        [1.15, 0.55],
        [-0.25, 0.80],
    ],
    dtype=DTYPE,
)
affine_offset = torch.tensor([-1.0, 1.25], dtype=DTYPE)

affine_gmm = affine_transform_gmm(reference_gmm, matrix, affine_offset)

expected_mean = reference_gmm.expectation() @ matrix.mT + affine_offset
torch.testing.assert_close(affine_gmm.expectation(), expected_mean)

plot_2d_gmm_density_panels(
    [reference_gmm, affine_gmm],
    ["Reference", "Affine transform"],
)

# %%
# 4. Chain in-place transformations
# ---------------------------------
#
# In-place methods return ``self``, so they can be chained. The order matters: here the
# mixture is scaled around the origin, then rotated around the origin, and finally
# shifted.

combined_gmm = (
    reference_gmm.clone()
    .scale_(0.85)
    .rotate_2d_(torch.deg2rad(torch.tensor(-30.0, dtype=DTYPE)))
    .shift_(torch.tensor([1.5, 0.75], dtype=DTYPE))
)

plot_2d_gmm_density_panels(
    [reference_gmm, combined_gmm],
    ["Reference", "Scaled, rotated, then shifted"],
)

# %%
# 5. Compose exact operations with ``closed_form_propagation``
# -------------------------------------------------------------
#
# Similarly we can use the function ``closed_form_propagation`` from `gmtorch.propagation` which applies a sequence
# of exact GMM operations and returns the resulting mixture.

propagated_gmm = closed_form_propagation(
    reference_gmm,
    [
        Scale(0.85),
        Rotate2D(torch.deg2rad(torch.tensor(-30.0, dtype=DTYPE))),
        Shift(torch.tensor([1.5, 0.75], dtype=DTYPE)),
    ],
)

torch.testing.assert_close(propagated_gmm.weights, combined_gmm.weights)
torch.testing.assert_close(propagated_gmm.means, combined_gmm.means)
torch.testing.assert_close(propagated_gmm.covariances, combined_gmm.covariances)

plot_2d_gmm_density_panels(
    [reference_gmm, propagated_gmm],
    ["Reference", "closed_form_propagation result"],
)
