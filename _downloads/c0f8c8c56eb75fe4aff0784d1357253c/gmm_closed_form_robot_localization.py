# ruff: noqa: D205, D400, INP001, T201
"""
Robot Localization Under Uncertainty
====================================

This example models 2D robot localization with Gaussian mixtures. A robot starts with a
multimodal position belief, propagates that belief through an exact odometry update with
mixture-valued process noise, and then uses Bayesian fusion to combine the prediction
with an ambiguous sensor likelihood.
"""

import torch
from _plotting import plot_2d_gmm_density_panels

from gmtorch import GaussianMixture, closed_form_propagation
from gmtorch.propagation import BayesianFusion, Rotate2D, Shift, SumIndependent

DIM = 2
DTYPE = torch.float64

# %%
# 1. Build the localization ingredients
# -------------------------------------
#
# The prior has three plausible robot locations. The process-noise GMM represents mostly
# centered odometry noise plus a less likely biased mode, such as wheel slip. The sensor
# likelihood has two modes because the landmark observation is ambiguous.

prior_gmm = GaussianMixture(
    n_components=3,
    dim=DIM,
    weights=torch.tensor([0.45, 0.35, 0.20], dtype=DTYPE),
    means=torch.tensor(
        [
            [-2.2, -0.6],
            [0.1, 1.3],
            [2.1, -1.0],
        ],
        dtype=DTYPE,
    ),
    covariances=torch.tensor(
        [
            [[0.45, 0.12], [0.12, 0.32]],
            [[0.55, -0.18], [-0.18, 0.40]],
            [[0.35, 0.05], [0.05, 0.28]],
        ],
        dtype=DTYPE,
    ),
)

process_noise_gmm = GaussianMixture(
    n_components=2,
    dim=DIM,
    weights=torch.tensor([0.75, 0.25], dtype=DTYPE),
    means=torch.tensor([[0.0, 0.0], [0.25, -0.15]], dtype=DTYPE),
    covariances=torch.tensor(
        [
            [[0.10, 0.02], [0.02, 0.08]],
            [[0.18, -0.03], [-0.03, 0.12]],
        ],
        dtype=DTYPE,
    ),
)

sensor_likelihood = GaussianMixture(
    n_components=2,
    dim=DIM,
    weights=torch.tensor([0.70, 0.30], dtype=DTYPE),
    means=torch.tensor([[0.75, 1.55], [3.05, -0.15]], dtype=DTYPE),
    covariances=torch.tensor(
        [
            [[0.35, 0.05], [0.05, 0.45]],
            [[0.45, -0.08], [-0.08, 0.35]],
        ],
        dtype=DTYPE,
    ),
)

plot_2d_gmm_density_panels(
    [prior_gmm, process_noise_gmm, sensor_likelihood],
    ["Prior belief", "Process noise", "Sensor likelihood"],
)

# %%
# 2. Predict the robot position
# -----------------------------
#
# The motion model is represented as a chain of exact operations:
#
# - ``Rotate2D`` applies a small heading or frame correction.
# - ``Shift`` applies the odometry translation.
# - ``SumIndependent`` adds independent process noise.
#
# The independent sum multiplies component counts, so the 3-component prior and
# 2-component process-noise model produce a 6-component prediction.

yaw_delta = torch.deg2rad(torch.tensor(17.0, dtype=DTYPE))
odometry_delta = torch.tensor([2.0, 0.35], dtype=DTYPE)

motion_operations = [
    Rotate2D(yaw_delta),
    Shift(odometry_delta),
    SumIndependent(process_noise_gmm),
]

prediction_gmm = closed_form_propagation(
    prior_gmm,
    motion_operations,
    max_components=8,
)

print("Motion update")
print(f"  yaw correction: {torch.rad2deg(yaw_delta).item():.1f} deg")
print(f"  odometry shift: {odometry_delta.tolist()}")
print(f"  prediction components: {prediction_gmm.n_components}")

# %%
# 3. Fuse the sensor likelihood
# -----------------------------
#
# ``BayesianFusion`` multiplies the predicted belief by the sensor likelihood and
# normalizes the result. Since the prediction has six components and the likelihood has
# two, the exact posterior has twelve components.

posterior_gmm = closed_form_propagation(
    prior_gmm,
    [*motion_operations, BayesianFusion(sensor_likelihood, block_size=4)],
    max_components=32,
)

for name, gmm in [
    ("prior", prior_gmm),
    ("prediction", prediction_gmm),
    ("sensor likelihood", sensor_likelihood),
    ("posterior", posterior_gmm),
]:
    mean = gmm.expectation()
    print(f"{name:>17}: {gmm.n_components:2d} components, mean={mean.tolist()}")

# %%
# 4. Visualize the belief update
# ------------------------------
#
# The posterior keeps the hypotheses that are compatible with both the odometry
# prediction and the sensor likelihood.

fig, axes = plot_2d_gmm_density_panels(
    [prior_gmm, prediction_gmm, sensor_likelihood, posterior_gmm],
    ["Prior belief", "After motion", "Sensor likelihood", "Fused posterior"],
    n_rows=2,
    n_cols=2,
)

beliefs = [prior_gmm, prediction_gmm, sensor_likelihood, posterior_gmm]
for ax, gmm in zip(axes, beliefs, strict=True):
    expected_position = gmm.expectation().detach().cpu()
    ax.scatter(
        expected_position[0].item(),
        expected_position[1].item(),
        marker="*",
        s=130,
        color="white",
        edgecolors="black",
        linewidths=0.8,
        label="mixture mean",
    )

axes[0].legend(loc="best")
fig.suptitle("Closed-form 2D localization update", y=1.02)

# %%
# 5. Inspect the posterior modes
# ------------------------------
#
# Sorting posterior components by weight gives a compact view of the remaining
# localization hypotheses.

component_order = torch.argsort(posterior_gmm.weights, descending=True)

print("Largest posterior components:")
for rank, component in enumerate(component_order[:5], start=1):
    weight = posterior_gmm.weights[component].item()
    mean = posterior_gmm.means[component].tolist()
    print(f"  {rank}. weight={weight:.3f}, mean={[round(value, 3) for value in mean]}")
