# ruff: noqa: D205, D400, INP001
"""
Introduction
============

This example introduces the main ``gmtorch`` class, ``GaussianMixture``. It creates a
two-dimensional Gaussian mixture and showcases the following basic methods:

- ``random``
- ``sample``
- ``expectation``
- ``covariance``
- ``prob`` and ``log_prob``
"""

import warnings

import matplotlib.pyplot as plt
import torch
from _plotting import plot_2d_gmm, sample_bounds

from gmtorch import GaussianMixture

N_COMPONENTS = 3
DIM = 2
N_SAMPLES = 12_000
N_DISPLAY_SAMPLES = 2_000
SAMPLE_SEED = 24
RANDOM_SEED = 3
GRID_SIZE = 180
MOMENT_ATOL = 0.18
MOMENT_RTOL = 0.12

# %%
# 1. Create a random GMM
# ----------------------
#
# ``GaussianMixture.random`` is a convenient way to create a valid mixture with
# random weights, means, and covariance matrices. It samples the weights from a
# Dirichlet distribution, the means from a uniform distribution, and the
# covariances from a Wishart distribution. A fixed seed makes this example
# reproducible.

with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    gmm = GaussianMixture.random(
        n_components=N_COMPONENTS,
        dim=DIM,
        mu_rng=(-2.0, 2.0),
        var_df=8.0,
        var_scale=0.25 * torch.eye(DIM),
        seed=RANDOM_SEED,
    )

ax = plot_2d_gmm(gmm)
ax.set_title("Random two-dimensional Gaussian mixture")

# %%
# 2. Draw samples and overlay them
# --------------------------------
#
# Samples are returned as a tensor with shape ``(n_samples, dim)``. The plot
# overlays a subset of samples on the component contours so the figure remains
# readable.

samples = gmm.sample(N_SAMPLES, seed=SAMPLE_SEED)
display_samples = samples[:N_DISPLAY_SAMPLES].detach().cpu()

ax = plot_2d_gmm(gmm)
ax.scatter(
    display_samples[:, 0].numpy(),
    display_samples[:, 1].numpy(),
    s=8,
    alpha=0.20,
    color="black",
    edgecolors="none",
    label="Samples",
)
ax.set_title("Samples from the Gaussian mixture")
ax.legend(loc="best")

# %%
# 3. Estimate the expectation from samples
# ----------------------------------------
#
# ``expectation`` computes the analytical mixture mean. The empirical mean of
# enough samples should be close to it.

true_expectation = gmm.expectation()
sample_expectation = samples.mean(dim=0)

assert torch.allclose(sample_expectation, true_expectation, atol=MOMENT_ATOL, rtol=MOMENT_RTOL)

ax = plot_2d_gmm(gmm)
ax.scatter(
    [sample_expectation[0].item()],
    [sample_expectation[1].item()],
    s=150,
    marker="o",
    color="white",
    edgecolors="black",
    linewidths=1.2,
    label="Sample mean",
)
ax.scatter(
    [true_expectation[0].item()],
    [true_expectation[1].item()],
    s=50,
    marker="*",
    color="gold",
    edgecolors="black",
    linewidths=1.0,
    label="Analytical expectation",
)
ax.set_title("Analytical expectation vs. sample mean")
ax.legend(loc="best")
print("Analytical expectation:", true_expectation)
print("Sample mean:", sample_expectation)

# %%
# 4. Estimate the covariance from samples
# ---------------------------------------
#
# ``covariance`` computes the analytical covariance matrix of the whole
# mixture, including both within-component covariance and between-component
# spread.

true_covariance = gmm.covariance()
sample_covariance = torch.cov(samples.T)
covariance_error = (sample_covariance - true_covariance).abs()

assert torch.allclose(sample_covariance, true_covariance, atol=MOMENT_ATOL, rtol=MOMENT_RTOL)

fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), constrained_layout=True)
covariance_panels = (
    ("Analytical covariance", true_covariance),
    ("Sample covariance", sample_covariance),
    ("Absolute error", covariance_error),
)
color_limit = float(torch.stack((true_covariance.abs(), sample_covariance.abs())).max().item())

for ax, (title, matrix) in zip(axes, covariance_panels, strict=True):
    image = ax.imshow(matrix.detach().cpu().numpy(), vmin=0.0, vmax=color_limit, cmap="viridis")
    ax.set_title(title)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

fig.colorbar(image, ax=axes, shrink=0.82)

# %%
# 5. Evaluate density with ``prob`` and ``log_prob``
# --------------------------------------------------
#
# ``log_prob`` evaluates the log density, while ``prob`` returns the density
# itself. The density should match ``exp(log_prob)``.

x_limits, y_limits = sample_bounds(samples.detach().cpu())
x_values = torch.linspace(*x_limits, GRID_SIZE)
y_values = torch.linspace(*y_limits, GRID_SIZE)
x_grid, y_grid = torch.meshgrid(x_values, y_values, indexing="xy")
grid_points = torch.stack((x_grid, y_grid), dim=-1)

log_density = gmm.log_prob(grid_points)
density = gmm.prob(grid_points)

assert torch.allclose(density, log_density.exp())

fig, ax = plt.subplots(figsize=(7, 6))
density_plot = ax.contourf(
    x_grid.numpy(), y_grid.numpy(), density.detach().cpu().numpy(), levels=40
)
ax.contour(
    x_grid.numpy(), y_grid.numpy(), density.detach().cpu().numpy(), levels=12, colors="white"
)
ax.scatter(
    gmm.means[:, 0].detach().cpu().numpy(),
    gmm.means[:, 1].detach().cpu().numpy(),
    marker="x",
    s=90,
    color="crimson",
    linewidths=2.0,
    label="Component means",
)
fig.colorbar(density_plot, ax=ax, label="Density")
ax.set(xlabel="x0", ylabel="x1", title="Mixture density from prob")
ax.set_aspect("equal", adjustable="box")
ax.legend(loc="best")
