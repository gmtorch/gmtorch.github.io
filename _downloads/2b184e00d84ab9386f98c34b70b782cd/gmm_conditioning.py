# ruff: noqa: D205, D400, INP001, T201
"""
Conditioning a Gaussian Mixture
===============================

This example conditions a two-dimensional Gaussian mixture on one observed coordinate.
The original mixture models a joint belief over position ``(x, y)``. After observing a
particular value of ``y``, ``GaussianMixture.cond`` returns a one-dimensional mixture for
the remaining coordinate, representing ``p(x | y)``.
"""

import matplotlib.pyplot as plt
import torch
from _plotting import plot_2d_gmm_density_panels
from torch.distributions import Normal

from gmtorch import GaussianMixture

DIM = 2
N_COMPONENTS = 3
DTYPE = torch.float64
GRID_SIZE = 350

# %%
# 1. Build a joint 2D belief
# --------------------------
#
# The mixture has three spatial hypotheses. The covariances include correlations between
# ``x`` and ``y``, so conditioning on ``y`` can shift the conditional mean of ``x`` inside
# each component.

joint_gmm = GaussianMixture(
    n_components=N_COMPONENTS,
    dim=DIM,
    weights=torch.tensor([0.34, 0.40, 0.26], dtype=DTYPE),
    means=torch.tensor(
        [
            [-2.3, -1.1],
            [0.1, 0.7],
            [2.2, 1.5],
        ],
        dtype=DTYPE,
    ),
    covariances=torch.tensor(
        [
            [[0.48, 0.26], [0.26, 0.36]],
            [[0.58, -0.22], [-0.22, 0.32]],
            [[0.42, 0.18], [0.18, 0.30]],
        ],
        dtype=DTYPE,
    ),
)

observed_y_values = torch.tensor([-1.05, 1.25], dtype=DTYPE)
observed_index = torch.tensor([1])

fig, axes = plot_2d_gmm_density_panels([joint_gmm], ["Joint belief p(x, y)"])
fig.set_size_inches(7.2, 6.4)
colorbar_ax = fig.axes[-1]
colorbar_position = colorbar_ax.get_position()
colorbar_height = 0.62 * colorbar_position.height
colorbar_ax.set_position(
    [
        colorbar_position.x0,
        colorbar_position.y0 + 0.5 * (colorbar_position.height - colorbar_height),
        colorbar_position.width,
        colorbar_height,
    ]
)
ax = axes[0]
line_colors = ["cyan", "lime"]

for observed_y, color in zip(observed_y_values, line_colors, strict=True):
    ax.axhline(
        observed_y.item(),
        color=color,
        linewidth=2.0,
        linestyle="--",
        label=f"observed y = {observed_y.item():.2f}",
    )

ax.legend(loc="best")
fig.suptitle("Conditioning slices through a 2D GMM", y=1.02)

# %%
# 2. Condition on one coordinate
# ------------------------------
#
# Passing ``obs_idx=torch.tensor([1])`` tells ``cond`` that the observed coordinate is
# ``y``. The output keeps the unobserved coordinate ``x``, so each result is a
# one-dimensional GMM with the same number of components and updated weights, means, and
# variances.

conditioned_gmms = [
    joint_gmm.cond(torch.tensor([observed_y.item()], dtype=DTYPE), obs_idx=observed_index)
    for observed_y in observed_y_values
]

for observed_y, conditioned_gmm in zip(observed_y_values, conditioned_gmms, strict=True):
    assert conditioned_gmm.dim == 1
    assert conditioned_gmm.n_components == joint_gmm.n_components
    torch.testing.assert_close(conditioned_gmm.weights.sum(), torch.tensor(1.0, dtype=DTYPE))
    print(f"p(x | y = {observed_y.item():.2f})")
    print("  weights:", conditioned_gmm.weights)
    print("  means:", conditioned_gmm.means.squeeze(-1))

# %%
# 3. Plot the conditional densities
# ---------------------------------
#
# The observed value ``y = -1.05`` mostly supports the lower-left hypothesis. The
# observed value ``y = 1.25`` instead supports the central and upper-right hypotheses.
# This is the most important effect of GMM conditioning: each Gaussian component is
# conditioned, and the mixture weights are reweighted by how likely the observation is
# under that component.

x_values = torch.linspace(-4.2, 4.0, GRID_SIZE, dtype=DTYPE)
x_points = x_values[:, None]

fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True, constrained_layout=True)

for ax, observed_y, conditioned_gmm, color in zip(
    axes, observed_y_values, conditioned_gmms, line_colors, strict=True
):
    density = conditioned_gmm.prob(x_points)
    ax.plot(
        x_values.numpy(),
        density.detach().cpu().numpy(),
        color=color,
        linewidth=2.4,
        label="conditional density",
    )

    component_means = conditioned_gmm.means[:, 0]
    component_stds = torch.sqrt(conditioned_gmm.covariances[:, 0, 0])
    for component in range(conditioned_gmm.n_components):
        component_distribution = Normal(component_means[component], component_stds[component])
        component_density = (
            conditioned_gmm.weights[component] * component_distribution.log_prob(x_values).exp()
        )
        ax.fill_between(
            x_values.numpy(),
            component_density.detach().cpu().numpy(),
            alpha=0.20,
            label=f"component {component}",
        )

    conditional_mean = conditioned_gmm.expectation()[0].item()
    ax.axvline(
        conditional_mean,
        color="black",
        linestyle=":",
        linewidth=1.6,
        label="conditional mean",
    )
    ax.set(
        xlabel="x",
        title=f"p(x | y = {observed_y.item():.2f})",
        xlim=(x_values[0].item(), x_values[-1].item()),
    )

axes[0].set_ylabel("density")
axes[0].legend(loc="best")

# %%
# 4. Compare posterior weights
# ----------------------------
#
# The original component weights describe the prior mixture. The conditional weights
# describe which components remain plausible after observing ``y``.

component_positions = torch.arange(N_COMPONENTS)
bar_width = 0.25

fig, ax = plt.subplots(figsize=(7, 3.6), constrained_layout=True)
ax.bar(
    (component_positions - bar_width).numpy(),
    joint_gmm.weights.detach().cpu().numpy(),
    width=bar_width,
    label="prior",
)

for offset, observed_y, conditioned_gmm, color in zip(
    [0.0, bar_width],
    observed_y_values,
    conditioned_gmms,
    line_colors,
    strict=True,
):
    ax.bar(
        (component_positions + offset).numpy(),
        conditioned_gmm.weights.detach().cpu().numpy(),
        width=bar_width,
        color=color,
        label=f"y = {observed_y.item():.2f}",
    )

ax.set(
    xlabel="component",
    ylabel="weight",
    title="Component weights before and after conditioning",
    xticks=component_positions.numpy(),
)
ax.legend(loc="best")
