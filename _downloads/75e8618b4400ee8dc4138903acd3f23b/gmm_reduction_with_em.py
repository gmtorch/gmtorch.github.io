# ruff: noqa: D205, D400, INP001, T201
"""
Reducing a GMM With EM
======================

This example reduces a four-component two-dimensional Gaussian mixture to two
components. The reduction strategy samples from the original mixture, fits a
``GaussianMixtureEstimator`` with EM, and returns a smaller ``GaussianMixture``.
"""

import matplotlib.pyplot as plt
import torch
from _plotting import plot_2d_gmm, plot_2d_gmm_density_panels

from gmtorch import GaussianMixture, GaussianMixtureEstimator
from gmtorch.reduction import EMBasedReductionStrategy, reduce_gmm

N_COMPONENTS = 4
TARGET_COMPONENTS = 2
DIM = 2
SAMPLES_PER_PARAMETER = 200
N_DISPLAY_SAMPLES = 2_500
DTYPE = torch.float64

# %%
# 1. Build an L-shaped mixture
# ----------------------------
#
# The original distribution uses four localized components arranged in an L shape.
# Reducing to two components should preserve the main probability mass while smoothing
# over some local structure.

weights = torch.full((N_COMPONENTS,), 1.0 / N_COMPONENTS, dtype=DTYPE)
means = torch.tensor(
    [
        [0.0, 0.0],
        [0.0, 2.5],
        [0.0, 5.0],
        [2.5, 0.0],
    ],
    dtype=DTYPE,
)
covariances = torch.tensor(
    [
        [[0.25, 0.00], [0.00, 0.25]],
        [[0.20, 0.00], [0.00, 0.35]],
        [[0.20, 0.00], [0.00, 0.30]],
        [[0.35, 0.00], [0.00, 0.20]],
    ],
    dtype=DTYPE,
)

original_gmm = GaussianMixture(
    n_components=N_COMPONENTS,
    dim=DIM,
    weights=weights,
    means=means,
    covariances=covariances,
)

ax = plot_2d_gmm(original_gmm)
ax.set_title("Original four-component L-shaped GMM")

# %%
# 2. Configure EM-based reduction
# --------------------------------
#
# ``EMBasedReductionStrategy`` owns both sampling configuration and the estimator used
# for fitting. If ``n_samples`` is omitted, the strategy uses the number of free
# parameters in the reduced full-covariance mixture:
#
# ``(K - 1) + K * D + K * D * (D + 1) / 2``.

free_parameters = (TARGET_COMPONENTS - 1) + TARGET_COMPONENTS * DIM + (
    TARGET_COMPONENTS * DIM * (DIM + 1) // 2
)
n_samples = SAMPLES_PER_PARAMETER * free_parameters

estimator = GaussianMixtureEstimator(
    n_components=TARGET_COMPONENTS,
    init_method="random",
    n_init=10,
    max_iter=150,
    tol=1e-5,
    reg_covar=1e-5,
    seed=13,
    dtype=DTYPE,
)
strategy = EMBasedReductionStrategy(
    estimator=estimator,
    samples_per_parameter=SAMPLES_PER_PARAMETER,
    sample_seed=7,
)

print(f"Free parameters in reduced GMM: {free_parameters}")
print(f"Automatically selected samples: {n_samples}")

# %%
# 3. Reduce from four components to two
# -------------------------------------
#
# ``reduce_gmm`` validates the requested reduction and delegates the actual work to the
# strategy. With ``return_info=True`` it also returns the estimator's ``EMFitInfo``.

reduced_gmm, fit_info = reduce_gmm(original_gmm, strategy, return_info=True)

assert fit_info is strategy.estimator.fit_info
assert reduced_gmm.n_components == TARGET_COMPONENTS
assert reduced_gmm.dim == original_gmm.dim

print(f"Components: {original_gmm.n_components} -> {reduced_gmm.n_components}")
print(f"Converged: {fit_info.converged}")
print(f"EM iterations: {fit_info.n_iter}")
print(f"Final average log likelihood: {fit_info.lower_bound.item():.4f}")

# %%
# 4. Inspect the reduced parameters
# ---------------------------------
#
# The learned covariance matrices are broader than the original component covariances.
# They summarize groups of localized components rather than matching each original
# Gaussian one-for-one.

print("Reduced weights:")
print(reduced_gmm.weights)
print("\nReduced means:")
print(reduced_gmm.means)
print("\nReduced covariances:")
print(reduced_gmm.covariances)

# %%
# 5. Plot the EM lower-bound trace
# --------------------------------
#
# The best initialization's lower-bound values are stored in ``fit_info``. They should
# increase until EM reaches the configured tolerance or ``max_iter``.

lower_bounds = torch.stack(fit_info.lower_bounds).detach().cpu()
iterations = torch.arange(1, lower_bounds.shape[0] + 1)

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(iterations.numpy(), lower_bounds.numpy(), marker="o", linewidth=1.8)
ax.set(
    xlabel="EM iteration",
    ylabel="Average log likelihood",
    title="EM lower-bound trace",
)
ax.grid(alpha=0.25)
fig.tight_layout()

# %%
# 6. Compare original and reduced densities
# -----------------------------------------
#
# The reduced GMM has fewer components, but it should keep probability mass in the same
# L-shaped region. The approximation is deliberately smoother.

plot_2d_gmm_density_panels(
    [original_gmm, reduced_gmm],
    ["Original density", "Reduced density"],
)

# %%
# 7. Compare samples from both mixtures
# -------------------------------------
#
# Sampling both mixtures gives another view of what is retained and what is smoothed by
# the reduction.

original_samples = original_gmm.sample(N_DISPLAY_SAMPLES, seed=101).detach().cpu()
reduced_samples = reduced_gmm.sample(N_DISPLAY_SAMPLES, seed=101).detach().cpu()

fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2), constrained_layout=True)
sample_panels = (
    ("Original samples", original_samples),
    ("Reduced samples", reduced_samples),
)
x_limits = (-2.0, 5.0)
y_limits = (-2.0, 7.0)

for ax, (title, samples) in zip(axes, sample_panels, strict=True):
    ax.scatter(
        samples[:, 0].numpy(),
        samples[:, 1].numpy(),
        s=8,
        alpha=0.20,
        edgecolors="none",
    )
    ax.set(xlim=x_limits, ylim=y_limits, xlabel="x0", ylabel="x1", title=title)
    ax.set_aspect("equal", adjustable="box")

# %%
# 8. Use the convenience constructor
# ----------------------------------
#
# When default EM hyperparameters are sufficient, ``from_n_components`` builds the
# estimator internally. Use the estimator-based constructor above when you need to tune
# initialization, iteration limits, tolerances, or dtype.

simple_strategy = EMBasedReductionStrategy.from_n_components(
    TARGET_COMPONENTS,
    samples_per_parameter=SAMPLES_PER_PARAMETER,
    sample_seed=7,
)

print(f"Convenience strategy target: {simple_strategy.target_n_components}")

# %%
# 9. Compare different reduction targets
# --------------------------------------
#
# We can repeat the same EM-based reduction with three and one target components. The
# function ``l2_divergence`` returns the integrated squared error between two densities,
# so its square root is the L2 distance.

from gmtorch.divergences import l2_divergence


def reduce_to_n_components(n_components):
    estimator = GaussianMixtureEstimator(
        n_components=n_components,
        init_method="random",
        n_init=10,
        max_iter=150,
        tol=1e-5,
        reg_covar=1e-5,
        seed=13,
        dtype=DTYPE,
    )
    strategy = EMBasedReductionStrategy(
        estimator=estimator,
        samples_per_parameter=SAMPLES_PER_PARAMETER,
        sample_seed=7,
    )
    return reduce_gmm(original_gmm, strategy)


reduced_3_gmm = reduce_to_n_components(3)
reduced_1_gmm = reduce_to_n_components(1)

reduced_gmms = {
    3: reduced_3_gmm,
    2: reduced_gmm,
    1: reduced_1_gmm,
}
l2_squared_distances = {
    n_components: l2_divergence(original_gmm, reduced).clamp_min(0.0)
    for n_components, reduced in reduced_gmms.items()
}
l2_distances = {
    n_components: squared_distance.sqrt()
    for n_components, squared_distance in l2_squared_distances.items()
}

print("L2 distances to the original GMM:")
for n_components, distance in l2_distances.items():
    squared_distance = l2_squared_distances[n_components]
    component_label = "component" if n_components == 1 else "components"
    print(
        f"{n_components} {component_label}: {distance.item():.6f} "
        f"(squared: {squared_distance.item():.6f})"
    )

plot_2d_gmm_density_panels(
    [original_gmm, reduced_3_gmm, reduced_gmm, reduced_1_gmm],
    ["Original", "Reduced to 3", "Reduced to 2", "Reduced to 1"],
    n_rows=2,
    n_cols=2,
)
