# ruff: noqa: D205, D400, INP001, T201
"""
Fitting a GMM With EM
=====================

This example builds a known two-dimensional Gaussian mixture, samples from it, and fits a
new mixture with ``GaussianMixtureEstimator``. Since mixture component labels are
arbitrary, the fitted component order may differ from the ground truth even when the
geometry is recovered well.
"""

import matplotlib.pyplot as plt
import torch
from _plotting import plot_2d_gmm, plot_2d_gmm_density_panels, plot_2d_samples

from gmtorch import GaussianMixture, GaussianMixtureEstimator

N_COMPONENTS = 3
DIM = 2
N_SAMPLES = 6_000
N_DISPLAY_SAMPLES = 2_000
N_INIT = 6
SAMPLE_SEED = 42
EM_SEED = 7
DTYPE = torch.float64

# %%
# 1. Build the ground-truth mixture
# ---------------------------------
#
# The mixture has three full-covariance components. The covariance matrices are
# deliberately non-diagonal so the fitted contours need to recover orientation as
# well as location and scale.

true_weights = torch.tensor([0.30, 0.45, 0.25], dtype=DTYPE)
true_means = torch.tensor(
    [
        [-3.0, -1.0],
        [0.75, 2.4],
        [3.0, -1.4],
    ],
    dtype=DTYPE,
)
true_covariances = torch.tensor(
    [
        [[0.65, 0.18], [0.18, 0.85]],
        [[0.85, -0.32], [-0.32, 0.55]],
        [[0.45, 0.02], [0.02, 0.35]],
    ],
    dtype=DTYPE,
)

true_gmm = GaussianMixture(
    n_components=N_COMPONENTS,
    dim=DIM,
    weights=true_weights,
    means=true_means,
    covariances=true_covariances,
)

ax = plot_2d_gmm(true_gmm)
ax.set_title("Ground-truth GMM")

# %%
# 2. Draw training samples
# ------------------------
#
# ``sample`` returns a tensor with shape ``(n_samples, dim)``. A subset is displayed so
# the scatter plot stays readable.

samples = true_gmm.sample(N_SAMPLES, seed=SAMPLE_SEED)
display_samples = samples[:N_DISPLAY_SAMPLES]

ax = plot_2d_samples(display_samples)
ax.set_title("Samples from the ground-truth GMM")

# %%
# 3. Fit a mixture with EM
# ------------------------
#
# ``GaussianMixtureEstimator`` owns the EM loop and fitted state. After fitting,
# ``to_mixture`` converts the learned parameters into the regular ``GaussianMixture``
# object used elsewhere in ``gmtorch``.

estimator = GaussianMixtureEstimator(
    n_components=N_COMPONENTS,
    init_method="random_from_data",
    n_init=N_INIT,
    max_iter=100,
    tol=1e-4,
    reg_covar=1e-5,
    seed=EM_SEED,
    dtype=DTYPE,
)

labels = estimator.fit_predict(samples)
fitted_gmm = estimator.to_mixture()

fit_info = estimator.fit_info
assert fit_info is not None

print(f"Converged: {fit_info.converged}")
print(f"EM iterations: {fit_info.n_iter}")
print(f"Final average log likelihood: {fit_info.lower_bound.item():.3f}")
print(f"Label shape: {tuple(labels.shape)}")

# %%
# 4. Inspect the fitted parameters
# --------------------------------
#
# The component labels are arbitrary, so sorting the means by their first coordinate
# gives a simple way to compare the fitted parameters with the ground truth in this
# example.

true_order = torch.argsort(true_gmm.means[:, 0])
fitted_order = torch.argsort(fitted_gmm.means[:, 0])

print("True weights, sorted by mean x0:")
print(true_gmm.weights[true_order])
print("Fitted weights, sorted by mean x0:")
print(fitted_gmm.weights[fitted_order])

print("\nTrue means, sorted by x0:")
print(true_gmm.means[true_order])
print("Fitted means, sorted by x0:")
print(fitted_gmm.means[fitted_order])

# %%
# 5. Plot the EM lower-bound trace
# --------------------------------
#
# The estimator stores the best initialization's per-iteration lower bounds in
# ``fit_info``. The curve should increase until the EM loop reaches the configured
# tolerance or ``max_iter``.

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
# 6. Compare fitted contours with the samples
# -------------------------------------------
#
# The fitted component contours should line up with the dense regions in the sample
# cloud. The fitted means are marked with crosses.

ax = plot_2d_gmm(fitted_gmm)
ax.scatter(
    display_samples[:, 0].detach().cpu().numpy(),
    display_samples[:, 1].detach().cpu().numpy(),
    s=8,
    alpha=0.12,
    color="black",
    edgecolors="none",
    label="Training samples",
)
ax.set_title("Fitted GMM with training samples")
ax.legend(loc="best")

# %%
# 7. Compare the true and fitted densities
# ----------------------------------------
#
# A density view makes the comparison independent of component ordering. The fitted
# mixture should put probability mass in the same regions as the ground-truth mixture.

plot_2d_gmm_density_panels(
    [true_gmm, fitted_gmm],
    ["Ground-truth density", "Fitted density"],
)
