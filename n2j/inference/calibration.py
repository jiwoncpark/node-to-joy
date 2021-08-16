import numpy as np
import matplotlib.pyplot as plt


def get_p_Y_val_approx_mahalanobis(post_samples, y_mean, y_truth, cov):
    """Calculate the percentage of draws from the predicted distribution that
    encompasses the truth, for all of the examples in the validation set.

    Parameters
    ----------
    post_samples : np.array of shape [n_samples, n_sightlines, Y_dim]
        BNN posterior samples
    y_mean : np.array of shape [n_lenses, Y_dim]
        Central prediction to use in the distance calculation
    y_truth: np.array of shape [n_lenses, Y_dim]
        True values
    cov : float
        Scale factor to use in the distance calculation

    Notes
    -----
    Adapted from https://github.com/swagnercarena/ovejero

    """

    bnn_coverage = (post_samples-y_mean)/cov
    truth_coverage = np.expand_dims((y_truth-y_mean)/cov, axis=0)

    p_Y_val = (bnn_coverage < truth_coverage)
    p_Y_val = np.mean(p_Y_val, axis=0)
    return p_Y_val


def plot_calibration(post_samples, y_mean, y_truth, cov,
                     color_map=["#377eb8", "#4daf4a"], n_perc_points=20,
                     figure=None, ls='--', legend=None, show_plot=True,
                     block=True, title=None, dpi=200):
    """Plot the calibration metric for a grid of p_X percentages,
    with error bars
    obtained through jackknife sampling

    Parameters
    ----------
    See the docstring for `get_p_Y_val_approx_mahalanobis`.
    n_perc_points : int
        Grid size of p_X (probability volume) to compare p_Y against

    Notes
    -----
    Adapted from https://github.com/swagnercarena/ovejero

    """
    p_Y_val = get_p_Y_val_approx_mahalanobis(post_samples, y_mean,
                                             y_truth, cov=cov)

    # Plot what percentage of images have at most x% of draws with
    # p(draws)>p(true).
    percentages = np.linspace(0.0, 1.0, n_perc_points)
    p_images = np.zeros_like(percentages)
    if figure is None:
        fig = plt.figure(figsize=(8, 8), dpi=dpi)
        plt.plot(percentages, percentages, c=color_map[0],
                 ls='--', label=legend[0])
    else:
        fig = figure

    # We'll estimate the uncertainty in our plot using a jacknife method.
    p_images_jn = np.zeros((len(p_Y_val), n_perc_points))
    for pi in range(n_perc_points):
        percent = percentages[pi]
        p_images[pi] = np.mean(p_Y_val <= percent)
        for ji in range(len(p_Y_val)):
            samp_p_Y_val = np.delete(p_Y_val, ji)
            p_images_jn[ji, pi] = np.mean(samp_p_Y_val <= percent)
    # Estimate the standard deviation from the jacknife
    p_Y_val_std = np.sqrt((len(p_Y_val)-1)*np.mean(np.square(p_images_jn - np.mean(p_images_jn, axis=0)), axis=0))
    plt.plot(percentages, p_images, c=color_map[1], ls=ls, label=legend[1])
    # Plot the 1 sigma contours from the jacknife estimate to get an idea of our sample variance.
    plt.fill_between(percentages,
                     p_images+p_Y_val_std,
                     p_images-p_Y_val_std,
                     color=color_map[1], alpha=0.2)
    if figure is None:
        plt.grid(True, ls='dotted', alpha=0.5)
        plt.xlabel(r'Fraction of posterior volume = $p_X$', fontsize=15)
        plt.ylabel(r'Fraction of validation lenses with truth in the volume = $p_Y^{\mathrm{val}}$',
                   fontsize=15)
        plt.text(-0.03, 1, 'Underconfident', fontsize=15)
        plt.text(0.80, 0, 'Overconfident', fontsize=15)
    plt.legend(fontsize=15, loc=9)
    if show_plot:
        plt.show(block=block)

    return fig
