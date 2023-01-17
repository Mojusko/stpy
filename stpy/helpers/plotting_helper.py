import matplotlib.pyplot as plt
import sklearn


def plot_R2(vals, lcb, ucb, truth, s, truth_lcb=None, truth_ucb=None, show=False, save_file_name=None, name=None):
    r2 = sklearn.metrics.r2_score(truth, vals)
    if save_file_name is not None:
        filename = save_file_name

    if name is None:
        plt.title(str(r2))
    else:
        plt.title(name + " " + str(r2))

    plt.xlabel("true")
    plt.ylabel("predicted")

    plt.plot(truth, truth, 'k-')
    plt.plot(truth, truth + s, 'k--')
    plt.plot(truth, truth - s, 'k--')
    plt.plot(truth, vals, color='k', marker='o', linestyle='')

    plt.errorbar(truth, vals, yerr=vals - lcb, color='k', marker='o', linestyle='')


    if save_file_name is not None:
        plt.savefig(filename + "_0.png", dpi=150)

    plt.errorbar(truth, vals, yerr=vals - lcb + 2 * s, color='r', marker='o', linestyle='', zorder = -10)

    if save_file_name is not None:
        plt.savefig(filename + "_1.png", dpi=150)


    if show:
        plt.show()
