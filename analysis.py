import matplotlib.pyplot as plt


def plot_timeseries(x, ticks):
    plt.plot(range(len(x)), x)
    plt.xticks(range(len(ticks)), ticks, rotation=90, fontsize=9)
    plt.show()
