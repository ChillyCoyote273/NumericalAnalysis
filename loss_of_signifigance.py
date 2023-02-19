import numpy as np
import matplotlib.pyplot as plt


def main():
    xs  = np.linspace(-0.000001, 0.000001, 1000)
    ys = (1-np.cos(xs)) / xs**2
    plt.plot(xs, ys)
    plt.show()


if __name__ == '__main__':
    main()
