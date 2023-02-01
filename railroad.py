import numpy as np
import matplotlib.pyplot as plt


def func(x):
	return x * np.sin(5281*0.5 / x) - 5280*0.5


def func_prime(x):
	return -5281*0.5 / x * np.cos(5281*0.5 / x) + np.sin(5281*0.5 / x)


def get_dist(x):
	theta = 5281*0.5 / x
	return x * (1 - np.cos(theta))


def main():
	num_iterations = 20
	radii = np.zeros(num_iterations + 1)
	radii[0] = 1
	for i in range(num_iterations):
		num_iterations
		radii[i + 1] = radii[i] - func(radii[i]) / func_prime(radii[i])

	dists = get_dist(radii)

	for i, estimate in enumerate(dists):
		print(f'{i}: {estimate}')

	plt.plot(np.linspace(0, num_iterations, num_iterations + 1), dists)

	plt.show()


if __name__ == '__main__':
	main()
