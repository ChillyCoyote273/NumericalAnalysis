import numpy as np
from timeit import timeit


def twist(x: np.ndarray, u: np.ndarray) -> np.ndarray:
	theta = x[2]
	rotation = np.array([
		[np.cos(theta), -np.sin(theta), 0],
		[np.sin(theta), np.cos(theta), 0],
		[0, 0, 1]
	])
	return rotation.dot(u)


def pose_exponential(x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
	u = u * dt

	theta = x[2]
	rotation = np.array([
		[np.cos(theta), -np.sin(theta), 0],
		[np.sin(theta), np.cos(theta), 0],
		[0, 0, 1]
	])
	
	phi = u[2]
	correction = np.array([
		[1, 0, 0],
		[0, 1, 0],
		[0, 0, 1]
	])
	if phi != 0:
		correction = np.array([
			[np.sin(phi) / phi, (np.cos(phi) - 1) / phi, 0],
			[(1 - np.cos(phi)) / phi, np.sin(phi) / phi, 0],
			[0, 0, 1]
		])
		
	return x + rotation.dot(correction.dot(u))


def runge_kutta(x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
	k1 = twist(x, u)
	k2 = twist(x + dt/2 * k1, u)
	k3 = twist(x + dt/2 * k2, u)
	k4 = twist(x + dt * k3, u)
	
	return x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)


def three_eighths(x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
	k1 = twist(x, u)
	k2 = twist(x + dt * (k1 / 3), u)
	k3 = twist(x + dt* (-k1 / 3 + k2), u)
	k4 = twist(x + dt * (k1 - k2 + k3), u)
	
	return x + dt * (k1 + 3 * k2 + 3 * k3 + k4) / 8


def euler_step(x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
	k = twist(x, u)
	return x + dt * k


def euler(x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
	dt /= 6
	for _ in range(6):
		x = euler_step(x, u, dt)
	return x


def main() -> None:
	x = np.array([0.0, 0.0, np.pi / 3])
	u = np.array([40.0, 10.0, 10.0])
	dt = 0.05
	
	x_exponential = pose_exponential(x, u, dt)
	x_runge = runge_kutta(x, u, dt)
	x_three_eighths = three_eighths(x, u, dt)
	x_euler = euler(x, u, dt)
	x_exponential[2] = np.rad2deg(x_exponential[2])
	x_runge[2] = np.rad2deg(x_runge[2])
	x_three_eighths[2] = np.rad2deg(x_three_eighths[2])
	x_euler[2] = np.rad2deg(x_euler[2])
	
	print(f'\n\npose exponential: {x_exponential}\n\nrunge kutta: {x_runge}\n\nthree eighths: {x_three_eighths}\n\neuler: {x_euler}\n\n\n\nrunge error: {np.abs(x_runge - x_exponential)}\n\nthree eighths error: {np.abs(x_three_eighths - x_exponential)}\n\neuler error: {np.abs(x_euler - x_exponential)}\n\n\n\n')

	print(f'{timeit(lambda: pose_exponential(x, u, dt), number=1_000_000)} us')
	print(f'{timeit(lambda: runge_kutta(x, u, dt), number=1_000_000)} us')
	print(f'{timeit(lambda: three_eighths(x, u, dt), number=1_000_000)} us')
	print(f'{timeit(lambda: euler(x, u, dt), number=1_000_000)} us')


if __name__ == "__main__":
	main()
