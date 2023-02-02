import numpy as np


def main():
	length = 4
	width = 3
	div = length**2 + width**2
	inverse_kinematics = np.array([
		[1, 0, -width],
		[0, 1, length],
		[1, 0, -width],
		[0, 1, -length],
		[1, 0, width],
		[0, 1, -length],
		[1, 0, width],
		[0, 1, length],
	])
	forward_kinematics = np.linalg.pinv(inverse_kinematics)
	forward_kinematics_estimate = inverse_kinematics.transpose()
	forward_kinematics_estimate = forward_kinematics_estimate / 4.0
	forward_kinematics_estimate[2] /= div
	print(forward_kinematics_estimate)
	print(forward_kinematics)


if __name__ == '__main__':
	main()
