import numpy as np


class Integrator:
    def __init__(self):
        self.value = 0
        self.started = False
        self.prev_data = None
        self.prev_prev_data = None
        self.prev_time = None
        self.prev_prev_time = None
    
    def next(self, time: float, data: float) -> None:
        if self.prev_data is None:
            self.prev_data = data
            self.prev_time = time
            return
        
        if self.prev_prev_data is None:
            self.prev_prev_data = self.prev_data
            self.prev_prev_time = self.prev_time
            self.prev_data = data
            self.prev_time = time
            self.value = (self.prev_time - self.prev_prev_time) * (self.prev_data + self.prev_prev_data) / 2
            return
        
        if not self.started:
            pass


def function(x: float) -> float:
    return x * np.log(x)# + np.random.normal(0, 0.001)


def derivative(x: float) -> float:
    return 1 + np.log(x)


def antiderivative(x: float) -> float:
    return x**2 / 2 * np.log(x) - x**2 / 4


def reverse_simpson(func, a: float, h: float) -> float:
    # g'(a) = (3f(a) - 4f(a-h) + f(a-2h)) / (2h)
    return (3 * func(a) - 4 * func(a - h) + func(a - 2 * h)) / (2 * h)


def reverse_trapezoid(func, a: float, h: float) -> float:
    return (func(a) - func(a - h)) / h


def trapezoid_rule(func, a: float, b: float, num_points: int) -> float:
    h = (b - a) / (num_points - 1)
    points = np.linspace(a, b, num_points)
    values = func(points)
    evaluations = (values[1:] + values[:-1]) / 2
    return h * np.sum(evaluations)


def midpoint_rule(func, a: float, b: float, num_points: int) -> float:
    h = (b - a) / (num_points - 1)
    points = np.linspace(a, b, num_points)
    points = (points[1:] + points[:-1]) / 2
    evaluations = func(points)
    return h * sum(evaluations)


def simpsons_rule(func, a: float, b: float, num_points: int) -> float:
    h = (b - a) / (num_points - 1)
    points = np.linspace(a, b, num_points)
    evaluations = func(points)
    coefficient_vector = np.array([2 if i % 2 == 0 else 4 for i in range(num_points)])
    coefficient_vector[np.array([0, -1])] = 1
    return h / 3 * (evaluations @ coefficient_vector)


def alt_simpsons_rule(func, a: float, b: float, num_points: int) -> float:
    h = (b - a) / (num_points - 1)
    points = np.linspace(a, b, num_points)
    evaluations = func(points)
    return h / 48 * (
        np.array([17, 59, 43, 49]) @ evaluations[:4] +\
        np.array([49, 43, 59, 17]) @ evaluations[-4:] +\
        48 * sum(evaluations[4:-4])
    )


def modified_simpsons_rule(func, a: float, b: float, num_points: int) -> float:
    h = (b - a) / (num_points - 1)
    points = np.linspace(a, b, num_points)
    offsets = np.random.normal(0, h / 10, num_points)
    points = points + offsets
    hs = np.diff(points)
    evaluations = func(points)
    result = sum(
        (h0 + h1) / 6 * ((2 - h1 / h0) * f0 + (h0 + h1)**2 / (h0 * h1) * f1 + (2 - h0 / h1) * f2) for f0, f1, f2, h0, h1 in zip(evaluations[:-2:2], evaluations[1:-1:2], evaluations[2::2], hs[:-1:2], hs[1::2])
    )
    if num_points % 2 == 0:
        alpha = (2 * hs[-1]**2 + 3 * hs[-1] * hs[-2]) / (6 * (hs[-2] + hs[-1]))
        beta = (hs[-1]**2 + 3 * hs[-1] * hs[-2]) / (6 * hs[-2])
        eta = hs[-1]**3 / (6 * hs[-2] * (hs[-2] + hs[-1]))
        result += alpha * evaluations[-1] + beta * evaluations[-2] - eta * evaluations[-3]
    return result, sum(
        h0 / 2 * (f0 + f1) for f0, f1, h0 in zip(evaluations[:-1], evaluations[1:], hs)
    )


def test_differentiation():
    a = 1
    h = 0.01

    actual = derivative(a)

    trap = reverse_trapezoid(function, a, h)
    simp = reverse_simpson(function, a, h)

    print(actual)
    print(trap)
    print(simp)
    print(np.abs(actual - trap))
    print(np.abs(actual - simp))


def test_quadrature():
    a = 1
    b = 2
    n = 20
    ant = antiderivative(b) - antiderivative(a)

    divisions = 6
    simp, trap = modified_simpsons_rule(function, a, b, 3)
    previous = np.abs(ant - simp), np.abs(ant - trap)
    for i in range(10):
        simp, trap = modified_simpsons_rule(function, a, b, divisions)
        error = np.abs(ant - simp), np.abs(ant - trap)
        print(f'simp: {previous[0] / error[0]}   \ttrap: {previous[1] / error[1]}   \tdiff: {error[1] / error[0]}')
        # print(f'error: {error}\tdiff: {previous / error}')
        previous = error
        divisions *= 2
    
    # simp = simpsons_rule(function, a, b, n)
    # mod = modified_simpsons_rule(function, a, b, n)
    # print(ant)
    # print(simp)
    # print(mod)
    # print(np.abs(ant - simp))
    # print(np.abs(ant - mod))
    
    # print(f'trapezoid rule: {trap}\nmidpoint rule: {rect}\nsimpson\'s rule: {simp}\ntrapezoid error: {np.abs(trap - ant)}\nmidpoint error: {np.abs(rect - ant)}\nsimpson\'s error: {np.abs(simp - ant)}')


def main():
    test_quadrature()


if __name__ == "__main__":
    main()
