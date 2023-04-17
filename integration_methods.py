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


def modified_simpsons_rule(func, a: float, b: float, num_points: int) -> float:
    points = np.linspace(a, b, num_points)
    offsets = np.zeros(num_points)#np.random.normal(0, h / 10, num_points)
    points = points + offsets
    hs = np.diff(points)
    evaluations = func(points)
    coefficients = np.array([
        np.linalg.inv(np.array([
        [points[i]**2, points[i], 1],
        [points[i + 1]**2, points[i + 1], 1],
        [points[i + 2]**2, points[i + 2], 1]
        ])) @ np.array([evaluations[i], evaluations[i + 1], evaluations[i + 2]])
        for i in range(0, num_points - 2, 1)
    ])
    ends = points[2:]
    beginnings = points[:-2]
    areas = coefficients[:, 0] / 3 * (ends**3 - beginnings**3) + coefficients[:, 1] / 2 * (ends**2 - beginnings**2) + coefficients[:, 2] * (ends - beginnings)
    inner = areas.sum()
    inner += coefficients[0, 0] / 3 * (points[1]**3 - points[0]**3) + coefficients[0, 1] / 2 * (points[1]**2 - points[0]**2) + coefficients[0, 2] * (points[1] - points[0])
    inner += coefficients[-1, 0] / 3 * (points[-1]**3 - points[-2]**3) + coefficients[-1, 1] / 2 * (points[-1]**2 - points[-2]**2) + coefficients[-1, 2] * (points[-1] - points[-2])
    return inner / 2


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
    n = 2 + 1
    ant = antiderivative(b) - antiderivative(a)
    
    simp = simpsons_rule(function, a, b, n)
    mod = modified_simpsons_rule(function, a, b, n)
    print(ant)
    print(simp)
    print(mod)
    print(np.abs(ant - simp))
    print(np.abs(ant - mod))
    
    # print(f'trapezoid rule: {trap}\nmidpoint rule: {rect}\nsimpson\'s rule: {simp}\ntrapezoid error: {np.abs(trap - ant)}\nmidpoint error: {np.abs(rect - ant)}\nsimpson\'s error: {np.abs(simp - ant)}')


def main():
    test_quadrature()


if __name__ == "__main__":
    main()
