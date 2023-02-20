import numpy as np


def function(x: float) -> float:
    return x**6 - x - 1


def function_derivative(x: float) -> float:
    return 6 * x**5 - 1


def newton_raphson(x: float) -> float:
    return x - function(x) / function_derivative(x)


def secant_method(x_0: float, x_1: float) -> float:
    f_x_0 = function(x_0)
    f_x_1 = function(x_1)
    return (x_1 * f_x_0 - x_0 * f_x_1) / (f_x_0 - f_x_1)


def bisection_method(a: float, b: float) -> tuple[float, float]:
    f_a = np.sign(function(a))
    f_b = np.sign(function(b))
    c = (a + b) / 2
    f_c = np.sign(function(c))
    if f_a == f_c:
        return c, b
    return a, c


def main() -> None:
    x = 0
    for i in range(8):
        x = newton_raphson(x)
        print(f'{i}: {x}')
    
    print('\n')
    
    x_0 = 0
    x_1 = 0.1
    for i in range(11):
        x_1, x_0 = secant_method(x_0, x_1), x_1
        print(f'{i}: {x_1}')
    
    print('\n')

    a = -1
    b = 0
    for i in range(53):
        a, b = bisection_method(a, b)
        print(f'{i}: {(a + b) / 2}')


if __name__ == "__main__":
    main()
