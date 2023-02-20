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


if __name__ == "__main__":
    main()
