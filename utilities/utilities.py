from sympy import *


def cal(V, W_v, O_v1, T_v, U_v, P, O_p1, T_p, W_p, O_v2, O_p2):
    x = Symbol('x')
    y = Symbol('y')
    print(solve([x * (V - 2 * W_v - O_v1) + (1 - x) * V - x * (V - (T_v + 2) * W_v - U_v)
                 - (1 - x) * ((V - (T_v + 3) * W_v) - U_v)], x))
    print(solve([y * (P - O_p1) + (1 - y) * P - y * (P - (T_p + 2) * W_p) - (1 - y) * (P - W_p)], y))

    a = Symbol('a')
    b = Symbol('b')
    print(solve([a * (V - 2 * W_v - O_v2) + (1 - a) * V - a * (V - (T_v + 2) * W_v - U_v) -
                 (1 - a) * (V - 0.25 * ((2 * T_v + 11) * W_v + 2 * U_v + O_v1))], a))
    print(solve([b * (P - O_p2) + (1 - b) * P - b * (P - (T_p + 2) * W_p) - (1 - b) * (P - 0.25 * (O_p1 + (T_p + 7) * W_p))],
          b))

    return [x, y, a, b]
