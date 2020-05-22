import numpy as np
from collections import Counter
import warnings
from tabulate import tabulate
from termcolor import colored

warnings.simplefilter("ignore")

supply = np.array([23., 25., 12., 30.])
demand = np.array([18., 18., 18., 18., 18.])
costs = np.array([[3., 25., 11., 22., 12.],
                 [9., 15., 4., 26., 12.],
                 [13., 22., 15., 12., 27.],
                 [6., 19., 8., 11., 8.]])


def print_initial_solution(costs, X, supply, demand, str):
    n, m = costs.shape
    arr_of_res = [[0] * m for i in range(n)]
    for i in range(n):
        for j in range(m):
            if X[i, j] == 0:
                arr_of_res[i][j] = f"-[{costs[i, j]}]"
            else:
                arr_of_res[i][j] = f"{X[i, j]}[{costs[i, j]}]"
    arr_of_res = np.array(arr_of_res)
    a = np.column_stack((arr_of_res, supply))
    a = np.vstack((a, np.append(demand, str)))
    print(tabulate(a, tablefmt='grid'))
    print("Сумма перевозок:", np.sum(X * costs))


def print_optimal_solution(costs, X, u, v, str, neg, pos, q):
    n, m = costs.shape
    arr_of_res = [[0] * m for i in range(n)]
    for i in range(n):
        for j in range(m):
            if X[i, j] == 0:
                arr_of_res[i][j] = f"-[{costs[i, j]}]"
            else:
                arr_of_res[i][j] = f"{X[i, j]}[{costs[i, j]}]"
    arr_of_res = np.array(arr_of_res)
    a = np.column_stack((u, arr_of_res))
    a = np.vstack((np.append(str, v), a))

    if q == 0:
        print("Задача после улучшения")
        print(tabulate(a, tablefmt='grid'))
        print("Сумма перевозок:", np.sum(X * costs),"\n")

    else:
        print(tabulate(a, tablefmt='grid'))
        print("Положительные элементы цыкла:", colored(arr_of_res[list(zip(*pos))], 'green'))
        print("Отрицательные элементы цыкла:", colored(arr_of_res[list(zip(*neg))], 'red'))
        print("Дельта:", q)


def find_initial_solution(costs, demand, supply):
    C = np.copy(costs)
    d = np.copy(demand)
    s = np.copy(supply)

    # Get the shape of costs-matrix
    n, m = C.shape

    # Create the matrix of basic values and convert cost to one-dim array
    X = np.zeros((n, m))
    indices = [(i, j) for i in range(n) for j in range(m)]
    xs = sorted(zip(indices, C.flatten()), key=lambda kv: kv[1])

    # Find initial solution
    for (i, j), _ in xs:
        if d[j] == 0:
            continue
        else:
            # Reserving supplies in a greedy way
            remains = s[i] - d[j] if s[i] >= d[j] else 0
            grabbed = s[i] - remains
            X[i, j] = grabbed
            s[i] = remains
            d[j] -= grabbed
    return X


def find_potential(X, C):
    n, m = X.shape

    u = np.array([np.nan] * n)
    v = np.array([np.nan] * m)

    _x, _y = np.where(X > 0)
    nonzero = list(zip(_x, _y))
    f = nonzero[0][0]
    u[f] = 0

    while any(np.isnan(u)) or any(np.isnan(v)):
        for i, j in nonzero:
            if np.isnan(u[i]) and not np.isnan(v[j]):
                u[i] = C[i, j] - v[j]
            elif not np.isnan(u[i]) and np.isnan(v[j]):
                v[j] = C[i, j] - u[i]
            else:
                continue
    return u, v


def main():
    # Get initials solution
    n, m = costs.shape
    X = find_initial_solution(costs, demand, supply)

    # print initial solution
    print("Опорный план")
    print_initial_solution(costs, X, supply, demand, "a{i}/b{j}")
    print("Нахождение оптимального плана")

    while True:
        S = np.zeros((n, m))

        # Find potentials
        u, v = find_potential(X, costs)

        # Find S - matrix
        for i in range(n):
            for j in range(m):
                S[i, j] = costs[i, j] - u[i] - v[j]

        # Condition to break
        s = np.min(S)
        if s >= 0:
            print("Минимальная сумма перевозок найдена")
            break

        i, j = np.argwhere(S == s)[0]
        start = (i, j)

        # print(start)
        # Find cycle elements

        T = np.copy(X)
        T[start] = 1
        while True:
            _xs, _ys = np.nonzero(T)
            xcount, ycount = Counter(_xs), Counter(_ys)

            for x, count in xcount.items():
                if count <= 1:
                    T[x, :] = 0
            for y, count in ycount.items():
                if count <= 1:
                    T[:, y] = 0

            if all(x > 1 for x in xcount.values()) \
                    and all(y > 1 for y in ycount.values()):
                break
        # print(T)

        # Finding cycle order
        dist = lambda kv1, kv2: abs(kv1[0] - kv2[0]) + abs(kv1[1] - kv2[1])
        fringe = [tuple(p) for p in np.argwhere(T > 0)]
        # print(fringe)

        size = len(fringe)

        path = [start]
        while len(path) < size:
            last = path[-1]
            if last in fringe:
                fringe.remove(last)
            next = min(fringe, key=lambda kv: dist(last, (kv[0], kv[1])))
            path.append(next)

        # Improving solution on cycle elements
        neg = path[1::2]
        pos = path[::2]
        q = min(X[list(zip(*neg))])

        # Print optimal solution
        print_optimal_solution(costs, X, u, v, "v{i}/u{j}", neg, pos, q)

        # Improve solution
        X[list(zip(*neg))] -= q
        X[list(zip(*pos))] += q

        # Print table after improving
        print_optimal_solution(costs, X, u, v, "v{i}/u{j}", [], [], 0)


main()
