from math import comb
import numpy as np


def get_n_possible_delays(n):
    sum = 2 ** n - 1
    l1 = []
    for k in range(1, n + 1):
        l2 = []
        for s in range(1, k):
            l2.append(comb(n, s) * comb(n - s, k - s))
        l1.append(np.array(l2).sum())
    diff = int(0.5 * np.array(l1).sum())

    print(n, ':', sum + diff)


if __name__ == '__main__':
    for i in range(1, 20):
        get_n_possible_delays(i)
        # print(1 / 2 * (2 ** (2 * i) - (2 * i)))

