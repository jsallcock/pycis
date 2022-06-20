from math import comb
import matplotlib.pyplot as plt
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


def get_n_possible_delays_v2(n):
    t1 = 1 / 2 * (2 ** n - 1)
    l1 = []
    for k in range(1, n + 1):
        l2 = []
        for s in range(1, k + 1):
            l2.append(comb(n, k) * comb(k, s))
        l1.append(np.array(l2).sum())
    t2 = 0.5 * np.array(l1).sum()
    print(n, ':', t1 + t2)


def get_n_possible_delays_v3(n):
    l1 = []
    for k in range(0, n + 1):
        l2 = []
        for s in range(0, k + 1):
            l2.append(comb(n, s) * comb(n-s, k-s))
        l1.append(np.array(l2).sum())
    t = 0.5 * (np.array(l1).sum() - 1)
    print(n, ':', t)


def test():
    def gen(n):
        l1 = []
        for k in range(1, n + 1):
            l2 = []
            for s in range(1, k + 1):
                l2.append(comb(n, k) * comb(k, s))
            l1.append(np.array(l2).sum())
        return np.array(l1).sum()

    input = np.arange(1, 12)
    output = np.array([gen(n) for n in input], dtype=int)


    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(input, output, 'x-')
    ax.plot(input, 2 ** (1.582 * input), 'x-')
    plt.show()


def plot():

    def unique(n): return 0.5 * (3 ** n - 1)
    def possible(n): return 2 ** (2 * n - 1) - 2 ** (n - 1)

    x = np.arange(1, 6)

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(x, unique(x), 'o-', label='unique')
    ax.plot(x, possible(x), 'o-', label='possible')
    leg = plt.legend(frameon=False)
    plt.show()


if __name__ == '__main__':
    # plot()
    for i in range(1, 10):
        # print('i', i)
        get_n_possible_delays(i)
        get_n_possible_delays_v2(i)
        get_n_possible_delays_v3(i)
        print(i, ':', (0.5 * (3 ** i - 1)))

