import matplotlib.pyplot as plt
import math
import numpy as np

x_max: float = 0.255  # in m
y_max: float = 0.150  # in m
step: float = 0.005  # in m

epsilon_0 = 8.854 * 10 ** -12
epsilon_r: int = 10
epsilon = epsilon_0 * epsilon_r

U: float = 5  # in V
A = 0.002 * 0.008  # in m

electrode_1 = (0.070, 0.075)
electrode_2 = (0.170, 0.075)


def get_d(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    potenz1 = math.pow((p1[0]-p2[0]), 2)
    potenz2 = math.pow((p1[1]-p2[1]), 2)
    return math.sqrt(potenz1 + potenz2)


def magic_math_function(p: tuple[float, float]) -> float:
    dr = 0.002  # radius of electrode in m
    k = 1 / (4 * np.pi * epsilon)
    Q = U * 2 * np.pi * epsilon * dr
    V1 = - k * Q / get_d(electrode_1, p)
    V2 = k * Q / get_d(electrode_2, p)
    return V1 + V2 + U / 2  # due to zero-point


def main():
    data = []
    for x in range(0, int(x_max*1000), int(step*1000)):
        for y in range(0, int(y_max*1000), int(step*1000)):
            if (x/1000, y/1000) == electrode_1 or (x/1000, y/1000) == electrode_2:
                continue
            v = magic_math_function(p=(x/1000, y/1000))
            print(f"{v = }")
            data.append([x, y, v])

    # Just plotting.
    ax = plt.axes(projection="3d")
    ax.set_ylabel("y-distance [mm]")
    ax.set_xlabel("x-distance [mm]")
    ax.set_zlabel("voltage [V]")

    for i in data:
        ax.scatter(i[0], -i[1], i[2])

    plt.show()


if __name__ == "__main__":
    main()
