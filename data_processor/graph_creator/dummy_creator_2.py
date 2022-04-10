import matplotlib.pyplot as plt
import math
import numpy as np

x_max: int = 255  # in mm
y_max: int = 150  # in mm
step: int = 5  # in mm

epsilon_0 = 8.854 * 10 ** -12
epsilon_r: int = 10
epsilon = epsilon_0 * epsilon_r

U: float = 10  # in V
A = 2 * 8  # in mm

electrode_1 = (70, 60)
electrode_2 = (170, 60)


def get_d(p1: tuple[int, int], p2: tuple[int, int]) -> float:
    potenz1 = math.pow((p1[0]-p2[0]), 2)
    potenz2 = math.pow((p1[1]-p2[1]), 2)
    return math.sqrt(potenz1 + potenz2)


def magic_math_function(p: tuple[int, int]) -> float:
    k = 1/(4 * math.pi * epsilon)
    C = epsilon * A / get_d(electrode_1, electrode_2)
    Q1 = -U * C
    Q2 = U * C

    V1 = k * Q1 / get_d(p, electrode_1)
    V2 = k * Q2 / get_d(p, electrode_2)

    return V1 + V2  # +-?


if __name__ == "__main__":
    # get input
    # ipt = input("1: 3d, 2: heatmap")
    # if ipt == '':
    #     ipt = '1'
    #
    # if ipt == '1':
    data = []
    for x in range(0, x_max, step):
        for y in range(0, y_max, step):
            if (x, y) == electrode_1 or (x, y) == electrode_2:
                continue
            v = magic_math_function(p=(x, y))
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

    # elif ipt == '2':
    #     data = []
    #     for x in range(0, x_max, step):
    #         row = []
    #         for y in range(0, y_max, step):
    #             if (x, y) == electrode_1 or (x, y) == electrode_2:
    #                 v = 0
    #             else:
    #                 v = magic_math_function(p=(x, y))
    #             print(f"{v = }")
    #             row.append([x, y, v])
    #         data.append(row)
    #
    #     data = np.array(data)
    #     fig, ax = plt.subplots()
    #     img = ax.imshow(data)
    #     fig.tight_layout()
    #     plt.show()
