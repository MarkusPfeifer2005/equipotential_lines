"""Just copy file into folder and run it!"""

import csv
import os
import matplotlib.pyplot as plt


def main():
    # Get data from file.
    with open(os.getcwd().split("\\")[-1] + ".csv", 'r', newline='') as csv_file:
        reader = csv.reader(csv_file)
        data = list(reader)

    data = [[j.replace("ï»¿", '') for j in i] for i in data]  # first item might have strange characters - remove them
    data = [[float(j) for j in i] for i in data]  # strings to float

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
