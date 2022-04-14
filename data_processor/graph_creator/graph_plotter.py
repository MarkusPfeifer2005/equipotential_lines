from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt


def load_data(filename: str) -> np.array:
    with np.loadtxt(filename, delimiter=',') as f:
        data = f
    return data


def extract_data_dummy() -> np.array:
    return np.random.random((140, 250))


def show_2d_heatmap(data: np.array):
    ...


def show_heatmap(data: np.array):
    ax = plt.axes(projection='3d')

    X = np.arange(-5, 5, 0.25)
    Y = np.arange(-5, 5, 0.25)
    X, Y = np.meshgrid(X, Y)
    Z = np.ones((40, 40))

    C = np.random.random(size=40 * 40 * 3).reshape((40, 40, 3))  # color

    print(f"{X.shape = }, {Y.shape = }")

    ax.plot_surface(X=X, Y=Y, Z=Z)

    plt.show()


def main():
    # Get data from file.
    # with open(os.getcwd().split("\\")[-1] + ".csv", 'r', newline='') as csv_file:
    #     reader = csv.reader(csv_file)
    #     data = list(reader)

    # data = [[j.replace("ï»¿", '') for j in i] for i in data]  # first item might have strange characters - remove them
    # data = [[float(j) for j in i] for i in data]  # strings to float

    show_heatmap(extract_data_dummy())


if __name__ == "__main__":
    main()
