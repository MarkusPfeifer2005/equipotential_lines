import os
import matplotlib.pyplot as plt
from data_handling import CSV, JSON


class Plot:
    def __init__(self, path_to_session: str):
        if os.path.isdir(path_to_session) and "session" in os.path.split(path_to_session)[1]:
            session_name: str = os.path.split(path_to_session)[1]
            self.data_csv = CSV(os.path.join(path_to_session, session_name+".csv"))
            self.label_json = JSON(os.path.join(path_to_session, session_name+".json"))
        else:
            raise Exception("No session selected!")

    def plot(self) -> None:
        raise Exception("plot-function not implemented!")

    def __call__(self, *args, **kwargs):
        self.plot()


class ThreeD(Plot):
    def __init__(self, path_to_session: str):
        super(Plot, self).__init__(path_to_session)

    def plot(self) -> None:
        ax = plt.axes(projection="3d")
        ax.set_ylabel("y-distance [mm]")
        ax.set_xlabel("x-distance [mm]")
        ax.set_zlabel("voltage [V]")

        for row in self.data_csv:
            ax.scatter(*row)

    plt.show()


class TwoD(Plot):
    pass


class SpreadTwoD:
    pass
