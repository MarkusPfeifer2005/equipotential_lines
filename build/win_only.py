import os
import cv2
from build.data_accessories import RpiSession, JSON, Directory, get_desktop
import matplotlib.pyplot as plt
import numpy as np


class Session(RpiSession):
    """
    Special type of directory that contains a json and a csv.
    kwargs: path_to_dir
    If path_to_dir is not set, a filedialog box opens.
    """

    def __init__(self, **kwargs):
        super(Session, self).__init__(**kwargs)
        self.w, self.h = 150, 300
        self.xy = [(215, 230), (310, 230), (400, 230)]

    def prepare_for_ml(self, **kwargs):
        """
        Creates folder with images that can be used to broaden the capabilities of cnn. The method only works if
        a completely filled .csv file is available. The images must be in x,y,z.jpg format and the rois get saved in
        i,n.jpg format.
        """
        img_idx_json = {"img_idx": kwargs["img_idx"]} if "img_idx" in kwargs else JSON("../build/image_index.json")

        # create folders
        ml_dir = Directory(os.path.join(kwargs["target_dir"], "ml_" + self.name)) if "target_dir" in kwargs \
            else Directory(os.path.join(get_desktop(), "ml_" + self.name))

        for i in range(10):
            Directory(os.path.join(ml_dir.path, str(i)))

        # fill folders
        for x, y, z, v in self.csv:  # Attention: v is a float value like: v.vv
            image = cv2.imread(os.path.join(self.path, f"{x},{y},{z}{self.image_ext}"))

            for idx, (x1, y1) in enumerate(self.xy):
                digit_roi = image[y1 - self.h // 2:y1 + self.h // 2, x1 - self.w // 2:x1 + self.w // 2]  # digit roi
                digit = format(float(v), ".2f").replace('.', '')[idx]  # format makes it a string with 2 decimals
                img_path = os.path.join(ml_dir.path, digit, f"i{img_idx_json['img_idx']}n{digit}.jpg")
                cv2.imwrite(img_path, digit_roi)
                img_idx_json["img_idx"] += 1


class Plot:
    def __init__(self, session: Session):
        self.session = session

    def plot(self) -> None:
        pass

    def __call__(self, *args, **kwargs):
        self.plot()

    def prepare_data(self) -> np.array:
        x_max, y_max, z_max = self.session.json["area_to_map"]
        x_stp, y_stp, z_stp = self.session.json["step_size"]

        data = [[[self.session.csv.get_value(pos=(x, y, z)) for x in range(0, x_max, x_stp)]
                 for y in range(y_max-y_stp, 0, -y_stp)] for z in range(0, z_max, z_stp)]  # y is iterated in reverse

        return np.array(data)


class HeatMap(Plot):
    def __init__(self, session=None):
        super().__init__(session)

    def plot(self) -> None:
        data = self.prepare_data()
        fig, ax = plt.subplots(nrows=1, ncols=len(data))
        planes = [i for i in range(0, self.session.json["area_to_map"][2], self.session.json["step_size"][2])]
        plt.tight_layout(pad=0.1)

        for i, (d, debt) in enumerate(zip(data, planes)):
            im = ax[i].imshow(d)
            ax[i].set_title(f"depth {debt}mm")
            ax[i].set_xlabel("x-distance [mm]")
            ax[i].set_ylabel("y-distance [mm]")

        try:
            plt.suptitle(f"voltage: {self.session.json['voltage']}\n"
                         f"electrode type: {self.session.json['electrode_type']}\n"
                         f"liquid: {self.session.json['liquid']}\n"
                         f"liquid temperature: {self.session.json['liquid_temp']}°C")
        except KeyError:
            pass

        plt.colorbar(im, ax=ax.ravel().tolist()).set_label("Voltage [V]")
        plt.show()


def main() -> None:
    active_session = Session()
    # model = torch.load(r"../models/lcd_cnn_5_98.pt").to("cpu")
    # active_session.fill_csv(model=model)
    # active_session.prepare_for_ml()

    p = HeatMap(session=active_session)
    p.plot()


if __name__ == "__main__":
    main()
