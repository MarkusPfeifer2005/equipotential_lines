import tkinter as tk
from tkinter import ttk


class FrmParameters(ttk.Frame):
    def __init__(self, container):
        super().__init__(container)
        self.lbl_parameter = tk.Label(self, text="Parameters")
        self.lbl_value = tk.Label(self, text="Values")

        # preparing the entries
        self._entries = {"button": ttk.Button(self, text="apply", command=self.apply_changes)}
        self._entries["button"].grid(row=0, column=1)
        self.default_params = [
            "area_to_map",
            "step_size",
            "last_pos",
            "voltage",
            "electrode_type",
            "liquid",
            "ground_clearance",
            "liquid_debt",
            "liquid_temp",
        ]

        for param in self.default_params:
            self.add_entry(param)

    def add_entry(self, key):
        """Key get converted to string."""
        # add items
        self._entries[key] = [
            ttk.Label(self, text=key),
            ttk.Entry(self)
        ]

        # display items
        self._entries[key][0].grid(row=len(self._entries), column=0)
        self._entries[key][1].grid(row=len(self._entries), column=1)

    def update_entries(self):
        ...

    def apply_changes(self):
        ...


class FrmActuators(ttk.Frame):
    def __init__(self, container):
        super().__init__(container)

        self.buttons = {
            "start": ttk.Button(self, text="start"),
            "stop": ttk.Button(self, text="stop"),
            "pause": ttk.Button(self, text="pause")
        }
        for idx, button in enumerate(self.buttons.values()):
            button.grid(row=idx, column=0)


def main() -> None:
    root = tk.Tk()  # main window
    root.title("Equipotential Lines")

    actuators = FrmActuators(root)
    params = FrmParameters(root)
    actuators.pack()
    params.pack()

    root.mainloop()


if __name__ == "__main__":
    main()
