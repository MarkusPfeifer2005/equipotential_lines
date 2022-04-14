import tkinter as tk
from tkinter import filedialog


def get_file_path(filetypes: list[tuple[str, str]] = None) -> str:
    tk.Tk().withdraw()  # Stops additional tkinter window from popping up.

    if filetypes:
        return filedialog.askopenfilename(filetypes=filetypes)
    else:
        return filedialog.askopenfilename()


def main():
    print(get_file_path())


if __name__ == "__main__":
    main()
