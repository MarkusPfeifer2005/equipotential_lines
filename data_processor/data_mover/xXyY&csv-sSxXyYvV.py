"""Just copy file into folder and run it!"""

import os
import csv


def main():
    # get current working directory
    cwd = os.getcwd()

    # get current session number
    session = ""
    for char in cwd.split("\\")[-1]:
        if char.isdigit():
            session += char
    session = int(session)

    # get csv values
    with open(f"session{session}.csv", 'r') as csv_file:
        reader = csv.reader(csv_file)
        labels = list(reader)

    # rename all files to sSxXyYvV-convention
    for file_name in os.listdir():
        # allow only .jpg files
        if ".jpg" not in file_name or 'v' in file_name:  # prevent already labeled images from interrupting process
            continue

        X = file_name.split('x')[1].split('y')[0]
        Y = file_name.split('y')[1].replace(".jpg", '')

        for label in labels:
            x, y, v = label
            v = v.replace('.', '')
            x = x.replace("ï»¿", '')  # the first entry might contain: ï»¿ so remove it

            # append 0s that got lost in Excel
            if len(v) == 2:
                v += '0'
            elif len(v) == 1:
                v += "00"

            if x == X and y == Y:
                os.rename(file_name, f"s{session}x{x}y{y}v{v}.jpg")


if __name__ == "__main__":
    main()
