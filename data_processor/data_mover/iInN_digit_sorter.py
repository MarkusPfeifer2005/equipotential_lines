"""Just copy file into folder and run it!"""

# Imports:
import os


def main():
    # get current working directory
    cwd = os.getcwd()

    # change working directory
    nums = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    for num in nums:
        try:
            os.mkdir(path=os.path.join(cwd, str(num)))
        except FileExistsError:
            pass

    # sort-in .jpg files
    for file_name in os.listdir():
        if ".jpg" not in file_name or 'n' not in file_name:  # check if .jpg and has an n-value
            continue

        n = file_name.split('n')[1].replace(".jpg", '')  # get the number-value as str
        os.rename(file_name, os.path.join(cwd, n, file_name))  # move the file into corresponding folder


# Script:
if __name__ == "__main__":
    main()
