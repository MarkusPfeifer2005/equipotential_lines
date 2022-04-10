"""Just copy file into folder and run it!"""

import os
import cv2
import json

# Parameters
cwd = os.getcwd().split("\\")[-1]

# getting image names
jpg_names = [file for file in os.listdir() if ".jpg" in file]  # get all .jpg files

# create new directory
try:
    os.mkdir(f"../{cwd}/as_dataset_2_{cwd}")
except FileExistsError:
    pass
tar_dir = f"../{cwd}/as_dataset_2_{cwd}"

# handle image index
with open(r"D:\OneDrive - brg14.at\Desktop\data_processor\data_mover\image_index.json") as json_file:
    image_index = json.load(json_file)
    img_idx = image_index["img_idx"]

# rename file and move to new folder
for jpg_name in jpg_names:
    image = cv2.imread(jpg_name)
    v = jpg_name.split('v')[1].replace(".jpg", '')
    n1, n2, n3 = v[0], v[1], v[2]  # all are numbers as strings

    h, w = 300, 150  # height & width

    # extract digit 1
    x, y = 215, 230
    digit1 = image[y - h // 2:y + h // 2, x - w // 2:x + w // 2]
    cv2.imwrite(f"{tar_dir}/i{img_idx}n{n1}.jpg", digit1)
    img_idx += 1

    # extract digit 2
    x, y = 310, 230
    digit2 = image[y - h // 2:y + h // 2, x - w // 2:x + w // 2]
    cv2.imwrite(f"{tar_dir}/i{img_idx}n{n2}.jpg", digit2)
    img_idx += 1

    # extract digit 3
    x, y = 400, 230
    digit3 = image[y - h // 2:y + h // 2, x - w // 2:x + w // 2]
    cv2.imwrite(f"{tar_dir}/i{img_idx}n{n3}.jpg", digit3)
    img_idx += 1

# save the image index
with open(r"D:\OneDrive - brg14.at\Desktop\data_processor\data_mover\image_index.json", 'w') as json_file:
    json.dump({"img_idx": img_idx}, json_file)
