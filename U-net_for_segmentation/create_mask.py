import os
import cv2
from PIL import Image

source_path = "datasets_for_U-net/training/images"
save_path = "datasets_for_U-net/training/mask/"
img_names = [i for i in os.listdir(source_path) if i.endswith(".tif")]
for img in img_names:
    image = cv2.imread(source_path + "/" + img)
    image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    ret, image = cv2.threshold(image, 25, 255, cv2.THRESH_BINARY)
    image = Image.fromarray(image)
    image.save(save_path + img.split(".")[0]+".gif")
