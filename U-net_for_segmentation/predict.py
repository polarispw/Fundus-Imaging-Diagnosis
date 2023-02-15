import os

import torch
from torchvision import transforms
import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2
from src import UNet

def main():
    classes = 1  # exclude background
    mean = (0.709, 0.381, 0.224)
    std = (0.127, 0.079, 0.043)

    # get devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # create model
    weights_path = "./best_model.pth"
    assert os.path.exists(weights_path), f"weights {weights_path} not found."
    model = UNet(in_channels=3, num_classes=classes+1, base_c=32)
    model.load_state_dict(torch.load(weights_path, map_location='cpu')['model'])
    model.to(device)

    data_path = "../../test"
    img_names = os.listdir(data_path)
    save_path = "results"
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    for i in tqdm(img_names):
        img_path = data_path + "/" + i
        assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)

        img = cv2.imread(img_path)
        original_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        # load roi mask
        mask = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        ret, mask = cv2.threshold(mask, 25, 255, cv2.THRESH_BINARY)
        roi_img = Image.fromarray(mask).convert('L')
        roi_img = np.array(roi_img)

        data_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        img = data_transform(original_img)
        img = torch.unsqueeze(img, dim=0)

        model.eval()
        with torch.no_grad():
            # init model
            img_height, img_width = img.shape[-2:]
            init_img = torch.zeros((1, 3, img_height, img_width), device=device)
            model(init_img)

            output = model(img.to(device))

            prediction = output['out'].argmax(1).squeeze(0)
            prediction = prediction.to("cpu").numpy().astype(np.uint8)
            prediction[prediction == 1] = 255
            prediction[roi_img == 0] = 0
            res = Image.fromarray(prediction)
            res.save(os.path.join(save_path, i.split(".")[0]+".png"))

if __name__ == '__main__':
    main()
