import os
import json

import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import cv2
from model import efficientnetv2_s as create_model

def scaleRadius(img,scale):
    x = img[int(img.shape[0]/2),:,:].sum(1) # 图像中间1行的像素的3个通道求和。输出（width*1）
    r = (x>x.mean()/10).sum()/2 # x均值/10的像素是为眼球，计算半径
    s = scale*1.0/r
    return cv2.resize(img,(0,0),fx=s,fy=s)

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    img_size = {"s": [300, 384],  # train_size, val_size
                "m": [384, 480],
                "l": [384, 480]}
    num_model = "s"

    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # create model
    model = create_model(num_classes=4).to(device)
    # load model weights
    model_weight_path = "best_acc.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()

    data_transform = transforms.Compose(
        [transforms.Resize(img_size[num_model][1]),
         transforms.CenterCrop(img_size[num_model][1]),
         transforms.ToTensor(),
         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    # procedure
    data_path = "../../test"
    img_names = os.listdir(data_path)
    res = []
    for i in tqdm(img_names):
        img_path = data_path + "/" + i
        assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
        img = Image.open(img_path)

        img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        scale = 300
        a = scaleRadius(img, scale)
        # subtract local mean color
        a = cv2.addWeighted(a, 4, cv2.GaussianBlur(a, (0, 0), scale / 30),-4, 128)
        # remove out er 10%
        b = np.zeros(a.shape)
        cv2.circle(b, (int(a.shape[1] / 2), int(a.shape[0] / 2)), int(scale * 0.9), (1, 1, 1), -1, 8, 0)
        a = a * b + 128 * (1 - b)
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        img = data_transform(img)
        # expand batch dimension
        img = torch.unsqueeze(img, dim=0)

        with torch.no_grad():
            # predict class
            output = torch.squeeze(model(img.to(device))).cpu()
            predict = torch.softmax(output, dim=0)
            predict_cls = torch.argmax(predict).item()
        res.append([str(i), int(predict_cls)])

    np.save("./pre.npy", res)
    print("Results have been saved in pre.txt")

if __name__ == '__main__':
    main()
    # res = np.load('pre.npy')
    # print(res)
