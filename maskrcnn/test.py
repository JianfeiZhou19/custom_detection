import time
import os
import torch
import torchvision
from PIL import Image
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.transforms import functional as F

from train import get_model_instance_segmentation

@torch.no_grad()
def test(model, dataset):
    start = time.time()
    for data_file in dataset:
        start = time.time()
        img_input = Image.open(data_file).convert('RGB')
        img = F.to_tensor(img)
        out = model([img.cuda()])
        labels = out[0]['labels'].data.cpu().numpy()
        mask_out = out[0]['masks'].data.cpu().numpy()
        w, h = img_input.size
        mask = np.zeros((h, w), dtype=np.int8)
        for idx, label in enumerate(labels):
            mask[mask_out[idx, 0] > 0.1] = label
        img = np.asarray(img_input)
        plt.subplot(121)
        plt.imshow(img)
        plt.subplot(122)
        plt.imshow(mask)
        print(time.time()-start)
        plt.show()
        
    print('average time:', (time.time()-start)/len(dataset))
        
if __name__ == '__main__':
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    num_classes = 3
    backbone = torchvision.models.mobilenet_v2(pretrained=True).features
    backbone.out_channels = 1280  #v3_small: 576
    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                    aspect_ratios=((0.5, 1.0, 2.0),))
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                output_size=7,
                                                sampling_ratio=2)
    # get the model using our helper function
    model = get_model_instance_segmentation(num_classes, backbone, anchor_generator, roi_pooler)
    model.load_state_dict(torch.load('mobile_v3_small.pth'))
    # move model to the right device
    model.to(device)
    model.eval()
    dataset = glob(os.path.join('D:/BaiduNetdiskDownload/MaskDatas/img/test', '*.jpg'))
    test(model, dataset)
