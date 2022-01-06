import time
import os
import torch
import torchvision
from PIL import Image
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

# from train import get_model_instance_segmentation
from torchvision.transforms import functional as F

def get_model_instance_segmentation(num_classes, backbone, anchor_generator, roi_pooler):
    # load an instance segmentation model pre-trained pre-trained on COCO
    # model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    model = MaskRCNN(backbone=backbone, 
                     num_classes=num_classes, 
                     rpn_anchor_generator=anchor_generator,
                     box_roi_pool=roi_pooler)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model

@torch.no_grad()
def test(model, dataset):
    start1 = time.time()
    for data_file in dataset:
        start = time.time()
        img_input = Image.open(data_file).convert('RGB')
        img = F.to_tensor(img_input)
        out = model([img.cuda()])
        labels = out[0]['labels'].data.cpu().numpy()
        mask_out = out[0]['masks'].data.cpu().numpy()
        w, h = img_input.size
        mask = np.zeros((h, w), dtype=np.int8)
        for idx, label in enumerate(labels):
            mask[mask_out[idx, 0] > 0.3] = label
        img = np.asarray(img_input)
        plt.subplot(121)
        plt.imshow(img)
        plt.subplot(122)
        plt.imshow(mask)
        print(time.time()-start)
        plt.show()
        
    print('average time:', (time.time()-start1)/len(dataset))
        
if __name__ == '__main__':
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    num_classes = 3
    backbone = torchvision.models.mobilenet_v2(pretrained=True).features
    backbone.out_channels = 1280
    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                    aspect_ratios=((0.5, 1.0, 2.0),))
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                output_size=7,
                                                sampling_ratio=2)
    # get the model using our helper function
    model = get_model_instance_segmentation(num_classes, backbone, anchor_generator, roi_pooler)
    model.load_state_dict(torch.load('mobile.pth'))
    # move model to the right device
    model.to(device)
    model.eval()
    dataset = glob(os.path.join('D:/BaiduNetdiskDownload/MaskDatas/img/test', '*.jpg'))
    test(model, dataset)
