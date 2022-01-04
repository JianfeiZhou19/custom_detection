import time
import os
import torch
import torchvision
from PIL import Image
from glob import glob
from torchvision.models.detection.rpn import AnchorGenerator


from train import get_model_instance_segmentation

@torch.no_grad()
def test(model, dataset):
    start = time.time()
    for data_file in dataset:
        img = Image.open(data_file).convert('RGB')
        out = model(img)
    print('average time:', (time.time()-start)/len(dataset))
        
if __name__ == '__main__':
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    num_classes = 2
    backbone = torchvision.models.mobilenet_v2(pretrained=True).features
    backbone.out_channels = 1280
    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                    aspect_ratios=((0.5, 1.0, 2.0),))
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                output_size=7,
                                                sampling_ratio=2)
    # get the model using our helper function
    model = get_model_instance_segmentation(num_classes, backbone, anchor_generator, roi_pooler)
    model.load_state_dict(torch.load('model.pth'))
    # move model to the right device
    model.to(device)
    dataset = glob(os.path.join('', '*.jpg'))
    test(model, dataset)
