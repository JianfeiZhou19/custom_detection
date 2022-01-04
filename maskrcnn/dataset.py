import os
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
import json
import matplotlib.pyplot as plt
import pycocotools.mask as mask_util
from pycocotools import mask as maskUtils
from torchvision.models.vgg import make_layers
from coco_utils import ConvertCocoPolysToMask


class HookDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms, phas):
        self.root = root
        self.transforms = transforms
        self.phas = phas
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, 'img', self.phas))))
        # self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.root, "img", self.phas, self.imgs[idx])
        # mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        anno_path = img_path.replace('img', 'json').replace('jpg', 'json')
        # mask = Image.open(mask_path)
        # convert the PIL Image into a numpy array
        mask = convert(anno_file=anno_path)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)

def polygons_to_mask(img_shape, polygons, label):
 
    mask = np.zeros(img_shape, dtype=np.uint8)
    mask = Image.fromarray(mask)
    xy = list(map(tuple, polygons))
    ImageDraw.Draw(mask).polygon(xy=xy, outline=label, fill=label)
    mask = np.array(mask, dtype=np.uint8)
    return mask

def bounding_box(img):
    # return max and min of a mask to draw bounding box
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax

def convert(anno_file):
    with open(anno_file, 'r') as f:
        target = json.load(f)
    h, w = target['imageHeight'], target['imageWidth']
    label_dict = {'hook':1, 'hookcolor':1, 'trunnion':2, 'trunnion1':2, 'trunnioncolor':2}
    mask_res = np.zeros((h, w), dtype=np.uint8)
    for obj in target['shapes']:
        label = obj['label']
        points = obj['points']
        mask_res += polygons_to_mask((h, w), points, label_dict[label])
    mask_res = np.clip(mask_res, 0, 2)
    return mask_res

if __name__ == '__main__':
    mask_res = convert('./data/192.168.13.10_01_2021110509243822_21.json')
    plt.imshow(mask_res)
    plt.show()
    
