import os
import sys
import tarfile
import collections
import torch.utils.data as data
import shutil
import numpy as np
import cv2
W = int(1216/4)
H = int(1936/4)

def voc_cmap(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap

class SegmentDataLoader(data.Dataset):
    """`Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Segmentation Dataset.
    Args:
        root (string): Root directory of the VOC Dataset.
        year (string, optional): The dataset year, supports years 2007 to 2012.
        image_set (string, optional): Select the image_set to use, ``train``, ``trainval`` or ``val``
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
    """
    cmap = voc_cmap()
    def __init__(self, 
                 image_dir, 
                 mask_dir, 
                 width,
                 height,
                 transform=None,
                 gammas=3.0,
                 crop=35):
        
        self.transform = transform
        self.gammas = gammas
        self.crop = crop
        self.width = width
        self.height = height
        train_img_dir = os.listdir(image_dir)
        train_img_dir.sort()
        # jpg image
        self.images = [os.path.join(image_dir, path) for path in train_img_dir]
        # png mask
        load_mask_path = os.listdir(mask_dir)
        load_mask_path.sort()
        self.masks = [os.path.join(mask_dir, path) for path in load_mask_path]
        assert (len(self.images) == len(self.masks))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        img = cv2.imread(self.images[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.height, self.width), cv2.INTER_NEAREST)
        if self.gammas is not None:
            img = self.gamma(img, gamma=self.gammas)
        img = img.astype(np.float32)/255
        target = cv2.imread(self.masks[index], 0)
        target = cv2.resize(target, (self.height, self.width), cv2.INTER_NEAREST)
        
        if self.transform is not None:
            augment = self.transform(image=img, mask=target)
            img, target = augment['image'], augment['mask']
        if self.crop:
            return img[:W-self.crop,:], target.reshape(self.width, self.height)[:W-self.crop,:]
        else:
            return img, target.reshape(self.width, self.height)
            
    def __len__(self):
        return len(self.images)
    
    def gamma(self, img, gamma = 3.0):
        gamma_cvt = np.zeros((256,1),dtype = 'uint8')
        for i in range(256):
             gamma_cvt[i][0] = 255 * (float(i)/255) ** (1.0/gamma)
        gamma_img = cv2.LUT(img, gamma_cvt)
        return gamma_img

    @classmethod
    def decode_target(cls, mask):
        """decode semantic mask to RGB image"""
        return cls.cmap[mask]



