import os
import os.path as osp
import numpy as np
import random
import matplotlib.pyplot as plt
import collections
import torch
import torchvision
from torch.utils import data
from PIL import Image, ImageFile

from .augmentations import *
from skimage.segmentation import find_boundaries
# from .HIPe import HIPePeeler
from typing import Tuple, Callable
ImageFile.LOAD_TRUNCATED_IMAGES = True
to_tensor = torchvision.transforms.ToTensor()
valid_colors = [[128, 64, 128],
                [244, 35, 232],
                [70, 70, 70],
                [102, 102, 156],
                [190, 153, 153],
                [153, 153, 153],
                [250, 170, 30],
                [220, 220, 0],
                [107, 142, 35],
                [152, 251, 152],
                [70, 130, 180],
                [220, 20, 60],
                [255, 0, 0],
                [0, 0, 142],
                [0, 0, 70],
                [0, 60, 100],
                [0, 80, 100],
                [0, 0, 230],
                [119, 11, 32]]
label_colours = dict(zip(range(19), valid_colors))

def colorful_texture_structure_mix(img1_structure, img1_texture,
                                   texture : np.ndarray,
                                   ratio : float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Mixing two images with texture-structure decomposition.
    Args:
        img : original image decoupled by HIPe, Tuple[PIL.Image, PIL.Image]
        texture : textural image, PIL.Image
        ratio : mixup ratio
    """
    
    # img1, img2 = to_tensor(img1).unsqueeze(0), to_tensor(img2).unsqueeze(0)
    # guide_edge = guide_edge.unsqueeze(0) # boundary_gt is already in [C, H, W] 
    # print(img1.shape, img2.shape, guide_edge.shape)
    
    img1_structure, img1_texture = np.asarray(img1_structure, dtype=np.float32), np.asarray(img1_texture, dtype=np.float32)
    img2 = np.asarray(texture, dtype=np.float32)
    img2_structure = img2.mean(axis=(0, 1), keepdims=True)
    img2_texture = img2 - img2_structure
    img12 = img1_structure + img2_texture * ratio + img1_texture * (1 - ratio)
    img21 = img2_structure + img1_texture * ratio + img2_texture * (1 - ratio)
    return img12, img21

def colorful_spectrum_mix(img1, img2, low=0, alpha=0.3, ratio=1.0, deg=None):
    """Input image size: ndarray of [H, W, C]"""
    lam = np.random.uniform(low, alpha)

    # lam = alpha

    assert img1.shape == img2.shape
    h, w, c = img1.shape
    h_crop = int(h * sqrt(ratio))
    w_crop = int(w * sqrt(ratio))
    h_start = h // 2 - h_crop // 2
    w_start = w // 2 - w_crop // 2

    img1_fft = np.fft.fft2(img1, axes=(0, 1))
    img2_fft = np.fft.fft2(img2, axes=(0, 1))
    img1_abs, img1_pha = np.abs(img1_fft), np.angle(img1_fft)
    img2_abs, img2_pha = np.abs(img2_fft), np.angle(img2_fft)

    img1_abs = np.fft.fftshift(img1_abs, axes=(0, 1))
    img2_abs = np.fft.fftshift(img2_abs, axes=(0, 1))

    img1_abs_ = np.copy(img1_abs)
    img2_abs_ = np.copy(img2_abs)
    img1_abs[h_start:h_start + h_crop, w_start:w_start + w_crop] = \
        lam * img2_abs_[h_start:h_start + h_crop, w_start:w_start + w_crop] + (1 - lam) * img1_abs_[
                                                                                          h_start:h_start + h_crop,
                                                                                          w_start:w_start + w_crop]
    img2_abs[h_start:h_start + h_crop, w_start:w_start + w_crop] = \
        lam * img1_abs_[h_start:h_start + h_crop, w_start:w_start + w_crop] + (1 - lam) * img2_abs_[
                                                                                          h_start:h_start + h_crop,
                                                                                          w_start:w_start + w_crop]

    img1_abs = np.fft.ifftshift(img1_abs, axes=(0, 1))
    img2_abs = np.fft.ifftshift(img2_abs, axes=(0, 1))

    img21 = img1_abs * (np.e ** (1j * img1_pha))
    img12 = img2_abs * (np.e ** (1j * img2_pha))
    img21 = np.real(np.fft.ifft2(img21, axes=(0, 1)))
    img12 = np.real(np.fft.ifft2(img12, axes=(0, 1)))
    img21 = np.uint8(np.clip(img21, 0, 255))
    img12 = np.uint8(np.clip(img12, 0, 255))

    return img21, img12


class TextureLoader(data.Dataset):
    def __init__(self, root, list_path, texture_list_path, max_iters=None, crop_size=None, mean=(128, 128, 128), transform=None):
        self.n_classes = 19
        self.root = root
        self.list_path = list_path
        self.crop_size = crop_size
        self.mean = mean
        self.transform = transform
        #self.peeler = self.peeler.to('cuda:0').eval()
        # self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        self.img_ids = [i_id.strip() for i_id in open(list_path)]

        if not max_iters == None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))

        self.files = []

        self.id_to_trainid = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
                              19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
                              26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}

        # self.id_to_maskid = {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 9: 1, 10: 1,
        #                      14: 1, 15: 1, 16: 1, 18: 1, 29: 1, 30: 1,
        #                      7: 0, 8: 0, 11: 0, 12: 0, 13: 0, 17: 0,
        #                      19: 0, 20: 0, 21: 0, 22: 0, 23: 0, 24: 0, 25: 0,
        #                      26: 0, 27: 0, 28: 0, 31: 0, 32: 0, 33: 0}
        self.id_to_maskid = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 9: 0, 10: 0,
                             14: 0, 15: 0, 16: 0, 18: 0, 29: 0, 30: 0,
                             7: 0, 8: 0, 11: 0, 12: 0, 13: 0, 17: 0,
                             19: 0, 20: 0, 21: 0, 22: 0, 23: 0, 24: 0, 25: 0,
                             26: 0, 27: 0, 28: 0, 31: 0, 32: 0, 33: 0}
        # self.id_to_maskverid = {7: 0, 8: 1, 11: 1, 12: 1, 13: 1, 17: 1,
        #                         19: 1, 20: 1, 21: 1, 22: 1, 23: 0, 24: 1, 25: 1,
        #                         26: 1, 27: 1, 28: 1, 31: 1, 32: 1, 33: 1}

        self.id_to_maskid1 = {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 9: 1, 10: 1,
                             14: 1, 15: 1, 16: 1, 18: 1, 29: 1, 30: 1,
                             7: 1, 8: 1, 11: 0, 12: 0, 13: 0, 17: 0,
                              19: 0, 20: 0, 21: 0, 22: 0, 23: 1, 24: 0, 25: 0,
                              26: 0, 27: 0, 28: 0, 31: 0, 32: 0, 33: 0}
        self.id_to_maskid2 = {11: 1, 12: 1, 13: 1}
        self.id_to_maskid3 = {17: 1, 19: 1, 20: 1}
        self.id_to_maskid4 = {21: 1, 22: 1}
        # self.id_to_maskid5 = {23: 1}  # sky ignore
        self.id_to_maskid6 = {24: 1, 25: 1}
        self.id_to_maskid7 = {26: 1, 27: 1, 28: 1, 31: 1, 32: 1, 33: 1}
        # self.id_to_maskverid = {7: 1, 8: 1, 11: 1, 12: 1, 13: 1, 17: 1,
        #                      19: 1, 20: 1, 21: 1, 22: 1, 23: 1, 24: 1, 25: 1,
        #                      26: 1, 27: 1, 28: 1, 31: 1, 32: 1, 33: 1}
        self.id_to_maskverid = {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 9: 1, 10: 1,
                             14: 1, 15: 1, 16: 1, 18: 1, 29: 1, 30: 1,
                             7: 1, 8: 1, 11: 1, 12: 1, 13: 1, 17: 1,
                             19: 1, 20: 1, 21: 1, 22: 1, 23: 1, 24: 1, 25: 1,
                             26: 1, 27: 1, 28: 1, 31: 1, 32: 1, 33: 1}
        
        # for split in ["train", "trainval", "val"]:
        for name in self.img_ids:
            img_file = osp.join(self.root, "images/%s" % name)
            label_file = osp.join(self.root, "labels/%s" % name)
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })

    def __len__(self):
        return len(self.files)


    def __getitem__(self, index):
        datafiles = self.files[index]

        image = Image.open(datafiles["img"]).convert('RGB')
        label = Image.open(datafiles["label"])
        # name = datafiles["name"]
        # resize
        if self.crop_size != None:
            image_PIL = image.resize((self.crop_size[1], self.crop_size[0]), Image.BICUBIC)
            label_PIL = label.resize((self.crop_size[1], self.crop_size[0]), Image.NEAREST)

        i_iter = 0
        while (1):
            i_iter = i_iter + 1
            print(i_iter)
            if i_iter > 5:
                print(datafiles["img"])
                break
            # transform
            if self.transform != None:
                image, label = self.transform(image_PIL, label_PIL)

            image = np.asarray(image, np.uint8)
            label_copy = 255 * np.ones(label.shape, dtype=np.long)

            # texture1 = np.asarray(texture1, np.uint8)
            
            #texture2 = np.asarray(texture2, np.uint8)
            # dtexture3 = np.asarray(texture3, np.uint8)
            #texture4 = np.asarray(texture4, np.uint8)
            # texture5 = np.asarray(texture5, np.uint8)
            #texture6 = np.asarray(texture6, np.uint8)
            #texture7 = np.asarray(texture7, np.uint8)
            
            #image_trans, texture = colorful_texture_structure_mix(img_structure, img_texture, texture)#[np.newaxis, :, : , :].transpose((0, 3, 1, 2))
            # image_trans1, texture1 = colorful_spectrum_mix(image, texture1)
            #image_trans2, texture2 = colorful_texture_structure_mix(image, texture2, boundary_gt, self.peel, 1)#[np.newaxis, :, : , :].transpose((0, 3, 1, 2))
            #image_trans3, texture3 = colorful_texture_structure_mix(image, texture3, boundary_gt, self.peel, 1)#[np.newaxis, :, : , :].transpose((0, 3, 1, 2))
            #image_trans4, texture4 = colorful_texture_structure_mix(image, texture4, boundary_gt, self.peel, 1)#[np.newaxis, :, : , :].transpose((0, 3, 1, 2))
            # image_trans5, texture5 = colorful_spectrum_mix(image, texture5)
            #image_trans6, texture6 = colorful_texture_structure_mix(image, texture6, boundary_gt, self.peel, 1)#[np.newaxis, :, : , :].transpose((0, 3, 1, 2))
            #image_trans7, texture7 = colorful_texture_structure_mix(image, texture7, boundary_gt, self.peel, 1)#[np.newaxis, :, : , :].transpose((0, 3, 1, 2))
            # print('image', image.shape)
            # print('texture', texture.shape)
            image = np.asarray(image, np.float32)
            #image_trans = np.asarray(image_trans, np.float32)
            # image_trans1 = np.asarray(image_trans1, np.float32)
            #image_trans2 = np.asarray(image_trans2, np.float32)
            #image_trans3 = np.asarray(image_trans3, np.float32)
            #image_trans4 = np.asarray(image_trans4, np.float32)
            # image_trans5 = np.asarray(image_trans5, np.float32)
            #image_trans6 = np.asarray(image_trans6, np.float32)
            #image_trans7 = np.asarray(image_trans7, np.float32)
            # texture = np.asarray(texture, np.float32)
            label = np.asarray(label, np.long)

            # re-assign labels to match the format of Cityscapes
            label_copy = 255 * np.ones(label.shape, dtype=np.long)
            label_copy_mask = np.zeros(label.shape, dtype=np.long)
            label_copy_maskver = np.zeros(label.shape, dtype=np.long)
            label_copy_mask1 = np.zeros(label.shape, dtype=np.long)
            label_copy_mask2 = np.zeros(label.shape, dtype=np.long)
            label_copy_mask3 = np.zeros(label.shape, dtype=np.long)
            label_copy_mask4 = np.zeros(label.shape, dtype=np.long)
            # label_copy_mask5 = np.zeros(label.shape, dtype=np.long)
            label_copy_mask6 = np.zeros(label.shape, dtype=np.long)
            label_copy_mask7 = np.zeros(label.shape, dtype=np.long)
            # label_copy_maskver = np.zeros(label.shape, dtype=np.long)
            for k, v in self.id_to_trainid.items():
                label_copy[label == k] = v
            for k, v in self.id_to_maskid.items():
                label_copy_mask[label == k] = v
            for k, v in self.id_to_maskverid.items():
                label_copy_maskver[label == k] = v
            for k, v in self.id_to_maskid1.items():
                label_copy_mask1[label == k] = v
            for k, v in self.id_to_maskid2.items():
                label_copy_mask2[label == k] = v
            for k, v in self.id_to_maskid3.items():
                label_copy_mask3[label == k] = v
            for k, v in self.id_to_maskid4.items():
                label_copy_mask4[label == k] = v
            # for k, v in self.id_to_maskid5.items():
            #     label_copy_mask5[label == k] = v
            for k, v in self.id_to_maskid6.items():
                label_copy_mask6[label == k] = v
            for k, v in self.id_to_maskid7.items():
                label_copy_mask7[label == k] = v

            # for k, v in self.id_to_maskverid.items():
            #     label_copy_maskver[label == k] = v
            #label_copy_mask1 = np.concatenate([label_copy_mask1[:, :, np.newaxis]] * 3, axis=-1)
            #label_copy_mask2 = np.concatenate([label_copy_mask2[:, :, np.newaxis]] * 3, axis=-1)
            #label_copy_mask3 = np.concatenate([label_copy_mask3[:, :, np.newaxis]] * 3, axis=-1)
            #label_copy_mask4 = np.concatenate([label_copy_mask4[:, :, np.newaxis]] * 3, axis=-1)
            # label_copy_mask5 = np.concatenate([label_copy_mask5[:, :, np.newaxis]] * 3, axis=-1)
            #label_copy_mask6 = np.concatenate([label_copy_mask6[:, :, np.newaxis]] * 3, axis=-1)
            #label_copy_mask7 = np.concatenate([label_copy_mask7[:, :, np.newaxis]] * 3, axis=-1)
            # label_copy_maskver = np.concatenate([label_copy_maskver[:, :, np.newaxis]] * 3, axis=-1)
            #label_copy_mask1 = np.asarray(label_copy_mask1, np.float32)
            #label_copy_mask2 = np.asarray(label_copy_mask2, np.float32)
            #label_copy_mask3 = np.asarray(label_copy_mask3, np.float32)
            #label_copy_mask4 = np.asarray(label_copy_mask4, np.float32)
            # label_copy_mask5 = np.asarray(label_copy_mask5, np.float32)
            #label_copy_mask6 = np.asarray(label_copy_mask6, np.float32)
            #label_copy_mask7 = np.asarray(label_copy_mask7, np.float32)
            # label_copy_maskver = np.asarray(label_copy_maskver, np.float32)

            label_copy_mask = np.concatenate([label_copy_mask[:, :, np.newaxis]] * 3, axis=-1)
            label_copy_maskver = np.concatenate([label_copy_maskver[:, :, np.newaxis]] * 3, axis=-1)
            label_copy_mask = np.asarray(label_copy_mask, np.float32)
            label_copy_maskver = np.asarray(label_copy_maskver, np.float32)
            #image_trans_sky = image * label_copy_mask + image_trans * label_copy_maskver
            #image_trans_class = image * label_copy_mask1 + image_trans2 * label_copy_mask2 + image_trans3 * label_copy_mask3 + \
            #                    image_trans4 * label_copy_mask4 + image_trans6 * label_copy_mask6 + image_trans7 * label_copy_mask7
            label_cat, label_time = np.unique(label_copy, return_counts=True)
            label_p = 1.0 * label_time / np.sum(label_time)
            pass_c, pass_t = np.unique(label_p > 0.02, return_counts=True)
            boundary_gt = label_copy.astype(np.uint8)
            boundary_gt = (find_boundaries(boundary_gt, mode='inner')).astype(np.uint8)
            boundary_gt = torch.from_numpy(boundary_gt).unsqueeze(0).float()
            # print('label_copy_mask', label_copy_mask.shape)
            # print(label_copy_mask.max(), label_copy_mask.min())
            # print('label_copy_maskver', label_copy_maskver.shape)
            # print(label_copy_maskver.max(), label_copy_maskver.min())

            if pass_c[-1] == True:
                if pass_t[-1] >= 3:
                    break
                elif pass_t[-1] == 2:
                    if not (label_cat[-1] == 255 and label_p[-1] > 0.02):
                        break
            
        # size = image_trans.shape
        image = image[:, :, ::-1]  # change to BGR
        image -= 128#self.mean
        image = image.transpose((2, 0, 1)) / 128.0

        #image_trans_class = image_trans_class[:, :, ::-1]  # change to BGR
        #image_trans_class -= self.mean
        #image_trans_class = image_trans_class.transpose((2, 0, 1)) / 128.0


        #image_trans_sky = image_trans_sky[:, :, ::-1]  # change to BGR
        #image_trans_sky -= self.mean
        #image_trans_sky = image_trans_sky.transpose((2, 0, 1)) / 128.0

        # if random.random() < 0.9:
        #     return image.copy(), image_trans_sky.copy(), image_trans_class.copy(), label_copy.copy(), boundary_gt
        #print(torch.sum(boundary_gt[boundary_gt > 0]))
        return image.copy(),  label_copy.copy(), boundary_gt 

    def decode_segmap(self, img):
        map = np.zeros((img.shape[0], img.shape[1], img.shape[2], 3))
        for idx in range(img.shape[0]):
            temp = img[idx, :, :]
            r = temp.copy()
            g = temp.copy()
            b = temp.copy()
            for l in range(0, self.n_classes):
                r[temp == l] = label_colours[l][0]
                g[temp == l] = label_colours[l][1]
                b[temp == l] = label_colours[l][2]

            rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
            rgb[:, :, 0] = r / 255.0
            rgb[:, :, 1] = g / 255.0
            rgb[:, :, 2] = b / 255.0
            map[idx, :, :, :] = rgb
        return map


if __name__ == '__main__':
    dst = GTA5DataSet("./data", is_transform=True)
    trainloader = data.DataLoader(dst, batch_size=4)
    for i, data in enumerate(trainloader):
        imgs, labels = data
        if i == 0:
            img = torchvision.utils.make_grid(imgs).numpy()
            img = np.transpose(img, (1, 2, 0))
            img = img[:, :, ::-1]
            plt.imshow(img)
            plt.show()
