#!/usr/bin/python3
# -*- coding: utf-8 -*-

import random
import torchvision.transforms.functional as TF
from PIL import Image, ImageEnhance
import os
import numpy as np


def erase_and_save(image_path, target_dir, position, size):
    image = TF.to_tensor(Image.open(image_path))
    erased_image = TF.to_pil_image(TF.erase(img=image,
                                            i=position[0],
                                            j=position[1],
                                            h=size[0],
                                            w=size[1],
                                            v=1))
    erased_image.save(os.path.join(target_dir, os.path.basename(image_path[:-4])) + "_erased_.jpg")
    return erased_image


def rotate_and_save(image_path, target_dir, angles_list):
    image = Image.open(image_path)
    image_list = []
    for angle in angles_list:
        rotated_image = TF.rotate(img=image, angle=angle, resample=Image.NEAREST)
        rotated_image.save(os.path.join(target_dir, os.path.basename(image_path[:-4])) + "_" + str(angle) + "_.jpg")
        image_list.append(rotated_image)
    return image_list


def vflip_and_save(image_path, target_dir):
    image = Image.open(image_path)
    vertical_image = TF.vflip(img=image)
    vertical_image.save(os.path.join(target_dir, os.path.basename(image_path[:-4])) + "_v_.jpg")

    return vertical_image


def hflip_and_save(image_path, target_dir):
    image = Image.open(image_path)
    horizontal_image = TF.hflip(img=image)
    horizontal_image.save(os.path.join(target_dir, os.path.basename(image_path[:-4])) + "_h_.jpg")

    return horizontal_image


def randomColor(image):
    random_factor = np.random.randint(0, 31) / 10.
    color_image = ImageEnhance.Color(image).enhance(random_factor)
    random_factor = np.random.randint(10, 21) / 10.
    brightness_image = ImageEnhance.Brightness(color_image).enhance(random_factor)
    random_factor = np.random.randint(10, 21) / 10.
    contrast_image = ImageEnhance.Contrast(brightness_image).enhance(random_factor)
    random_factor = np.random.randint(0, 31) / 10.
    return ImageEnhance.Sharpness(contrast_image).enhance(random_factor)


def brightness_and_save(image_path, target_dir):
    image = Image.open(image_path)
    factor = random.uniform(0, 2)
    brightness_image = TF.adjust_brightness(image, brightness_factor=factor)
    brightness_image.save(os.path.join(target_dir, os.path.basename(image_path[:-4])) + "_brightness_.jpg")
    return brightness_image


def adjust_hue_and_save(image_path, target_dir):
    image = Image.open(image_path)
    adjust_hue_image = TF.adjust_hue(image, hue_factor=0.5)
    adjust_hue_image.save(os.path.join(target_dir, os.path.basename(image_path[:-4])) + "_adjust_.jpg")
    return adjust_hue_image


def gamma_and_save(image_path, target_dir, gamma_value):
    image = Image.open(image_path)
    gamma_image = TF.adjust_gamma(img=image, gamma=gamma_value)
    gamma_image.save(os.path.join(target_dir, os.path.basename(image_path[:-4])) + "_gamma_.jpg")
    return gamma_image


if __name__ == '__main__':
    import glob

    names = glob.glob(r'data for augmentation')
    random.shuffle(names)
    for name in names:
        image_path = name
        target_dir = '\\'.join(name.split('\\')[:-1])
        rotate_and_save(image_path, target_dir, angles_list=[-10, -20, -30, -40])
        vflip_and_save(image_path, target_dir)
        brightness_and_save(image_path, target_dir)
        hflip_and_save(image_path, target_dir)
        x = random.randint(1, 155)
        y = random.randint(1, 155)
        l = random.randint(1, 100)
        w = random.randint(1, 100)
        res2 = erase_and_save(image_path, target_dir, position=[x, y], size=[l, w])
