# -*- coding: utf-8 -*-
import os
import json
import random

def get_image_label_list(root):
    flower_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    flower_class.sort()
    class_indices = dict((k, v) for v, k in enumerate(flower_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)
    train_images_path = []
    train_images_label = []
    every_class_num = []
    supported = [".jpg", ".JPG", ".png", ".PNG"]
    for cla in flower_class:
        cla_path = os.path.join(root, cla)
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
        image_class = class_indices[cla]
        every_class_num.append(len(images))
        for img_path in images:
            train_images_path.append(img_path)
            train_images_label.append(image_class)

    return train_images_path, train_images_label

def read_split_data_mineral(root: str, val_rate: float = 0.2):
    random.seed(0)
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)
    train_path = os.path.join(root, 'train')
    test_path = os.path.join(root, 'test')
    train_images_path, train_images_label = get_image_label_list(train_path)
    test_images_path, test_images_label = get_image_label_list(test_path)
    print("{} images for training.".format(len(train_images_path)))
    print("{} images for validation.".format(len(test_images_path)))
    return train_images_path, train_images_label, test_images_path, test_images_label
