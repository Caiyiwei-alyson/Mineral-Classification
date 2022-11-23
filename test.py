#!/usr/bin/python3
# coding=utf-8
from __future__ import absolute_import, division, print_function
import logging
import argparse
import os
import random
import numpy as np
from datetime import timedelta
import torch
import torch.distributed as dist
from tqdm import tqdm
from models.model_resnet import resnet50
from models.model_mobilenet import MobileNetV2
from models.model_swin import swin_tiny_patch4_window7_224
from utils.data_utils import get_loader
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
logger = logging.getLogger(__name__)


def setup(args):
    # Prepare model
    if args.dataset == "mineral":
        num_classes = 7
    if args.method == 'resnet50':
        model = resnet50(num_classes=num_classes, include_top=True)
        model_weight_path = args.test_pretrain_weights
        assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
        pretrained_dict = torch.load(model_weight_path, map_location=args.device)['model']
        model.load_state_dict(pretrained_dict)
    elif args.method == 'swin':
        print('we use swin')
        model = swin_tiny_patch4_window7_224(num_classes=num_classes)
        '''
        pretraining
        '''
        model_weight_path = args.test_pretrain_weights
        assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
        pretrained_dict = torch.load(model_weight_path, map_location=args.device)['model']
        model.load_state_dict(pretrained_dict)
    elif args.method == 'mobilenet':
        print('we use mobilenet')
        model = MobileNetV2(num_classes=num_classes)
        model_weight_path = args.test_pretrain_weights
        assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
        pretrained_dict = torch.load(model_weight_path, map_location=args.device)['model']
        model.load_state_dict(pretrained_dict)
    else:
        print('model chose error！')
    model.to(args.device)
    num_params = count_parameters(model)
    logger.info("Training parameters %s", args)
    logger.info("Total Parameter: \t%2.1fM" % num_params)
    return args, model


def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params / 1000000


def valid(args, model):
    # Validation!
    # Prepare dataset
    train_loader, test_loader = get_loader(args)
    model.eval()
    all_preds, all_label = [], []
    epoch_iterator = tqdm(test_loader,
                          desc="Validating... (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=False,
                          disable=args.local_rank not in [-1, 0],
                          ncols=80,
                          position=0, leave=True)

    with torch.no_grad():
        i = 0
        for x, y in epoch_iterator:
            x = x.to(args.device)
            y = y.to(args.device)
            logits = model(x)
            i += 1
            preds = torch.argmax(logits, dim=-1)
            print('this image mineral type: ', preds)
            if len(all_preds) == 0:
                all_preds.append(preds.detach().cpu().numpy())
                all_label.append(y.detach().cpu().numpy())
            else:
                all_preds[0] = np.append(
                    all_preds[0], preds.detach().cpu().numpy(), axis=0
                )
                all_label[0] = np.append(
                    all_label[0], y.detach().cpu().numpy(), axis=0
                )

    all_preds, all_label = all_preds[0], all_label[0]
    classes = ['arsenopyrite', 'chalcopyrite', 'galena', 'gold', 'pyrite', 'sphalerite', 'stbnite']
    plt.figure()
    report = classification_report(all_label, all_preds, target_names=classes, output_dict=True)
    print(report)
    print('val accuracy:', report['accuracy'])


if __name__ == "__main__":

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--dataset", choices=['mineral'],
                        default="mineral",
                        help="Which dataset.")
    parser.add_argument('--data_root', type=str, default=r'C:\data\mineral\data')
    parser.add_argument('--method', type=str, default="swin_base",
                        choices=["transfg", "resnet50", "densenet", "mobilenet"])
    parser.add_argument("--test_pretrain_weights", type=str, default=r"test_pretrain_weights path",
                        help="test_pretrain_weights path")
    parser.add_argument("--this_val_save_path", type=str, default=r"this_val_save_path",
                        help="this_val_save_path")
    parser.add_argument("--img_size", default=[256, 256], type=int,
                        help="Resolution size")
    parser.add_argument("--num_workers", default=0, type=int,
                        help="num_workers")
    parser.add_argument("--train_batch_size", default=2, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=1, type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--eval_every", default=500, type=int,
                        help="Run prediction on validation set every so many steps."
                             "Will always run one evaluation at the end of training.")
    parser.add_argument("--learning_rate", default=3e-2, type=float,
                        help="The initial learning rate for SGD.")
    parser.add_argument("--weight_decay", default=0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--num_steps", default=1000000, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--decay_type", choices=["cosine", "linear"], default="cosine",
                        help="How to decay the learning rate.")
    parser.add_argument("--warmup_steps", default=500, type=int,
                        help="Step of training to perform learning rate warmup for.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O2',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument('--loss_scale', type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--smoothing_value', type=float, default=0.0,
                        help="Label smoothing value\n")
    parser.add_argument('--split', type=str, default='non-overlap',
                        help="Split method")
    parser.add_argument('--slide_step', type=int, default=12,
                        help="Slide step for overlap split")

    args = parser.parse_args()
    args.this_val_save_path = args.this_val_save_path + '\\'
    os.makedirs(args.this_val_save_path, exist_ok=True)
    args.data_root = '{}/{}'.format(args.data_root, args.dataset)
    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl',
                                             timeout=timedelta(minutes=60))
        args.n_gpu = 1
    args.device = device
    args.nprocs = torch.cuda.device_count()
    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s" %
                   (args.local_rank, args.device, args.n_gpu, bool(args.local_rank != -1)))
    # Model & Tokenizer Setup
    args, model = setup(args)
    valid(args, model)
