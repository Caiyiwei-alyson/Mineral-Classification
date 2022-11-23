import logging
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from utils.read_data import read_split_data_mineral
from utils.autoaugment import AutoAugImageNetPolicy
from utils.my_dataset import MyDataSet
logger = logging.getLogger(__name__)


def get_loader(args):
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    if args.dataset == 'mineral':
        train_images_path, train_images_label, val_images_path, val_images_label = read_split_data_mineral(
            args.root_path)
        trainset = MyDataSet(images_path=train_images_path,
                             images_class=train_images_label,
                             transform=transforms.Compose([
                                 transforms.Resize((256, 256), Image.BILINEAR),
                                 transforms.RandomCrop((256, 256)),
                                 transforms.RandomHorizontalFlip(),
                                 AutoAugImageNetPolicy(),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                             )
        testset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=transforms.Compose([
                                transforms.Resize((256, 256), Image.BILINEAR),
                                transforms.RandomCrop((256, 256)),
                                transforms.RandomHorizontalFlip(),
                                AutoAugImageNetPolicy(),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                            )
    else:
        print('data set error')

    if args.local_rank == 0:
        torch.distributed.barrier()
    train_sampler = RandomSampler(trainset)
    test_sampler = SequentialSampler(testset)
    train_loader = DataLoader(trainset,
                              sampler=train_sampler,
                              batch_size=args.train_batch_size,
                              num_workers=args.num_workers,
                              drop_last=True,
                              pin_memory=True)
    test_loader = DataLoader(testset,
                             sampler=test_sampler,
                             batch_size=args.eval_batch_size,
                             num_workers=args.num_workers,
                             pin_memory=True) if testset is not None else None
    return train_loader, test_loader