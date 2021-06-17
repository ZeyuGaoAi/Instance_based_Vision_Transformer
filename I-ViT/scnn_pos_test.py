import argparse
import os
from tqdm import tqdm
import time
import datetime
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import tensorboardX
import numpy as np
from load_dataset import load_prcc_dataset, load_prcc_dataset_scnn_pos
import vit
import my_transform as T
from sklearn.metrics import classification_report
from vit.utils import (
    adjust_learning_rate,
    cross_entropy_with_label_smoothing,
    accuracy,
    save_model,
    load_model,
    resume_model,
)


best_val_acc = 0.0
best_test_acc = 0.0


def parse_args():
    parser = argparse.ArgumentParser(description='Image classification')
    parser.add_argument(
        '--dataset', default='cifar', help='Dataset names.'
    )
    parser.add_argument('--local_rank', type=int, default = -1)
    parser.add_argument('--num_heads', type=int, default = 12)
    parser.add_argument('--hidden_dim', type=int, default = 128)
    parser.add_argument('--num_layers', type=int, default = 24)
    parser.add_argument('--img_size', type=int, default = 2000)
    parser.add_argument('--nuclues_size', type=int, default = 32)
    parser.add_argument('--crop_path', default = '../dataset/crops/', help='path to crop_nuclues')
    parser.add_argument(
        '--num_classes',
        type=int,
        default=2,
        help='The number of classes in the dataset.',
    )
    parser.add_argument(
        '--train_dirs',
        default='/home5/hby/PRCC/New_Data/trainset.txt',
        help='path to training data',
    )
    parser.add_argument(
        '--val_dirs',
        default='/home5/hby/PRCC/New_Data/validset.txt',
        help='path to validation data',
    )
    parser.add_argument(
        '--test_dirs',
        default='/home5/hby/PRCC/New_Data/testset.txt',
        help='path to test data',
    )
    parser.add_argument(
        '--num_nuclei',
        type=int,
        default=1000,
        help='The max number of nuclei to use.',
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=8,
        help='input batch size for training',
    )
    parser.add_argument(
        '--val_batch_size',
        type=int,
        default=4,
        help='input batch size for val',
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=4,
        help='input batch size for training',
    )
    parser.add_argument(
        "--color_jitter",
        action='store_true',
        default=False,
        help="To apply color augmentation or not.",
    )
    parser.add_argument('--model', default='VitsCNN_pos', help='Model names.')
    parser.add_argument(
        '--epochs', type=int, default=50, help='number of epochs to train'
    )
    parser.add_argument(
        '--test_epochs',
        type=int,
        default=1,
        help='number of internal epochs to test',
    )
    parser.add_argument(
        '--save_epochs',
        type=int,
        default=10,
        help='number of internal epochs to save',
    )
    parser.add_argument('--optim', default='sgd', help='Model names.')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument(
        '--warmup_epochs',
        type=float,
        default=5,
        help='number of warmup epochs',
    )
    parser.add_argument(
        '--momentum', type=float, default=0.9, help='SGD momentum'
    )
    parser.add_argument(
        '--weight_decay', type=float, default=0.00008, help='weight decay'
    )
    parser.add_argument(
        "--label_smoothing",
        action='store_true',
        default=False,
        help="To use label smoothing or not.",
    )
    parser.add_argument(
        '--nesterov',
        action='store_true',
        default=False,
        help='To use nesterov or not.',
    )
    parser.add_argument(
        '--work_dirs', default='./work_dirs4', help='path to work dirs'
    )
    parser.add_argument(
        '--name', default='scnn_newdata_test', help='the name of work_dir'
    )
    parser.add_argument(
        '--no_cuda',
        action='store_true',
        default=False,
        help='disables CUDA training',
    )
    parser.add_argument(
        '--lr_scheduler',
        type=str,
        default="linear",
        choices=["linear", "cosine"],
        help='how to schedule learning rate',
    )
    parser.add_argument(
        '--test', action='store_true', default=False, help='Test'
    )
    parser.add_argument(
        '--test_model', type=int, default=-1, help="Test model's epochs"
    )
    parser.add_argument(
        '--resume', action='store_true', default=False, help='Resume training'
    )
    parser.add_argument(
        '--gpu_id', default='5', type=str, help='id(s) for CUDA_VISIBLE_DEVICES'
    )

    args = parser.parse_args()


    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    return args


def get_transform(train):
    base_size = args.img_size + 64
    crop_size = args.img_size
    transforms = []
    if train:
        transforms.append(T.Resize(1000))
        transforms.append(T.RandomHorizontalFlip(0.5))
        transforms.append(T.RandomCrop(base_size, crop_size))
        transforms.append(T.RandomRotation((-5,5)))
    transforms.append(T.ToTensor())
    transforms.append(T.Normalize(mean=[0.681, 0.486, 0.630], std=[0.213, 0.256, 0.196]))
    return T.Compose(transforms)


def build_dataset_scnn(train_files, val_files, test_files, N, nuclues_size,crop_path):
    
    train_data = load_prcc_dataset_scnn_pos(train_files, transform = get_transform, train = True, N = N, nuclues_size = nuclues_size, crop_path = crop_path)
    train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True)
    
    val_data = load_prcc_dataset_scnn_pos(val_files, transform = get_transform, train = False, N = N, nuclues_size = nuclues_size, crop_path = crop_path)
    val_loader = torch.utils.data.DataLoader(
            val_data, batch_size=args.val_batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True)
    
    test_data = load_prcc_dataset_scnn_pos(test_files, transform = get_transform, train = False, N = N, nuclues_size = nuclues_size, crop_path = crop_path)
    test_loader = torch.utils.data.DataLoader(
            test_data, batch_size=args.val_batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True)
    
    return train_loader, val_loader, test_loader


        
def test(model, test_loader, criterion, epoch, args, log_writer=False):
    global best_test_acc
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    predicted_all = []
    outputs_all = []
    target_names = ['t1','t2']

    with torch.no_grad():
        for batch_idx, (data, cls,pos,target) in enumerate(tqdm(test_loader)):
            data, cls,pos, target = data.cuda(), cls.cuda(),pos.cuda(), target.cuda()
            outputs = model(data, cls,pos)
            loss = criterion(outputs, target)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            predicted_all += predicted.cpu().data.numpy().tolist()
            outputs_all += target.cpu().data.numpy().tolist()
        print('Test Loss: %.3f | Test Acc: %.3f%% (%d/%d)'
              % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
        print(classification_report(outputs_all, predicted_all, target_names=target_names,digits=4))

    
    


def test_net(args):
    print("Init...")
    _, _, test_loader= build_dataset_scnn(args.train_dirs, args.val_dirs, args.test_dirs, args.num_nuclei,args.nuclues_size,args.crop_path)
    
    model = vit.build_model(args)
    file_path = '/home/hby/test3/ViT-main/work_dirs4/scnn_pos_adjust_N1000_h128_L24_head12_20/27.pth'
    checkpoint = torch.load(file_path)
    
    model.load_state_dict({k.replace('module.',''):v for k,v in checkpoint['net'].items()})
    criterion = nn.CrossEntropyLoss().cuda()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        model.cuda()

    print("Start testing...")
    test(model, test_loader, criterion, args.test_model, args)




if __name__ == "__main__":
    args = parse_args()
    print(args)
    test_net(args)

