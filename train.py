import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from trainer import trainer


parser = argparse.ArgumentParser()

parser.add_argument('--num_classes', type=int, default=5, help='number of output channels of network')

parser.add_argument('--max_epochs', type=int, default=250, help='maximum number of epochs to train')

parser.add_argument('--batch_size', type=int, default=8, help='batch size')

parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')

parser.add_argument('--base_lr', type=float,  default=0.0033, help='segmentation network learning rate')

parser.add_argument('--img_size', type=int, default=256, help='input patch size of network input')

parser.add_argument('--seed', type=int, default=1234, help='random seed')

parser.add_argument('--n_skip', type=int, default=3, help='using number of skip-connect, default is 3')

parser.add_argument('--vit_name', type=str, default='R50-ViT-B_16', help='select one vit model')

parser.add_argument('--vit_patches_size', type=int, default=16, help='vit_patches_size, default is 16')

parser.add_argument('--train_data_path', type=str, default='./data/train.pkl', help='path for the training data')

parser.add_argument('--val_data_path', type=str, default='./data/val.pkl', help='path for the validation data')

args = parser.parse_args()


if __name__ == "__main__":

    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    # set the path for saving the trained model
    args.exp = 'model_'+str(args.max_epochs)+"epochs_"+str(args.base_lr)+"base_lr"+"_cropped_l1"
    snapshot_path = "./trained_models/{}".format(args.exp)

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
        
    #Configure ViT parameters...
    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    if args.vit_name.find('R50') != -1:
        config_vit.patches.grid = (int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))
        
    #Create a ViT model and load it's initial weights
    net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()
    net.load_from(weights=np.load(config_vit.pretrained_path))
    net = net.double()

    # train the model using the trainer function
    trainer = {'LBP': trainer}
    trainer['LBP'](args, net, snapshot_path)