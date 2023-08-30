import numpy as np
import copy
import logging
import os
import sys
from tqdm import tqdm

import torch
import torch.optim as optim
from torch.nn.modules.loss import CrossEntropyLoss
from torch.nn import L1Loss

from monai.data import Dataset, DataLoader
from monai.transforms import (
    Compose,
    AddChanneld,
    ScaleIntensityd,
    RandGaussianNoised,
    RandShiftIntensityd,
    RandAffined,
    RandGaussianSmoothd,
    RandRotated,
    ToTensord,
    CropForeground
)

from utils import DiceLoss, data_loader


################################### GPU configuration #############################################

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
device = torch.device("cuda")


def trainer(args, model, snapshot_path):

    ############################# primary configurations ##########################################
    
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size

    ############################# data load and augmentation ######################################

    train_path = args.train_data_path
    val_path = args.val_data_path

    train_dict, val_dict = data_loader(train_path, val_path)

    train_transforms = Compose(

    [
        AddChanneld(keys=['mri', 'maj']),
        ScaleIntensityd(keys='mri', channel_wise=True),
        
        RandGaussianNoised(keys='mri'),
        RandShiftIntensityd(keys='mri', offsets=0.2),
        RandAffined(keys=['mri', 'maj', 'var'], translate_range=[0,(-50,0),0]),
        RandGaussianSmoothd(keys='mri'),
        RandRotated(keys=['mri', 'maj', 'var'], range_x=0.09),
        
        ToTensord(keys=['mri', 'maj', 'var']),
    ],
        log_stats=True
    )
    
    val_transforms = Compose(

    [
        
        AddChanneld(keys=['mri']),
        ScaleIntensityd(keys='mri', channel_wise=True),
        ToTensord(keys=['mri', 'maj', 'var'])
    ],
        log_stats=True
    )

    train_set = Dataset(data=train_dict, transform=train_transforms)
    val_set = Dataset(data=val_dict, transform=val_transforms)
    
    
    trainloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True)
    valloader = DataLoader(val_set, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True)

    
    ##################################### model training ###################################
        
    model.train()
    model = model.double()
    model_best = copy.deepcopy(model)
    
    # loss functions used in task 1 (segmentation)
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)

    # loss function used in task 2 (variance map estimation)
    def threshold_fn(x):
         return x != 0
    cropper = CropForeground(select_fn=threshold_fn, margin=10, return_coords=True)
    l1 = L1Loss()

    # using stochastic gradient descent as the optimizer.
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = max_epoch * len(trainloader)
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    
    # for task1 (segmentation), a combination of cross entropy and dice loss is used
    train_loss_ce = []
    train_loss_dice = []
    train_loss_task1 = []
    # task 2 (variance map estimation) uses l1 loss
    train_loss_task2 = []
    
    # same losses are used in validation
    val_loss_ce = []
    val_loss_dice = []
    val_loss_task1 = []
    val_loss_task2 = []
    
    tolerance=0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        
        # in the first epoch, evaluate the model with the validation dataset as an initial assessment
        if epoch_num == 0:
            model.eval()
            loss_ce_, loss_dice_, loss_task1_, loss_task2_ = 0, 0, 0, 0
            i_batch=0
            for data_dict in valloader:

                    data = data_dict["mri"].double().to(device)
                    mask_majority = data_dict["maj"].long().to(device)
                    var_gt = data_dict["var"].double().to(device)
                    
                    mask_pred, var_pred = model(data)


                    # loss for task1
                    # mask_pred size: (1, num_classes, img_size, img_size), mask_majority size: (1, 1, img_size, img_size)
                    loss_ce = ce_loss(mask_pred, mask_majority)
                    loss_ce_ += loss_ce.item()

                    loss_dice = dice_loss(mask_pred, mask_majority, softmax=True)
                    loss_dice_ += loss_dice.item()

                    loss = 0.5 * loss_ce + 0.5 * loss_dice
                    loss_task1_ += loss.item()

                    # loss for task 2
                    # var_pred size: (1, num_classes-1, img_size, img_size), var_gt size: (1, num_classes-1, img_size, img_size)
                    # cropped_mask = torch.zeros((1, num_classes-1, args.img_size, args.img_size), dtype=torch.float64).cuda()
                    # cropped_gt = torch.zeros((1, num_classes-1, args.img_size, args.img_size), dtype=torch.float64).cuda()
                    loss_task2 = 0
                    for i in range(num_classes-1):
                        bounds1 = cropper.compute_bounding_box(var_pred[0,i].unsqueeze(0))
                        bounds2 = cropper.compute_bounding_box(var_gt[0,i].unsqueeze(0))
                        start_bound = bounds1[0] if np.linalg.norm(bounds1[0]) < np.linalg.norm(bounds2[0]) else bounds2[0]
                        end_bound = bounds1[1] if np.linalg.norm(bounds1[1]) > np.linalg.norm(bounds2[1]) else bounds2[1]
                        cropped_mask = cropper.crop_pad(var_pred[0,i].unsqueeze(0), start_bound, end_bound)
                        cropped_gt= cropper.crop_pad(var_gt[0,i].unsqueeze(0), start_bound, end_bound)
                        loss_task2 += l1(cropped_mask, cropped_gt)
                    # loss_task2 = l1(var_pred, var_gt)
                    loss_task2_ += loss_task2.item()

                    i_batch = i_batch+1

            val_loss_ce.append(loss_ce_ / (i_batch+1))
            val_loss_dice.append(loss_dice_ / (i_batch+1))
            val_loss_task1.append(loss_task1_ / (i_batch+1))
            val_loss_task2.append(loss_task2_/(i_batch+1))
            total_loss = 0.6*(loss_task2_/(i_batch+1)) + 0.4*(loss_task1_/(i_batch+1))
            
            loss_best = total_loss
            loss_old = loss_best
       

        model.train()
        i_batch, loss_ce_, loss_dice_, loss_task1_, loss_task2_ = 0, 0, 0, 0, 0
        for data_dict in trainloader:
            
            data = data_dict["mri"]
            mask_majority = data_dict["maj"]
            var_gt = data_dict["var"]
            
            #data size: (b, 1, img_size, img_size)
            data = data.double().to(device)
            # an extra channel dimension was added to the train majority masks in order for the augmentations to be applied
            # mask_majority size: (b, 1, img_size, img_size)
            mask_majority = mask_majority.long().to(device)
            # var_gt size: (b, num_classes - 1, img_size, img_size)
            var_gt = var_gt.double().to(device)
            
            mask_pred, var_pred = model(data)  # mask_pred size: (b, 5, img_size, img_size), var_pred size: (b, num_classes - 1, img_size, img_size)

            # loss for task 1
            loss_ce = ce_loss(mask_pred, torch.squeeze(mask_majority, 1))
            loss_ce_ += loss_ce.item()
            
            loss_dice = dice_loss(mask_pred, torch.squeeze(mask_majority, 1), softmax=True)
            loss_dice_ += loss_dice.item()
            
            loss = 0.5 * loss_ce + 0.5 * loss_dice
            loss_task1_ += loss.item()
            
            # loss for task 2
            # loss_task2 = l1(var_pred, var_gt)
            loss_task2 = 0
            for i in range(num_classes-1):
                bounds1 = cropper.compute_bounding_box(var_pred[0,i].unsqueeze(0))
                bounds2 = cropper.compute_bounding_box(var_gt[0,i].unsqueeze(0))
                start_bound = bounds1[0] if np.linalg.norm(bounds1[0]) < np.linalg.norm(bounds2[0]) else bounds2[0]
                end_bound = bounds1[1] if np.linalg.norm(bounds1[1]) > np.linalg.norm(bounds2[1]) else bounds2[1]
                cropped_mask = cropper.crop_pad(var_pred[0,i].unsqueeze(0), start_bound, end_bound)
                cropped_gt= cropper.crop_pad(var_gt[0,i].unsqueeze(0), start_bound, end_bound)
                loss_task2 += l1(cropped_mask, cropped_gt)
            loss_task2_ += loss_task2.item()
            
            final_loss = 0.6*loss_task2 + 0.4*loss
            optimizer.zero_grad()
            final_loss.backward()
            optimizer.step()

            # gradually decrease the learning rate
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            i_batch = i_batch+1
            

            logging.info('iteration: %d, epoch: %d, batch: %d, total_loss: %f, loss_task1 : %f, loss_task2:%f' % (iter_num, epoch_num+1, i_batch, final_loss.item(), loss.item(), loss_task2.item()))

           
        train_loss_ce.append(loss_ce_ / (i_batch+1))
        train_loss_dice.append(loss_dice_ / (i_batch+1))
        train_loss_task1.append(loss_task1_ / (i_batch+1))
        train_loss_task2.append(loss_task2_/(i_batch+1))
        total_loss = 0.6*(loss_task2_/(i_batch+1)) + 0.4*(loss_task1_/(i_batch+1))
        
       
        i_batch, loss_ce_, loss_dice_, loss_task1_, loss_task2_ = 0, 0, 0, 0, 0
        model.eval()
        for data_dict in valloader:

                data = data_dict["mri"].double().to(device)
                mask_majority = data_dict["maj"].long().to(device)
                var_gt = data_dict["var"].double().to(device)

                mask_pred, var_pred = model(data)

                #loss for task1
                loss_ce = ce_loss(mask_pred, mask_majority)
                loss_ce_ += loss_ce.item()

                loss_dice = dice_loss(mask_pred, mask_majority, softmax=True)
                loss_dice_ += loss_dice.item()

                loss = 0.5 * loss_ce + 0.5 * loss_dice
                loss_task1_ += loss.item()

                #loss for task2
                # loss_task2 = l1(var_pred, var_gt)
                loss_task2 = 0
                for i in range(num_classes-1):
                    bounds1 = cropper.compute_bounding_box(var_pred[0,i].unsqueeze(0))
                    bounds2 = cropper.compute_bounding_box(var_gt[0,i].unsqueeze(0))
                    start_bound = bounds1[0] if np.linalg.norm(bounds1[0]) < np.linalg.norm(bounds2[0]) else bounds2[0]
                    end_bound = bounds1[1] if np.linalg.norm(bounds1[1]) > np.linalg.norm(bounds2[1]) else bounds2[1]
                    cropped_mask = cropper.crop_pad(var_pred[0,i].unsqueeze(0), start_bound, end_bound)
                    cropped_gt= cropper.crop_pad(var_gt[0,i].unsqueeze(0), start_bound, end_bound)
                    loss_task2 += l1(cropped_mask, cropped_gt)
                loss_task2_ += loss_task2.item()
                
                i_batch = i_batch+1

        val_loss_ce.append(loss_ce_ / (i_batch+1))
        val_loss_dice.append(loss_dice_ / (i_batch+1))
        val_loss_task1.append(loss_task1_ / (i_batch+1))
        val_loss_task2.append(loss_task2_/(i_batch+1))
        total_loss = 0.6*(loss_task2_/(i_batch+1)) + 0.4*(loss_task1_/(i_batch+1))
        
        loss_new = total_loss

        # compare the validation loss to the best loss to detect improvements and keep the best model
        if loss_best > loss_new:
            del model_best
            model_best = copy.deepcopy(model)
            loss_best = loss_new
            epoch_best = epoch_num

        # detect improvements ...
        if loss_old < loss_new:
            tolerance += 1
        elif loss_old > loss_new:
            tolerance = 0
        loss_old = loss_new
        
        #if done with 250 epochs or if 50 epochs of not improving...
        if (epoch_num + 1) == 250 or tolerance==50:
            save_model_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num+1) + '.pth')
            torch.save(model.state_dict(), save_model_path)
            logging.info("save model to {}".format(save_model_path))
            save_model_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_best+1) + '_best.pth')
            torch.save(model_best.state_dict(), save_model_path)
            logging.info("save best model to {}".format(save_model_path))
            
            # save all the metrics in a file for later evaluations. ".npy" files are easy to work with!
            metrics_path = os.path.join(snapshot_path,'metrics_{}.npy'.format(epoch_num+1))
            with open(metrics_path, 'wb') as f:
                np.save(f, train_loss_ce)
                np.save(f, train_loss_dice)
                np.save(f, train_loss_task1)
                np.save(f, train_loss_task2)
                np.save(f, val_loss_ce)
                np.save(f, val_loss_dice)
                np.save(f, val_loss_task1)
                np.save(f, val_loss_task2)

        if epoch_num >= max_epoch - 1:
            iterator.close()
            break
            

    print('done!')
    
    return "Training Finished"