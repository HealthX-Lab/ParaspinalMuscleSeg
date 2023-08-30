import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import cv2 as cv


def find_majority(mask_array):

    """ Finds the per-pixel majority vote of the masks and returns the result as a torch tensor.

    :param mask_array: input array containing discrete masks of multiple raters; dimensions should be (img_num, rater_num, img_size, img_size)
    :type mask_array: array_like
    :return: majority vote of the raters (dims: (img_num, img_size, img_size))
    :rtype: torch tensor 
    """

    img_size = mask_array.shape[2]
    rater_num = mask_array.shape[1]
    img_num = mask_array.shape[0]
    mask_majority = torch.zeros((img_num, img_size, img_size), dtype=torch.long)
    for s in range(img_num):
        print('processing {}'.format(s+1))
        for i in range(img_size):
            for j in range(img_size):
                counts = {}
                counts[mask_array[s,0,i,j]] = 1
                for r in range(1,rater_num):
                    if mask_array[s,r,i,j] not in list(counts.keys()):
                        counts[mask_array[s,r,i,j]] = 1
                    else:
                        counts[mask_array[s,r,i,j]] += 1
                val_list = list(counts.values())
                mask_majority[s,i,j] = list(counts.keys())[np.argmax(val_list)]

    mask_majority = mask_majority.numpy()
    print('majority vote done!')
    return mask_majority


def find_variance_map(mask_array, num_classes):

    """ Finds the variance map for each mask and returns the result as a torch tensor.

    :param mask_array: input array containing discrete masks of multiple raters; dimensions should be (img_num, rater_num, img_size, img_size)
    :type mask_array: array_like
    :param num_classes: number of classes in each image (excluding background)
    :type num_classes: int
    :return: variance maps (dims: (img_num, num_classes, img_size, img_size))
    :rtype: torch tensor
    """

    img_size = mask_array.shape[2]
    rater_num = mask_array.shape[1]
    img_num = mask_array.shape[0]
    #generate the GT variance maps for task2(variance map prediction)
    mask_1hot = np.zeros((img_num, rater_num, num_classes, img_size, img_size), dtype = np.uint8)
    for s in range(img_num):
        for r in range(rater_num):
            mask_temp = mask_array[s,r]
            for i in range(1,num_classes+1):
                arr = np.full((img_size, img_size), i)
                mask_temp_i = (mask_temp == arr).astype(int)
                mask_1hot[s,r,i-1]= mask_temp_i         

    log_odds = torch.zeros((img_num, rater_num, num_classes, img_size, img_size), dtype = torch.float64)
    for s in range(img_num):
        for r in range(rater_num):
            for c in range(num_classes):
                distp = cv.distanceTransform(mask_1hot[s,r,c], cv.DIST_L2, num_classes+1)
                mask_complement = 1 - mask_1hot[s,r,c]
                distn = cv.distanceTransform(mask_complement, cv.DIST_L2, num_classes+1)
                distn = -1*distn
                dist = distp + distn
                log_odd = torch.special.expit(torch.from_numpy(dist))
                log_odds[s,r,c] = log_odd

    var_array = torch.zeros((img_num, num_classes, img_size, img_size), dtype=torch.float64)
    for s in range(img_num):
        for i in range(num_classes):
            mean = (log_odds[s,0,i] + log_odds[s,1,i] + log_odds[s,2,i])/rater_num
            variance = ((log_odds[s,0,i] - mean)**2 + (log_odds[s,1,i] - mean)**2 + (log_odds[s,2,i] - mean)**2)/rater_num
            var_array[s,i] = variance
 
    var_array = var_array.numpy()
    print('variance map done!')
    return var_array


def data_loader(train_path, val_path):

    """ Loads the train and validation data from corresponding pandas dataframes into lists of per-image dictionaries to be used with monai dataloaders. In each dataframe, the data should be stored subject-wise.

    :param train_path: path for the train dataframe
    :type train_path: string
    :param val_path: path for the validation dataframe
    :type val_path: string
    :return: lists of per-image dictionaries (train_dict, val_dict)
    """

    train = pd.read_pickle(train_path)
    val = pd.read_pickle(val_path)
    
    mri_train, var_train, maj_train = [], [], []
    for i in range(len(train)):
        if(type(train.loc[i, 'l3l4_MRI']) != int):
            mri_train.append(train.loc[i, 'l3l4_MRI'])
            var_train.append(train.loc[i, 'l3l4_var'])
            maj_train.append(train.loc[i, 'l3l4_maj'])
            
        if(type(train.loc[i, 'l4l5_MRI']) != int):
            mri_train.append(train.loc[i, 'l4l5_MRI'])
            var_train.append(train.loc[i, 'l4l5_var'])
            maj_train.append(train.loc[i, 'l4l5_maj'])
          
        if(type(train.loc[i, 'l5s1_MRI']) != int):
            mri_train.append(train.loc[i, 'l5s1_MRI'])
            var_train.append(train.loc[i, 'l5s1_var'])
            maj_train.append(train.loc[i, 'l5s1_maj'])
           
        if(type(train.loc[i, 'l4upper_MRI']) != int):
            mri_train.append(train.loc[i, 'l4upper_MRI'])
            var_train.append(train.loc[i, 'l4upper_var'])
            maj_train.append(train.loc[i, 'l4upper_maj'])
           
        if(type(train.loc[i, 'l5upper_MRI']) != int):
            mri_train.append(train.loc[i, 'l5upper_MRI'])
            var_train.append(train.loc[i, 'l5upper_var'])
            maj_train.append(train.loc[i, 'l5upper_maj'])
           
        if(type(train.loc[i, 's1_MRI']) != int):
            mri_train.append(train.loc[i, 's1_MRI'])
            var_train.append(train.loc[i, 's1_var'])
            maj_train.append(train.loc[i, 's1_maj'])

    train_dict = [{'mri': mri_train[i], 'maj': maj_train[i], 'var':var_train[i]} for i in range(len(mri_train))]

    mri_val, var_val, maj_val = [], [], []
    for i in range(len(val)):
        if(type(val.loc[i, 'l3l4_MRI']) != int):
            mri_val.append(val.loc[i, 'l3l4_MRI'])
            var_val.append(val.loc[i, 'l3l4_var'])
            maj_val.append(val.loc[i, 'l3l4_maj'])
           
        if(type(val.loc[i, 'l4l5_MRI']) != int):
            mri_val.append(val.loc[i, 'l4l5_MRI'])
            var_val.append(val.loc[i, 'l4l5_var'])
            maj_val.append(val.loc[i, 'l4l5_maj'])
            
        if(type(val.loc[i, 'l5s1_MRI']) != int):
            mri_val.append(val.loc[i, 'l5s1_MRI'])
            var_val.append(val.loc[i, 'l5s1_var'])
            maj_val.append(val.loc[i, 'l5s1_maj'])
            
        if(type(val.loc[i, 'l4upper_MRI']) != int):
            mri_val.append(val.loc[i, 'l4upper_MRI'])
            var_val.append(val.loc[i, 'l4upper_var'])
            maj_val.append(val.loc[i, 'l4upper_maj'])
            
        if(type(val.loc[i, 'l5upper_MRI']) != int):
            mri_val.append(val.loc[i, 'l5upper_MRI'])
            var_val.append(val.loc[i, 'l5upper_var'])
            maj_val.append(val.loc[i, 'l5upper_maj'])
            
        if(type(val.loc[i, 's1_MRI']) != int):
            mri_val.append(val.loc[i, 's1_MRI'])
            var_val.append(val.loc[i, 's1_var'])
            maj_val.append(val.loc[i, 's1_maj'])

    val_dict = [{'mri': mri_val[i], 'maj': maj_val[i], 'var':var_val[i]} for i in range(len(mri_val))]

    return train_dict, val_dict


def test_loader(test_path):
    """ Loads the test data from a pandas dataframe into a list of per-image dictionaries to be used with monai dataloaders. In the dataframe, the data should be stored subject-wise.

    :param test_path: path for the test dataframe
    :type test_path: string
    :return: a list of per-image dictionaries (test_dict)
    """
    test = pd.read_pickle(test_path)
    mri_test, var_test, maj_test, name_test, level_test = [], [], [], [], []
    for i in range(len(test)):
        if(type(test.loc[i, 'l3l4_MRI']) != int):
            mri_test.append(test.loc[i, 'l3l4_MRI'])
            var_test.append(test.loc[i, 'l3l4_var'])
            maj_test.append(test.loc[i, 'l3l4_maj'])
            name_test.append(test.loc[i, 'name'])
            level_test.append(1)
        if(type(test.loc[i, 'l4l5_MRI']) != int):
            mri_test.append(test.loc[i, 'l4l5_MRI'])
            var_test.append(test.loc[i, 'l4l5_var'])
            maj_test.append(test.loc[i, 'l4l5_maj'])
            name_test.append(test.loc[i, 'name'])
            level_test.append(2)
        if(type(test.loc[i, 'l5s1_MRI']) != int):
            mri_test.append(test.loc[i, 'l5s1_MRI'])
            var_test.append(test.loc[i, 'l5s1_var'])
            maj_test.append(test.loc[i, 'l5s1_maj'])
            name_test.append(test.loc[i, 'name'])
            level_test.append(3)
        if(type(test.loc[i, 'l4upper_MRI']) != int):
            mri_test.append(test.loc[i, 'l4upper_MRI'])
            var_test.append(test.loc[i, 'l4upper_var'])
            maj_test.append(test.loc[i, 'l4upper_maj'])
            name_test.append(test.loc[i, 'name'])
            level_test.append(4)
        if(type(test.loc[i, 'l5upper_MRI']) != int):
            mri_test.append(test.loc[i, 'l5upper_MRI'])
            var_test.append(test.loc[i, 'l5upper_var'])
            maj_test.append(test.loc[i, 'l5upper_maj'])
            name_test.append(test.loc[i, 'name'])
            level_test.append(5)
        if(type(test.loc[i, 's1_MRI']) != int):
            mri_test.append(test.loc[i, 's1_MRI'])
            var_test.append(test.loc[i, 's1_var'])
            maj_test.append(test.loc[i, 's1_maj'])
            name_test.append(test.loc[i, 'name'])
            level_test.append(6)
        
        
    test_dict = [{'mri': mri_test[i], 'maj': maj_test[i], 'var':var_test[i], 'level':level_test[i]} for i in range(len(mri_test))]

    return test_dict


class DiceLoss(nn.Module):
    """ Class for calculating the dice loss between a GT mask and the model outputs. The outputs can be drawn from the model either before or after applying the softmax."""

    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes