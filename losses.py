import torch
from torch.autograd import Variable
import numpy as np
from scipy.ndimage import distance_transform_edt as distance


###############################################################################
# SURFACE LOSS

def one_hot2dist(seg):

    #n_classes = seg.shape[1]
    #posmask = seg[:, 1:n_classes, :, :]  # BACKGROUND is skipped (!)

    C = seg.shape[1]

    res = np.zeros_like(seg)
    for c in range(1, C):  # background is excluded (C=0)
        posmask = seg[c].astype(np.bool)

        if posmask.any():
            negmask = ~posmask
            res[c] = distance(negmask) * negmask - (distance(posmask) - 1) * posmask

    return res


def surface_loss(y_true, y_pred):

    n_classes = y_pred.shape[1]

    y_pred_prob = torch.softmax(y_pred, axis=1)

    N = y_true.shape[0]

    loss = 0.0
    for i in range(N):

        y_true_onehot = make_one_hot(y_true, n_classes)
        y_true_onehot_numpy = y_true_onehot.cpu().numpy()
        dist_maps = one_hot2dist(y_true_onehot_numpy)  # it works on a numpy array
        dist_maps_tensor = torch.from_numpy(dist_maps).to(torch.float32)
        dist_maps_tensor = dist_maps_tensor.to(device='cuda:0')
        #dist_maps_tensor = Variable(dist_maps_tensor)

        loss += dist_maps_tensor * y_pred_prob[i]

    return loss.mean()


###############################################################################
# DICE LOSS

def make_one_hot(labels, C=2):

    one_hot = torch.FloatTensor(labels.size(0), C, labels.size(1), labels.size(2)).zero_()
    one_hot = one_hot.to('cuda:0')
    target = one_hot.scatter_(1, labels.unsqueeze(1), 1.0)

    target = Variable(target)

    return target


def dice_loss(input, target):

    """

    :param input: input is a torch variable of size Batch x nclasses x H x W representing log probabilities for each class
    :param target:  target is a 1-hot representation of the groundtruth, shoud have same size as the input
    :return: Dice loss

    """

    # input: torch.Tensor,
    # target: torch.Tensor -> torch.Tensor


    nclasses = input.shape[1]

    input = torch.softmax(input, axis=1)
    target_onehot = make_one_hot(target, nclasses)

    # exclude Background (assumed = 0)
    input_no_back = input[:, 1:, ...]
    target_onehot_no_back = target_onehot[:, 1:, ...]

    #input_no_back = input_no_back.view(-1)
    #target_onehot_no_back = target_onehot_no_back.view(-1)

    smooth = 1.0
    intersection = (input_no_back * target_onehot_no_back).sum()
    L = 1.0 - ((2.0 * intersection) + smooth) / (input_no_back.sum() + target_onehot_no_back.sum() + smooth)

    return L


def generalized_dice_loss(input, target, weights):

    """

    :param input: input is a torch variable of size Batch x nclasses x H x W representing log probabilities for each class
    :param target:  target is a 1-hot representation of the groundtruth, shoud have same size as the input
    :return: Dice loss

    """

    # input: torch.Tensor,
    # target: torch.Tensor -> torch.Tensor


    nclasses = input.shape[1]

    input = torch.softmax(input, axis=1)
    target_onehot = make_one_hot(target, nclasses)

    # exclude Background (assumed = 0)
    input_no_back = input[:, 1:, ...]
    target_onehot_no_back = target_onehot[:, 1:, ...]

    intersection = weights[0] * (input_no_back[:, 0, :, :] * target_onehot_no_back[:, 0, :, :]).sum()
    union = weights[0] * (input_no_back[:, 0, :, :].sum() + target_onehot_no_back[:, 0, :, :].sum())

    for j in range(1, nclasses-1):
        intersection += weights[j] * (input_no_back[:, j, :, :] * target_onehot_no_back[:, j, :, :]).sum()
        union += weights[j] * (input_no_back[:, j, :, :].sum() + target_onehot_no_back[:, j, :, :].sum())

    smooth = 1.0
    L = 1.0 - ((2.0 * intersection) + smooth) / (union + smooth)

    return L