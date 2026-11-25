import numpy as np

def compute_mIoU(preds, labels, num_classes=4):
    iou_list = []
    preds = preds.view(-1)
    labels = labels.view(-1)
    for cls in range(num_classes):
        pred_inds = (preds == cls)
        target_inds = (labels == cls)
        intersection = (pred_inds & target_inds).sum().item()
        union = (pred_inds | target_inds).sum().item()
        if union == 0:
            iou = float('nan')
        else:
            iou = intersection / union
        iou_list.append(iou)
    return iou_list, np.nanmean(iou_list)

def compute_dice(preds, labels, num_classes=4):
    dice_scores = []
    preds = preds.view(-1)
    labels = labels.view(-1)
    for cls in range(num_classes):
        pred_inds = (preds == cls)
        target_inds = (labels == cls)
        intersection = (pred_inds & target_inds).sum().item()
        pred_sum = pred_inds.sum().item()
        target_sum = target_inds.sum().item()
        if pred_sum + target_sum == 0:
            dice = float('nan')
        else:
            dice = 2 * intersection / (pred_sum + target_sum)
        dice_scores.append(dice)
    return dice_scores, np.nanmean(dice_scores)