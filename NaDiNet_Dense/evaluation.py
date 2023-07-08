import os
import torch
import numpy as np
from PIL import Image


def batch_pix_accuracy(prediction, target):
    pixel_labeled = (target > 0).sum()
    pixel_correct = ((prediction == target) * (target > 0)).sum()
    pixel_acc = np.divide(pixel_correct.cpu().numpy(), pixel_labeled.cpu().numpy())
    assert pixel_correct <= pixel_labeled, "Correct area should be smaller than Labeled"
    return pixel_correct.cpu().numpy(), pixel_labeled.cpu().numpy(), pixel_acc


def batch_intersection_union(prediction, target, num_class):
    # prediction = prediction * (target > 0).long()
    intersection = prediction * (prediction == target).long()
    area_inter = torch.histc(intersection.float(), bins=num_class - 1, max=num_class - 0.9, min=0.1)
    # print(area_inter[0])
    area_pred = torch.histc(prediction.float(), bins=num_class - 1, max=num_class - 0.9, min=0.1)
    area_lab = torch.histc(target.float(), bins=num_class - 1, max=num_class - 0.9, min=0.1)
    area_union = area_pred + area_lab - area_inter
    # print(area_union.float())
    IoU = area_inter.float() / (np.spacing(1) + area_union.float())
    mIoU = IoU.sum() / torch.nonzero(area_lab).size(0)

    assert (area_inter <= area_union).all(), "Intersection area should be smaller than Union area"

    return mIoU.cpu().numpy()


if __name__ == '__main__':
    label_dir_165 = ''
    label_dir_965 = ''

    file_dir_165 = ''
    file_dir_965 = ''

    curImgDir = file_dir_165
    num_test = 0
    avg_pa_165 = 0
    avg_mIou_165 = 0

    for num_test, img in enumerate(os.listdir(curImgDir)):
        image_path = os.path.join(curImgDir, img)
        label_path = os.path.join(label_dir_165, img)

        img = np.asarray(Image.open(image_path), dtype=np.float32)
        label = np.asarray(Image.open(label_path), dtype=np.int32)  # - 1 # from -1 to 149
        label = label / 255
        image = img / 255

        image = torch.from_numpy(image).cuda().long()
        true_masks = torch.from_numpy(label).cuda().long()

        pa = batch_pix_accuracy(image, true_masks)
        iou = batch_intersection_union(image, true_masks, 2)
        avg_pa_165 = avg_pa_165 + pa[2]
        avg_mIou_165 = avg_mIou_165 + iou

    avg_pa_165 = avg_pa_165 / (num_test + 1)
    avg_mIou_165 = avg_mIou_165 / (num_test + 1)

    curImgDir = file_dir_965
    num_test = 0
    avg_pa_965 = 0
    avg_mIou_965 = 0

    for num_test, img in enumerate(os.listdir(curImgDir)):
        image_path = os.path.join(curImgDir, img)
        label_path = os.path.join(label_dir_965, img)

        img = np.asarray(Image.open(image_path), dtype=np.float32)
        label = np.asarray(Image.open(label_path), dtype=np.int32)  # - 1 # from -1 to 149
        label = label / 255
        image = img / 255

        image = torch.from_numpy(image).cuda().long()
        true_masks = torch.from_numpy(label).cuda().long()

        pa = batch_pix_accuracy(image, true_masks)
        iou = batch_intersection_union(image, true_masks, 2)
        avg_pa_965 = avg_pa_965 + pa[2]
        avg_mIou_965 = avg_mIou_965 + iou

    avg_pa_965 = avg_pa_965 / (num_test + 1)
    avg_mIou_965 = avg_mIou_965 / (num_test + 1)

    print('Dataset 965: PA:{:.3f}, mIoU:{:.3f}; Dataset 165: PA:{:.3f}, mIoU:{:.3f}'.format(avg_pa_965, avg_mIou_965,
                                                                                            avg_pa_165, avg_mIou_165))
