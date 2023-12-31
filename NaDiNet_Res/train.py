import torch
import os
from torch.autograd import Variable
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
import glob
from data_loader import RescaleT
from data_loader import ToTensorLab
from data_loader import SalObjDataset
from model import NaDiNet_Res
import pytorch_iou


# ------- 1. define loss function --------

bce_loss = nn.BCELoss(size_average=True)
iou_loss = pytorch_iou.IOU(size_average=True)


def bce_iou_loss(pred, target):
    bce_out = bce_loss(pred, target)
    iou_out = iou_loss(pred, target)

    loss = bce_out + iou_out

    return loss


def muti_loss(d1, d2, d3, d4, d5, db, labels_v):
    loss1 = bce_iou_loss(d1, labels_v)
    loss2 = bce_iou_loss(d2, labels_v)
    loss3 = bce_iou_loss(d3, labels_v)
    loss4 = bce_iou_loss(d4, labels_v)
    loss5 = bce_iou_loss(d5, labels_v)
    lossb = bce_iou_loss(db, labels_v)

    loss = loss1 + loss2 + loss3 + loss4 + loss5 + lossb
    print("l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, lb: %3f" % (
        loss1.item(), loss2.item(), loss3.item(),
        loss4.item(), loss5.item(), lossb.item()))
    print("l_total: %3f" % (loss.item()))

    return loss1, loss


# ------- 2. set the directory of training dataset --------

data_dir = ''
tra_image_dir = ''
tra_label_dir = ''

image_ext = '.png'
label_ext = '.jpg'

model_dir = ''

epoch_num = 100000
batch_size_train = 4
batch_size_val = 1
train_num = 0
val_num = 0

tra_img_name_list = glob.glob(data_dir + tra_image_dir + '*' + image_ext)

tra_lbl_name_list = []
for img_path in tra_img_name_list:
    img_name = img_path.split("/")[-1]

    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]
    for i in range(1, len(bbb)):
        imidx = imidx + "." + bbb[i]
    name = imidx.split("\\")[-1]
    tra_lbl_name_list.append(data_dir + tra_label_dir + name + label_ext)

print("---")
print("train images: ", len(tra_img_name_list))
print("train labels: ", len(tra_lbl_name_list))
print("---")

train_num = len(tra_img_name_list)
salobj_dataset = SalObjDataset(
    img_name_list=tra_img_name_list,
    lbl_name_list=tra_lbl_name_list,
    transform=transforms.Compose([
        RescaleT(384),
        # RandomCrop(224),
        ToTensorLab(flag=0)]))
salobj_dataloader = DataLoader(salobj_dataset, batch_size=batch_size_train, shuffle=True, num_workers=4)

# ------- 3. define model --------
# define the net
net = NaDiNet_Res.NaDiNet()
if torch.cuda.is_available():
    net.cuda()

# ------- 4. define optimizer --------
print("---define optimizer...")
optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

# ------- 5. training process --------
print("---start training...")
ite_num = 0
running_loss = 0.0
running_tar_loss = 0.0
ite_num4val = 0

if __name__ == '__main__':
    for epoch in range(0, epoch_num):
        net.train()

        for i, data in enumerate(salobj_dataloader):
            ite_num = ite_num + 1
            ite_num4val = ite_num4val + 1

            inputs, labels = data['image'], data['label']

            inputs = inputs.type(torch.FloatTensor)
            labels = labels.type(torch.FloatTensor)

            # wrap them in Variable
            if torch.cuda.is_available():
                inputs_v, labels_v = Variable(inputs.cuda(), requires_grad=False), \
                                     Variable(labels.cuda(), requires_grad=False)
            else:
                inputs_v, labels_v = Variable(inputs, requires_grad=False), \
                                     Variable(labels, requires_grad=False)

            # y zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            d1, d2, d3, d4, d5, db = net(inputs_v)
            loss2, loss = muti_loss(d1, d2, d3, d4, d5, db, labels_v)

            loss.backward()
            optimizer.step()

            # # print statistics
            running_loss += loss.item()
            running_tar_loss += loss2.item()

            # del temporary outputs and loss
            del d1, d2, d3, d4, d5, db, loss2, loss

            print("[epoch: %3d/%3d, batch: %5d/%5d, ite: %d] train loss: %3f, tar: %3f\n" % (
                epoch + 1, epoch_num, (i + 1) * batch_size_train, train_num, ite_num, running_loss / ite_num4val,
                running_tar_loss / ite_num4val))

        if (epoch + 1) % 5 == 0:
            torch.save(net.state_dict(), model_dir + "epoch_%d.pth" % (epoch + 1))
            running_loss = 0.0
            running_tar_loss = 0.0
            # net.train()  # resume train
            ite_num4val = 0

    print('-------------Congratulations! Training Done!!!-------------')
