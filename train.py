import os
import torch
import numpy as np
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.autograd import Variable
from utils.utils import AvgMeter
from utils.dataloader import BaseSegmentationExperiment

from model import IPDFRNet

def structure_loss(pred, mask):
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter+1)
    return (wbce + wiou).mean()

def train(train_loader, model, optimizer, epoch, best_dice, step):
    model.train()
    accum = 0
    total_step = len(train_loader)

    for i, pack in enumerate(train_loader, start=1):
        # ---- data prepare ----
        images, gts = pack
        images = Variable(images).cuda()
        gts = Variable(gts).cuda()

        # ---- forward --
        outputs,samouts= model(images)

        # ---- loss function ----
        loss1 = structure_loss(outputs, gts)
        loss2 = structure_loss(samouts, gts)
        loss=loss1+loss2

        # ---- backward ----
        loss.backward() 
        optimizer.step()
        optimizer.zero_grad()

        # ---- train visualization ----
        if i % 20 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '.format(datetime.now(), epoch, opt.epoch, i, total_step))

    save_path = '{}/{}/'.format(opt.train_save, opt.train_data_type)
    os.makedirs(save_path, exist_ok=True)
    if (epoch+1) % 1 == 0:
        step+=1
        meandice = test(model, opt)
        if meandice > best_dice:
            print('new best dice: ', meandice)
            best_dice = meandice
            torch.save(model.state_dict(), save_path + 'best.pth')
            print('[Saving Snapshot:]', save_path + 'best.pth')
        torch.save(model.state_dict(), save_path + 'last.pth')
    return best_dice

def test(model, opt):

    model.eval()

    test_loader = BaseSegmentationExperiment(opt).test_loader

    DSC = 0.0
    JACARD = 0.0
    preds = []
    gts = []
    num1 = len(test_loader)
    

    for i, pack in enumerate(test_loader, start=1):
        image, gt = pack
        image = Variable(image).cuda()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
      
        with torch.no_grad():

            outputs,samouts= model(images)

        res = outputs

        res = res.sigmoid().data.cpu().numpy().squeeze()

        input = np.where(res >= 0.5, 1, 0)
        target = np.where(np.array(gt) >= 0.5, 1, 0)
        
        preds.append(input)
        gts.append(gt)
        
        smooth = 1
        input_flat = np.reshape(input, (-1))
        target_flat = np.reshape(target, (-1))
        intersection = (input_flat * target_flat)   
        union = input_flat + target_flat - intersection
        
        jacard = ((np.sum(intersection)+smooth)/(np.sum(union)+smooth))
        jacard = '{:.4f}'.format(jacard)
        jacard = float(jacard)
        JACARD += jacard
        
        dice = (2 * intersection.sum() + smooth) / (input.sum() + target.sum() + smooth)
        dice = '{:.4f}'.format(dice)
        dice = float(dice)
        DSC += dice
        
    print('*****************************************************')
    print('Dice Score: ' + str(DSC/num1))
    print('Jacard Score: ' + str(JACARD/num1))
    print('*****************************************************')

    return DSC/num1 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=str, default='./dataset/BioMedicalDataset')
    parser.add_argument('--train_data_type', type=str, required=False, choices=['ISIC2018','BUSI','PH2','STU'])
    parser.add_argument('--test_data_type', type=str, required=False, choices=['ISIC2018','BUSI','PH2','STU'])
    parser.add_argument('--img_size', type=int, default=352, help='input patch size of network input')#352
    parser.add_argument('--num_classes', type=int, default=1, help='epoch number')#1 samlst0
    parser.add_argument('--epoch', type=int, default=100, help='epoch number')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--batchsize', type=int, default=4, help='training batch size')
    parser.add_argument('--grad_norm', type=float, default=2.0, help='gradient clipping norm')
    parser.add_argument('--train_save', type=str, default='./IPDFRNet')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 of adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 of adam optimizer')
    parser.add_argument('--vit_name', type=str, default='vit_b', help='select one vit model')
    parser.add_argument('--ckpt', type=str, default='./basemodel/segment_anything/modeling/sam_vit_b_01ec64.pth',
                        help='Pretrained checkpoint')
    parser.add_argument('--rank', type=int, default=8, help='Rank for LoRA adaptation')
    parser.add_argument('--module', type=str, default='sam_lora_image_encoder')


    opt = parser.parse_args()

    # ---- build models ----

    torch.cuda.set_device(1) 
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    net = IPDFRNet().cuda()

    params = net.parameters()
    optimizer = torch.optim.AdamW(params, opt.lr, weight_decay=1e-4)

    opt.train_dataset_dir = os.path.join(opt.data_path, opt.train_data_type)
    opt.test_dataset_dir = os.path.join(opt.data_path, opt.test_data_type)

    train_loader = BaseSegmentationExperiment(opt).train_loader

    print("#"*20, "Start Training", "#"*20)
    step = 0
    best_dice = 0.0
    for epoch in range(1, opt.epoch + 1):

        best_dice = train(train_loader, net, optimizer, epoch, best_dice, step)
