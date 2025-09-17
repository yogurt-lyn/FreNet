import torch
import imageio
import numpy as np
import os, argparse
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torchvision.transforms.functional as TF
from utils.dataloader import BaseSegmentationExperiment

from model import IPDFRNet

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./dataset/BioMedicalDataset')
    parser.add_argument('--train_data_type', type=str, required=False, choices=['ISIC2018','BUSI','PH2','STU'])
    parser.add_argument('--test_data_type', type=str, required=False, choices=['ISIC2018','BUSI','PH2','STU'])
    parser.add_argument('--img_size', type=int, default=352, help='input patch size of network input')
    parser.add_argument('--batchsize', type=int, default=1, help='training batch size')
    parser.add_argument('--save_path', type=str, default='./IPDFRNetISIC2018vis', help='path to save inference segmentation')
    parser.add_argument('--num_classes', type=int, default=1, help='epoch number')
    parser.add_argument('--ckpt', type=str, default='./basemodel/segment_anything/modeling/sam_vit_b_01ec64.pth',help='Pretrained checkpoint')
    parser.add_argument('--lora_ckpt', type=str, default='./IPDFRNet/ISIC2018/best.pth', help='The checkpoint from LoRA')
    parser.add_argument('--vit_name', type=str, default='vit_b', help='Select one vit model')#
    parser.add_argument('--rank', type=int, default=8, help='Rank for LoRA adaptation')
    parser.add_argument('--module', type=str, default='sam_lora_image_encoder')
# 
    opt = parser.parse_args()

    torch.cuda.set_device(1)  # 设置当前使用的 GPU 设备为 1
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    net = IPDFRNet().cuda()

    net.load_state_dict(torch.load(opt.lora_ckpt))

    net.eval()

    if opt.save_path is not None:
        os.makedirs(opt.save_path, exist_ok=True)

    # print('evaluating model: ', opt.ckpt_path)

    opt.train_dataset_dir = os.path.join(opt.data_path, opt.train_data_type)
    opt.test_dataset_dir = os.path.join(opt.data_path, opt.test_data_type)

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
            outputs, sam_outs = net(image)

        outputs = outputs.sigmoid().data.cpu().numpy().squeeze()

        image = image.sigmoid().data.cpu().numpy().squeeze()

        if opt.save_path is not None:
            # 转换为 0-255 的 uint8 格式
            sample_res = outputs
            # sample_sam = sam_outs
            sample_gt = gt.squeeze()
            sample_img = np.transpose(image, (1, 2, 0)) 

            sample_res = (sample_res * 255).astype(np.uint8)
            # sample_sam = (sample_sam * 255).astype(np.uint8)
            sample_gt = (sample_gt * 255).astype(np.uint8)
            sample_img = (sample_img * 255).astype(np.uint8)
            imageio.imwrite(opt.save_path + '/' + str(i) + '_img.jpg', sample_img)
            # imageio.imwrite(opt.save_path + '/' + str(i) + '_sam.jpg', sample_sam)
            imageio.imwrite(opt.save_path + '/' + str(i) + '_pred.jpg', sample_res)
            imageio.imwrite(opt.save_path + '/' + str(i) + '_gt.jpg', sample_gt)

        input = np.where(outputs >= 0.5, 1, 0)
        target = np.where(np.array(gt) >= 0.5, 1, 0)

        preds.append(input)
        gts.append(gt)

        smooth = 1
        input_flat = np.reshape(input, (-1))
        target_flat = np.reshape(target, (-1))
        intersection = (input_flat * target_flat)
        union = input_flat + target_flat - intersection

        jacard = ((np.sum(intersection) + smooth) / (np.sum(union) + smooth))
        jacard = '{:.4f}'.format(jacard)
        jacard = float(jacard)
        JACARD += jacard

        dice = (2 * intersection.sum() + smooth) / (input.sum() + target.sum() + smooth)
        dice = '{:.4f}'.format(dice)
        dice = float(dice)
        DSC += dice

    print('*****************************************************')
    print('Dice Score: ' + str(DSC / num1))
    print('Jacard Score: ' + str(JACARD / num1))
