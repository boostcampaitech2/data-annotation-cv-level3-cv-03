import os
import os.path as osp
import time
import math
from datetime import timedelta
from argparse import ArgumentParser

import torch
from torch import cuda
from torch.utils.data import DataLoader,ConcatDataset
from torch.optim import lr_scheduler
from tqdm import tqdm

from east_dataset import EASTDataset
from dataset import SceneTextDataset, AIHubDataset
from model import EAST

# custom
import wandb



import numpy as np
import cv2

def gray_mask_to_heatmap(x):
    x = cv2.cvtColor(cv2.applyColorMap(x, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    return x

def get_superimposed_image(image, score_map, heatmap=True, w_image=None, w_map=None):
    """
    Args:
        image (ndarray): (H, W, C) shaped, float32 or uint8 dtype is allowed.
        score_map (ndarray): (H, W) shaped, float32 or uint8 dtype is allowed.
        heatmap (boot): Wheather to convert `score_map` into a heatmap.
        w_image (float)
        w_map (float)

    Blending weights(`w_image` and `w_map`) are default to (0.4, 0.6).
    """

    assert w_image is None or (w_image > 0 and w_image < 1)
    assert w_map is None or (w_map > 0 and w_map < 1)

    image = cv2.resize(image, dsize=(score_map.shape[1], score_map.shape[0]))

    if image.dtype != np.uint8:
        image = (255 * np.clip(image, 0, 1)).astype(np.uint8)

    if score_map.dtype != np.uint8:
        score_map = (255 * np.clip(score_map, 0, 1)).astype(np.uint8)
    if heatmap:
        score_map = gray_mask_to_heatmap(score_map)
    elif score_map.ndim == 2 or score_map.shape[2] != 3:
        score_map = cv2.cvtColor(score_map, cv2.COLOR_GRAY2RGB)

    if w_image is None and w_map is None:
        w_image, w_map = 0.4, 0.6
    elif w_image is None:
        w_image = 1 - w_map
    elif w_map is None:
        w_map = 1 - w_image

    return cv2.addWeighted(image, w_image, score_map, w_map, 0)

def parse_args():
    parser = ArgumentParser()

    # Conventional args
    parser.add_argument('--data_dir', type=str,
                        default=os.environ.get('SM_CHANNEL_TRAIN', '../input/data/ICDAR17_Korean'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR',
                                                                        'trained_models'))

    parser.add_argument('--device', default='cuda' if cuda.is_available() else 'cpu')
    parser.add_argument('--num_workers', type=int, default=4)

    parser.add_argument('--image_size', type=int, default=1024)
    parser.add_argument('--input_size', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--max_epoch', type=int, default=200)
    parser.add_argument('--save_interval', type=int, default=5)

    parser.add_argument('--model_name',default='model_name')

    args = parser.parse_args()

    if args.input_size % 32 != 0:
        raise ValueError('`input_size` must be a multiple of 32')

    return args


def do_training(data_dir, model_dir, device, image_size, input_size, num_workers, batch_size,
                learning_rate, max_epoch, save_interval,**kwargs):
    dataset = SceneTextDataset(data_dir, split='train', image_size=image_size, crop_size=input_size)
    dataset = EASTDataset(dataset)
    ## custom
    aihub_dataset = AIHubDataset(copy_obj = ['word','ignored']) # copy_obj = ['character','ignored'] or copy_obj = ['word','ignored']
    aihub_dataset = EASTDataset(aihub_dataset)

    dataset = ConcatDataset([dataset,aihub_dataset])
    
    num_batches = math.ceil(len(dataset) / batch_size)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EAST()
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    # scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[max_epoch // 2], gamma=0.1)
    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=max_epoch,eta_min=1e-4)

    ## custom
    wandb.watch(model)

    model.train()
    for epoch in range(max_epoch):
        epoch_loss, epoch_start = 0, time.time()
        with tqdm(total=num_batches) as pbar:
            for img, gt_score_map, gt_geo_map, roi_mask in train_loader:
                pbar.set_description('[Epoch {}]'.format(epoch + 1))

                loss, extra_info = model.train_step(img, gt_score_map, gt_geo_map, roi_mask)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_val = loss.item()
                epoch_loss += loss_val

                pbar.update(1)
                val_dict = {
                    'Cls loss': extra_info['cls_loss'], 'Angle loss': extra_info['angle_loss'],
                    'IoU loss': extra_info['iou_loss']
                }

                ## custom
                
                wandb.log({
                    "train/loss" : loss_val,
                    "train/Cls_loss": extra_info['cls_loss'],
                    "train/Angle_loss" : extra_info['angle_loss'],
                    "train/IoU_loss" : extra_info['iou_loss'],
                    "lr" : scheduler.get_last_lr()[0]
                    })

                pbar.set_postfix(val_dict)

        scheduler.step()

        print('Mean loss: {:.4f} | Elapsed time: {}'.format(
            epoch_loss / num_batches, timedelta(seconds=time.time() - epoch_start)))

        ## custom
        wandb_images=[]
        wandb_images.append(
            wandb.Image(
                get_superimposed_image(
                    img[0].detach().cpu().numpy().transpose(1,2,0),
                    gt_score_map[0].detach().cpu().numpy().squeeze() * 0.8,
                    heatmap=False),
                caption='gt_superimposed_image'))

        wandb_images.append(
            wandb.Image( 
                get_superimposed_image(
                img[0].detach().cpu().numpy().transpose(1,2,0),
                extra_info['score_map'][0].detach().cpu().numpy().squeeze(),
                heatmap=False)
            ,caption='pred_superimposed_image'))
        
        wandb.log({ "train/images" : wandb_images })
        ##

        if (epoch + 1) % save_interval == 0:
            if not osp.exists(model_dir):
                os.makedirs(model_dir)

            ckpt_fpath = osp.join(model_dir, f'latest_{epoch+1}.pth')
            torch.save(model.state_dict(), ckpt_fpath)


def main(args):
    wandb.init(
        project='p-stgae_ocr',
        name=args.model_name)
    wandb.config.update(args)
    
    do_training(**args.__dict__)


if __name__ == '__main__':
    args = parse_args()
    main(args)



