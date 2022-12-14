import os
import os.path as osp
import time
import math
from datetime import timedelta
from argparse import ArgumentParser

import torch
from torch import cuda
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from tqdm import tqdm

from east_dataset import EASTDataset
from dataset import SceneTextDataset
from model import EAST
import wandb
import numpy as np
import random
from inference import do_inference
from deteval import calc_deteval_metrics
import json

wandb.login()
wandb.init(
    project='OCR_project',
    entity="aitech4_cv3",
    name='base(ICDAR+upstage)',
    config={
        'lr': 0.001,
        'batch_size':24,
        'epoch' : 200,
        'seed' : 42,
    }
)

config = wandb.config

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def parse_args():
    parser = ArgumentParser()

    # Conventional args
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--data_dir', type=str,
                        default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data'))
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

    args = parser.parse_args()

    if args.input_size % 32 != 0:
        raise ValueError('`input_size` must be a multiple of 32')

    return args

def calc_score(gt_anno, pred_anno):
    precision, recall, f1_score = 0, 0, 0
    num_images = len(gt_anno['images'])
    
    for gt_image, pred_image in zip(sorted(gt_anno['images'].items()), sorted(pred_anno['images'].items())):
        gt_bboxes_dict, pred_bboxes_dict, transcriptions_dict  = {}, {}, {}
        gt_bboxes_list, pred_bboxes_list, transcriptions_list  = [], [], []
        
        for gt_point_index in gt_image[1]['words']:
            gt_bboxes_list.extend([gt_image[1]['words'][str(gt_point_index)]['points']])
            transcriptions_list.extend([gt_image[1]['words'][str(gt_point_index)]['transcription']])
        
        for pred_point_index in pred_image[1]['words']:
            pred_bboxes_list.extend([pred_image[1]['words'][pred_point_index]['points']])
        
        gt_bboxes_dict[gt_image[0]] = gt_bboxes_list
        transcriptions_dict[gt_image[0]] = transcriptions_list
        pred_bboxes_dict[gt_image[0]] = pred_bboxes_list
        
        metrics = calc_deteval_metrics(pred_bboxes_dict=pred_bboxes_dict, gt_bboxes_dict=gt_bboxes_dict, transcriptions_dict=transcriptions_dict)
        
        precision += metrics['total']['precision']
        recall += metrics['total']['recall']
        f1_score += metrics['total']['hmean']
    
    return precision/num_images, recall/num_images, f1_score/num_images

def do_training(seed, data_dir, model_dir, device, image_size, input_size, num_workers, batch_size,
                learning_rate, max_epoch, save_interval):
    
    seed_everything(seed)

    train_dataset = SceneTextDataset(f'{data_dir}/train_image', split='train', image_size=image_size, crop_size=input_size)
    train_dataset = EASTDataset(train_dataset)
    train_num_batches = math.ceil(len(train_dataset) / batch_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    valid_dataset = SceneTextDataset(f'{data_dir}/valid_image', split='valid', image_size=image_size, crop_size=input_size)
    valid_dataset = EASTDataset(valid_dataset)
    valid_num_batches = math.ceil(len(valid_dataset) / batch_size)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EAST()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[max_epoch // 2], gamma=0.1)


    mean_loss = float('inf')
    max_f1 = 0
    
    for epoch in range(max_epoch):
        model.train()
        epoch_loss, epoch_cls_loss, epoch_angle_loss, epoch_iou_loss, epoch_start = 0, 0, 0, 0, time.time()
        
        with tqdm(total=train_num_batches) as pbar:
            for img, gt_score_map, gt_geo_map, roi_mask in train_loader:
                pbar.set_description('[[train] Epoch {}]'.format(epoch + 1))

                loss, extra_info = model.train_step(img, gt_score_map, gt_geo_map, roi_mask)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_val = loss.item()
                epoch_loss += loss_val
                epoch_cls_loss += extra_info['cls_loss']
                epoch_angle_loss += extra_info['angle_loss']
                epoch_iou_loss += extra_info['iou_loss']
                
                pbar.update(1)
                val_dict = {
                    'Cls loss': epoch_cls_loss/train_num_batches, 'Angle loss': epoch_angle_loss/train_num_batches,
                    'IoU loss': epoch_iou_loss/train_num_batches
                }
                pbar.set_postfix(val_dict)

        print('Mean loss: {:.4f} | Elapsed time: {}'.format(
            epoch_loss / train_num_batches, timedelta(seconds=time.time() - epoch_start)))

        scheduler.step()
        
        model.eval()
        val_epoch_loss, val_epoch_cls_loss, val_epoch_angle_loss, val_epoch_iou_loss, val_epoch_start = 0, 0, 0, 0, time.time()
        
        with tqdm(total=valid_num_batches) as pbar:
            for img, gt_score_map, gt_geo_map, roi_mask in valid_loader:
                pbar.set_description('[[valid] Epoch {}]'.format(epoch + 1))
                
                with torch.no_grad():
                    loss, extra_info = model.train_step(img, gt_score_map, gt_geo_map, roi_mask)
                
                loss_val = loss.item()
                val_epoch_loss += loss_val
                val_epoch_cls_loss += extra_info['cls_loss']
                val_epoch_angle_loss += extra_info['angle_loss']
                val_epoch_iou_loss += extra_info['iou_loss']
                
                pred_score_map = extra_info['score_map']
                pred_geo_map = extra_info['geo_map']
                
                pbar.update(1)
                val_dict = {
                    'Cls loss': val_epoch_cls_loss/valid_num_batches, 'Angle loss': val_epoch_angle_loss/valid_num_batches,
                    'IoU loss': val_epoch_iou_loss/valid_num_batches
                }
                pbar.set_postfix(val_dict)
        
        data_dir = '/opt/ml/input/data'
        with open('/opt/ml/input/data/ufo/valid.json', 'r') as f:
            gt_anno = json.load(f)
        pred_anno = do_inference(model=model, data_dir=data_dir, ckpt_fpath=None, input_size=input_size, batch_size=valid_num_batches, split='valid_image')
        precision, recall, f1_score = calc_score(gt_anno, pred_anno)
        
        if mean_loss >= val_epoch_loss / valid_num_batches:
            mean_loss = val_epoch_loss / valid_num_batches
            
        wandb.log({
                    'Mean_loss': mean_loss,
                    'Cls_loss': val_epoch_cls_loss/valid_num_batches,
                    'Angle_loss': val_epoch_angle_loss/valid_num_batches,
                    'IoU_loss': val_epoch_iou_loss/valid_num_batches,
                    'precision' : precision,
                    'recall' : recall,
                    'f1_score' : f1_score
                    })
        
        wandb.watch(model)
        
        print('f1 score: {:.4f} | Elapsed time: {}'.format(
            f1_score, timedelta(seconds=time.time() - val_epoch_start)))
        
        if max_f1 <= f1_score:
            max_f1 = f1_score
            if not osp.exists(model_dir):
                os.makedirs(model_dir)
            print(f"Update latest.pth! f1 score:{f1_score}")
            ckpt_fpath = osp.join(model_dir, 'latest.pth')
            torch.save(model.state_dict(), ckpt_fpath)
            

def main(args):
    do_training(**args.__dict__)


if __name__ == '__main__':
    args = parse_args()
    main(args)
    wandb.save('/opt/ml/code/trained_models/latest.pth')
    wandb.finish()
