# ENGN8501-Advanced Computer Vision 2020 Project - Group 24
# Google Landmark Recognition 2020
# Create By: Tianyi,Qi

# ==== Code Reference Start ====
# https://github.com/lyakaap/Landmark2019-1st-and-3rd-Place-Solution
# ==== Code Reference End ======

import argparse
import os
from collections import OrderedDict

import torch.nn as nn
import torch.optim as optim
import logging
import torch.utils.data
from tqdm import tqdm

import data_utils
from extra import utils, metrics
from parameters import params
import dataset_connector
from model import model

if __name__ == '__main__':
    # Config Logger
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
    # Parse Args
    parser = argparse.ArgumentParser(description="ENGN8501-2020-Landmark Training")
    parser.add_argument('--devices', '-d', type=str,
                        help='Comma delimited GPU device list you want to use.(e.g. "0,1")')
    parser.add_argument('--resume', type=str, default="",
                        help='Set to a pretrained model or a previous checkpoint to continue training.')
    parser.add_argument('--resume_epoch', type=int, default=0,
                        help='Set the epoch to resume. The program will resume training at this epoch.')
    args = parser.parse_args()

    result_folder = os.path.join(dataset_connector.result_dir, f'{params["loss"]}/')

    if not torch.cuda.is_available():
        logging.error("NO CUDA DEVICE AVAILABLE. REQUIRES AT LEAST 1 CUDA DEVICE.")
        exit(-1)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.devices

    # Build Data Augmentation (Data Transformer)
    if params['augmentation'] == 'soft':
        params['scale_limit'] = 0.2
        params['brightness_limit'] = 0.1
    elif params['augmentation'] == 'middle':
        params['scale_limit'] = 0.3
        params['shear_limit'] = 4
        params['brightness_limit'] = 0.1
        params['contrast_limit'] = 0.1
    else:
        logging.error("Data Augmentation Config Not Defined. Must set to soft/middle.")
        raise ValueError
    train_transform, eval_transform = data_utils.build_transforms(
        scale_limit=params['scale_limit'],
        shear_limit=params['shear_limit'],
        brightness_limit=params['brightness_limit'],
        contrast_limit=params['contrast_limit'],
    )

    # Build Pytorch Dataloader
    data_loaders = data_utils.make_train_loaders(
        params=params,
        train_transform=train_transform,
        eval_transform=eval_transform,
        scale='S2',
        test_size=0.1,
        num_workers=os.cpu_count() * 2)

    # Build Model, loss and optimizer.
    model = model.LandmarkNet(n_classes=params['class_topk'],
                              model_name=params['model_name'],
                              pooling=params['pooling'],
                              loss_module=params['loss'],
                              s=params['s'],
                              margin=params['margin'],
                              theta_zero=params['theta_zero'],
                              use_fc=params['use_fc'],
                              fc_dim=params['fc_dim'],
                              ).cuda()
    optimizer = optim.SGD(model.parameters(), params['lr'], momentum=0.9, weight_decay=params['wd'])
    criterion = nn.CrossEntropyLoss()

    # Load Pretrained Weights
    if len(args.resume) != 0:
        state_dict = torch.load(args.resume)
        if state_dict.keys()[0][:7] == "nn.Module":
            # Previously trained on a multi-GPU.
            clean_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]
                clean_state_dict[name] = v
            model.load_state_dict(clean_state_dict)
        else:
            model.load_state_dict(state_dict)

    # Set Parameters. And Sync GPUs.
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=params['epochs'] * len(data_loaders['train']), eta_min=3e-6)

    start_epoch, end_epoch = (args.resume_epoch, params['epochs'])

    if len(args.devices.split(',')) > 1:
        model = nn.DataParallel(model)

    # Train.
    for epoch in range(start_epoch, end_epoch):

        model.train(True)

        losses = utils.AverageMeter()
        prec1 = utils.AverageMeter()

        for i, (_, x, y) in tqdm(enumerate(data_loaders['train']),
                                 total=len(data_loaders['train']),
                                 miniters=None, ncols=55):
            x = x.to('cuda')
            y = y.to('cuda')

            if params["loss"] in ["AdditiveMarginSoftmaxLoss"]:
                outputs, loss = model(x, y)
                loss = loss.mean()
            elif params["loss"] in ["LSoftmax", "arcface", "cosface"]:
                outputs = model(x, y)
                loss = criterion(outputs, y)
            elif params["loss"] in ["Softmax"]:
                outputs = model(x)
                loss = criterion(outputs, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            acc = metrics.accuracy(outputs, y)
            losses.update(loss.item(), x.size(0))
            prec1.update(acc, x.size(0))

            logging.info("Epoch:{}/{} Iter:{}/{} Loss:{} Acc:{}".format(epoch + 1, end_epoch, i,
                                                                        len(data_loaders["train"]), loss.item(),
                                                                        prec1.avg))

        model_save_path = result_folder + f'Epoch{epoch}_' + str(params["loss"]) + '.pth'
        logging.info("Saving Model {} ...".format(model_save_path))
        utils.save_checkpoint(path=model_save_path,
                              model=model,
                              epoch=epoch,
                              optimizer=optimizer,
                              params=params)
        logging.info("Model Saved:{}".format(model_save_path))

    logging.info("Training Done.")
