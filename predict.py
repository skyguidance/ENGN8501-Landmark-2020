import argparse
import logging
import os
import pickle
import numpy as np
import time
import torch
from tqdm import tqdm
from scipy.special import softmax
import torch.nn.functional as F
from sklearn.metrics import *
import pandas as pd

import data_utils
import dataset_connector
from model import model

if __name__ == '__main__':
    # Config Logger
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
    # Parse Args
    parser = argparse.ArgumentParser(description="ENGN8501-2020-Landmark Predict")
    parser.add_argument('--devices', '-d', type=str,
                        help='Comma delimited GPU device list you want to use.(e.g. "0,1")')
    parser.add_argument('--model-path', '-m', type=str,
                        help='The model path for prediction.')
    parser.add_argument('--ms', action="store_true",
                        help='Enable Multi-Scale Prediction.')
    args = parser.parse_args()
    # Load Module.
    os.environ['CUDA_VISIBLE_DEVICES'] = args.devices
    ckpt = torch.load(args.model_path)
    params, state_dict = ckpt['params'], ckpt['state_dict']
    params['test_batch_size'] = 32
    # Load Net.
    model = model.LandmarkNet(n_classes=params['class_topk'],
                              model_name=params['model_name'],
                              pooling=params['pooling'],
                              loss_module=params["loss"],
                              s=params['s'],
                              margin=params['margin'],
                              theta_zero=params['theta_zero'],
                              use_fc=params['use_fc'],
                              fc_dim=params['fc_dim'],
                              )
    model.load_state_dict(state_dict, strict=False)
    model = model.to('cuda').eval()

    # Import class_id to landmark_id translator.
    classid_2_landmarkid = pickle.load(
        open(os.path.join(dataset_connector.result_dir, "classid_2_landmarkid.pkl"), "rb"))

    train_transform, eval_transform = data_utils.build_transforms()

    data_loaders = data_utils.make_train_loaders(
        params=params,
        train_transform=train_transform,
        eval_transform=eval_transform,
        scale='S2',
        test_size=0.1,
        num_workers=os.cpu_count() * 2)

    min_size = 128
    scales = [0.75, 1.0, 1.25] if args.ms else [1.0]

    ids, labels, scores, gts, inference_time = [], [], [], [], []

    for i, (img_id, x, y) in tqdm(enumerate(data_loaders['val']),
                                  total=len(data_loaders['val']),
                                  miniters=None, ncols=55):

        batch_size, _, h, w = x.shape
        score_blend = np.zeros((batch_size, params['class_topk']))
        start_time = time.time()
        with torch.no_grad():
            x = x.to('cuda')
            for s in scales:
                th = max(min_size, int(h * s // model.DIVIDABLE_BY * model.DIVIDABLE_BY))
                tw = max(min_size, int(w * s // model.DIVIDABLE_BY * model.DIVIDABLE_BY))  # round off

                scaled_x = F.interpolate(x, size=(th, tw), mode='bilinear', align_corners=True)
                score = model(scaled_x).cpu().numpy()
                # score = score.max(axis=1)
                score_blend += score

        score_blend /= len(scales)
        label = score_blend.argmax(axis=1)
        score_blend = softmax(score_blend).max(axis=1)
        end_time = time.time()
        time_per_image = (end_time - start_time) / batch_size
        inference_time.append(time_per_image)
        # score_blend = score_blend.max(axis=1)
        ids.extend(img_id)
        scores.extend(score_blend.tolist())
        labels.extend(label.tolist())
        gts.append(y.tolist())

    predict_landmark_id = []
    for i, label in enumerate(labels):
        predict_landmark_id.append(str(classid_2_landmarkid[labels[i]]))
    gts = np.array(gts).flatten()
    df = pd.DataFrame(
        {'id': ids, 'PredictLandmarkID': predict_landmark_id, 'label': labels, 'score': scores, 'gt': gts})
    df.to_csv(os.path.join(dataset_connector.result_dir, "result.csv"), index=None)
    print("CSV Reult Saved: {}".format(str(os.path.join(dataset_connector.result_dir, "result.csv"))))
    print("Model:Results\n")
    print("Accuracy:{}".format(accuracy_score(gts, labels)))
    print("Time per Image:{}s".format(np.average(inference_time)))
