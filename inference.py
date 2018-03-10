""" To make a submission
"""
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F

from func import Recommend, Recommend_2, DataLoader, FocalLoss
from metric import mapk
import numpy as np
import pandas as pd
from tqdm import tqdm
import random
import pickle
import os

n_factors = 50
save_loc = "trained_weights/model2.pth"
use_gpu = True

xtrain, xtest, embed = pickle.load(open("data/torch_dump.pkl", "rb"))
print(xtrain.shape, xtest.shape)

model = Recommend(10, n_factors, len(embed))

state_dict = torch.load(save_loc)
model.load_state_dict(state_dict)
model_pred = model.eval()

train_fe = list(range(1, 11))
print(train_fe)

inputs = torch.from_numpy(xtest[train_fe].values)
print(inputs.size())

preds = model_pred(inputs)
final = F.sigmoid(preds)

predictions = []
for i in tqdm(final):
    array = i.data.cpu().numpy()
    k = np.argsort(array)[::-1][0:3]
    predictions.append(k)

final_preds = np.concatenate(predictions)
print(final_preds.shape)

embed_inv = {k:v for v, k in embed.items()}
pred_challenge = [embed_inv[i] for i in final_preds]

#TODO boring stufff. Need to modify this
# train = pd.read_csv("data/train.csv")
# train = train[train["challenge_sequence"].isin([11, 12, 13])]
# train["predicted"] = pred_challenge
# train.to_csv("train_sub.csv", index=False)
# true = list(train["challenge"].values.reshape(-1, 3))
# pred = list(train["predicted"].values.reshape(-1, 3))
# mapk(true, pred, k=3)

submission = pd.read_csv("data/sample_submission_J0OjXLi_DDt3uQN.csv")

submission["challenge"] = pred_challenge
submission.to_csv("sub1.csv", index=False)
