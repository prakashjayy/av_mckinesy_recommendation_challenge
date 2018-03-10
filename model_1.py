""" To train the model.
"""
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F

from func import Recommend, DataLoader, FocalLoss

import numpy as np
import random
import pickle
import os

random.seed(50)

batch_size = 256
epochs = 200
n_factors = 50
save_loc = "trained_weights/"
use_gpu = True

xtrain, xtest, embed = pickle.load(open("data/torch_dump.pkl", "rb"))
print(xtrain.shape, xtest.shape)

userid = list(xtrain["user_id"].values)
random.shuffle(userid)
tuid = userid[:int(len(userid)*0.7)]
teuid = userid[int(len(userid)*0.7):]
print(len(tuid), len(teuid))

x_train = xtrain[xtrain["user_id"].isin(tuid)]
x_valid = xtrain[xtrain["user_id"].isin(teuid)]
print(xtrain.shape, x_valid.shape)


train_fe = list(range(1, 11))
output = list(range(11, 14))
print(train_fe)
print(output)


train_dl = torch.utils.data.DataLoader(DataLoader(x_train[train_fe].values,
                                                  x_train[output].values, embed),
                                      batch_size= batch_size,
                                      shuffle = True,
                                      num_workers = 2)

valid_dl = torch.utils.data.DataLoader(DataLoader(x_valid[train_fe].values,
                                                  x_valid[output].values, embed),
                                      batch_size= batch_size,
                                      shuffle = True,
                                      num_workers = 2)


model = Recommend(10, n_factors, len(embed))

if use_gpu:
    model = model.cuda()

opt = torch.optim.SGD(list(filter(lambda p: p.requires_grad, model.parameters())),
                               lr = 0.1,
                               momentum= 0.9,
                               weight_decay=0.0001)
loss = FocalLoss(len(embed), use_gpu=use_gpu)

dl = {"train": train_dl, "valid": valid_dl}

best_loss=float("Inf")
print("[Network training began]")

for ep in range(epochs):
    print("[Epoch {}]".format(ep))
    for phase in ["train", "valid"]:
        if phase == "train":
            model.train()
        else:
            model.eval()
        loss_meter = []
        for inputs, outputs in dl[phase]:
            if use_gpu:
                inputs = Variable(inputs.cuda())
                outputs = Variable(outputs.cuda())
            else:
                inputs = Variable(inputs)
                outputs = Variable(outputs)

            if phase =="train":
                opt.zero_grad()
            preds = model(inputs)
            loss_calc = loss(preds, outputs)
            if phase =="train":
                loss_calc.backward()
                opt.step()
            loss_meter.append(loss_calc.data[0])
            #print("{} loss: {}".format(phase, loss_calc.data[0]))
        val_loss = np.mean(loss_meter)
        print("{} loss: {}".format(phase, val_loss))

        if phase == "valid":
            if val_loss < best_loss:
                print("Saving model...")
                if not os.path.isdir(save_loc):
                    os.mkdir(save_loc)

                    torch.save(model.state_dict(), save_loc+"model2.pth")
                    best_loss = val_loss
                    print("[Training finished.......]")
