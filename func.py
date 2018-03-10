import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math
from torch.autograd import Variable

pi = 0.01
class Recommend(nn.Module):
    """A model to build Recommendation system
    """
    def __init__(self, past_observations, n_factors,  output_dim):
        super().__init__()

        self.past_observations = past_observations
        self.n_factors = n_factors
        self.output_dim = output_dim
        self.embedding = torch.nn.Embedding(self.output_dim, self.n_factors)
        self.n1 = nn.Linear(self.n_factors * self.past_observations, 100)
        self.n2 = nn.Linear(100, 50)
        self.output = nn.Linear(50, self.output_dim)
        init.constant(self.output.bias, -math.log((1-pi)/pi))

    def forward(self, x):
        """ We will have one Embedding matrix.
        """
        k = []
        for i in x:
            val = self.embedding(i)
            k.append(val.view(1, -1))

        x = torch.cat(k)
        x = self.n1(x)
        x = F.relu(x)
        x = self.n2(x)
        x = F.relu(x)
        x = self.output(x)
        return x

class DataLoader():
    def __init__(self, inputs, output, embed):
        self.inputs = inputs
        self.output = output
        self.embed = embed

    def __getitem__(self, idx):
        o_in = torch.from_numpy(self.inputs[idx, :])
        o_out = torch.from_numpy(self.output[idx, :])
        return o_in, o_out

    def __len__(self):
        return self.inputs.shape[0]


class FocalLoss(nn.Module):
    def __init__(self,
                 classes,
                 focusing_param=2.0,
                 balance_param=0.25,
                use_gpu=False):
        super().__init__()
        self.focusing_param = focusing_param
        self.balance_param = balance_param
        self.classes = classes
        self.use_gpu = use_gpu

    def forward(self, x, y):
        batch_size, next_best = y.size()[0], y.size()[1]
        t = torch.FloatTensor(batch_size, self.classes)
        t.zero_()
        t.scatter_(1, y.data.cpu(), 1)
        t = Variable(t)
        sigmoid_p = F.sigmoid(x)

        zeros = Variable(torch.zeros(sigmoid_p.size()))

        if self.use_gpu:
            zeros = zeros.cuda()
            t = t.cuda()

        pos_p_sub = ((t >= sigmoid_p).float() * (t-sigmoid_p)) + ((t < sigmoid_p).float() * zeros)
        neg_p_sub = ((t >= zeros).float() * zeros) + ((t <= zeros).float() * sigmoid_p)
        ce = (-1) * self.balance_param * (pos_p_sub ** self.focusing_param) * torch.log(torch.clamp(sigmoid_p, 1e-4, 1.0)) -(1-self.balance_param) * (neg_p_sub ** self.focusing_param) * torch.log(torch.clamp(1.0-sigmoid_p, 1e-4, 1.0))
        pos_samples = float(batch_size * next_best)
        return ce.sum()/pos_samples
