import torch
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

class dataset_csv(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.Y)

    def __getitem__(self,index):
        x = self.X[index]
        y = self.Y[index]
        return x,y


class DNN(nn.Module):
    def __init__(self, input_size, tgt_type, model_list, output_size=1):
        super(DNN, self).__init__()
        self.tgt_type = tgt_type
        layers = []
        prev_map = input_size
        for n,i in enumerate(model_list):
            layers.extend([
                nn.Linear(prev_map, i),
                nn.Tanh(),
            ])
            prev_map = i
        
        if tgt_type=="regression":
            layers.extend([nn.Linear(prev_map, output_size),])
        elif tgt_type=="binary classification":
            layers.extend([nn.Linear(prev_map, output_size),nn.Sigmoid(),])
        else:
            layers.extend([nn.Linear(prev_map, output_size),nn.LogSoftmax(dim=1),])
        self.model = nn.Sequential(
            *layers
        )
    def forward(self,x):
        return self.model(x)


# if __name__=="__main__":
#     net = DNN(5, 5, 'regression', [16,64,128,16])