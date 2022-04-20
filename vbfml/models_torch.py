import torch
import torch.nn as nn
import torch.nn.functional as F


def swish(x):
    return x * torch.sigmoid(x)


class Net(nn.Module):
    def __init__(self, n_features, n_classes, dropout=0.5):
        super(Net, self).__init__()

        self.layers = []
        self.n_classes = n_classes
        self.n_features = n_features
        self.n_nodes = [18,10,5,2]
        self.dropout = dropout
        self._build_layers()

    def _build_layers(self):
        self.layers.append(nn.Linear(self.n_features, self.n_nodes[0]))
        last_nodes = self.n_nodes[0]
        for i_n_nodes in self.n_nodes[1:]:
            self.layers.append(nn.BatchNorm1d(last_nodes))
            self.layers.append(nn.Linear(last_nodes, i_n_nodes))
            last_nodes = i_n_nodes
            self.layers.append(nn.Dropout(self.dropout))
        self.layers.append(nn.Linear(last_nodes, self.n_classes))
        self.layers.append(nn.BatchNorm1d(self.n_classes))

        for i, layer in enumerate(self.layers):
            setattr(self, "layer_%d" % i, layer)

    def forward(self, x):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                x = swish(layer(x))
            else:
                x = layer(x)
        #x = F.softmax(x,dim=1)
        x = F.softmax(x,dim=1)
        #x = torch.sigmoid(x)
    # print(x)
        return x

    def predict(self, x):
        x = torch.Tensor(x).to(torch.device("cuda:0"))
        self.eval()
        return self(x).cpu().detach().numpy()


