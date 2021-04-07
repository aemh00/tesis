import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import torch.nn.functional as F
from torch import nn, cuda

import sys
sys.path.append("../libs")
from vvv_utils import parse_metadata, parse_light_curve_data, plot_light_curve, get_train_test_ids

# DATASET

class LightCurve_Dataset(Dataset):
    def __init__(self, df_metadata,p_label, transform=None):
        self.data = list()
        self.name = df_metadata.index.values
        self.period = df_metadata[p_label]
        self.amplitud = list()
        self.VVV_dict = {idx: i for i,idx in enumerate(self.name)}
        self.label = df_metadata["label"].values
        self.transform = transform
        column_names_lc = ["mjd", "mag", "err"]
        column_names_fill = ["mjd", "mag", "err", "phase", "mask"]
        df_zeros = pd.DataFrame(np.zeros((1, 5)),columns=column_names_fill)

        for i in range(len(df_metadata)):
            lc_data = parse_light_curve_data(self.name[i])
            lc_data["phase"] = np.mod(lc_data["mjd"],self.period[i])/self.period[i]
            #display(lc_data)
            #normalize
            mag_std = lc_data["mag"].std()
            self.amplitud.append(mag_std)
            lc_data["mjd"] = lc_data["mjd"]-lc_data["mjd"].min()
            lc_data["mag"] = (lc_data["mag"]-lc_data["mag"].mean())/mag_std
            lc_data["err"] = lc_data["err"]/mag_std
            lc_data.sort_values(by="phase", inplace=True)
            # ajustar todas a largo 335, rellenando con 0s las que sean mas pequeñas,
            # asignando un label '1' si es dato real y '0' si es dato rellenado.
            lc_data = lc_data[["phase","mag","err","mjd"]]
            lc_data_large = np.zeros(shape=(335, 5), dtype='float32')
            lc_data_large[:lc_data.shape[0], :4] = lc_data.values
            lc_data_large[:lc_data.shape[0], 4] = 1.
            self.data.append(torch.from_numpy(lc_data_large))

    def __getitem__(self, idx):
        if np.isnan(idx):
            idx = self.VVV_dict[idx]
        sample = {'data': self.data[idx], 'label': self.label_to_num(self.label[idx]),'name': self.name[idx],'period':self.period[idx],'amplitud':self.amplitud[idx]}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def label_to_num(self, label):
        if label=='binary':
            return 0
        elif label=='rrlyrae':
            return 1
        else:
            return 2

    def period(self,idx):
        return self.period[idx]

    def plot(self, idx, ax):
        assert len(ax)==2, "Needs two subaxis"
        ax[0].cla()
        ax[0].errorbar(self.data[idx][:, 0], self.data[idx][:, 1], self.data[idx][:, 2], fmt='.')
        ax[0].invert_yaxis()
        ax[1].cla()
        ax[1].errorbar(self.data[idx][:, 3], self.data[idx][:, 1], self.data[idx][:, 2], fmt='.')
        ax[1].invert_yaxis()
        ax[0].set_title("%d %s %0.4f" %(idx, self.name[idx],self.period[idx]))

    def __len__(self):
        return len(self.data)


# MODELO

# implementacion adaptada a 1D de https://github.com/naoto0804/pytorch-inpainting-with-partial-conv

class PartialConv(nn.Module):
    def __init__(self, in_channels_C,in_channels_M, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super().__init__()
        self.input_conv = nn.Conv1d(in_channels_C, out_channels, kernel_size,
                                    stride, padding, dilation, groups, bias)
        self.mask_conv = nn.Conv1d(in_channels_M, out_channels, kernel_size,
                                   stride, padding, dilation, groups, False)
        # self.input_conv.apply(weights_init('kaiming'))

        torch.nn.init.constant_(self.mask_conv.weight, 1.0)

        # mask is not updated
        for param in self.mask_conv.parameters():
            param.requires_grad = False

    def forward(self,input, mask):
        # http://masc.cs.gmu.edu/wiki/partialconv
        # C(X) = W^T * X + b, C(0) = b, D(M) = 1 * M + 0 = sum(M)
        # W^T* (M .* X) / sum(M) + b = [C(M .* X) – C(0)] / D(M) + C(0)
        #print(input.shape, mask.shape)
        output = self.input_conv(input * mask)
        if self.input_conv.bias is not None:
            output_bias = self.input_conv.bias.view(1, -1, 1).expand_as(output)
        else:
            output_bias = torch.zeros_like(output)

        with torch.no_grad():
            output_mask = self.mask_conv(mask)

        no_update_holes = output_mask == 0
        mask_sum = output_mask.masked_fill_(no_update_holes, 1.0)

        output_pre = (output - output_bias) / mask_sum + output_bias
        output = output_pre.masked_fill_(no_update_holes, 0.0)

        new_mask = torch.ones_like(output)
        new_mask = new_mask.masked_fill_(no_update_holes, 0.0)

        return output, new_mask

class MLP_last(nn.Module):
    def __init__(self, in_channels_C=2,in_channels_M=1,c1=128,c2=64, c3=32, c4=16,c5=8, kernel_size=3, hid_dim=32,hid2_dim=16,hid3_dim=8,output_dim=3):
        super(MLP_last, self).__init__()
        self.pconv1 = PartialConv(in_channels_C,in_channels_M, c4, kernel_size, stride=2, padding=0, dilation=1, bias=True)
        self.pconv2 = PartialConv(c4, c4, c4, kernel_size, stride=2, padding=0, dilation=1, bias=True)
        self.pconv3 = PartialConv(c4, c4, c2, kernel_size, stride=2, padding=0, dilation=1, bias=True)
        #self.pconv4 = PartialConv(c4, c4, c4, kernel_size, stride=2, padding=0, dilation=1, bias=True)
        #self.pool1 = torch.nn.AvgPool1d(kernel_size, stride=None, padding=0,
        #                                 ceil_mode=False, count_include_pad=True)
        self.gap = nn.AdaptiveMaxPool1d(1)

        self.hidden1 = nn.Linear(c2*1+1, hid_dim, bias=True)
        self.hidden2 = nn.Linear(hid_dim, hid2_dim, bias=True)
        #self.hidden3 = torch.nn.Linear(hid2_dim, hid3_dim, bias=True)
        self.output = nn.Linear(hid2_dim, output_dim, bias=True)

        self.activation = nn.ReLU()
        #self.soft = torch.nn.Softmax(dim=1)

    def forward(self, x,mask, p,amp, t=0):
        if t==1:
            x = x.transpose(0,1)
            mask = mask.transpose(0,1)
        elif t==2:
            #x = x.transpose(1,2)
            mask = mask.transpose(1,2)

        #print("B4 C1:", x.shape,mask.shape)
        x, mask = self.pconv1(x, mask)
        #print("after C1:", x.shape,mask.shape)
        x = self.activation(x)
        x, mask = self.pconv2(x, mask)
        #print("after C2:", x.shape,mask.shape)
        x = self.activation(x)
        x, mask = self.pconv3(x, mask)
        #print("after C3:", x.shape)
        x = self.activation(x)
        #x, mask = self.pconv4(x, mask)
        z = self.gap(x)
        #print("after gap:", z.shape)
        z = z.reshape(-1,64*1)
        #print("after reshape:", z.shape)
        z = torch.cat((z,p),1)
        #print("after cat:", z.shape)
        z = self.activation(self.hidden1(z))
        z = self.activation(self.hidden2(z))
        #z = self.activation(self.hidden3(z))
        #print("after L2:", z.shape)
        fuera = self.output(z)
        #print("after L3:", fuera.shape)
        return fuera
