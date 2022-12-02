import torch
from torch import nn
from torch.nn import functional as F
import numpy as np


class Conv(nn.Module):
    """
    Convolution module including conv2d, relu, batch_norm, and max_pool2d
    """

    def __init__(
        self,
        in_channel,
        out_channel,
        kernel,
        stride,
        padding,
        kernel_pool,
        stride_pool,
        padding_pool,
    ):
        self.conv = nn.Conv2d(
            in_channels=in_channel,
            out_channels=out_channel,
            kernel_size=kernel,
            stride=stride,
            padding=padding,
        )
        self.relu = nn.ReLU()
        self.norm = nn.BatchNorm2d(out_channel)
        self.pool = nn.MaxPool2d(kernel_size=kernel_pool, stride=stride_pool, padding=padding_pool)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.norm(x)
        x = self.pool(x)
        return x


class Model(nn.Module):
    def __init__(self, config):
        self.config = config
        self.layers = nn.ModuleList(
            modules=[
                Conv(
                    in_channel=3,
                    out_channel=32,
                    kernel=3,
                    stride=1,
                    padding=0,
                    kernel_pool=2,
                    stride_pool=2,
                    padding_pool=0,
                ),
                Conv(
                    in_channel=32,
                    out_channel=32,
                    kernel=3,
                    stride=1,
                    padding=0,
                    kernel_pool=2,
                    stride_pool=2,
                    padding_pool=0,
                ),
                Conv(
                    in_channel=32,
                    out_channel=32,
                    kernel=3,
                    stride=1,
                    padding=0,
                    kernel_pool=2,
                    stride_pool=2,
                    padding_pool=0,
                ),
                Conv(
                    in_channel=32,
                    out_channel=32,
                    kernel=3,
                    stride=1,
                    padding=0,
                    kernel_pool=2,
                    stride_pool=2,
                    padding_pool=0,
                ),
                nn.Flatten(),
                nn.Linear(in_features=32 * 5 * 5, out_features=5),
            ]
        )

    def forward(self, x, parameter=None):
        if not parameter:
            parameter = self.parameters()
        for layer in self.layers:
            x = layer(x)
        return x


class Learner(nn.Module):
    """ """

    def __init__(self, config, imgc, imgsz):
        """

        :param config: network config file, type:list of (string, list)
        :param imgc: 1 or 3
        :param imgsz:  28 or 84
        """
        super(Learner, self).__init__()

        self.config = config

        # this dict contains all tensors needed to be optimized
        self.vars = nn.ParameterList()
        # running_mean and running_var
        self.vars_bn = nn.ParameterList()

        # save weights according to layers specified in self.config
        for i, (name, param) in enumerate(self.config):
            if name == "conv2d":
                # [ch_out, ch_in, kernelsz, kernelsz]
                w = nn.Parameter(torch.ones(*param[:4]))
                # gain=1 according to cbfin's implementation
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0])))

            elif name == "convt2d":
                # [ch_in, ch_out, kernelsz, kernelsz, stride, padding]
                w = nn.Parameter(torch.ones(*param[:4]))
                # gain=1 according to cbfin's implementation
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                # [ch_in, ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[1])))

            elif name == "linear":
                # [ch_out, ch_in]
                w = nn.Parameter(torch.ones(*param))
                # gain=1 according to cbfinn's implementation
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0])))

            elif name == "bn":
                # [ch_out]
                w = nn.Parameter(torch.ones(param[0]))
                self.vars.append(w)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0])))

                # must set requires_grad=False
                running_mean = nn.Parameter(torch.zeros(param[0]), requires_grad=False)
                running_var = nn.Parameter(torch.ones(param[0]), requires_grad=False)
                self.vars_bn.extend([running_mean, running_var])

            elif name in [
                "tanh",
                "relu",
                "upsample",
                "avg_pool2d",
                "max_pool2d",
                "flatten",
                "reshape",
                "leakyrelu",
                "sigmoid",
            ]:
                continue
            else:
                raise NotImplementedError

    def forward(self, x, vars=None, bn_training=True):
        """
        This function can be called by finetunning, however, in finetunning, we dont wish to update
        running_mean/running_var. Thought weights/bias of bn is updated, it has been separated by fast_weights.
        Indeed, to not update running_mean/running_var, we need set update_bn_statistics=False
        but weight/bias will be updated and not dirty initial theta parameters via fast_weiths.
        :param x: [b, 1, 28, 28]
        :param vars:
        :param bn_training: set False to not update
        :return: x, loss, likelihood, kld
        """

        if vars is None:
            vars = self.vars

        idx = 0
        bn_idx = 0

        for name, param in self.config:
            if name == "conv2d":
                w, b = vars[idx], vars[idx + 1]
                # remember to keep synchrozied of forward_encoder and forward_decoder!
                x = F.conv2d(x, w, b, stride=param[4], padding=param[5])
                idx += 2
                # print(name, param, '\tout:', x.shape)
            elif name == "convt2d":
                w, b = vars[idx], vars[idx + 1]
                # remember to keep synchrozied of forward_encoder and forward_decoder!
                x = F.conv_transpose2d(x, w, b, stride=param[4], padding=param[5])
                idx += 2
                # print(name, param, '\tout:', x.shape)
            elif name == "linear":
                w, b = vars[idx], vars[idx + 1]
                x = F.linear(x, w, b)
                idx += 2
                # print('forward:', idx, x.norm().item())
            elif name == "bn":
                w, b = vars[idx], vars[idx + 1]
                running_mean, running_var = self.vars_bn[bn_idx], self.vars_bn[bn_idx + 1]
                x = F.batch_norm(
                    x, running_mean, running_var, weight=w, bias=b, training=bn_training
                )
                idx += 2
                bn_idx += 2

            elif name == "flatten":
                # print(x.shape)
                x = x.view(x.size(0), -1)
            elif name == "reshape":
                # [b, 8] => [b, 2, 2, 2]
                x = x.view(x.size(0), *param)
            elif name == "relu":
                x = F.relu(x, inplace=param[0])
            elif name == "leakyrelu":
                x = F.leaky_relu(x, negative_slope=param[0], inplace=param[1])
            elif name == "tanh":
                x = F.tanh(x)
            elif name == "sigmoid":
                x = torch.sigmoid(x)
            elif name == "upsample":
                x = F.upsample_nearest(x, scale_factor=param[0])
            elif name == "max_pool2d":
                x = F.max_pool2d(x, param[0], param[1], param[2])
            elif name == "avg_pool2d":
                x = F.avg_pool2d(x, param[0], param[1], param[2])

            else:
                raise NotImplementedError

        # make sure variable is used properly
        assert idx == len(vars)
        assert bn_idx == len(self.vars_bn)

        return x

    # def weight_clustering(self, weight_1, weight_2):
    #     # Flatten weight_1
    #     weight_flat = []
    #     for i in range(len(weight_1)):
    #         w = None
    #         for fw in weight_1[i]:
    #             if not torch.is_tensor(w):
    #                 w = torch.flatten(fw)
    #             else:
    #                 w = torch.cat([w, torch.flatten(fw)], dim=0)
    #         weight_flat.append(w)
    #     weight_flat = torch.stack(weight_flat, axis=1)
    #     # Flatten weight_2
    #     weight_flat_2 = []
    #     for i in range(len(weight_2)):
    #         w = None
    #         for fw in weight_2[i]:
    #             if not torch.is_tensor(w):
    #                 w = torch.flatten(fw)
    #             else:
    #                 w = torch.cat([w, torch.flatten(fw)], dim=0)
    #         weight_flat_2.append(w)
    #     weight_flat_2 = torch.stack(weight_flat_2, axis=1)

    #     #        diff = weight_flat - weight_flat_2
    #     st = torch.stack([weight_flat, weight_flat_2])
    #     diff = torch.norm(st, p="fro", dim=0)
    #     diff = torch.norm(diff, p="fro", dim=0)

    #     norm = torch.mean(diff)
    #     return norm

    def zero_grad(self, vars=None):
        """

        :param vars:
        :return:
        """
        with torch.no_grad():
            if vars is None:
                for p in self.vars:
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in vars:
                    if p.grad is not None:
                        p.grad.zero_()

    def parameters(self):
        """
        override this function since initial parameters will return with a generator.
        :return:
        """
        return self.vars


# class ResNet(nn.Module):
#     def __init__(self):
#         super(ResNet, self).__init__()
#         self.norm = nn.BatchNorm2d
#         self.conv = nn.Conv2d(...)  #
#         self.bn - nn.BatchNorm2d(...)   # 64

#     def get_block(self):
