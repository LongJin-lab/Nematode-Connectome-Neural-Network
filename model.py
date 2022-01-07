import torch
import torch.nn as nn
import torch.nn.functional as F

from connectomeblock import ConnectomeBlock


class Model(nn.Module):
    def __init__(self, node_num, p, in_channels, out_channels, graph_mode, graph_type, dataset_mode, is_train):
        super(Model, self).__init__()
        self.node_num = node_num
        self.p = p
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.graph_mode = graph_mode
        self.graph_type = graph_type
        self.is_train = is_train
        self.dataset_mode = dataset_mode

        self.num_classes = 1000
        self.dropout_rate = 0.2

        if self.dataset_mode == "CIFAR10":
            self.num_classes = 10
        elif self.dataset_mode == "CIFAR100":
            self.num_classes = 100
        elif self.dataset_mode == "IMAGENET":
            self.num_classes = 1000
        elif self.dataset_mode == "MNIST":
            self.num_classes = 10

        self.CIFAR_conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=self.out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU()
        )
        self.CIFAR_conv2 = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.out_channels),
        )
        # self.CIFAR_conv2 = nn.Sequential(
        #     RandWire(self.node_num, self.p, self.in_channels, self.out_channels, self.graph_mode, self.is_train, name="CIFAR10_conv2")
        # )
        self.CIFAR_conv3 = nn.Sequential(
            ConnectomeBlock(self.node_num, self.p, self.in_channels, self.out_channels * 2, self.graph_mode, self.graph_type, self.is_train, name="CIFAR10_conv3")
        )
        self.CIFAR_conv4 = nn.Sequential(
            ConnectomeBlock(self.node_num, self.p, self.in_channels * 2, self.out_channels * 4, self.graph_mode, self.graph_type, self.is_train, name="CIFAR10_conv4")
        )

        self.CIFAR_classifier = nn.Sequential(
            nn.Conv2d(self.in_channels * 4, 1280, kernel_size=1),
            nn.BatchNorm2d(1280)
        )

        self.output = nn.Sequential(
            nn.Dropout(self.dropout_rate),
            nn.Linear(1280, self.num_classes)
        )

    def forward(self, x):
        out = self.CIFAR_conv1(x)
        out = self.CIFAR_conv2(out)
        out = self.CIFAR_conv3(out)
        out = self.CIFAR_conv4(out)
        out = self.CIFAR_classifier(out)

        # global average pooling
        batch_size, channels, height, width = out.size()
        out = F.avg_pool2d(out, kernel_size=[height, width])
        out = torch.squeeze(out)
        out = self.output(out)

        return out
