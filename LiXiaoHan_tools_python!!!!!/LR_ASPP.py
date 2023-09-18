import torch
import torch.nn as nn
import torch.nn.functional as F
from bilinear_upsampling import BilinearUpSampling2D

class LiteRASSP(nn.Module):
    def __init__(self, input_shape, n_class=19, alpha=1.0, weights=None, backbone='small'):
        super(LiteRASSP, self).__init__()
        self.shape = input_shape
        self.n_class = n_class
        self.alpha = alpha
        self.weights = weights
        self.backbone = backbone

        # extract backbone
        if self.backbone == 'large':
            from model.mobilenet_v3_large import MobileNetV3_Large

            self.backbone_model = MobileNetV3_Large(self.shape, self.n_class, alpha=self.alpha,
                                                    include_top=False).build()
            layer_name8 = 'batch_normalization_13'
            layer_name16 = 'add_5'
        elif self.backbone == 'small':
            from model.mobilenet_v3_small import MobileNetV3_Small

            self.backbone_model = MobileNetV3_Small(self.shape, self.n_class, alpha=self.alpha,
                                                    include_top=False).build()
            layer_name8 = 'batch_normalization_7'
            layer_name16 = 'add_2'
        else:
            raise Exception('Invalid backbone: {}'.format(self.backbone))

        if self.weights is not None:
            self.backbone_model.load_state_dict(torch.load(self.weights), strict=False)

        self.out_feature8 = None
        self.out_feature16 = None
        self._set_out_features(layer_name8, layer_name16)

        # define layers
        self.conv1 = nn.Conv2d(in_channels=self.out_feature16.shape[1], out_channels=128, kernel_size=(1, 1))
        self.bn1 = nn.BatchNorm2d(128)
        self.relu1 = nn.ReLU()

        self.ap = nn.AvgPool2d(kernel_size=(49, 49), stride=(16, 20))
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1, 1))
        self.sigmoid = nn.Sigmoid()

        x2_s = self.out_feature16.shape
        self.bilinear_upsampling2d = BilinearUpSampling2D(target_size=(int(x2_s[2]), int(x2_s[3])))

        self.conv3 = nn.Conv2d(in_channels=self.out_feature8.shape[1], out_channels=self.n_class, kernel_size=(1, 1))

        self.multiply = Multiply()
        self.bilinear_upsampling2d_2 = BilinearUpSampling2D(size=(2, 2))
        self.conv4 = nn.Conv2d(in_channels=self.n_class, out_channels=self.n_class, kernel_size=(1, 1))

        self.add = Add()
        self.softmax = nn.Softmax(dim=1)

    def _set_out_features(self, layer_name8, layer_name16):
        feature8, feature16 = None, None
        for m in self.backbone_model.modules():
            if m._get_name() == layer_name8:
                feature8 = m
            elif m._get_name() == layer_name16:
                feature16 = m
            if feature8 is not None and feature16 is not None:
                break
        self.out_feature8 = feature8
        self.out_feature16 = feature16

    def forward(self, x):
        # pass inputs through backbone
        x8 = self.out_feature8(x)
        x16 = self.out_feature16(x)

        # branch1
        x1 = self.conv1(x16)
        x1 = self.bn1(x1)
        x1 = self.relu1(x1)

        # branch2
        s = x1.shape
        x2 = self.ap(x16)
        x2 = self.conv2(x2)
        x2 = self.sigmoid(x2)
        x2 = self.bilinear_upsampling2d(x2)

        # branch3
        x3 = self.conv3(x8)

        # merge1
        x = self.multiply([x1, x2])
        x = self.bilinear_upsampling2d_2(x)
        x = self.conv4(x)

        # merge2
        x = self.add([x, x3])

        # out
        x = self.softmax(x)

        return x