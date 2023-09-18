import torch
import torch.nn as nn
import torch.nn.functional as F

class BilinearUpSampling2D(nn.Module):
    def __init__(self, size=(1, 1), target_size=None, data_format='default', **kwargs):
        super(BilinearUpSampling2D, self).__init__()
        if data_format == 'default':
            data_format = 'channels_last'
        self.size = tuple(size)

        if target_size is not None:
            self.target_size = tuple(target_size)
        else:
            self.target_size = None
        assert data_format in {'channels_last', 'channels_first'}, 'data_format must be in {tf, th}'

        self.data_format = data_format

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_first':
            width = int(self.size[0] * input_shape[2] if input_shape[2] is not None else None)
            height = int(self.size[1] * input_shape[3] if input_shape[3] is not None else None)

            if self.target_size is not None:
                width = self.target_size[0]
                height = self.target_size[1]
            return (input_shape[0],
                    input_shape[1],
                    width,
                    height)
        elif self.data_format == 'channels_last':
            width = int(self.size[0] * input_shape[1] if input_shape[1] is not None else None)
            height = int(self.size[1] * input_shape[2] if input_shape[2] is not None else None)

            if self.target_size is not None:
                width = self.target_size[0]
                height = self.target_size[1]
            return (input_shape[0],
                    width,
                    height,
                    input_shape[3])
        else:
            raise Exception('Invalid data_format: ' + self.data_format)

    def forward(self, x):
        if self.target_size is not None:
            return F.interpolate(x, size=self.target_size, mode='bilinear', align_corners=True)
        else:
            return F.interpolate(x, scale_factor=self.size, mode='bilinear', align_corners=True)