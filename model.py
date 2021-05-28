import torch
import torch.nn as nn
import blocks
from collections import OrderedDict
import math

class TPSR(nn.Module):
    def __init__(self, n_feats=16, scale=4):
        super(TPSR, self).__init__()

        self.sub_mean = blocks.MeanShift()
        self.add_mean = blocks.MeanShift(sign=1)
        self.first_layer = nn.Conv2d(3, n_feats, 3, padding=1)

        self.early_layers = nn.Sequential(OrderedDict([
            ('Node_0_Operation_0_type_Sep2d_5x5', blocks.BaseConv(n_feats, 5, True)),
            ('Node_1_Operation_0_type_Conv2d_7x7', blocks.BaseConv(n_feats, 7, False)),
            ('Node_2_Operation_0_type_Conv2d_7x7', blocks.BaseConv(n_feats, 7, False))
        ]))

        self.middle_main_layers = nn.Sequential(OrderedDict([
            ('Node_4_Operation_0_type_Conv2d_3x3', blocks.BaseConv(n_feats, 3, False)),
            ('Node_5_Operation_0_type_Conv2d_5x5', blocks.BaseConv(n_feats, 5, False)),
            ('Node_6_Operation_0_type_Sep2d_3x3', blocks.BaseConv(n_feats, 3, True))
        ]))

        self.middle_side_layers = nn.Sequential(OrderedDict([
            ('Node_3_Operation_0_type_Conv2d_3x3', blocks.BaseConv(n_feats, 3, False)),
            ('Node_7_Operation_0_type_Conv2d_5x5', blocks.BaseConv(n_feats, 5, False)),
            ('Node_8_Operation_0_type_Conv2d_3x3', blocks.BaseConv(n_feats, 3, False))
        ]))

        self.skip_layer = nn.Sequential(OrderedDict([
            ('Node_9_Operation_0_type_Conv2d_7x7', blocks.BaseConv(n_feats, 7, False))
        ]))


        no_of_upsampling = int(math.log(scale,2))
        up_mod = [blocks.Upsamplingx2(n_feats)]
        for _ in range(no_of_upsampling - 1):
            up_mod.append(blocks.Upsamplingx2(3))
        self.merge_op_act = nn.PReLU()
        self.up_mod = nn.Sequential(*up_mod)        

    def forward(self, input):
        i = self.sub_mean(input)
        i = self.first_layer(i)
        after_n2 = self.early_layers(i)
        after_n6 = self.middle_main_layers(after_n2)
        after_n8 = self.middle_side_layers(after_n2)
        after_n9 = self.skip_layer(i)
        after_merge = self.merge_op_act(after_n9 + after_n8 + after_n6)

        o = self.up_mod(after_merge)
        return self.add_mean(o)