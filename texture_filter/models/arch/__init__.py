# Add your custom network here
import os

from models.arch.smoothing_nafnet import NAFNet
def smooth_nafnet(in_channels, out_channels):
    img_channel = 3
    width = 32

    enc_blks = [2, 2, 2, 2]
    middle_blk_num = 2
    dec_blks = [2, 2, 2, 2]

    print('enc blks', enc_blks, 'middle blk num', middle_blk_num, 'dec blks', dec_blks, 'width', width)
    return NAFNet(img_channel=img_channel, width=width, middle_blk_num=middle_blk_num,
                      enc_blk_nums=enc_blks, dec_blk_nums=dec_blks)
