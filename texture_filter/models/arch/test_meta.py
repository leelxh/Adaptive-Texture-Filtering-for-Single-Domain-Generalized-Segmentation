
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from functools import partial
from functorch import vmap

class MetaConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, groups=1, bias=True, 
                act_func=nn.Identity):
        super(MetaConv2d, self).__init__()
        self.linear_w = nn.Linear(1, in_channels * out_channels // groups * kernel_size * kernel_size, bias = True)
        self.linear_b = nn.Linear(1, out_channels) if bias else (lambda dumb : None)
        self.reshape = lambda para : para.reshape(out_channels, in_channels, kernel_size, kernel_size)
        self.fwd_transform = lambda lamb, img : F.conv2d(img, 
                                                    self.reshape(self.linear_w(lamb)),
                                                    self.linear_b(lamb),
                                                    stride=stride, padding=padding)        
        self.act_func = act_func()
    
    def forward(self, x, lamb):
        return self.act_func(vmap(partial(self.fwd_func, lamb))(x))

    def forward_d(self, x, lamb):
        return torch.stack([self.fwd_func(lamb, item) for item in x.unbind(0)])


meta_conv = MetaConv(3, 3).to("cuda")
meta_conv_2 = MetaConv(3, 3).to("cuda")
meta_conv_2.load_state_dict(meta_conv.state_dict())
lamb = torch.tensor((1., ), device='cuda')
loss_1, loss_2 = nn.L1Loss().to("cuda"), nn.L1Loss().to("cuda")
batch_size = 10
x = torch.randn(batch_size, 3, 224, 224).to('cuda')
y = x.detach().clone().to('cuda')




x_o = meta_conv.forward_d(x, lamb)
l_1 = loss_1(x_o, x)
l_1.backward()
y_o = meta_conv_2.forward(y, lamb)
l_2 = loss_2(y_o, y)
l_2.backward()




print(torch.allclose(meta_conv.linear_w.weight.grad, meta_conv_2.linear_w.weight.grad))




