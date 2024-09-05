# import torch
# from modules.lpips import LPIPS

# lpips = LPIPS('vgg')

# tracer = torch.jit.trace(lpips, (torch.rand(1, 3, 64, 64), torch.rand(1, 3, 64, 64)))

# # Save the model
# tracer.save("lpips.pt")

import torch
import torch.nn as nn

from modules.networks import get_network, LinLayers
from modules.utils import get_state_dict


net_type = 'vgg'

# pretrained network
net = get_network(net_type)

# print(net(torch.rand(1, 3, 64, 64)))

# linear layers
lin = LinLayers(net.n_channels_list)
lin.load_state_dict(get_state_dict(net_type))

lpips = nn.Sequential(net, lin)
lpips = lpips.eval()

# lpips(torch.rand(1, 3, 64, 64))
    # def forward(self, x: torch.Tensor, y: torch.Tensor):
    #     feat_x, feat_y = self.net(x), self.net(y)

    #     diff = [(fx - fy) ** 2 for fx, fy in zip(feat_x, feat_y)]
    #     res = [l(d).mean((2, 3), True) for d, l in zip(diff, self.lin)]

    #     return torch.sum(torch.cat(res, 0), 0, True)
