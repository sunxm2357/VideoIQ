
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import temporal_unpooling

class FPN(nn.Module):

    def __init__(self, fpn_dim):
        super().__init__()
        self.fpn_dim = fpn_dim
        # Top layer, Reduce channels
        self.toplayer = nn.Sequential(
            nn.Conv2d(2048, fpn_dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(fpn_dim)
        )
        self.toplayer_relu = nn.ReLU(inplace=True)
        # Lateral layers
        self.latlayer1 = nn.Sequential(
            nn.Conv2d(1024, fpn_dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(fpn_dim)
        )
        self.latlayer1_relu = nn.ReLU(inplace=True)
        self.latlayer2 = nn.Sequential(
            nn.Conv2d(512, fpn_dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(fpn_dim)
        )
        self.latlayer2_relu = nn.ReLU(inplace=True)
        self.latlayer3 = nn.Sequential(
            nn.Conv2d(256, fpn_dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(fpn_dim)
        )
        self.latlayer3_relu = nn.ReLU(inplace=True)
        # Smooth layers
        self.smooth1 = nn.Sequential(
            nn.Conv2d(fpn_dim, fpn_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(fpn_dim),
            nn.ReLU(inplace=True)
        )
        self.smooth2 = nn.Sequential(
            nn.Conv2d(fpn_dim, fpn_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(fpn_dim),
            nn.ReLU(inplace=True)
        )
        self.smooth3 = nn.Sequential(
            nn.Conv2d(fpn_dim, fpn_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(fpn_dim),
            nn.ReLU(inplace=True)
        )

    def _upsample_add(self, x, y, batch_size=None):
        n_y, _, h_y, w_y = y.size()
        n_x = x.size()[0]
        if n_x == n_y:
            return F.interpolate(x, size=(h_y, w_y), mode='bilinear', align_corners=True) + y
        else:
            return temporal_unpooling(x, n_x // batch_size, (n_y // batch_size, h_y, w_y)) + y

    def forward(self, fp2, fp3, fp4, fp5, batch_size):
        fp5 = self.toplayer(fp5)
        fp4 = self.smooth1(self.latlayer1_relu(self._upsample_add(fp5, self.latlayer1(fp4), batch_size)))
        fp3 = self.smooth2(self.latlayer2_relu(self._upsample_add(fp4, self.latlayer2(fp3), batch_size)))
        fp2 = self.smooth3(self.latlayer3_relu(self._upsample_add(fp3, self.latlayer3(fp2), batch_size)))
        fp5 = self.toplayer_relu(fp5)

        # FxHxW
        fp2 = F.adaptive_avg_pool2d(fp2, 1).view(-1, self.fpn_dim)
        # F/2xH/2xW/2
        fp3 = F.adaptive_avg_pool2d(fp3, 1).view(-1, self.fpn_dim)
        # F/4xH/4xW/4
        fp4 = F.adaptive_avg_pool2d(fp4, 1).view(-1, self.fpn_dim)
        # F/8xH/8xW/8
        fp5 = F.adaptive_avg_pool2d(fp5, 1).view(-1, self.fpn_dim)
        # out = torch.cat((fp2, fp3, fp4, fp5), dim=1)
        # out = out.view(-1, out.shape[-1])
        return fp2, fp3, fp4, fp5