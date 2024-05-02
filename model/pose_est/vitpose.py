import torch
import torch.nn as nn

from cfg import point_name

from .backbone_vit import ViT_b, ViT_l

class ViTpose(nn.Module):
    def __init__(self,backbone='b',mode = "heatmap"):
        super(ViTpose,self).__init__()
        if backbone == 'b':
            self.backbone = ViT_b()
            self.out_channels = 768
        elif backbone == 'l':
            self.backbone = ViT_l()
            self.out_channels = 1024

        self.view = nn.Linear(self.out_channels,5)


        self.mode = mode
        if self.mode == "heatmap" :
            self.head = nn.Sequential(
                nn.ReLU(),
                nn.Upsample(scale_factor=4,mode='bilinear'),
                nn.Conv2d(self.out_channels,len(point_name),3,1,1)
            )
        elif self.mode == "both":
            def group_conv(i,o,k,s,p,g,n):

                layers = []

                layers.append(nn.Conv2d(i,o,k,s,p,groups=g))
                layers.append(nn.ReLU())

                for i in range(n-1):
                    layers.append(nn.Conv2d(o,o,k,s,p,groups=g))
                    layers.append(nn.ReLU())
                return nn.Sequential(*layers)

            def offset_block(in_channels,n):
                layers = []
                for i in range(n):
                    layers.append(group_conv(in_channels*(2**i),in_channels*(2**(i+1)),3,1,1,g=len(point_name),n=2))
                    layers.append(nn.MaxPool2d(2,2))
                return nn.Sequential(*layers)

            self.head = nn.Sequential(
                nn.ReLU(),
                nn.Upsample(scale_factor=4,mode='bilinear'),
                nn.Conv2d(self.out_channels,len(point_name),3,1,1)
            )

            #Method2,3
            self.offset_head = nn.Sequential(
                offset_block(len(point_name),4),
                nn.Conv2d(len(point_name)*(2**4),len(point_name)*2,(4,3),1,groups=len(point_name))
            )
        else :
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.head = nn.Sequential(
                nn.ReLU(),
                nn.Linear(self.out_channels,len(point_name)*2)
            )



    def forward(self,x,view=None):

        b,_,_,_ = x.shape

        out = {}
        xp,cls = self.backbone(x,view)
        out['feature'] = xp

        if cls is not None:
            out['view'] = self.view(cls)

        if self.mode == "heatmap":
            out['heatmap'] = self.head(out['feature'])
        elif self.mode == "both":
            out['heatmap'] = self.head(out['feature'])

            with torch.no_grad():
                bridge = torch.zeros_like(out['heatmap'])
                bridge.set_(out['heatmap'])

            #Method 2,3
            out['coord'] = self.offset_head(bridge).view(b,-1,2)
        else :
            out['coord'] = self.pool(out['feature']).view(b,-1)
            out['coord'] = self.head(out['coord']).view(b,-1,2)

        return out