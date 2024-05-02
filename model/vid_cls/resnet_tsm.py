import torch
import torch.nn as nn

class bottleneck_block_tsm(nn.Module):
    def __init__(self,i,o,s,e,stage,frame=8,div=8):
        super(bottleneck_block_tsm,self).__init__()

        self.frame = frame
        self.div = div
        self.conv1 = nn.Conv2d(i,o,1,s)
        self.bn1 = nn.BatchNorm2d(o)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(o,o,3,1,1)
        self.bn2 = nn.BatchNorm2d(o)
        self.conv3 = nn.Conv2d(o,o*e,1,1)
        self.bn3 = nn.BatchNorm2d(o*e)
        if s == 2 or i==o:
          self.identity = nn.Sequential(
              nn.Conv2d(i,o*e,1,s),
              nn.BatchNorm2d(o*e)
          )
        else :
          self.identity = nn.Sequential()

    def shift(self,x,frame=8,div=8):
        bt,c,h,w = x.size()
        b = bt//frame
        t = frame

        div_c = c//div

        x = x.view(b,t,c,h,w)
        ret = torch.zeros_like(x)
        #ret = x.detach().clone()

        #bi-shift
        ret[:,:-1,:div_c,:,:] = x[:,1:,:div_c,:,:]
        ret[:,1:,div_c:div_c*2,:,:] = x[:,:-1,div_c:div_c*2,:,:]
        ret[:,:,div_c*2:,:,:] = x[:,:,div_c*2:,:,:]

        ret = ret.view(bt,c,h,w)
        return ret

    def forward(self,x):

        identity = self.identity(x)
        out = self.shift(x,self.frame,self.div)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)


        out += identity
        out = self.relu(out)

        return out


class ResNet_TSM(nn.Module): #정확성 향상을 위해 resnet 101로도 고려해볼것
    def __init__(self,e=4,num_layers=[3,4,6,3],frame=8,div=8,heatmap=False, num_classes = 12): #50 [3,4,6,3]101은 [3,4,23,3 ]152는 [3,8,36,3]
        super(ResNet_TSM,self).__init__()
        self.name = "TSM"
        def n_blocks(i,o,s,stage):
            layers = []
            layers.append(bottleneck_block_tsm(i,o,s,e,stage,frame=frame,div=div))

            for _ in range(1,num_layers[stage]):
                layers.append(bottleneck_block_tsm(o*e,o,1,e,stage,frame=frame,div=div))

            return nn.Sequential(*layers)


        self.conv1 = nn.Sequential(
            nn.Conv2d(3+24*heatmap,64,7,2,3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3,2,1)
        )

        self.stage1 = n_blocks(64,64,1,0)
        self.stage2 = n_blocks(64*e,128,2,1)
        self.stage3 = n_blocks(128*e,256,2,2)
        self.stage4 = n_blocks(256*e,512,2,3)

        self.F = nn.AdaptiveAvgPool2d(1)

        self.FC_cls = nn.Sequential(
            nn.Linear(512*e,num_classes) #운동 종류들() , 자세(5) ,활동상태(1,s)
        )


    def forward(self,x,cls_label=None):
        b,t,c,h,w = x.size()

        out = x.view(b*t,c,h,w)

        out = self.conv1(out)

        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)

        out = self.F(out)

        out = out.view(out.size(0),-1) #(b*t,512*e)

        out_cls = self.FC_cls(out).view(b,t,-1).mean(1)  # b*t,len(classes) -> b, t, len(classes) -> b, len(classes)

        return {"feature":out, "class":out_cls}