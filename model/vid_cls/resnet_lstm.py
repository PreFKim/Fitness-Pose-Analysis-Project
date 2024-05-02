import torch
import torch.nn as nn


class bottleneck_block(nn.Module):
    def __init__(self,i,o,s,e,stage):
        super(bottleneck_block,self).__init__()

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

    def forward(self,x):

        identity = self.identity(x)
        out = self.conv1(x)
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


class ResNet(nn.Module): #정확성 향상을 위해 resnet 101로도 고려해볼것
    def __init__(self,image_features,e=4,num_layers=[3,4,6,3]): #50 [3,4,6,3]101은 [3,4,23,3 ]152는 [3,8,36,3]
        super(ResNet,self).__init__()
        def n_blocks(i,o,s,stage):
            layers = []
            layers.append(bottleneck_block(i,o,s,e,stage))

            for _ in range(1,num_layers[stage]):
                layers.append(bottleneck_block(o*e,o,1,e,stage))

            return nn.Sequential(*layers)


        self.conv1 = nn.Sequential(
            nn.Conv2d(3,64,7,2,3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3,2,1)
        )

        self.stage1 = n_blocks(64,64,1,0)
        self.stage2 = n_blocks(64*e,128,2,1)
        self.stage3 = n_blocks(128*e,256,2,2)
        self.stage4 = n_blocks(256*e,512,2,3)

        self.F = nn.AdaptiveAvgPool2d(1)

        self.image_features = image_features
        self.bottleneck = nn.Linear(512*e, self.image_features)


    def forward(self,x,cls_label=None):
        b,c,h,w = x.size()

        out = self.conv1(x)

        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)

        out = self.F(out)

        out = out.view(out.size(0),-1) #(b*t,512*e)

        #out = self.bottleneck(out)

        return {"feature":out}

class Resnet_LSTM(nn.Module):
    def __init__(self, image_features=512*4, hidden_size=512, output_size=12, num_layers=1, dropout=0, bidirectional=True):

        if bidirectional:
            self.name = "BI-LSTM"
        else:
            self.name = "Uni-LSTM"
        super(Resnet_LSTM, self).__init__()


        self.D = (1 + bidirectional)

        self.num_layers = num_layers

        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(image_features, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout, bidirectional=bidirectional)

        self.fc = nn.Sequential(
            nn.Linear(self.D*hidden_size, self.D*hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.D*hidden_size, output_size)
        )

        self.extract_from_image = nn.Sequential(
            ResNet(image_features)
        )

        self.embed_hidden = nn.Embedding(5,self.hidden_size)

    def forward(self,x ):
        out = {}
        b, t, c, h, w = x.shape #batch, frame, points, coord
        x = x.view(b*t,c,h,w)

        hidden = self.init_hidden(b, x.device)

        feature = self.extract_from_image(x)
        feature = feature['feature'].view(b,t,-1)
        lstm_out, hidden = self.lstm(feature, hidden)

        out['feature'] = lstm_out[:, -1, :]

        out['class'] = self.fc(out['feature'])

        return out

    # def init_hidden(self,view , device):
    #     hidden = torch.tile(self.embed_hidden(view).unsqueeze(0),(self.D*self.num_layers,1,1))
    #     # batch, -> batch,hidden_size -> 1,batch,hidden_size -> (self.D*self.num_layers,batch,hidden_size
    #     return (hidden,hidden)
    def init_hidden(self,batch_size , device):
        return (torch.zeros(self.D*self.num_layers, batch_size, self.hidden_size, device=device),
                torch.zeros(self.D*self.num_layers, batch_size, self.hidden_size, device=device))