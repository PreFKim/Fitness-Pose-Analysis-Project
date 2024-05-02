from cfg import IMAGE_HEIGHT, IMAGE_WIDTH
import torch

def MPJPE_heatmap(pred,true,image_size=[IMAGE_WIDTH,IMAGE_HEIGHT],mean=True): #Mean Per Joint Position Error ( 낮은게 좋음 )
    with torch.no_grad():
        b,c,h,w = pred.shape

        pred = pred.reshape(b,c,h*w)

        pred_flatten_idx = torch.argmax(pred,-1).reshape(b,c,1) # heatmap에서 제일 큰 값이 있는 index
        pred_xy = torch.cat([pred_flatten_idx%w/w,pred_flatten_idx//w/h],-1) # b, 24, 2 [x,y]

        wh = torch.tensor(image_size,dtype=torch.float32).to(pred.device)

        metric = torch.mean(torch.sqrt(torch.sum(((pred_xy-true)*wh)**2,-1)+1e-5),1)

        if mean:
            metric = torch.mean(metric)



    return metric

def MPJPE_coord(pred,true,image_size=[IMAGE_WIDTH,IMAGE_HEIGHT],mean= True): #Mean Per Joint Position Error ( 낮은게 좋음 )
    with torch.no_grad():
        b,j,c = pred.shape

        wh = torch.tensor(image_size,dtype=torch.float32).to(pred.device)

        metric = torch.mean(torch.sqrt(torch.sum(((pred-true)* wh)**2,-1)+1e-5),1)
        if mean:
            metric = torch.mean(metric)

    return metric


def accuracy(pred, label):
    with torch.no_grad():
        _,pred = torch.max(pred, dim=1)
        accuracy = torch.sum(pred==label)/pred.shape[0]
    return accuracy