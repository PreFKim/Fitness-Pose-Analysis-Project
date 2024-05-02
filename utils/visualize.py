import torch
import cv2
import numpy as np

from cfg import point_name, linked_name, palette


def draw_joint(img,pts,rad=3,line=1):
    out = img.copy()
    h,w,c = img.shape
    used = np.zeros_like(point_name,dtype=np.int32())

    if torch.max(pts) <=1:
        pts = pts * torch.tensor([w,h])

    for i,se in enumerate(linked_name):
        from_xy = (int(pts[se[0]][0]),int(pts[se[0]][1]))
        to_xy = (int(pts[se[1]][0]),int(pts[se[1]][1]))

        out = cv2.line(out,from_xy,to_xy,[255,255,255],line,cv2.LINE_AA)
        if (used[se[0]]==0):
            out= cv2.circle(out,from_xy,rad,palette[se[0]],-1,cv2.LINE_AA)
            used[se[0]] = 1

        if (used[se[1]]==0):
            out= cv2.circle(out,to_xy,rad,palette[se[1]],-1,cv2.LINE_AA)
            used[se[1]] = 1

    return out

def denormalization(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):

    out_img =  img * std  + mean
    return out_img


def get_xy(heatmap):
    if (len(heatmap.shape)==4):
        b,c,h,w = heatmap.shape
        dim = 2
    else :
        c,h,w = heatmap.shape
        dim = 1

    heatmap_f = torch.flatten(heatmap,dim)
    heatmap_f = torch.argmax(heatmap_f,dim)
    xy = torch.stack([heatmap_f%w/w,heatmap_f//w/h],dim)
    return xy