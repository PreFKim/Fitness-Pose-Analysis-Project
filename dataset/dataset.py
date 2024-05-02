
import torch
from torch.utils.data import Dataset

import albumentations as A
import numpy as np

from albumentations.pytorch import ToTensorV2

import os, cv2

from cfg import classes, IMAGE_HEIGHT, IMAGE_WIDTH


class Pose(Dataset):
    def __init__(self, data, views = [1, 2, 3, 4, 5], training = True, frame_mode=1): # ,frame_mode=0 0번프레임만 1 랜덤 2 전체

        self.data = data

        self.views = [view-1 for view in views]

        self.training = training
        self.frame_mode = frame_mode

        self.len_data = 0
        self.idx_list = []
        for view in self.views:
            for idx in range(len(self.data[view])):
                if (self.frame_mode<=1):
                    self.idx_list.append([view,idx,-1])
                    self.len_data += 1
                else:
                    for frame in range(len(self.data[view][idx][2])):
                        self.idx_list.append([view,idx,frame])
                        self.len_data += 1

        if training :
            transform_list = [
                A.ColorJitter(p=0.5),
                #A.HorizontalFlip(p=0.5),
                A.Affine(mode=0),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ]
        else :
            transform_list = [A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                              ToTensorV2()]

        self.transforms = A.Compose(transform_list, keypoint_params=A.KeypointParams(format='xy',remove_invisible=False))


    def __len__(self):
        return self.len_data

    def gaussian_heatmap(self, center,img_size = (224,224),sig=5):
        x_axis = np.linspace(0,img_size[0]-1,img_size[0])-center[0]
        y_axis = np.linspace(0,img_size[1]-1,img_size[1])-center[1]
        xx,yy = np.meshgrid(x_axis,y_axis)
        kernel = np.exp(-0.5*(np.square(xx)+np.square(yy))/np.square(sig))
        return kernel #좌표 오버나면 알아서 없애줌

    def __getitem__(self, idx):

        view_idx, data_idx, frame_idx = self.idx_list[idx]

        if self.frame_mode==0:
            frame_idx = 0
        elif self.frame_mode==1:
            frame_idx = np.random.randint(0,len(self.data[view_idx][data_idx][2]))

        x1,y1,x2,y2 = self.data[view_idx][data_idx][1]

        #사전에 미리 Crop한 이미지 x1,y1,x2,y2
        crop_min_x = max(0,x1-200)
        crop_min_y = max(0,y1-200)
        crop_max_x = min(1920,x2+200)
        crop_max_y = min(1080,y2+200)

        path = self.data[view_idx][data_idx][0]

        if (os.path.exists(path)==False):
            print(path,"가 없습니다.")


        cap = cv2.VideoCapture(path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx*2) # 프레임 시작위치를 idx 프레임으로 옮김
        ret, img = cap.read()
        cap.release()

        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img = cv2.resize(img,(IMAGE_WIDTH,IMAGE_HEIGHT))

        coord = np.array(self.data[view_idx][data_idx][2][frame_idx]) - [crop_min_x,crop_min_y]

        coord = coord * [IMAGE_WIDTH/(crop_max_x-crop_min_x),IMAGE_HEIGHT/(crop_max_y-crop_min_y)]

        transformed = self.transforms(image=img,keypoints=coord)
        img = transformed['image']
        coord = transformed['keypoints']

        heatmap = np.zeros((IMAGE_HEIGHT,IMAGE_WIDTH,24),dtype= np.float32)



        for i in range(24):
            heatmap[:,:,i] = self.gaussian_heatmap((np.round(coord[i][0],0),np.round(coord[i][1],0)),(IMAGE_WIDTH,IMAGE_HEIGHT))

        coord = torch.tensor(coord,dtype=torch.float32) / torch.tensor([IMAGE_WIDTH,IMAGE_HEIGHT])

        heatmap = cv2.resize(heatmap,(IMAGE_WIDTH//4,IMAGE_HEIGHT//4))
        heatmap = torch.from_numpy(heatmap)
        heatmap = torch.permute(heatmap,(2,0,1))

        dataset = {"idx" : [view_idx, data_idx, frame_idx],
                   "image" : img,
                   "coord" : coord,
                   "heatmap" : heatmap,
                   "view" : view_idx,
                   "class" : self.data[view_idx][data_idx][3],
                   "type" : self.data[view_idx][data_idx][4],
                   "pose" : self.data[view_idx][data_idx][5],
                   "label_path" : self.data[view_idx][data_idx][6]}


        return dataset



class Video(Dataset):
    def __init__(self, data, num_frames = 8, views = [2, 3, 4], training = True, frame_mode=1):

        self.data = data

        self.views = [view-1 for view in views]

        self.len_data = 0
        self.idx_list = []
        self.num_frames = num_frames
        self.training = training
        self.frame_mode = frame_mode

        for cls in range(len(classes)):
            for view in self.views:
                for idx in range(len(self.data[cls][view])):
                    if (self.frame_mode<=1):
                        self.idx_list.append([cls,view,idx,-1])
                        self.len_data += 1
                    else:
                        for frame in range(max(1,len(self.data[cls][view][idx][3])-self.num_frames+1)):
                            self.idx_list.append([cls,view,idx,frame])
                            self.len_data += 1

        if training :
            transform_list = [
                A.ColorJitter(p=0.5),
                #A.HorizontalFlip(p=0.5),
                A.Affine(mode=0),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ]
        else :
            transform_list = [A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                              ToTensorV2()]

        additional_targets = {}
        for i in range(1,self.num_frames):
            additional_targets[f'image{i}'] = "image"
            additional_targets[f'keypoints{i}'] = "keypoints"
        self.transforms = A.Compose(transform_list, additional_targets = additional_targets,keypoint_params=A.KeypointParams(format='xy',remove_invisible=False))


    def __len__(self):
        return self.len_data

    def gaussian_heatmap(self, center,img_size = (224,224),sig=5):
        x_axis = np.linspace(0,img_size[0]-1,img_size[0])-center[0]
        y_axis = np.linspace(0,img_size[1]-1,img_size[1])-center[1]
        xx,yy = np.meshgrid(x_axis,y_axis)
        kernel = np.exp(-0.5*(np.square(xx)+np.square(yy))/np.square(sig))
        return kernel #좌표 오버나면 알아서 없애줌

    def __getitem__(self, idx):

        cls_idx, view_idx, data_idx, frame_idx = self.idx_list[idx]

        if self.frame_mode==0:
            frame_idx = 0
        elif self.frame_mode==1:
            if len(self.data[cls_idx][view_idx][data_idx][3])-self.num_frames+1 <= 0:
                frame_idx = 0
            else:
                frame_idx = np.random.randint(0,len(self.data[cls_idx][view_idx][data_idx][3])-self.num_frames+1)

        x1,y1,x2,y2 = self.data[cls_idx][view_idx][data_idx][2]

        #사전에 미리 Crop한 이미지 x1,y1,x2,y2
        crop_min_x = max(0,x1-200)
        crop_min_y = max(0,y1-200)
        crop_max_x = min(1920,x2+200)
        crop_max_y = min(1080,y2+200)


        path = self.data[cls_idx][view_idx][data_idx][0]



        if (os.path.exists(path)==False):
            print(path,"가 없습니다.")

        imgs = []
        coord = []
        cap = cv2.VideoCapture(path)
        for idx in range(frame_idx,frame_idx+self.num_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx*2) # 프레임 시작위치를 idx 프레임으로 옮김
            ret, frame = cap.read()

            if ret == False:
                break

            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame,(IMAGE_WIDTH,IMAGE_HEIGHT))
            imgs.append(frame)
            pts = np.array(self.data[cls_idx][view_idx][data_idx][3][idx]) - [crop_min_x,crop_min_y]
            pts = pts * [IMAGE_WIDTH/(crop_max_x-crop_min_x),IMAGE_HEIGHT/(crop_max_y-crop_min_y)]
            coord.append(pts)
        cap.release()

        imgs = np.array(imgs)
        coord = np.array(coord) # frame,24,2


        if len(imgs) < self.num_frames:
            imgs = np.concatenate([np.zeros((self.num_frames-len(imgs),IMAGE_HEIGHT,IMAGE_WIDTH,3),dtype=np.uint8),imgs],0)
            coord = np.concatenate([np.zeros((self.num_frames-len(imgs),24,2),dtype=np.float32)-255,coord],0)


        input_image = {}
        for i in range(len(imgs)):
            if i == 0:
                input_image['image'] = imgs[0]
                input_image['keypoints'] = coord[0]
            else:
                input_image[f'image{i}'] = imgs[i]
                input_image[f'keypoints{i}'] = coord[i]
        augmentations = self.transforms(**input_image)



        imgs = torch.zeros((self.num_frames,3,IMAGE_HEIGHT,IMAGE_WIDTH),dtype=torch.float32)
        heatmap = np.zeros((self.num_frames,IMAGE_HEIGHT,IMAGE_WIDTH,24),dtype= np.float32)

        imgs[0, :, :, :] = augmentations["image"]
        coord[0, :, :] = augmentations["keypoints"]
        for i in range(1, len(imgs)):
            imgs[i, :, :, :] = augmentations[f"image{i}"]
            coord[i, :, :] = augmentations[f"keypoints{i}"]

        for i in range(self.num_frames):
            for j in range(24):
                heatmap[i,:,:,j] = self.gaussian_heatmap((np.round(coord[i,j,0],0),np.round(coord[i,j,1],0)),(IMAGE_WIDTH,IMAGE_HEIGHT))

        heatmap = torch.from_numpy(heatmap)
        heatmap = torch.permute(heatmap,(0,3,1,2))

        vid_with_heatmap = torch.cat([imgs,heatmap],1)


        dataset = {"idx" : [cls_idx,view_idx, data_idx, frame_idx],
                   "video" : imgs,
                   "video_heatmap" : vid_with_heatmap,
                   "coord" : coord,
                   "class" : classes.index(self.data[cls_idx][view_idx][data_idx][1]),
                   "view" : view_idx,
                   "class_name" : self.data[cls_idx][view_idx][data_idx][1],
                   "type" : self.data[cls_idx][view_idx][data_idx][4],
                   "pose" : self.data[cls_idx][view_idx][data_idx][5],
                   "label_path" : self.data[cls_idx][view_idx][data_idx][6]}


        return dataset

