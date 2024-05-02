import json
import os 

import pickle
import glob
import tqdm
import numpy as np

import concurrent.futures

from cfg import classes, point_name

def get_vid_dataset(data_path, num_workers=1):
    dataset = [[[],[],[],[],[]] for _ in range(len(classes))]
    error = []

    worker = num_workers
    def load_file(filename):
        with open(filename) as f:
            label_json = json.load(f)
            for view in range(5):
                if len(label_json['frames']) != 0 and label_json['type_info']['exercise'] in classes :
                    # 경로 Crop
                    path_split = filename.split("/")
                    cropped_path = ("/".join(path_split[:filename.count('/')-1])).replace("label","video")
                    minx=1920
                    miny=1080
                    maxx=0
                    maxy=0

                    video_path = os.path.dirname(os.path.join(cropped_path, label_json['frames'][0][f'view{view+1}']['img_key']))+'.avi'
                    
                    data = [video_path,label_json['type_info']['exercise'],[],[],
                        label_json['type_info']['type'],label_json['type_info']['pose'],filename]
                    

                    for frame in range(len(label_json['frames'])):
                        pts_list = []
                        # 좌표 추가
                        for p in point_name:
                            pts_list.append([label_json['frames'][frame][f'view{view+1}']['pts'][p]['x'],
                                            label_json['frames'][frame][f'view{view+1}']['pts'][p]['y']])
                            minx = min(minx,label_json['frames'][frame][f'view{view+1}']['pts'][p]['x'])
                            miny = min(miny,label_json['frames'][frame][f'view{view+1}']['pts'][p]['y'])
                            maxx = max(maxx,label_json['frames'][frame][f'view{view+1}']['pts'][p]['x'])
                            maxy = max(maxy,label_json['frames'][frame][f'view{view+1}']['pts'][p]['y'])
                        data[3].append(pts_list)
                    data[2] = [minx,miny,maxx,maxy]


                    dataset[classes.index(label_json['type_info']['exercise'])][view].append(data)
                else :
                    error.append([filename,len(label_json['frames']), label_json['type_info']['exercise']])

    if os.path.exists(os.path.join(data_path,"VC.pkl")):
        with open(os.path.join(data_path,"VC.pkl"),"rb") as f:
            dataset = pickle.load(f)
    else:
        label_names = sorted(glob.glob(os.path.join(data_path, "label/*/*[!d].json")))
        for i in tqdm.tqdm(range(0,len(label_names),worker)):
            with concurrent.futures.ThreadPoolExecutor(max_workers=worker) as executor:
                executor.map(load_file, label_names[i:i+worker])
        with open(os.path.join(data_path, "VC.pkl"),"wb") as f:
            pickle.dump(dataset, f)

    for i in range(len(classes)):
        print(i, classes[i], len(dataset[i][0]))
        for j in range(5):
            dataset[i][j] = sorted(dataset[i][j])
            if len(dataset[i][j]) != len(dataset[i][j-1]) :
                print(i,j, "error")

    print(dataset[0][0][0])
    for i in range(len(dataset[0][0][0])):
        print(i,":",dataset[0][0][0][i])

    return dataset


def get_pose_dataset(data_path, anomaly=None, num_workers=1):
    dataset = [[[],[],[],[],[]] for _ in range(len(classes))]
    error = []

    worker = num_workers
    def load_file(filename):
        with open(filename) as f:
            label_json = json.load(f)
            for view in range(5):
                if len(label_json['frames']) != 0 :
                    # 경로 Crop
                    path_split = filename.split("/")
                    cropped_path = ("/".join(path_split[:filename.count('/')-1])).replace("label","video")
                    minx=1920
                    miny=1080
                    maxx=0
                    maxy=0

                    video_path = os.path.dirname(os.path.join(cropped_path, label_json['frames'][0][f'view{view+1}']['img_key']))+'.avi'
                    if anomaly is None or anomaly.get(video_path) is None: 
                        data = [video_path,[],[],label_json['type_info']['exercise'],label_json['type_info']['type'],label_json['type_info']['pose'],filename]
                        

                        for frame in range(len(label_json['frames'])):
                            pts_list = []
                            # 좌표 추가
                            for p in point_name:
                                pts_list.append([label_json['frames'][frame][f'view{view+1}']['pts'][p]['x'],
                                                label_json['frames'][frame][f'view{view+1}']['pts'][p]['y']])
                                minx = min(minx,label_json['frames'][frame][f'view{view+1}']['pts'][p]['x'])
                                miny = min(miny,label_json['frames'][frame][f'view{view+1}']['pts'][p]['y'])
                                maxx = max(maxx,label_json['frames'][frame][f'view{view+1}']['pts'][p]['x'])
                                maxy = max(maxy,label_json['frames'][frame][f'view{view+1}']['pts'][p]['y'])
                            data[2].append(pts_list)
                        data[1] = [minx,miny,maxx,maxy]

                        dataset[view].append(data)

    if os.path.exists(os.path.join(data_path,"PE.pkl")):
        with open(os.path.join(data_path,"PE.pkl"),"rb") as f:
            dataset = pickle.load(f)
    else:
        label_names = sorted(glob.glob(os.path.join(data_path, "label/*/*[!d].json")))
        for i in tqdm.tqdm(range(0,len(label_names),worker)):
            with concurrent.futures.ThreadPoolExecutor(max_workers=worker) as executor:
                executor.map(load_file, label_names[i:i+worker])
        with open(os.path.join(data_path, "PE.pkl"),"wb") as f:
            pickle.dump(dataset, f)

    for i in range(1,5):
        if len(dataset[i]) != len(dataset[i-1]) :
            print(i, "error")

    dataset = [sorted(view) for view in dataset]

    print()
    print(len(dataset[0]))
    print(dataset[0][0][0]) #path
    print(dataset[0][0][1]) #min,max
    print(dataset[0][0][2]) #frame
    print(dataset[0][0][2][0]) #pts
    print(dataset[0][0][3]) #exercise

    return dataset


def split_data(dataset, valid_ratio=0.2, valid_selection=0):
    train = []
    valid = []
    ratio = valid_ratio

    for i, inner_list in enumerate(dataset):
        view_train = []
        view_valid = []
        assert int(1/ratio) > valid_selection, "Split된 데이터 개수보다 Selection Index가 더 큽니다."
        for j in range(5):
            view_train.append([data for k,data in enumerate(inner_list[j]) if k % int(1/ratio) != valid_selection])
            view_valid.append([data for k,data in enumerate(inner_list[j]) if k % int(1/ratio) == valid_selection])
        train.append(view_train)
        valid.append(view_valid)

    max_num_train = 0
    for i in range(len(classes)):
        max_num_train = max(len(train[i][0]), max_num_train)
        print(i, classes[i], len(train[i][0]),len(valid[i][0]))

    return train, valid, max_num_train


def over_sampling(train, max_num_train):
    # Rare Class ( Over Sampling)
    sampled_train = []

    for i, inner_list in enumerate(train):
        new_inner_list = [(item * int(np.ceil(max_num_train/len(item))))[:max_num_train] if len(item) !=0 else [] for j,item in enumerate(inner_list)]
        sampled_train.append(new_inner_list)


    for i in range(len(classes)):
        print(i, classes[i], len(sampled_train[i][0]))
        for j in range(1,5):
            if len(sampled_train[i][j]) != len(sampled_train[i][j-1]) :
                print(i,j, "error")
    return sampled_train