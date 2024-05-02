import glob, tqdm, time, os, random
import numpy as np
import pandas as pd


import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import argparse

from cfg import *
from dataset.get_data import get_vid_dataset, split_data, over_sampling
from dataset.dataset import Video
from metric import accuracy
from model.vid_cls import ResNet_TSM, Resnet_LSTM

def set_seed(seed: int = 32):
    if seed != -1:
        print(seed)
        random.seed(seed)
        np.random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)  # type: ignore
        torch.backends.cudnn.deterministic = True  # type: ignore
        torch.backends.cudnn.benchmark = True  # type: ignore

def train(args):

    def load_ckpt(args):
        if args.checkpoint != "":
            checkpoint_path = args.checkpoint

            checkpoint = torch.load(checkpoint_path,'cuda')

            learning_status = pd.read_csv(os.path.join(os.path.dirname(checkpoint_path)+f'status.csv'),index_col=0)
            learning_status.iloc[:,:checkpoint['epoch']]
            learning_status = learning_status.to_dict(orient='list')

            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            #scheduler.load_state_dict(checkpoint['scheduler'])

            for k in learning_status.keys():
                for i in range(len(learning_status[k])):
                    if k!='lrs' and k!='best':
                        learning_status[k][i] = np.array([item for item in learning_status[k][i][1:-1].split(' ') if item != ''],dtype=np.float32)


            min_epoch = np.argmin(np.sum(learning_status['valid_losses'],-1))

            for i in range(len(learning_status['valid_metrics'])):
                print(f"Epoch:{i+1} | Train_Acc : {np.round(learning_status['train_metrics'][i],4)} | Train_Loss : {np.round(learning_status['train_losses'][i],4)}")
                print(f"Epoch:{i+1} | Valid_Acc : {np.round(learning_status['valid_metrics'][i],4)} | Valid_Loss : {np.round(learning_status['valid_losses'][i],4)}")
                print()

            print(len(learning_status['lrs']),min_epoch+1,learning_status['train_metrics'][-1],learning_status['train_losses'][-1],learning_status['valid_metrics'][-1],learning_status['valid_losses'][-1],learning_status['lrs'][-1])
            print(len(learning_status['lrs']),min_epoch+1,learning_status['train_metrics'][min_epoch],learning_status['train_losses'][min_epoch],learning_status['valid_metrics'][min_epoch],learning_status['valid_losses'][min_epoch],learning_status['lrs'][min_epoch])
        
    def train_begin(training,loader,running_loss,running_metric):
        if training :
            desc = "Train"
        else:
            desc = "Valid"

        progress = tqdm.tqdm(loader,desc=f'Epoch:{epoch+1}/{epochs}')
        for i,data in enumerate(progress):


            if (training):
                optimizer.zero_grad()

            pred = model(*[data[key].to(device) for key in model_input])

            total_loss = torch.tensor(0.0).to(device)
            metric = torch.tensor(0.0).to(device)

            loss_list = []
            metric_list = []


            for key in criterions.keys():
                if (type(criterions[key]) == list):
                    loss = criterions[key][0](pred[criterions[key][1]],data[criterions[key][2]].to(device))
                else :
                    loss = criterions[key](pred[key],data[key].to(device))
                total_loss  = total_loss+loss
                loss_list.append(loss.detach().cpu().numpy())

            for key in metrics.keys():
                if (type(metrics[key]) == list):
                    metric = metrics[key][0](pred[metrics[key][1]],data[metrics[key][2]].to(device))
                else :
                    metric = metrics[key](pred[key],data[key].to(device))
                metric_list.append(metric.cpu())

            if (training):

                total_loss.backward()

                optimizer.step()

            running_loss += loss_list
            running_metric += metric_list
            progress.set_description(f'Epoch:{epoch+1}/{epochs} | {desc}_Metric{list(metrics.keys())}:{np.round(running_metric/(i+1),5)} | {desc}_Loss{list(criterions.keys())}:{np.round(running_loss/(i+1),5)}')
    dataset = get_vid_dataset(args.data_path, args.num_workers)
    t, v, max_num_train = split_data(dataset,args.valid_ration, args.valid_selection)
    t = over_sampling(t, max_num_train)

    
    train_set = Video(t, num_frames = args.num_frame, views = [1, 2, 3, 4, 5])
    valid_set = Video(v, num_frames = args.num_frame, views = [1, 2, 3, 4, 5], training =  False, frame_mode=0)
    #test_set = Video(test, num_frames = num_frame, views = [1, 2, 3, 4, 5], training =  False)
    print(len(train_set), len(valid_set))

    set_seed(args.seed)

    # model = Resnet_LSTM(image_features=2048,
    #                     hidden_size=1024,
    #                     num_layers=4,
    #                     dropout=0, #Dropout을 하기 위해서는 레이어의 수가 2 이상이어야함
    #                     bidirectional=True)
    model = ResNet_TSM(num_layers=[3,4,23,3], frame=args.num_frame, heatmap=True)

    model_input = ["video_heatmap"]

    metrics = {"class":accuracy} # [평가지표, Pred, Label]
    criterions = {"class":nn.CrossEntropyLoss()}

    min_epoch = 0
    learning_status = {
        'train_metrics' : [],
        'valid_metrics' : [],
        'train_losses' : [],
        'valid_losses' : [],
        'lrs' : [],
        'best' : []
    }

    optimizer = torch.optim.AdamW(params = model.parameters(), lr=args.lr, weight_decay=0.01)

    load_ckpt(args)

    train_loader = DataLoader(train_set,batch_size=args.batch_size,shuffle=True, num_workers=args.num_workers)
    valid_loader = DataLoader(valid_set,batch_size=args.batch_size,shuffle=False, num_workers=args.num_workers)

    model_save_path = f'./experiment/VC/{model.name}{len(glob.glob("./experiment/VC/*"))}/' # frame,layers,hidden

    model = model.to(device)

    epochs = args.epochs

    print("모델 저장 경로 : "+ model_save_path)
    fit_time = time.time()
    start_epoch = len(learning_status['lrs'])

    for epoch in range(start_epoch,epochs):
        print("모델명 :", model_save_path.split('/')[-2])

        #Warmup Schedule
        # if epoch < warmup_epochs :
        #     scheduler.step()
        #     print("lr이 변경되었습니다.",optimizer.param_groups[0]['lr'])


        # scheduler.step()
        # print("lr이 변경되었습니다.",optimizer.param_groups[0]['lr'])


        running_train_loss = np.zeros((len(criterions.keys())))
        running_valid_loss = np.zeros((len(criterions.keys())))

        running_train_metric = np.zeros((len(metrics.keys())))
        running_valid_metric = np.zeros((len(metrics.keys())))


        model.train()
        train_begin(True,train_loader,running_train_loss,running_train_metric)
        model.eval()
        with torch.no_grad():
            train_begin(False,valid_loader,running_valid_loss,running_valid_metric)


        learning_status['train_losses'].append((running_train_loss/len(train_loader)))
        learning_status['valid_losses'].append((running_valid_loss/len(valid_loader)))
        learning_status['train_metrics'].append((running_train_metric/len(train_loader)))
        learning_status['valid_metrics'].append((running_valid_metric/len(valid_loader)))
        learning_status['lrs'].append(optimizer.param_groups[0]['lr'])

        checkpoint = {
            'epoch': epoch+1 , #에폭
            'model': model.state_dict(),  # 모델
            'optimizer': optimizer.state_dict()  # 옵티마이저
        }
        if "scheduler" in globals():
            checkpoint['scheduler'] = scheduler.state_dict()  # 스케줄러

        if os.path.exists(model_save_path) == False:
            os.makedirs(model_save_path,exist_ok=True)


        if sum(learning_status['valid_losses'][min_epoch]) >= sum(learning_status['valid_losses'][-1]) and sum(learning_status['valid_losses'][-1] > 0):
            print(f"Valid Loss가 최소가 됐습니다. ({sum(learning_status['valid_losses'][min_epoch]):.5f}({min_epoch+1}) -> {sum(learning_status['valid_losses'][-1]):.5f}({len(learning_status['valid_losses'])}))")
            print(f'해당 모델이 {model_save_path}Best{epoch+1}.pth 경로에 저장됩니다.')
            min_epoch = len(learning_status['valid_losses'])-1
            torch.save(checkpoint, model_save_path+f'Best{epoch+1}.pth')
            learning_status['best'].append(sum(learning_status['valid_losses'][-1]))
        else:
            print(f"Valid_Loss가 최소가 되지 못했습니다.(최소 Epoch:{min_epoch+1} : {sum(learning_status['valid_losses'][min_epoch]):.5f}, 현재 : {sum(learning_status['valid_losses'][-1]):.5f})")
            learning_status['best'].append('False')

        torch.save(checkpoint, os.path.join(model_save_path, f'Last{epoch+1}.pth'))

        df = pd.DataFrame(learning_status)
        df.to_csv(os.path.join(model_save_path, 'status.csv'), index=True)

        print('')


    print('학습 최종 시간: {:.2f} 분\n' .format((time.time()- fit_time)/60))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default="-1")

    parser.add_argument("--num_workers", type=int, default="1")
    parser.add_argument("--data_path", type=str, default="./data")
    parser.add_argument("--valid_ratio", type=float, default="0.2")
    parser.add_argument("--valid_selection", type=int, default="0", hint="Validation set 선택") 

    parser.add_argument("--num_frame", type=int, default="8")
    parser.add_argument("--checkpoint", type=str, default="")

    parser.add_argument("--epochs", type=int, default="50")
    parser.add_argument("--batch_size", type=int, default="32")
    parser.add_argument("--lr", type=float, default="5e-5")


    args = parser.parse_args()

    train(args)

