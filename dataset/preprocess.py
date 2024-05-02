import argparse
import os
import time
import cv2
import json
import shutil
import glob

def im2vid(root_path, day, from_to=[], crop_margin=200):
    start_time = time.time()
    count=0
    original_size = 0.0
    croped_size = 0.0
    error = []
    count_error = [0,0,0,0,0]
    label_num = 0
    for filename in sorted(glob.glob(os.path.join(root_path,f'/label/Day{day:02d}*/*[!d].*'))):

        if label_num != int(filename.split('-')[-2]):
            label_num = int(filename.split('-')[-2])
            print(label_num,"탐색 시작")

        with open(filename, "r") as st_json:
            label = json.load(st_json)
            

        if (len(label['frames'])!=0):
            for num in [1,2,3,4,5]:
                x1=1920
                y1=1080
                x2=0
                y2=0
                view = f"view{num}"
                for i in range(len(label['frames'])):
                    for key in label['frames'][i][view]['pts'].keys():
                        x1 = min(x1,label['frames'][i][view]['pts'][key]['x'])
                        x2 = max(x2,label['frames'][i][view]['pts'][key]['x'])
                        y1 = min(y1,label['frames'][i][view]['pts'][key]['y'])
                        y2 = max(y2,label['frames'][i][view]['pts'][key]['y'])

                # 코드 좀 깔끔히 정리하고 파일 구조 확인해서 전처리하도록
                write_dir = os.path.dirname(os.path.join(root_path , 'video' ,label['frames'][0][view]['img_key']))+'.avi'
                os.makedirs(os.path.dirname(write_dir),exist_ok=True)

                y1 = max(0,y1-crop_margin)
                y2 = min(1080,y2+crop_margin)
                x1 = max(0,x1-crop_margin)
                x2 = min(1920,x2+crop_margin)

                if ((os.path.exists(write_dir)==False)):

                    current_size = 0

                    image_path = os.path.dirname(os.path.join(root_path, label['frames'][0][view]['img_key']))
                    if os.path.exists(image_path) == False:
                        for ft in from_to:
                            from_dir = image_path.replace(ft[0],ft[1])
                            if (os.path.exists(from_dir)):
                                image_path = from_dir
                            else:
                                error.append(image_path)
                                print("라벨에 해당하는 원본 이미지 파일이 존재하지 않습니다.",image_path)
                                del_dir = False
                                count_error[num-1]+=1
                                break

                    frames=[]
                    for img_filename in sorted(glob.glob(image_path+'/*.*')):
                        img = cv2.imread(img_filename)
                        img = img[y1:y2,x1:x2]
                        frames.append(img)

                        current_size += os.path.getsize(img_filename)


                    out = cv2.VideoWriter(write_dir, cv2.VideoWriter_fourcc(*'DIVX'), 1, (x2-x1, y2-y1))
                    for frame in frames:
                        out.write(frame)
                    out.release()

                    original_size += current_size/1024
                    croped_size += os.path.getsize(write_dir)/1024
                    count+=1


    print("처리시간",(time.time()-start_time)/60,"분")
    print("기존 파일 용량",original_size,"Kb")
    print("결과 파일 용량",croped_size,"Kb")
    if (croped_size!=0):
        print("용량 비율",original_size/croped_size)
    print("개수",count)
    print("error",error)
    print("항목별 error 수",count_error)

def main(args):
    day_list = [[4],[3,[["C","B"]]],[26],[25],[12,[["-Z","_Z"]]],[29],[30],[11],[15],[40],[39],[41],[42],[8],[16],[5],[1,[["_B","_A"],["_C","_A"],["_D","_A"],["_E","_A"]]],[2,[["B","A"]]],[23],[27],[28],[24],[33],[35],[36],[38],[37],[10],[9],[14],[13],[34],[17],[20],[22],[21],[7],[32],[19]]
    for i in range(0,len(day_list)):
        im2vid(
            args.data_path,
            day_list[i][0],
            day_list[i][1] if len(day_list[i])>1 else [],
            crop_margin = args.crop_margin 
        )


if __name__ == "main":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./data")
    parser.add_argument("--crop_margin", type=int, default="200")

    args = parser.parse_args()

    main(args)
    
