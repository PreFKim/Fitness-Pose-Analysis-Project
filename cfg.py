
# Video Classification
classes = ["프런트 레이즈",
           "업라이트로우",
           "바벨 스티프 데드리프트",
           "바벨 로우",
           "덤벨 벤트오버 로우",
           "바벨 데드리프트",
           "바벨 스쿼트",
           "오버 헤드 프레스",
           "사이드 레터럴 레이즈",
           "바벨 컬 ",
           "덤벨 컬",
           "덤벨 풀 오버"]


# Pose estimation
palette = [(255, 165, 0),(0, 128, 0),(128, 0, 128),(255, 0, 255),(0, 0, 128),(128, 128, 0),(0, 0, 255),(165, 42, 42),(0, 255, 255),(255, 192, 203),(70, 130, 90),(255, 99, 71),(0, 128, 128),(255, 0, 0),(0, 255, 0),(0, 0, 0),(255, 255, 255),(128, 128, 128),(210, 105, 30),(218, 165, 32),(173, 216, 230),(255, 20, 147),(0, 255, 127),(220, 20, 60)]
point_name = ['Nose', 'Left Eye', 'Right Eye', 'Left Ear', 'Right Ear', 'Left Shoulder', 'Right Shoulder', 'Left Elbow', 'Right Elbow', 'Left Wrist', 'Right Wrist', 'Left Hip', 'Right Hip', 'Left Knee', 'Right Knee', 'Left Ankle', 'Right Ankle', 'Neck', 'Left Palm', 'Right Palm', 'Back', 'Waist', 'Left Foot', 'Right Foot']
linked_name =  [["Left Ear","Left Eye"],["Left Eye","Nose"],["Right Ear","Right Eye"],["Right Eye","Nose"],["Nose","Neck"],["Left Palm","Left Wrist"],["Left Wrist","Left Elbow"],["Left Elbow","Left Shoulder"],["Left Shoulder","Neck"],["Right Palm","Right Wrist"],["Right Wrist","Right Elbow"],["Right Elbow","Right Shoulder"],["Right Shoulder","Neck"],["Neck","Back"],["Back","Waist"],["Left Foot","Left Ankle"],["Left Ankle","Left Knee"],["Left Knee","Left Hip"],["Left Hip","Waist"],["Right Foot","Right Ankle"],["Right Ankle","Right Knee"],["Right Knee","Right Hip"],["Right Hip","Waist"]]

for i in range(len(linked_name)):
    linked_name[i][0] = point_name.index(linked_name[i][0])
    linked_name[i][1] = point_name.index(linked_name[i][1])

# Shared param
IMAGE_WIDTH = 192
IMAGE_HEIGHT = 256

device = "cuda:0"