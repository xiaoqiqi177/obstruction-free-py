import cv2
import glob
import os

video_dir = './data/ObstructionFreePhotography_SIGGRAPH2015_Data'
output_dir = './data/images'
if os.path.exists(output_dir) is False:
    os.mkdir(output_dir)
video_paths = glob.glob(os.path.join(video_dir, '*_input.avi'))
video_paths.sort()

for video_path in video_paths:
    cap = cv2.VideoCapture(video_path)
    if cap.isOpened() is False:
        print("existing not video {}.".format(video_path))
    video_name = os.path.split(video_path)[-1].split('_')[0]
    if os.path.exists(os.path.join(output_dir, video_name)) is False:
        os.mkdir(os.path.join(output_dir, video_name))
    image_id = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret is True:
            cv2.imwrite(os.path.join(output_dir, video_name, video_name + '_'+str(image_id)+'.png'), frame)
            image_id += 1
        else:
            break
    cap.release()


