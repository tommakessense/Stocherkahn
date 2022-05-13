import cv2
video_name = "/home/tom/Desktop/video_25.mp4"
dest_dir = '/home/tom/Desktop/images/water/'
vidcap = cv2.VideoCapture(video_name)
count = 0
while True:
    success,image = vidcap.read()
    if success == False:
        break
    print('Read a new frame: ', success)
    count += 1
    if count % 30 == 0:
        cv2.imwrite(f"{dest_dir}frame_2_{count}.jpg", image)     # save frame as JPEG file
        print(f"added image")
