import cv2
import numpy as np
from typing import List
 
# Function to generate video
def generate_video(imgs:List[np.ndarray], video_path:str, framerate:float): #Images should be prepared to be opencv image layout

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(video_path, fourcc, framerate, (512, 512))
    if not video.isOpened():
        raise ValueError("CV2 video encoder Not supported")

    # Appending images to video
    for i in range(len(imgs)):
        video.write(imgs[i])

    # Release the video file
    video.release()
    cv2.destroyAllWindows()
    print("Video generated successfully!")
