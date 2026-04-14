from pathlib import Path
import cv2
import numpy as np

src_dir = "./datasets/Defects/Hole/videos"
output_dir = "./datasets/Defects/Hole/cut_videos"

Path(output_dir).mkdir(parents=True, exist_ok=True)

paths_to_videos = list(Path(src_dir).rglob("*.MOV"))

for path_to_video in paths_to_videos:
    video_name = path_to_video.stem
    cap = cv2.VideoCapture(str(path_to_video))
    Num_Frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    if cap.isOpened():
        for FrameNo in np.arange(1, Num_Frames + 1, 10):
            cap.set(cv2.CAP_PROP_POS_FRAMES, FrameNo)
            state, frame = cap.read()
            if state:
                name_frame = f"{output_dir}/{video_name}_frame_{FrameNo}.jpg"
                cv2.imwrite(name_frame, frame)
