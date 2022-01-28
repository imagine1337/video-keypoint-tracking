import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
import numpy as np
import tqdm
import cv2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.data import MetadataCatalog
import time

videoname='yourvideo.mp4'

video = cv2.VideoCapture(videoname)
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
frames_per_second = video.get(cv2.CAP_PROP_FPS)
num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

video_writer = cv2.VideoWriter(videoname, fourcc=cv2.VideoWriter_fourcc(*"mp4v"), fps=float(frames_per_second), frameSize=(width, height), isColor=True)

cfg = get_cfg()
cfg.MODEL.DEVICE="cpu"
cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set threshold for this model
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")
predictor = DefaultPredictor(cfg)

v = VideoVisualizer(MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), ColorMode.IMAGE)

def runOnVideo(video, mf):
    rf = 0
    while True:
        has_frame, frame = video.read()
        if not has_frame:
            break
        outputs = predictor(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        visualization = v.draw_instance_predictions(frame, outputs["instances"].to("cpu"))
        visualization = cv2.cvtColor(visualization.get_image(), cv2.COLOR_RGB2BGR)
        yield visualization
        rf += 1
        if rf > maxFrames:
            break

num_frames = 150

for visualization in tqdm.tqdm(runOnVideo(video, num_frames), total=num_frames):
    cv2.imwrite('POSE detectron2.png', visualization)
    video_writer.write(visualization)

video.release()
video_writer.release()
cv2.destroyAllWindows()
