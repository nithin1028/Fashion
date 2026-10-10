from time import process_time
from pathlib import Path
import mediapipe as mp 
from mediapipe.tasks import python 
from mediapipe.tasks.python import vision 
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import drawing_styles
import cv2
import os
import numpy as np
BASE=Path(__file__).resolve().parent    
MODEL = (BASE / ".." / ".." / "models" / "pose_landmarker_heavy.task").resolve()
if not MODEL.is_file():
    raise FileNotFoundError(f"Model not found {MODEL} is invalid")
print("Using Model:",MODEL) 
#<----Draw LandMarks -----> 
def draw_the_landmarks(images,result):
    annotated_image=np.copy(images)
    if result is None or not result.pose_landmarks:
        return annotated_image
    for pose_landmarks in result.pose_landmarks:
         drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=pose_landmarks,
            connections=vision.PoseLandmarker.POSE_CONNECTIONS 
            if False else vision.PoseLandmarksConnections.POSE_LANDMARKS,
            landmark_drawing_spec=drawing_styles.get_default_pose_landmarks_style()
            )
    return annotated_image        


#<------Model Intialisations------>
BaseOptions=mp.tasks.BaseOptions
PoseLandmarker=mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions=mp.tasks.vision.PoseLandmarkerOptions
PoseLandmarkerResult=mp.tasks.vision.PoseLandmarkerResult
VisionRunningMode=mp.tasks.vision.RunningMode  
#<---- Print the LandMarks X,Y,Z ----->
landmarks=None
def print_result(result:PoseLandmarkerResult,image:mp.Image,timestamp_mp:int):
    global landmarks
    landmarks=result
    if result.pose_landmarks:
        print("Pose detected",len(result.pose_landmarks)) 
    else :
        print("No Pose detected")   
    return result      
options=PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL)),
        running_mode=VisionRunningMode.LIVE_STREAM,
        num_poses=2,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=True,
        result_callback=print_result,  
    )   
#<-----Image Capture----->      
cap =None
for camera_index in range(5):
    temp_cap=cv2.VideoCapture(camera_index)
    if temp_cap.isOpened():
        cap=temp_cap
        print(f"Cemera is Working!!")
        break
    temp_cap.release()
if cap is None:
     raise IOError("No cemera found")
fps=cap.get(cv2.CAP_PROP_FPS)
if fps <=0:
   fps=30
frame_index=0 
last_timestamp_ms=-1
try :      
  with PoseLandmarker.create_from_options(options) as landmarker:
    while True:
       sucess,frame=cap.read()
       if not sucess :
           raise IOError("Cemera not sending data")
       image_rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
       mp_image=mp.Image(image_format=mp.ImageFormat.SRGB,data=image_rgb)
       timestamp_ms = max(
       int((frame_index * 1000) / fps),
       last_timestamp_ms + 1
       )
       last_timestamp_ms = timestamp_ms
       landmarker.detect_async(mp_image,timestamp_ms)
       annotated_image=draw_the_landmarks(image_rgb,landmarks)
       cv2.imshow('wecam feed',cv2.cvtColor(annotated_image,cv2.COLOR_RGB2BGR))
       if cv2.waitKey(1)&0xFF==ord("q"): 
           break
       frame_index+=1 
finally:       
 cap.release()
 cv2.destroyAllWindows()           