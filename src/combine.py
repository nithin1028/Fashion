import mediapipe as mp 
import numpy as np 
import threading
import sys
from pathlib import Path
import cv2 
sys.path.insert(0,str(Path(__file__).resolve().parent))
from skintone_analysis import (
    draw_facial_landmarks,
    FaceLandmarker,
    FaceLandmarkerOptions,
    FaceLandmarkerResult,
    BaseOptions as FaceBaseOptions,
    VisionRunningMode as FaceRunningModel,
    MODEL as FACE_MODEL
)
from pose_detection import (
    draw_the_landmarks,
    BaseOptions as PoseBaseOptions,
    PoseLandmarker,
    PoseLandmarkerOptions,
    PoseLandmarkerResult,
    VisionRunningMode as PoseRunningModel,
    MODEL as POSE_MODEL
)
face_result=None
pose_result=None
lock=threading.Lock()
def face_callback(result:FaceLandmarkerResult,image:mp.Image,timestamp_mp:int):
       global face_result
       with lock:
          face_result=result if result.face_landmarks else None
def pose_callback(result:PoseLandmarkerResult,image:mp.Image,timestamp_ms:int):
    global pose_result
    with lock:
          pose_result=result if result.pose_landmarks else None      
#<--Instailise the Face Mode-->
face_model=FaceLandmarkerOptions(
              base_options=FaceBaseOptions(model_asset_path=str(FACE_MODEL)),
              running_mode=FaceRunningModel.LIVE_STREAM,
              num_faces=3,
              min_face_detection_confidence=0.5,
              min_face_presence_confidence=0.5,
              min_tracking_confidence=0.5,
              output_facial_transformation_matrixes=True,
              result_callback=face_callback
)
pose_model=PoseLandmarkerOptions(
            base_options=PoseBaseOptions(model_asset_path=str(POSE_MODEL)),
            running_mode=PoseRunningModel.LIVE_STREAM,
            num_poses=3,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_segmentation_masks=True,
            result_callback=pose_callback
            )
#<--Camera Module-->
cap=cv2.VideoCapture(0)
if not cap.isOpened():
     raise IOError("Camera is not foud")
cap.set(cv2.CAP_PROP_FRAME_WIDTH,1200)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
fps=cap.get(cv2.CAP_PROP_FPS)
if fps<=0:
    fps=30
frame_index=0
last_timestamp_ms=-1
cv2.namedWindow("Face and Pose Detection:",cv2.WINDOW_NORMAL) 
cv2.resizeWindow("Face and Pose Detection:",1280,1000)
try:
  with FaceLandmarker.create_from_options(face_model) as face_options,\
       PoseLandmarker.create_from_options(pose_model) as pose_options:
            try :
                while True:
                    sucess,frame=cap.read()
                    if not sucess:
                        raise IOError("Cemera Stopperd")
                    rgb_image=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                    mp_image=mp.Image(mp.ImageFormat.SRGB,data=rgb_image)
                    timestamp_ms=int(max(int(frame_index*1000/fps),last_timestamp_ms+1))
                    last_timestamp_ms=timestamp_ms
                    face_options.detect_async(mp_image,timestamp_ms)
                    pose_options.detect_async(mp_image,timestamp_ms)
                    display=np.copy(frame)
                    with lock:
                        f=face_result
                        p=pose_result
                        if f is not None:
                           display= draw_facial_landmarks(display,f)
                        if p is not None:
                           display= draw_the_landmarks(display,p) 
                        cv2.putText(display,f"FACE:{'ON'if f else '--'}",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)
                        cv2.putText(display,f"POSE:{'ON'if p else '--'}",(10,60),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)
                        cv2.putText(display,f"QUIT:0",(10,90),cv2.FONT_HERSHEY_SIMPLEX,0.6,(180,180,180),1)
                        cv2.imshow("Face+Pose",display)
                        if cv2.waitKey(1)&0xFF==ord('q'):
                            break
                        frame_index+=1
            finally:
                pass

finally:
    cap.release()
    cv2.destroyAllWindows()            
                            