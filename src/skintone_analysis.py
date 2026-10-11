import mediapipe as mp 
from mediapipe.tasks import python 
from mediapipe.tasks.python import vision
from pathlib import Path
import cv2
import matplotlib.pyplot as plt 
from mediapipe.tasks.python.vision import drawing_utils,drawing_styles
import numpy as np
import threading
import time  
Base=Path(__file__).resolve().parent
MODEL=(Base/".."/".."/"models"/"face_landmarker.task").resolve()
if not  MODEL.is_file():
    raise FileNotFoundError(f"Model not found in this path:{MODEL}")
BaseOptions=mp.tasks.BaseOptions
FaceLandmarker=mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions=mp.tasks.vision.FaceLandmarkerOptions
FaceLandmarkerResult=mp.tasks.vision.FaceLandmarkerResult
VisionRunningMode=mp.tasks.vision.RunningMode
#<---Draw the LandMarks--->
def draw_facial_landmarks(image,result):
    face_landmarks_list=result.face_landmarks
    annotated_image=np.copy(image)
    for idx in range(len(face_landmarks_list)):
        face_landmarks=face_landmarks_list[idx]
        drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=drawing_styles.get_default_face_mesh_tesselation_style())
        drawing_utils.draw_landmarks(
                      image=annotated_image,
                      landmark_list=face_landmarks,
                      connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS,
                      landmark_drawing_spec=None,
                      connection_drawing_spec=drawing_styles.get_default_face_mesh_contours_style()
        )
        drawing_utils.draw_landmarks(
                     image=annotated_image,
                     landmark_list=face_landmarks,
                     connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_LEFT_IRIS,
                     landmark_drawing_spec=None,
                     connection_drawing_spec=drawing_styles.get_default_face_mesh_iris_connections_style()
        )
        drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_IRIS,
                landmark_drawing_spec=None,
                connection_drawing_spec=drawing_styles.get_default_face_mesh_iris_connections_style()
        )
    return annotated_image

#<--Blend-shape-->
def plot_face_blendshape_bar_graph(face_blendshape):
        face_blendshape_names=[face_blendshape_category.category_name for face_blendshape_category in face_blendshape]
        face_blendshape_scores=[face_blendshape_category.score for face_blendshape_category in face_blendshape]
        face_blendshape_rank=range(len(face_blendshape_names))
        fig,asix=plt.subplots(figsize=(12,12))
        bar=asix.barh(face_blendshape_rank,face_blendshape_scores,label=[str(x) for x in face_blendshape_rank])
        asix.set_yticks(list(face_blendshape_rank))
        asix.set_yticklabels(face_blendshape_names,fontsize=8)
        asix.invert_yaxis()
        for score,patch in zip(face_blendshape_scores,bar.patches):
                plt.text(patch.get_x()+patch.get_width(),patch.get_y(),f"{score:.4f}",va='top')
        asix.set_xlabel("Score")
        asix.set_title("Face BlednShape")
        plt.tight_layout()
        plt.show()        
#<---Model Instalisation--->
landmark=None
lock=threading.Lock()
def print_result(result:FaceLandmarkerResult,image:mp.Image,timestamp_mp:int):
    global landmark
    with lock:
        landmark= result if result.face_landmarks else None
        print(f"face dectection landmark_drawing_spec:{result}")
options=FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL)),
        running_mode=VisionRunningMode.LIVE_STREAM,
        num_faces=2,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        result_callback=print_result  
)
cap=cv2.VideoCapture(0)
if  not cap.isOpened():
        raise IOError("Cemara is not opend")       
fps=cap.get(cv2.CAP_PROP_FPS) 
if fps<=0:
    fps=30
frame_index=0
last_timestamp_ms=-1              
try :
   with FaceLandmarker.create_from_options(options) as  face_landmarks:
     while True:  
        sucess,frame=cap.read()
        if not sucess:
             raise IOError("Cemera not working")    
        rgb_image=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)  
        mp_image=mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb_image)
        timestamp_ms=int(max((int(frame_index*1000)/fps),last_timestamp_ms+1))
        last_timestamp_ms=timestamp_ms
        face_landmarks.detect_async(mp_image,timestamp_ms)
        with lock: 
          result=landmark
          display=frame   
          if result is not None:
                display=draw_facial_landmarks(frame,result)   
          cv2.imshow("wecam",display) 
          key=cv2.waitKey(1)&0xFF 
          if key==ord('q'):
                break 
          if key ==ord('b') and result is not None and result.face_blendshapes:
                plot_face_blendshape_bar_graph(landmark.face_blendshapes[0])
     frame_index+=1
finally:
    cap.release()
    cv2.destroyAllWindows()    



