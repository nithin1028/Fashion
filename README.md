# FashioNet
This project (`FashionNet`) is digital trailroom ,Now-a-days peoples feel insequerity about trail room 
with help this `FashionNet` project we slove this problem 

## Woriking
1. with help of `Mediapipe` we detect the body cordinates via capture the  webcam uisng`Opencv`
2. Draw the Tarso box for to find the accurate detection
3. Analysis the skintone using `Opencv`
4. Based on skin stone Recomend the dress
5. finally Virtual-try-on

## Requrimets
1. `python` version from 10 to up to latest version
2. `MediaPipe` Latest Version
3. `Opencv` Latest Version

## Installation 
   ### Mediapipe
       ```bash
          python -m pip install mediapipe 
       ```
   ### OpenCv
       ```bash
          python -m pip install opencv-python

       ```   
   ### numpy,pandas,pathlib 
       ```bash
          python -m pip install numpy pandas pathlib

       ```     
# NOTE: 
1. Models add we add in `.gitignore` file based your requrimets and project requrimets you will dowload the models 
2. POSE DETECTION MODEL `https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker/index#models`  
3. FACE DETECTION MODEL  `https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker/index#models`    