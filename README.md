<div align="center">

# 👗 FashioNet

### A Digital Trial Room powered by Computer Vision

*Try before you wear: no crowded changing rooms, no discomfort, just you and your webcam.*

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-Latest-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Latest-0097A7?style=for-the-badge&logo=google&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

</div>

---

## 📖 Overview

Many people feel uncomfortable or insecure using physical trial rooms. **FashioNet** addresses this by moving the trial room onto your screen. Using a standard webcam, it detects your body, analyses your skin tone, recommends outfits that suit you, and lets you preview them with a virtual try-on.

## ✨ Features

| Feature | Description |
|---|---|
| 🧍 **Body Landmark Detection** | Real-time pose detection with MediaPipe on a live OpenCV webcam feed |
| 📦 **Torso Bounding Box** | Draws a torso box from shoulder and hip landmarks for accurate clothing placement |
| 🎨 **Skin Tone Analysis** | Estimates skin tone from the face region using OpenCV |
| 👕 **Outfit Recommendation** | Suggests clothing colours and styles that match the detected skin tone |
| 🪞 **Virtual Try-On** | Overlays the selected garment on the user in real time |

## ⚙️ How It Works

```
 Webcam (OpenCV)
       │
       ▼
 Pose & Face Detection (MediaPipe)
       │
       ▼
 Torso Bounding Box (accurate body region)
       │
       ▼
 Skin Tone Analysis (OpenCV)
       │
       ▼
 Dress Recommendation
       │
       ▼
 Virtual Try-On
```

1. **Capture**: OpenCV reads frames from the webcam.
2. **Detect**: MediaPipe finds body and face landmarks.
3. **Locate**: A torso box is drawn from the shoulder and hip points for precise detection.
4. **Analyse**: OpenCV extracts the dominant skin tone.
5. **Recommend**: Dresses are suggested based on the skin tone.
6. **Try On**: The chosen outfit is overlaid on the live feed.

## 🛠️ Tech Stack

<div align="center">

<img src="https://skillicons.dev/icons?i=python,opencv,numpy,pandas,git,github,vscode" alt="Tech stack icons" />

</div>

- **Python**: core language
- **MediaPipe**: pose and face landmark detection
- **OpenCV**: webcam capture, image processing, skin tone analysis, overlay
- **NumPy / Pandas**: numerical operations and recommendation data handling

## 📋 Requirements

- Python **3.10 or later**
- MediaPipe (latest)
- OpenCV (latest)
- A working webcam

## 🚀 Installation

**1. Clone the repository**

```bash
git clone https://github.com/nithin1028/Fashion.git
cd Fashion
```

**2. (Recommended) Create a virtual environment**

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

**3. Install dependencies**

```bash
python -m pip install mediapipe
python -m pip install opencv-python
python -m pip install numpy pandas
```

> `pathlib` is part of the Python standard library, so it does not need to be installed with pip.

## 📥 Download Models

Model files are large, so they are listed in `.gitignore` and **not** stored in this repository. Download the ones you need and place them in a `models/` folder.

| Model | Purpose | Download |
|---|---|---|
| Pose Landmarker | Body landmark detection | [Official models page](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker/index#models) |
| Face Landmarker | Face detection for skin tone analysis | [Official models page](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker/index#models) |

Expected layout:

```
models/
├── pose_landmarker.task
└── face_landmarker.task
```

> Adjust the file names above to match the models you download and the paths used in your code.

## ▶️ Usage

```bash
python main.py
```

- Stand in front of the webcam so your upper body is visible.
- Wait for the torso box and landmarks to appear.
- Review the skin tone analysis and recommended outfits.
- Select an outfit to start the virtual try-on.
- Press `q` to quit.

> Replace `main.py` with your actual entry-point file name.

## 📁 Project Structure

```
Fashion/
├── models/            # Downloaded MediaPipe models (git-ignored)
├── assets/            # Clothing images / overlays
├── main.py            # Application entry point
├── requirements.txt   # Python dependencies
├── .gitignore
└── README.md
```

> Update this tree to match your real folders and files.

## 🗺️ Roadmap

- [x] Body landmark detection with MediaPipe
- [x] Torso bounding box
- [x] Skin tone analysis
- [x] Skin-tone-based dress recommendation
- [x] Virtual try-on
- [ ] Improved garment warping for different body poses
- [ ] Larger clothing catalogue
- [ ] Lighting-aware skin tone correction
- [ ] Web or mobile interface

## 🤝 Contributing

Contributions are welcome.

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m "Add your feature"`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a Pull Request

## 📄 License

This project is licensed under the **MIT License**. Add a `LICENSE` file to the repository, or change this section if you use a different license.

## 🙏 Acknowledgements

- [MediaPipe](https://developers.google.com/mediapipe) by Google
- [OpenCV](https://opencv.org/)

---

<div align="center">

⭐ If you find this project useful, please consider giving it a star!

</div>
