# 🛰️ Multimodal Satellite Image Analysis using AI

A powerful Streamlit-based application that leverages advanced deep learning models to analyze satellite images through **Image Classification**, **Segmentation**, and **Landslide Detection**.

---

## 🚀 Features

- 🌍 **Interactive Map Interface**  
  Capture and analyze real-world satellite data by clicking directly on the map.

- 🏷 **Image Classification**  
  Classify land types using **ViT**, **ResNet**, and **VGG** models (e.g., Forest, Residential, River, etc.).

- 🎭 **Image Segmentation with YOLO**  
  Detect and segment objects in satellite images pixel-by-pixel using YOLOv8.

- 🏔 **Landslide Detection**  
  Classify regions as **landslide** or **non-landslide** using a **Vision Transformer (ViT)** model.

- 📥 **Multiple Input Modes**  
  - Upload satellite images manually  
  - Select via interactive map  
  - Enter latitude & longitude directly

---

## 🧠 Models Used

| Model      | Task                      | Role                                                               |
|------------|---------------------------|--------------------------------------------------------------------|
| **ViT**    | Classification, Landslide | Captures global patterns & long-range dependencies                 |
| **ResNet** | Classification            | Learns deep spatial features using residual connections            |
| **VGGNet** | Classification            | Learns textures and patterns using deep CNN layers                 |
| **YOLOv8** | Segmentation & Detection  | Pixel-wise segmentation and bounding-box object detection          |

---


---

## 🛠️ Tech Stack

- **Frontend**: Streamlit, Folium
- **Backend**: Python
- **AI/ML**: PyTorch, TensorFlow, OpenCV, YOLOv8
- **Tools**: Selenium, Matplotlib, Seaborn

---

## 🖥 How to Run

1. 🔧 **Install Requirements**
   ```bash
   pip install -r requirements.txt

2. **Run command in terminal**
   ```bash
   streamlit run home.py 
---


<table>
  <tr>
    <td><img src="screenshort/home page.png" alt="Image 1" width="300"></td>
    <td><img src="screenshort/classification 1.png" alt="Image 2" width="300"></td>
  </tr>
  <tr>
    <td><img src="screenshort/classification 2.png" alt="Image 3" width="300"></td>
    <td><img src="screenshort/landslide.png" alt="Image 4" width="300"></td>
  </tr>
</table>
