# AI-Enabled Construction Material Identification and Volume Estimation

## Overview

This project is an AI-powered construction material analysis system that automatically detects cement bags from uploaded images and estimates the total material volume.

The system uses a custom-trained YOLOv8 object detection model to identify cement bags from different viewing angles and calculate the total count and volume.

The application is deployed as a web application using Streamlit and can be accessed from both desktop and mobile devices.

---

## Features

- Cement bag detection using YOLOv8
- Custom-trained object detection model
- Multi-image support
- Median-based counting for multiple angles
- Adjustable confidence threshold
- Automatic volume estimation
- Web-based interface
- Mobile-compatible deployment
- Real-time inference

---

## Project Workflow

### Step 1: Image Upload

Users upload one or more images of cement bag stacks.

### Step 2: Image Preprocessing

Images are preprocessed to improve detection quality.

Techniques used:

- Image resizing
- Color conversion
- Noise reduction
- Contrast enhancement

### Step 3: Object Detection

The trained YOLOv8 model identifies cement bags in the image.

Output:

- Bounding boxes
- Confidence scores
- Total detected bags

### Step 4: Multi-Angle Processing

When multiple images of the same stack are uploaded:

- Each image is analyzed separately
- Counts are stored
- Median count is selected

This prevents overcounting caused by summing detections from multiple angles.

### Step 5: Volume Estimation

Volume is calculated using:

```
Total Volume = Number of Bags × Volume per Bag
```

Default value:

```
1 Cement Bag = 0.04 m³
```

### Step 6: Result Generation

The system displays:

- Detected images
- Bounding boxes
- Total count
- Estimated volume

---

## AI Model

### Model Used

YOLOv8 Nano (YOLOv8n)

### Framework

Ultralytics YOLOv8

### Training Details

- Custom cement bag dataset
- Trained using transfer learning
- 50 epochs
- Image size: 416 × 416
- Confidence threshold adjustable by user

---

## Technologies Used

### Programming Language

- Python

### Computer Vision

- OpenCV

### Deep Learning

- YOLOv8
- PyTorch
- Ultralytics

### Data Processing

- NumPy
- Pandas

### Visualization

- Matplotlib

### Web Application

- Streamlit

### Report Generation

- ReportLab

---

## Project Structure

```text
construction_ai_project/

├── app.py
├── requirements.txt
├── dataset/
├── input_images/
├── outputs/
├── modules/
├── runs/
│   └── detect/
│       └── train-3/
│           └── weights/
│               └── best.pt
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
```

Move into the project directory:

```bash
cd construction_ai_project
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

---

## Deployment

The project is deployed using Streamlit Cloud.

The application can be accessed through a web browser on desktop or mobile devices.

---

## Results

The system successfully:

- Detects cement bags from uploaded images
- Handles multiple viewing angles
- Reduces duplicate counting using median estimation
- Estimates material volume automatically
- Provides real-time analysis through a web application

---

## Limitations

- Accuracy depends on image quality
- Extreme viewing angles may reduce detection accuracy
- Heavy occlusion can affect counting performance
- Performance depends on the quality and diversity of training data

---

## Future Enhancements

- 3D stack reconstruction
- Automatic height estimation
- Support for multiple construction materials
- PDF report generation
- Database integration
- Mobile application version
- Drone-based inventory analysis

---

## Author

Durga Ganapathi

B.Tech Computer Science and Engineering (AI & ML)

Vel Tech University

---

## License

This project is developed for academic and research purposes.
