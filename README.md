# AI-Based Intrusion Detection System

## Description 
An AI-powered Intrusion Detection System (IDS) built with Python, Flask, and machine learning/deep learning models.  
This system detects and classifies network attacks in real-time, providing a web-based dashboard to visualize detected threats and model performance.

---

## Features
- Detects multiple types of network attacks.  
- Preprocessing and scaling using saved `scaler.pkl` and `label_encoder.pkl`.  
- Predicts using a trained deep learning model (`saved_model.h5`).  
- Interactive Flask web app to upload network traffic CSV files and view results.  
- Plots for attack distribution, confusion matrix, and timeline of detected attacks.

---

## Installation

1. **Clone the repository**
```bash
git clone https://github.com/Shivanshssj/AI-Based-Intrusion-Detection-System.git
cd AI-Based-Intrusion-Detection-System
```
2. Create a virtual environment
```bash
python -m venv .venv
```

3. Activate the virtual environment

Windows:
```bash

.venv\Scripts\activate
```

Linux/Mac:
```bash

source .venv/bin/activate
```

4. Install dependencies
```bash

pip install -r requirements.txt
```
##Usage

Run the Flask app
```bash

python app.py
```

Open your browser
```bash

Visit: http://127.0.0.1:5000/
```
Upload CSV files

Upload your network traffic CSV file for detection.

View predictions and visualizations of:

Attack types distribution

Confusion matrix

Timeline of detected attacks


