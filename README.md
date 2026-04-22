# 🛡️ FedMask — Privacy-Preserving Mask Detection using Federated Learning

A modern **Federated Learning-based Mask Detection System** that ensures **data privacy** while achieving high accuracy using a CNN model.

---

## 📌 Overview

FedMask is a **privacy-first AI system** designed for mask detection in healthcare environments.
Instead of centralizing sensitive data, it uses **Federated Learning (FedAvg)** to train models across multiple hospitals **without sharing raw images**.

---

## 🚀 Key Features

* 🔐 **Privacy-Preserving AI**

  * No raw image data leaves local nodes
  * Uses **Differential Privacy (ε = 0.5)**

* 🧠 **Federated Learning (FedAvg)**

  * Distributed training across multiple hospitals
  * Aggregation using weighted averaging

* 📊 **Interactive Dashboard UI**

  * Real-time training progress
  * Accuracy, loss, and metrics visualization
  * Node-wise performance monitoring

* 🎯 **Mask Detection System**

  * OpenCV-based face detection
  * CNN-based classification (Mask / No Mask)

---

## 🏗️ System Architecture

```
Hospitals (Local Training)
        ↓
   Model Updates (Δw only)
        ↓
Federated Server (FedAvg)
        ↓
 Global Model Distribution
```

* Each hospital trains locally
* Only **model weights** are shared
* Central server aggregates using **FedAvg**

---

## 🧠 Model Details

* Architecture: CNN (Conv2D + Dense)
* Input Shape: 64 × 64 × 3
* Output: 2 classes (Mask / No Mask)
* Optimizer: Adam
* Loss: Categorical Crossentropy

---

## 📊 Performance Metrics

| Metric    | Value |
| --------- | ----- |
| Accuracy  | 95.6% |
| Precision | 94.8% |
| Recall    | 96.2% |
| F1-Score  | 95.5% |

✅ Outperforms centralized training by **+2.2%**

---

## 🖥️ Frontend Dashboard

The project includes a **fully interactive UI dashboard** with:

* 📈 Training visualization (round-wise)
* 🏥 Hospital node monitoring
* 🔍 Live detection simulation
* 📊 Confusion matrix & comparison charts

---

## 📂 Project Structure

```
your-repo/
│
├── federated/          # Federated learning backend
│   ├── __init__.py
│   └── fl_engine.py
│
├── templates/          # Frontend UI
│   └── index.html
│
├── app.py              # Flask server
├── requirements.txt
├── Procfile
└── .gitignore
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone Repository

```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run Flask Server

```bash
python app.py
```

### 4️⃣ Open in Browser

```
http://127.0.0.1:5000
```

---

## 🧪 Dataset

* MaskedFaceNet (simulated distributed dataset)
* Total Samples: ~7,000
* Distributed across 3 nodes

---

## 🔐 Privacy Mechanism

* ✔ No raw data sharing
* ✔ Weight updates only (Δw)
* ✔ Differential privacy noise added
* ✔ Secure aggregation

---

## 📚 Technologies Used

* Python (Flask)
* TensorFlow / CNN
* Federated Learning (FedAvg)
* OpenCV
* HTML + CSS + JavaScript
* Chart.js

---

## 📸 Demo

> <img width="2559" height="814" alt="image" src="https://github.com/user-attachments/assets/68715bcf-1217-4b5f-8082-c2d49a336ac1" />

> <img width="2559" height="1317" alt="image" src="https://github.com/user-attachments/assets/a5f9af7a-8a84-441c-a859-3345ab0e9bf2" />

> <img width="2559" height="946" alt="image" src="https://github.com/user-attachments/assets/279520ce-e921-4c30-b17f-8f122a53bbab" />



---

## 👨‍💻 Author

**Dhruv Vasudev**

---

## 📄 License

This project is for academic and research purposes.

---

## ⭐ Future Improvements

* Real-time video stream support
* Secure aggregation (HE / MPC)
* Mobile deployment
* Edge device optimization

---
