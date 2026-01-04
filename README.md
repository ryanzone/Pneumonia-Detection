# 🫁 Pneumonia Detection from Chest X-Ray Images

This project implements a **deep learning–based Pneumonia Detection system** using **PyTorch and Transfer Learning (ResNet-18)**.
The model classifies chest X-ray images into **Normal** or **Pneumonia** and is deployed as an interactive **Streamlit web application** for real-time inference.

---

## 📌 Project Description

Pneumonia is a lung infection that can be diagnosed through chest X-ray images. Manual diagnosis requires expert analysis and may be time-consuming.
This project aims to automate the detection process using a **Convolutional Neural Network (CNN)**, improving speed and consistency while serving as an educational demonstration of medical image classification.

---

## 🔑 Key Features

* Transfer learning using **ResNet-18 (ImageNet pretrained)**
* Fine-tuning only **Layer 4 and the final fully connected layer**
* **Class-weighted loss** to address dataset imbalance
* Threshold-based prediction to improve pneumonia sensitivity
* Validation-based best model saving
* Streamlit web interface for easy image upload and prediction
* GPU acceleration support (CUDA)

---

## 🛠 Technologies Used

* **Python**
* **PyTorch & Torchvision**
* **Scikit-learn**
* **Streamlit**
* **NumPy, Matplotlib**
* **PIL (Image Processing)**

---

## 📂 Dataset Structure

```
chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

The dataset contains labeled chest X-ray images categorized into Normal and Pneumonia classes.

---

## ⚙️ Model & Training Details

* Base model: **ResNet-18**
* Input size: `224 × 224`
* Loss function: **Weighted CrossEntropyLoss**
* Optimizer: **Adam**
* Batch size: `64`
* Epochs: `5`
* Best-performing model saved as `best_model.pth`

Only selected layers are unfrozen during training to reduce overfitting and improve generalization.

---

## 📊 Evaluation

* Predictions generated using softmax probabilities
* Pneumonia classification threshold set to **0.35**
* Evaluation includes:

  * Confusion Matrix
  * Precision, Recall, and F1-score

This threshold tuning improves detection of pneumonia cases, which is important in medical applications.

---

## 🌐 Streamlit Web Application

The trained model is deployed using **Streamlit**.

**Features:**

* Upload chest X-ray images (JPG / PNG)
* Displays predicted class and confidence score
* Shows class-wise probabilities
* Includes medical disclaimer and system information

---

## ▶️ How to Run the Project

```bash
pip install torch torchvision streamlit scikit-learn matplotlib pillow tqdm
```

```bash
python train.py
streamlit run app.py
```

---

## ⚠️ Disclaimer

This project is intended for **educational and research purposes only** and must not be used for real-world medical diagnosis.

---

