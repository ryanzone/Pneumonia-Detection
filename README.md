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
=======
# AI Chest X-Ray Classifier

An AI-powered diagnostic support application designed to classify chest X-ray images as **Normal** or **Pneumonia**. The project features a fine-tuned deep learning backend in PyTorch and a premium clinical dashboard built with Streamlit.

---

## 🚀 Key Features

* **Deep Learning Classification**: Powered by a fine-tuned **ResNet18** model trained to identify diagnostic indicators of pneumonia in chest radiographs.
* **Refined Clinical Dashboard**: A premium GitHub Dark styled medical web application layout with soft shadows, custom sliders, and a sleek dark mode palette tailored for clear data visualization.
* **Side-by-Side Image Viewer**: View the original X-ray alongside a real-time enhanced version.
* **Interactive Image Adjustments**: Fine-tune Brightness, Contrast, and Sharpness on the fly for closer visual inspection.
* **Visual Diagnosis & Probability Chart**: Get instant diagnosis feedback via soft glowing alert cards (green for Normal, red for Pneumonia) paired with an interactive probability distribution bar chart.
* **Grouped System Information**: A dedicated sidebar showing live hardware details (CPU/GPU status, VRAM consumption) and model settings.
* **Diagnostic Reports**: Generate and download diagnostic summary reports in Markdown (`.md`) or Plain Text (`.txt`) formats.
* **Session Classification History**: Track classification logs locally during the session with an option to download history as a CSV file.

---

## 🧠 Model Architecture & Training

The backend deep learning model is defined in `model_arch.py` and utilizes the following setup:

* **Base Model**: Pre-trained **ResNet18** architecture from PyTorch `torchvision`.
* **Fine-Tuning Strategy**: Feature extraction layers (Layers 1-3) are frozen to preserve generalized ImageNet representations, while **Layer 4** and the final **Fully Connected (FC) head** are unfrozen for chest X-ray domain specialization.
* **Loss Function**: Weighted Cross-Entropy Loss, automatically adjusting class weights based on label counts to mitigate dataset class imbalance:
  $$\text{Weight}_c = \frac{N_{\text{total}}}{N_c}$$
* **Optimizer**: Adam optimizer with a learning rate of $1\times10^{-4}$ trained for 5 epochs.
* **Decision Boundary**: The Pneumonia classification threshold is tuned to **$0.35$** (instead of standard $0.5$) to prioritize medical recall/sensitivity, ensuring fewer false negatives in clinical screenings.

---

## 🖼️ Dashboard Preview

### Updated GitHub Dark Theme
![Dashboard Overview](ss/up1.png)
![Diagnostic Details](ss/up2.png)
![Enhanced Image Analysis](ss/up3.png)

### Previous Theme (For Comparison)
![Old Dashboard](ss/old.png)

---

## 📂 Project Structure

```filepath
ICM/
├── streamlit_app.py     # Streamlit Clinical Dashboard (Web App)
├── model_arch.py        # Model definition, Data loaders, & Training pipeline
├── inference.py         # Lightweight script for loading model and testing images
├── best_model.pth       # Saved model checkpoint weights (approx. 45MB)
├── requirements.txt     # Python package dependencies
├── confusion_matrix.png # Generated performance visualization on the validation set
├── samples/             # Sample chest X-ray images (.jpeg) for dashboard testing
└── chest_xray/          # Kaggle dataset directory (Normal/Pneumonia train & test splits)
```

---

## 🛠️ Installation & Setup

### Prerequisites
* Python 3.10+
* CUDA-capable GPU (optional, but recommended for model training)

### 1. Clone & Navigate
```bash
git clone <repository-url>
cd ICM
```

### 2. Configure Virtual Environment
Create and activate a Python virtual environment to manage dependencies:
```powershell
# On Windows
python -m venv venv
.\venv\Scripts\activate

# On Linux/macOS
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
>>>>>>> ec2e48b (feat: add inference script, documentation, and Streamlit dashboard enhancements for the pneumonia classifier)
```

---

<<<<<<< HEAD
## ⚠️ Disclaimer

This project is intended for **educational and research purposes only** and must not be used for real-world medical diagnosis.

---

=======
## 🖥️ Running the Clinical Dashboard

To launch the web interface, run the following command within your activated virtual environment:
```bash
streamlit run streamlit_app.py
```

The application will build the custom CSS theme and open a browser window displaying the dashboard (usually at `http://localhost:8501`).

### Testing the App
1. You can upload a custom `.jpg` / `.png` chest X-ray or select one of the sample images pre-loaded in the dropdown menu.
2. The dashboard will display the Original and Enhanced images side by side.
3. Review the **Diagnosis Card** (showing prediction confidence) and the **Class Probabilities Chart** below the viewer.
4. Download the clinical report or check the **History** tab to see previously logged classifications.

---

## 🏋️ Training the Model from Scratch

If you want to re-train the classifier model:
1. Download the [Chest X-Ray Images (Pneumonia) dataset from Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia).
2. Unpack the dataset into a folder named `chest_xray` in the root of the project directory.
3. Run the training script:
```bash
python model_arch.py
```
This will train the model, tune the threshold, print a classification performance report, save the best weights to `best_model.pth`, and render the `confusion_matrix.png` plot.

