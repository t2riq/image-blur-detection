# 📷 Blur Detection using CNN

This project implements a **Convolutional Neural Network (CNN)** for classifying images as **blurry** or **sharp**. It is trained on a labeled image dataset and uses TensorFlow/Keras for model creation and training.

---

## 🧠 Project Summary

The goal of this project is to develop a deep learning model that can:
- Automatically detect whether an input image is **blurred or sharp**
- Classify images with high accuracy using a simple but effective CNN architecture

---

## 📂 Dataset

- Source: [Kaggle - Blur Detection Dataset](https://www.kaggle.com/datasets/kwentar/blur-dataset)
- Classes: `blur`, `sharp`
- Images are preprocessed to grayscale and resized to **64×64** pixels.

---

## 🧱 Model Architecture

A custom CNN model built using **Keras**:
- 3 Convolutional layers (Conv2D) with ReLU activation
- MaxPooling layers to reduce spatial dimensions
- Flatten layer followed by Dense (fully connected) layers
- Final output: 1 neuron with **sigmoid** activation for binary classification

---

## ⚙️ Technologies Used

- Python 3.x
- TensorFlow / Keras
- NumPy
- OpenCV (for image preprocessing)
- Matplotlib & Seaborn (for visualization)
- Scikit-learn (metrics and confusion matrix)

---

## 🚀 How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/blur-detection-cnn.git
   cd blur-detection-cnn
    ```
2. **Install dependencies**
    ```
    pip install -r requirements.txt
    ```
3. **Prepare the dataset**
    Download the dataset from Kaggle and place it in the data/ folder or set the correct path in the notebook/code.
4. **Run the training script**
   ```
    python train.py
   ```
---
## 📊 Evaluation

The model was evaluated on a held-out test set and achieved strong classification performance:

| Metric         | Score |
|----------------|-------|
| **Accuracy**   | 95%   |
| **Precision**  | 97% (Class 0 - Blurry), 92% (Class 1 - Sharp) |
| **Recall**     | 94% (Class 0), 97% (Class 1) |
| **F1-Score**   | 95% (Blurry), 94% (Sharp) |
| **Support**    | 79 blurry, 61 sharp images |

> **Macro Avg F1-Score**: 95%  
> **Weighted Avg F1-Score**: 95%

The confusion matrix and classification report demonstrate that the model is highly reliable at distinguishing between **blurry** and **sharp** images.

---
##📈 Results Visualization
-The training and validation accuracy and loss over epochs are plotted:
```
# Code used for plotting (already in notebook)
```
---

##📄 License

This project is licensed under the MIT License.
---
📬 Contact

For questions or suggestions:
📧 tariqalajam@gmail.com
