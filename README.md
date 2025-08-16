# Hand Sign Language Digit Classification (0–5) 

This project focuses on building a deep learning model to classify hand sign language digits (from 0 to 5). It demonstrates the use of **Convolutional Neural Networks (CNN)** and **ResNet50 (transfer learning)** for image classification tasks.

---

## 📂 Folder Structure

```
HandSignRecognition/
│── dataset/                  # Training dataset (from Kaggle)
│── testing-images/           # Test images for prediction
│── testing-images-results/   # Predicted results saved here
│── models/                   # Saved trained models
│── notebooks/                # Jupyter notebooks for experiments
│── app.py                    # Streamlit app for predictions
│── requirements.txt          # Required Python packages
│── README.md                 # Project documentation
```

---

## 📥 Dataset

The dataset is publicly available on **Kaggle**:
[Hand Sign Language Digit Dataset (0–5)](https://www.kaggle.com/datasets/shivam1711/hand-sign-language-digit-dataset-for-0-5)

---

## 🖥 Usage

1. Open the Streamlit app.
2. Upload an image of a hand sign (from `testing-images/` or any new image).
3. The app preprocesses the image and passes it to the trained ResNet50 model.
4. The predicted class label and confidence scores are displayed.
5. Predicted results are saved in the `testing-images-results/` folder.

---

## 🛠 Technologies Used

* Python 3
* TensorFlow / Keras
* NumPy, Pandas, Matplotlib
* Streamlit
* OpenCV

---

## 🚀 How to Run

1. Clone this repository:

   ```bash
  git clone https://github.com/Faizan-Ali-Memon/HandSignClassifier.git
  cd HandSignClassifier

   ```
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:

   ```bash
   streamlit run app.py
   ```
