# 🧠 SVM Digits Classifier

A Machine Learning project using **Support Vector Machine (SVM)** to classify handwritten digits (from the sklearn digits dataset).  
Deployed with **Streamlit** for interactive predictions.  

---

## 📊 Dataset
- Source: `sklearn.datasets.load_digits`
- Contains **8x8 pixel grayscale images** of digits (0–9).

Example visualization:

![Sample Digits](images/sample_digits.png)

---

## ⚙️ How It Works
1. Load the dataset  
2. Split into train/test  
3. Train an SVM classifier (`rbf` kernel)  
4. Evaluate accuracy  
5. Deploy model with **Streamlit**  

---

## 🚀 Running Locally

Clone the repo:
```bash
git clone https://github.com/yourusername/svm-digits-classifier.git
cd svm-digits-classifier
