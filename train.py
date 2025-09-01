import joblib
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

# Load digits dataset
digits = datasets.load_digits()
X, y = digits.data, digits.target

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train SVM model (RBF Kernel works well here)
svm_model = SVC(kernel='rbf', gamma=0.01, C=10)
svm_model.fit(X_train, y_train)

# Save model
joblib.dump(svm_model, "svm_digits_model.pkl")
print("✅ Model trained and saved as svm_digits_model.pkl")