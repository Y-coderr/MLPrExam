import matplotlib.pyplot as plt 
from sklearn.datasets import load_digits 
from sklearn.model_selection import train_test_split 
from sklearn.neural_network import MLPClassifier 
from sklearn.metrics import classification_report, accuracy_score 
# 1. Load digits dataset (0 to 9) 
digits = load_digits() 
# 2. Visualize some digits 
plt.gray() 
for i in range(4): 
    plt.matshow(digits.images[i]) 
plt.show() 
# 3. Features and labels 
X = digits.data  # 64 features (8x8 pixel images) 
y = digits.target  # Digit labels (0 to 9) 
# 4. Split the dataset 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) 
# 5. Create an ANN model (MLP) 
model = MLPClassifier(hidden_layer_sizes=(50,), activation='relu', max_iter=1000) 
# 6. Train the model 
model.fit(X_train, y_train) 
# 7. Predict 
y_pred = model.predict(X_test) 
# 8. Accuracy and Classification Report 
print("Accuracy:", accuracy_score(y_test, y_pred)) 
print("\nClassification Report:\n", classification_report(y_test, y_pred))
