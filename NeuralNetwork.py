import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.datasets import load_iris 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import LabelBinarizer, StandardScaler 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense 
from tensorflow.keras.utils import plot_model 
 
# 1. Load the dataset 
data = load_iris() 
X = data.data 
y = data.target 
 
# 2. Preprocess the data 
scaler = StandardScaler() 
X = scaler.fit_transform(X) 
 
# Convert labels to one-hot encoding 
encoder = LabelBinarizer() 
y = encoder.fit_transform(y) 
 
# 3. Split the dataset 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 
 
# 4. Create the Neural Network model 
model = Sequential() 
model.add(Dense(10, input_shape=(4,), activation='relu')) 
model.add(Dense(8, activation='relu')) 
model.add(Dense(3, activation='softmax')) 
 
# 5. Compile the model 
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) 
 
# 6. Train the model 
history = model.fit(X_train, y_train, epochs=50, batch_size=5, validation_data=(X_test, y_test), verbose=0) 
 
# 7. Plot training history 
plt.plot(history.history['accuracy'], label='Training Accuracy') 
plt.plot(history.history['val_accuracy'], label='Validation Accuracy') 
plt.title('Model Accuracy') 
plt.xlabel('Epoch') 
plt.ylabel('Accuracy') 
plt.legend() 
plt.grid(True) 
plt.show() 
 
# 8. Visualize the Model Architecture (text summary) 
model.summary() 
plot_model(model, to_file="model.png", show_shapes=True, show_layer_names=True)
