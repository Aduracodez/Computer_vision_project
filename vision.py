import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np

# 1. Load the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# 2. Normalize the data (scaling pixel values to [0, 1])
x_train, x_test = x_train / 255.0, x_test / 255.0

# 3. Define the CNN model
model = models.Sequential([
    # Convolutional layer with 32 filters, kernel size 3x3, and ReLU activation
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    
    # Another convolutional layer with 64 filterspyth
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # Another convolutional layer with 128 filters
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # Flatten the 3D output to 1D for the fully connected layers
    layers.Flatten(),
    
    # Fully connected layer with 128 neurons
    layers.Dense(128, activation='relu'),
    
    # Output layer with 10 units (one for each class) and softmax activation
    layers.Dense(10, activation='softmax')
])

# 4. Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 5. Train the model
history = model.fit(x_train, y_train, epochs=9, batch_size=64, validation_data=(x_test, y_test))

# 6. Evaluate the model on test data
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc}")

# 7. Visualize training history (accuracy and loss curves)
plt.figure(figsize=(12, 5))

# Plot training & validation accuracy values
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training accuracy')
plt.plot(history.history['val_accuracy'], label='Validation accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# 8. Make a prediction on a test image
# Class names for CIFAR-10 dataset
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Select a random image from the test set
image = x_test[6]
image = np.expand_dims(image, axis=0)  # Add batch dimension

# Predict the class
prediction = model.predict(image)
predicted_class = prediction.argmax()

# Display the image with the predicted class name
plt.imshow(x_test[1])
plt.title(f"Predicted Class: {class_names[predicted_class]}")
plt.show()
