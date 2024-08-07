import streamlit as st
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt

# Load the digits dataset
digits = load_digits()

# Streamlit title
st.title("Handwritten Digit Recognition")

# Split the dataset
X = digits.data
y = digits.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train the model
model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)

# Upload image section
st.subheader("Upload an Image of a Handwritten Digit")
uploaded_file = st.file_uploader("Choose an image...", type="png")

if uploaded_file is not None:
    # Convert the uploaded image to grayscale and invert it
    img = Image.open(uploaded_file).convert('L')
    img_inverted = ImageOps.invert(img)
    
    # Display the uploaded image
    st.image(img_inverted, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image for prediction
    arr = np.array(img_inverted.resize((8, 8), Image.Resampling.LANCZOS))  # Resize to 8x8 pixels
    arr1 = arr.flatten()  # Flatten the image
    
    # Normalize the pixel values to match the dataset's scale
    arr1 = arr1 / 16.0 * 16.0
    
    # Predict the digit
    prediction = model.predict([arr1])
    
    # Display the prediction
    st.subheader("Predicted Digit")
    st.write(prediction[0])

y_pred = model.predict(X_test)
st.subheader("Model Accuracy")
acc=accuracy_score(y_test,y_pred)
st.write(f"{acc*100:.2f}")

# Display the confusion matrix
st.subheader("Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
st.write(cm)

# Display a sample image from the dataset
st.subheader("Sample Image from Digits Dataset")
plt.figure(figsize=(2, 2))
plt.imshow(digits.images[8], cmap='gray')
st.pyplot(plt)
