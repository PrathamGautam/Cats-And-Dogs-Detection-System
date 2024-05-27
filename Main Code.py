import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageOps, ImageTk
from keras.models import load_model  # TensorFlow is required for Keras to work
import numpy as np

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("keras_model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

# Create the main window
root = tk.Tk()
root.title("Image Classifier")

# Function to load and classify the image
def classify_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        # Load the image
        image = Image.open(file_path).convert("RGB")
        
        # Resize and crop the image
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
        
        # Convert image to numpy array
        image_array = np.asarray(image)
        
        # Normalize the image
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
        
        # Create the array of the right shape to feed into the keras model
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        data[0] = normalized_image_array
        
        # Predict using the model
        prediction = model.predict(data)
        index = np.argmax(prediction)
        class_name = class_names[index].strip()
        confidence_score = prediction[0][index]
        
        # Determine if the image is a dog or cat
        if class_name==0:
            result_text = f"Dog\nConfidence Score: {confidence_score:.2f}"
        else:
            result_text = f"Cat\nConfidence Score: {confidence_score:.2f}"
        
        # Display the prediction and confidence score
        result_label.config(text=result_text)
        
        # Display the image
        tk_image = ImageTk.PhotoImage(image)
        image_label.config(image=tk_image)
        image_label.image = tk_image

# Create and place widgets
load_button = tk.Button(root, text="Load and Classify Image", command=classify_image)
load_button.pack(pady=20)

result_label = tk.Label(root, text="", font=("Helvetica", 14))
result_label.pack(pady=20)

image_label = tk.Label(root)
image_label.pack(pady=20)

# Run the application
root.mainloop()
