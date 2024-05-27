# Image Classifier using Tkinter and Keras

This is a simple image classifier application built using Python's Tkinter for the GUI, Pillow for image processing, and Keras (with TensorFlow backend) for machine learning. The application allows users to load an image and classify it as either a cat or a dog.

## Features

- Load an image from your file system.
- Resize and normalize the image to prepare it for classification.
- Predict if the image is a cat or a dog using a pre-trained Keras model.
- Display the image along with the prediction and confidence score.

## Requirements

- Python 3.x
- Tkinter (usually included with Python installations)
- Pillow
- TensorFlow
- Keras

## Installation

1. Clone the repository:

    ```sh
    git clone https://github.com/yourusername/image-classifier.git
    cd image-classifier
    ```

2. Install the required Python packages:

    ```sh
    pip install -r requirements.txt
    ```

3. Make sure you have `keras_model.h5` and `labels.txt` in the project directory.

## Usage

1. Run the application:

    ```sh
    python image_classifier.py
    ```

2. Click the "Load and Classify Image" button.
3. Select an image file from your computer.
4. The application will display the image along with the prediction (Cat or Dog) and the confidence score.

## Code Overview

The main components of the application are:

- **Loading and setting up the GUI**: The Tkinter library is used to create the main window and buttons.
- **Image processing**: Pillow is used to open, resize, and normalize the image.
- **Model prediction**: Keras is used to load the pre-trained model and make predictions.

### Main Application Code

```python
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageOps, ImageTk
from keras.models import load_model
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
        if class_name == '0':
            result_text = f"Cat\nConfidence Score: {confidence_score:.2f}"
        else:
            result_text = f"Dog\nConfidence Score: {confidence_score:.2f}"
        
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
```

## Files

- `image_classifier.py`: The main script containing the application code.
- `keras_model.h5`: The pre-trained Keras model file.
- `labels.txt`: A text file containing the class labels (0 for Cat, 1 for Dog).
- `requirements.txt`: The required Python packages.
## Output of The code
![image](https://github.com/PrathamGautam/Cats-And-Dogs-Detection-System/assets/142311958/0b5150cc-3010-4a8c-af1d-ec835861c2f6)
![image](https://github.com/PrathamGautam/Cats-And-Dogs-Detection-System/assets/142311958/4052a4af-41bb-4209-bc1e-c449b88d8f50)


## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- The Keras and TensorFlow teams for providing powerful machine learning tools.
- The Pillow team for providing a versatile image processing library.
- The Tkinter documentation and community for GUI support.

Feel free to open an issue or submit a pull request for any improvements or bug fixes. Enjoy classifying your images!
