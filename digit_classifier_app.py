import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

def predict_digit(image_path, model_path):
    """
    This function takes the path to an image and a trained model,
    preprocesses the image, and predicts the digit.

    Parameters:
    image_path (str): The file path to the handwritten digit image.
    model_path (str): The file path to the trained model (.h5 format).
    
    Returns:
    int: The predicted digit (0-9).
    """

    # Step 1: Load and Preprocess the Image
    try:
        image = Image.open(image_path).convert('L')  # Convert image to grayscale
        image = image.resize((28, 28))               # Resize to 28x28 pixels (same as MNIST)
        image = np.invert(image)                     # Invert colors if needed (white on black)
        image = np.array(image) / 255.0              # Normalize pixel values (0-1 range)
        
        # Reshape to match model input: (1, 28, 28, 1)
        image = np.expand_dims(image, axis=0)        # Add batch dimension
        image = np.expand_dims(image, axis=-1)       # Add channel dimension (grayscale)
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

    # Step 2: Load the Trained Model
    try:
        model = load_model(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

    # Step 3: Make Prediction
    prediction = model.predict(image)
    predicted_digit = np.argmax(prediction)

    # Step 4: Display the Image and the Prediction
    plt.imshow(image.squeeze(), cmap='gray')
    plt.title(f'Predicted Digit: {predicted_digit}')
    plt.axis('off')
    plt.show()

    return predicted_digit

# Example usage:
if __name__ == "__main__":
    # Replace 'your_digit_image.png' with the path to your image
    # and 'mnist_digit_classifier.h5' with your trained model
    image_path = 'image2.jpeg'
    model_path = 'mnist_digit_classifier.h5'
    
    predicted_digit = predict_digit(image_path, model_path)
    if predicted_digit is not None:
        print(f'The model predicted: {predicted_digit}')

    

    