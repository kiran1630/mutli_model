import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2

def load_model(model_path):
    """Load the landslide detection model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(model_path, map_location=device)
    model.eval()
    return model




transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Ensure correct size
        transforms.ToTensor(),  # Converts to (C, H, W) format
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def preprocess_image(image, target_size=(128, 128)):
    image = np.array(image)
    image = cv2.resize(image, target_size)
    image = image / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

def predict(image, model):
    # Ensure the image is in the correct shape and format
    if isinstance(image, np.ndarray):
        if image.ndim == 4:  # Remove unnecessary dimensions if present
            image = np.squeeze(image)  # Removes batch/channel dimensions
        if image.dtype != np.uint8:  # Convert to uint8 if needed
            image = (image * 255).astype(np.uint8)
        
        image = Image.fromarray(image)  # Convert NumPy array to PIL image

    image = transform(image).unsqueeze(0)  # Preprocess and add batch dimension

    with torch.no_grad():
        output = model(image)
        predicted_class = torch.argmax(output, dim=1).item()

    class_names = ["landslide", "non-landslide"]
    return class_names[predicted_class]