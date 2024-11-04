import torch
from facenet_pytorch import InceptionResnetV1
from PIL import Image
import numpy as np
import base64
import torchvision.transforms as transforms
import os

# Load your image
# Define the relative path from the project root
image_path = "stored_faces/zach-selfie.jpg"
image = Image.open(image_path)

# Define the necessary transformations for face embeddings
preprocess = transforms.Compose([
    transforms.Resize(160),
    transforms.CenterCrop(160),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Preprocess the image
input_tensor = preprocess(image)
input_batch = input_tensor.unsqueeze(0)  # Create a mini-batch as expected by the model

# Check if a GPU is available and use it if possible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the InceptionResnetV1 model pre-trained on VGGFace2
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Move the input to the same device as the model
input_batch = input_batch.to(device)

# Get the embeddings
with torch.no_grad():
    embeddings = model(input_batch)

# Convert the embeddings to a NumPy array and then to a Base64 string
embeddings_np = embeddings.cpu().numpy()
embeddings_bytes = embeddings_np.tobytes()
embeddings_base64 = base64.b64encode(embeddings_bytes).decode('utf-8')

print(embeddings_base64)
