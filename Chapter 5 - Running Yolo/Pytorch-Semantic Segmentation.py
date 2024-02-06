import torch
from torchvision import models, transforms
from PIL import Image
import cv2
import numpy as np

# Load a pre-trained DeepLabV3 model
model = models.segmentation.deeplabv3_resnet101(pretrained=True)
model.eval()

# Define the standard transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load the image
image_path = "../Images/The_War_Room.jpg"  # Update to your image path
input_image = Image.open(image_path)

# Preprocess the image and add a batch dimension
input_tensor = transform(input_image).unsqueeze(0)

# Forward pass: compute the output of the model
with torch.no_grad():
    output = model(input_tensor)['out'][0]
output_predictions = output.argmax(0)

# Create a color palette for visualization
palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
colors = (colors % 255).numpy().astype("uint8")

# Prepare the segmented image
r = Image.fromarray(output_predictions.byte().cpu().numpy()).resize(input_image.size)
r.putpalette(colors)
segmented_image = r.convert('RGB')

# Convert PIL images to OpenCV format
original_image = np.array(input_image)
segmented_image = np.array(segmented_image)

# Display the original and segmented images using OpenCV
cv2.imshow('Original Image', original_image)
cv2.imshow('Segmented Image', segmented_image)

# Wait for the ESC key and then destroy all windows
while True:
    if cv2.waitKey(1) == 27:  # ESC key code
        break

cv2.destroyAllWindows()
