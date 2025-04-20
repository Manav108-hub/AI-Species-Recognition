import torch
from torchvision import transforms
from PIL import Image
import os
import numpy as np

# Function to compute the mean and std of the training dataset
def compute_mean_std(data_dir):
    means = []
    stds = []
    for class_name in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, class_name)
        if os.path.isdir(class_dir):
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                img = Image.open(img_path).convert("RGB")
                img = np.array(img)
                means.append(np.mean(img, axis=(0, 1)))
                stds.append(np.std(img, axis=(0, 1)))

    means = np.mean(means, axis=0)
    stds = np.mean(stds, axis=0)
    return means, stds

# Compute mean and std based on your training dataset
data_dir = "E:/Minor-Project/Datasets/Train/animals"  # Path to training data directory
mean, std = compute_mean_std(data_dir)

# Define the CNN architecture (same as in training)
class CNNModel(torch.nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.relu = torch.nn.ReLU()
        self.pool = torch.nn.MaxPool2d(2, 2)

        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(64)

        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = torch.nn.BatchNorm2d(128)

        self.conv4 = torch.nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = torch.nn.BatchNorm2d(256)

        self.fc1 = torch.nn.Linear(256 * 14 * 14, 512)
        self.dropout = torch.nn.Dropout(0.5)
        self.fc2 = torch.nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = self.pool(self.relu(self.bn4(self.conv4(x))))

        x = x.view(-1, 256 * 14 * 14)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Paths
model_path = r"E:\Minor-Project\animal_species_cnn.pth"  # Path to your saved model
classes = sorted(os.listdir(data_dir))  # List of class names in sorted order
num_classes = len(classes)  # Number of classes

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNModel(num_classes=num_classes)

# Load only the model weights safely
model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))  # Load only the model weights
model.to(device)
model.eval()  # Set model to evaluation mode

# Define the image transformation with the calculated mean and std
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize image to match training size
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)  # Use dataset-specific normalization
])

# Function to predict the species from an image
def predict_species(image_path):
    try:
        # Load and preprocess the image
        image = Image.open(image_path).convert("RGB")
        image = transform(image).unsqueeze(0)  # Add batch dimension

        # Move the image to the same device as the model
        image = image.to(device)

        # Perform inference
        with torch.no_grad():
            outputs = model(image)
            _, predicted_class = torch.max(outputs, 1)  # Get the index of the highest score

        # Map the predicted index to the class name
        predicted_species = classes[predicted_class.item()]
        return predicted_species
    except Exception as e:
        return f"Error in prediction: {str(e)}"

# Test the prediction function
if __name__ == "__main__":
    test_image_path = r"E:\Minor-Project\Datasets\Train\animals\Elephant\ind_corb_03_2014_001_P003_78852219_29456361_c002a_03_04_2014_21_57_54_P_1045.jpg"  # Update with the path to your test image
    predicted_species = predict_species(test_image_path)
    print(f"The predicted species is: {predicted_species}")
