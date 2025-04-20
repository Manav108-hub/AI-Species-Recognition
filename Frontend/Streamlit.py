import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import zipfile
import io

# Page configuration: Must be the first Streamlit command
st.set_page_config(
    page_title="Species Recognition",
    page_icon="🦁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define the CNN Model
class SpeciesModel(nn.Module):
    def __init__(self, num_classes):
        super(SpeciesModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(512, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = self.pool(self.relu(self.bn4(self.conv4(x))))
        x = self.global_pool(x)
        x = x.view(-1, 512)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Load model with caching
@st.cache_resource
def load_model(model_directory, num_classes, device):
    model = SpeciesModel(num_classes=num_classes)
    if os.path.exists(model_directory):
        model.load_state_dict(torch.load(model_directory, map_location=device))
    else:
        st.error("Model file not found. Please check the path.")
        return None
    model.to(device)
    model.eval()
    return model

# Function to predict species
def predict_species(image, model, transform, classes, device, defined_threshold=0.7):
    try:
        image_tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)
        if confidence.item() < defined_threshold:
            return None
        return classes[predicted_class.item()]
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None

# Main function
def main():
    st.title("🦁 Species Recognition")

    # Sidebar information
    with st.sidebar:
        st.header("About")
        st.write("Upload an image or a zip file of images to identify their species.")
        st.markdown("---")
        st.write("**Model Details:**")
        st.write("- CNN Architecture")
        st.write("- Input Size: 224x224")
        st.write("- Confidence defined_threshold: 70%")
        st.write(f"Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_path = "../Datasets/Train/animals"
    model_directory = "../model/best_model.pth"

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    classes = sorted(os.listdir(data_path))
    num_classes = len(classes)

    with st.spinner("Loading model..."):
        model = load_model(model_directory, num_classes, device)

    # Upload image or zip file
    uploaded_file = st.file_uploader("Upload an image or a zip file:", type=["jpg", "jpeg", "png", "zip"])

    if uploaded_file:
        output_zip = io.BytesIO()
        with zipfile.ZipFile(output_zip, mode="w", compression=zipfile.ZIP_DEFLATED) as zipf:
            if uploaded_file.name.endswith(".zip"):
                with zipfile.ZipFile(uploaded_file) as z:
                    image_files = [f for f in z.namelist() if f.lower().endswith(('png', 'jpg', 'jpeg'))]
                    if not image_files:
                        st.error("No image files found in the uploaded zip.")
                        return

                    st.write(f"Found {len(image_files)} image(s) in the zip file.")

                    for file_name in image_files:
                        with z.open(file_name) as f:
                            image = Image.open(f).convert("RGB")
                            species = predict_species(image, model, transform, classes, device)

                            if species:
                                img_buffer = io.BytesIO()
                                image.save(img_buffer, format="JPEG")
                                zipf.writestr(f"{species}/{os.path.basename(file_name)}", img_buffer.getvalue())
                                st.markdown(f"**Image:** {file_name} - Predicted Species: **{species}**")
                            else:
                                st.warning(f"Low confidence for image: {file_name}.")
            else:
                image = Image.open(uploaded_file).convert("RGB")
                st.image(image, caption="Uploaded Image", use_container_width=True)
                with st.spinner("Analyzing image..."):
                    species = predict_species(image, model, transform, classes, device)
                    if species:
                        img_buffer = io.BytesIO()
                        image.save(img_buffer, format="JPEG")
                        zipf.writestr(f"{species}/{uploaded_file.name}", img_buffer.getvalue())
                        st.success(f"Predicted Species: **{species}**")
                    else:
                        st.warning("Low confidence for the uploaded image.")

        # Provide download link for the ZIP file
        output_zip.seek(0)
        st.download_button(
            label="Download ZIP with Recognized Species",
            data=output_zip,
            file_name="recognized_species.zip",
            mime="application/zip"
        )

if __name__ == "__main__":
    main()
