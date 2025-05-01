import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# CUDA Configuration - Add these lines to ensure GPU is used
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    torch.cuda.empty_cache()  # Clear GPU memory
    current_device = torch.cuda.current_device()
    print(f"Current CUDA device: {current_device} - {torch.cuda.get_device_name(current_device)}")
    print(f"Device capability: {torch.cuda.get_device_capability(current_device)}")
    print(f"GPU Memory Usage:")
    print(f"  Allocated: {torch.cuda.memory_allocated(current_device) / 1024**2:.2f} MB")
    print(f"  Cached: {torch.cuda.memory_reserved(current_device) / 1024**2:.2f} MB")
else:
    print("CUDA is not available. Using CPU instead.")

# For deterministic results
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Set to True for speed if not needing reproducibility

# Define the unified model architecture
class FusionNet(nn.Module):
    def __init__(self, num_classes):
        super(FusionNet, self).__init__()
        
        # Custom CNN Stream
        self.cnn_stream = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # EfficientNet Stream
        self.effnet = models.efficientnet_b0(weights="IMAGENET1K_V1")
        self.effnet.classifier = nn.Identity()  # Remove original classifier
        
        # Feature Fusion Module
        self.fusion = nn.Sequential(
            nn.Linear(512 + 1280, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        # Freeze EfficientNet initially
        for param in self.effnet.parameters():
            param.requires_grad = False

    def forward(self, x):
        # Process through CNN stream
        cnn_features = self.cnn_stream(x).flatten(1)
        
        # Process through EfficientNet stream
        eff_features = self.effnet(x)
        
        # Concatenate features
        combined = torch.cat((cnn_features, eff_features), dim=1)
        
        # Final classification
        return self.fusion(combined)

# Dataset and transforms
class AnimalDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.classes = sorted(os.listdir(data_dir))
        self.image_paths = []
        self.labels = []
        
        for label, cls in enumerate(self.classes):
            cls_dir = os.path.join(data_dir, cls)
            for img in os.listdir(cls_dir):
                self.image_paths.append(os.path.join(cls_dir, img))
                self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            return torch.zeros((3, 224, 224)), label

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(0.2),
    transforms.RandomRotation(15),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
    transforms.RandomAffine(0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Training function with GPU monitoring
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    best_acc = 0.0  # Track the best validation accuracy
    
    # Track GPU usage every epoch
    def print_gpu_stats():
        if device.type == 'cuda':
            print(f"  GPU Memory: {torch.cuda.memory_allocated(device) / 1024**2:.2f}MB allocated, "
                  f"{torch.cuda.memory_reserved(device) / 1024**2:.2f}MB cached")
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Enable CUDA stream synchronization for more accurate profiling
        torch.cuda.synchronize() if device.type == 'cuda' else None
        
        print(f"Starting Epoch {epoch+1}/{num_epochs}")
        print_gpu_stats()
        
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        
        train_loss = running_loss / len(train_loader)
        train_acc = correct / total
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")

        # Validation Phase
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total
        print(f"Validation Accuracy: {val_acc:.4f}")
        print_gpu_stats()

        # Save the best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            print(f"Model saved with Accuracy: {best_acc:.4f}")

    return model  # Return trained model


# Main execution
if __name__ == "__main__":
    # Set CUDA device priority
    if torch.cuda.is_available():
        # Force PyTorch to use your RTX 4060
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use the first (and likely only) GPU
        
        # Set device options
        device = torch.device('cuda')
        
        # Explicitly set CUDA performance options for your RTX 4060
        torch.backends.cudnn.benchmark = True  # Use cuDNN auto-tuner
        print("CUDA device set to:", torch.cuda.get_device_name(device))
    else:
        device = torch.device('cpu')
        print("CUDA not available, using CPU")
    
    # Configuration
    base_dir = os.path.dirname(os.path.abspath(__file__))  # Get script's directory
    data_dir = os.path.abspath(os.path.join(base_dir, "../Datasets"))  # Correct relative path

    # Debugging: Check if the path exists
    print("Resolved Data Directory:", data_dir)
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Dataset path does not exist: {data_dir}")

    # Optimize batch size for GPU training
    batch_size = 64  # Increased for better GPU utilization with RTX 4060
    num_epochs = 30
    num_classes = len(os.listdir(data_dir))
    
    # Dataset setup with worker optimization for faster loading
    full_dataset = AnimalDataset(data_dir, train_transform)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_set, val_set = random_split(full_dataset, [train_size, val_size])
    val_set.dataset.transform = val_transform  # Apply val transforms
    
    # Use more workers for data loading with pinned memory for faster GPU transfer
    num_workers = 4  # Usually set to number of CPU cores 
    train_loader = DataLoader(train_set, batch_size, shuffle=True, 
                             num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size, shuffle=False, 
                           num_workers=num_workers, pin_memory=True)
    
    # Model setup
    model = FusionNet(num_classes).to(device)
    
    # Check if the model is on CUDA
    print(f"Model is on CUDA: {next(model.parameters()).is_cuda}")
    
    # Training parameters 
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    
    # Optional: Add learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.1)
    
    # Start training
    print("=== Starting Training on", "GPU" if device.type == "cuda" else "CPU", "===")
    history = train_model(
        model, train_loader, val_loader,
        criterion, optimizer, num_epochs, device
    )
    
    # Final evaluation
    model = FusionNet(num_classes).to(device)  # Create fresh model instance
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()
    
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', 
                xticklabels=full_dataset.classes,
                yticklabels=full_dataset.classes)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('confusion_matrix.png')
    
    # Classification report
    report = classification_report(all_labels, all_preds, 
                                  target_names=full_dataset.classes)
    print("Classification Report:\n", report)
    with open('classification_report.txt', 'w') as f:
        f.write(report)
        
    # Final GPU stats
    if device.type == 'cuda':
        print(f"Peak GPU Memory Usage: {torch.cuda.max_memory_allocated(device) / 1024**3:.2f} GB")