import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.cuda.amp import autocast
from tqdm import tqdm
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from torch.nn.functional import mse_loss
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import timm

# Define the model with latent and classification heads
class ImprovedModel(nn.Module):
    def __init__(self, latent_dim=1024, num_classes=31):
        super(ImprovedModel, self).__init__()
        self.encoder = timm.create_model('convnext_base', pretrained=True)
        self.encoder_head_dim = self.encoder.get_classifier().in_features
        self.encoder.reset_classifier(0)  # Remove default classifier
        self.fc_latent = nn.Linear(self.encoder_head_dim, latent_dim)
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Linear(latent_dim, num_classes)
        )

    def forward(self, x):
        features = self.encoder(x)
        latent = self.fc_latent(features)
        output = self.classifier(latent)
        return output, latent

# Consistency loss (Mean Squared Error)
consistency_loss = nn.MSELoss()

# Training function with multiview consistency
def train_with_multiview_consistency(model, dataloader, n_epochs=10, lr=1e-4, weight_decay=1e-4, consistency_weight=0.5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler()  # Updated to the new version of GradScaler

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        correct_predictions = 0
        total_samples = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch [{epoch+1}/{n_epochs}]")

        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            # Generate two augmented views of the input images in PIL format
            pil_images = [transforms.ToPILImage()(img.cpu()) for img in images]
            aug_view_1 = torch.stack([transform_base(augmentation_1(img)) for img in pil_images]).to(device)
            aug_view_2 = torch.stack([transform_base(augmentation_2(img)) for img in pil_images]).to(device)

            with autocast():
                # Forward pass for both views
                _, latent_1 = model(aug_view_1)
                _, latent_2 = model(aug_view_2)
                
                # Classification output for the original image (use one of the augmented views for simplicity)
                outputs, _ = model(aug_view_1)
                
                # Classification loss
                classification_loss = criterion(outputs, labels)
                
                # Consistency loss between two latent views
                latent_consistency_loss = consistency_loss(latent_1, latent_2)
                
                # Combined loss
                loss = classification_loss + consistency_weight * latent_consistency_loss

            # Backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Accuracy': f'{(correct_predictions / total_samples * 100):.2f}%'
            })

        avg_loss = total_loss / len(dataloader)
        accuracy = correct_predictions / total_samples * 100
        print(f'Epoch [{epoch+1}/{n_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')


# Define transformations that apply on PIL Images
transform_base = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),  # Final conversion to tensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Different augmentation pipelines for creating views
augmentation_1 = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
])

augmentation_2 = transforms.Compose([
    transforms.RandomRotation(20),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
])

# Load target domain dataset
target_dir = "/kaggle/input/office31/Office31/Office31_dslr"  # Adjust as needed
dataset_target = datasets.ImageFolder(root=target_dir, transform=transform_base)
dataloader_target = DataLoader(dataset_target, batch_size=4, shuffle=True)

# Initialize model
model = ImprovedModel(latent_dim=1024, num_classes=31)

# Train model with multiview consistency
train_with_multiview_consistency(model, dataloader_target, n_epochs=10, lr=1e-4, weight_decay=1e-4, consistency_weight=0.5)


# Define transformation for the test set (ensure it matches the training normalization)
transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load the test dataset for the new domain
test_dir = "/kaggle/input/office31/Office31/Office31_webcam"  # Path to the new domain dataset
dataset_test = datasets.ImageFolder(root=test_dir, transform=transform_test)
dataloader_test = DataLoader(dataset_test, batch_size=64, shuffle=False)

# Evaluation function for accuracy on new domain
def evaluate_on_new_domain(model, dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    total_samples = 0
    correct_predictions = 0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating on new domain"):
            images, labels = images.to(device), labels.to(device)

            with autocast():
                outputs, _ = model(images)  # Get the prediction
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

    accuracy = correct_predictions / total_samples * 100
    print(f'Accuracy on new domain: {accuracy:.2f}%')
    return accuracy

# Visualize latent space on the new domain using t-SNE
def visualize_latent_space(model, dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    latents = []
    labels = []
    with torch.no_grad():
        for images, lbls in dataloader:
            images = images.to(device)
            _, latent = model(images)
            latents.append(latent.cpu())
            labels.append(lbls)

    latents = torch.cat(latents).numpy()
    labels = torch.cat(labels).numpy()
    
    # Use t-SNE for visualization
    tsne = TSNE(n_components=2, random_state=42)
    latents_2d = tsne.fit_transform(latents)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(latents_2d[:, 0], latents_2d[:, 1], c=labels, cmap="viridis", alpha=0.7)
    plt.colorbar(scatter, label="Class Labels")
    plt.title("t-SNE Visualization of Latent Space on Test Domain")
    plt.show()

# Consistency check across different augmentations on the new domain
def check_consistency(model, dataloader, augmentation_1, augmentation_2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    consistency_losses = []
    with torch.no_grad():
        for images, _ in tqdm(dataloader, desc="Consistency Check"):
            # Generate two augmented views of each image in PIL format
            pil_images = [transforms.ToPILImage()(img.cpu()) for img in images]
            aug_view_1 = torch.stack([transform_test(augmentation_1(img)) for img in pil_images]).to(device)
            aug_view_2 = torch.stack([transform_test(augmentation_2(img)) for img in pil_images]).to(device)

            # Pass through the model and get latent representations
            _, latent_1 = model(aug_view_1)
            _, latent_2 = model(aug_view_2)

            # Compute MSE consistency loss for the batch
            batch_consistency_loss = mse_loss(latent_1, latent_2).item()
            consistency_losses.append(batch_consistency_loss)

    avg_consistency_loss = sum(consistency_losses) / len(consistency_losses)
    print(f"Average Consistency Loss on New Domain: {avg_consistency_loss:.4f}")
    return avg_consistency_loss

# Define augmentations for consistency check
augmentation_1 = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)])
augmentation_2 = transforms.Compose([transforms.RandomRotation(20), transforms.RandomResizedCrop(224, scale=(0.8, 1.0))])

# Run the evaluations on the new domain
print("1. Evaluating model accuracy on new domain:")
evaluate_on_new_domain(model, dataloader_test)

print("\n2. Visualizing latent space with t-SNE:")
visualize_latent_space(model, dataloader_test)

print("\n3. Checking latent space consistency across augmentations:")
check_consistency(model, dataloader_test, augmentation_1, augmentation_2)
