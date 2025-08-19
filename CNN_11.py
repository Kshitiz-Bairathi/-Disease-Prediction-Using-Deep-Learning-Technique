import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from torch.optim.lr_scheduler import ReduceLROnPlateau

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "5"
# device = torch.device("cuda:0")  # Now PyTorch sees GPU 4 as GPU 0

basic_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # Normalize to [-1, 1]
])

# Load Dataset with Basic Transform
data_path = 'combined_images'
dataset = datasets.ImageFolder(root=data_path, transform=basic_transform)

# Convert Dataset Labels to Numpy Array for Stratification
labels = np.array([label for _, label in dataset.samples])

# Convert Original Dataset Images and Labels into Tensors
original_images, original_labels = [], []
for img, label in dataset:
    original_images.append(img)
    original_labels.append(label)

original_images = torch.stack(original_images)
original_labels = torch.tensor(original_labels)

# print("Before Augmentation:- ")
print(original_images.shape)  # Should match (N, 3, 256, 256)
print(original_labels.shape)  # Should match (N,)

combined_images = original_images
combined_labels = original_labels

# Define CNN Model
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),  # (3, 256, 256) -> (16, 256, 256)
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),  # (16, 256, 256) -> (16, 256, 256)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (16, 128, 128)

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),  # (16, 128, 128) -> (32, 128, 128)
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),  # (32, 128, 128) -> (32, 128, 128)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (32, 64, 64)

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # (32, 64, 64) -> (64, 64, 64)
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),  # (64, 64, 64) -> (64, 64, 64)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (64, 32, 32)

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # (64, 32, 32) -> (128, 32, 32)
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),  # (128, 32, 32) -> (128, 32, 32)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (128, 16, 16)

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # (128, 16, 16) -> (256, 16, 16)
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),  # (256, 16, 16) -> (256, 16, 16)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (256, 8, 8)

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),  # (256, 8, 8) -> (256, 8, 8)
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (256, 4, 4)
            
            nn.Flatten(),  # (256 * 4 * 4) = 4096
            nn.Linear(256 * 4 * 4, 512),  # Added an intermediate fully connected layer
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 4),  # Final output layer (4 classes)
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        return self.model(x)

# K-Fold Cross Validation
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()
all_accuracies = []
all_precisions = []
all_f1_scores = []
all_specificities = []
all_conf_matrices = []
all_roc_curves = []

def train_and_validate(train_loader, val_loader, model, criterion, optimizer, scheduler, device):
    latest_model = None  # To store the model state
    
    for epoch in range(100):  # Maximum number of epochs
        model.train()
        epoch_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        # Training Loop
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            # Calculate correct predictions for accuracy
            _, preds = torch.max(outputs, 1)  # Get the class with highest logit
            correct_predictions += (preds == labels).sum().item()  # Compare with ground truth
            total_samples += labels.size(0)  # Update total sample count
        
        # Calculate training accuracy
        train_accuracy = correct_predictions / total_samples
        
        # Validation Loop
        model.eval()
        val_loss = 0.0
        y_true, y_pred_probs = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, labels).item()
                y_true.extend(labels.cpu().numpy())
                y_pred_probs.extend(outputs.cpu().numpy())
        
        scheduler.step(val_loss)  # Update learning rate
        
        print(f"Epoch {epoch + 1}: Train Loss: {epoch_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save the model
        latest_model = model.state_dict()  # Save the model state
        
        # Stop training early if 99% training accuracy is achieved
        if train_accuracy >= 0.99:
            return latest_model
    
    return latest_model

# K-Fold Cross Validation with Best Model Saving
for fold, (train_idx, val_idx) in enumerate(kf.split(combined_images, combined_labels)):
    print(f"Fold {fold + 1}")
    
    # Prepare Data Loaders
    train_images, val_images = combined_images[train_idx], combined_images[val_idx]
    train_labels, val_labels = combined_labels[train_idx], combined_labels[val_idx]

    unique, counts = np.unique(train_labels.numpy(), return_counts=True)
    print(dict(zip(unique, counts)))  # Print label distribution
    
    train_dataset = TensorDataset(train_images, train_labels)
    val_dataset = TensorDataset(val_images, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, num_workers=0, pin_memory=True)
    
    model = CNNModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    # Train and Validate, Saving the Best Model
    latest_model = train_and_validate(train_loader, val_loader, model, criterion, optimizer, scheduler, device)
    
    # Load the model state for evaluation
    model.load_state_dict(latest_model)
    
    # Evaluate the best model
    model.eval()
    y_true, y_pred_probs = [], []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            y_true.extend(labels.cpu().numpy())
            y_pred_probs.extend(outputs.cpu().numpy())
    
    y_pred = np.argmax(y_pred_probs, axis=1)
    cm = confusion_matrix(y_true, y_pred)
    all_conf_matrices.append(cm)

    report = classification_report(y_true, y_pred, output_dict=True, zero_division=1)
    all_accuracies.append(report['accuracy'])
    print("The Accuracy is ", report['accuracy'])
    all_precisions.append(report['macro avg']['precision'])
    all_f1_scores.append(report['macro avg']['f1-score'])

    specificity_per_class = []
    for i in range(cm.shape[0]):
        tn = np.sum(cm) - np.sum(cm[i, :]) - np.sum(cm[:, i]) + cm[i, i]
        fp = np.sum(cm[:, i]) - cm[i, i]
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        specificity_per_class.append(specificity)
    all_specificities.append(np.mean(specificity_per_class))

    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(4):
        fpr[i], tpr[i], _ = roc_curve((np.array(y_true) == i).astype(int), np.array(y_pred_probs)[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    all_roc_curves.append((fpr, tpr, roc_auc))

# Aggregate and display results
print(f"\nFinal Metrics (Averaged Across Folds):")
print(f"Accuracy: {np.mean(all_accuracies):.4f}")
print(f"Precision: {np.mean(all_precisions):.4f}")
print(f"F1 Score: {np.mean(all_f1_scores):.4f}")
print(f"Specificity: {np.mean(all_specificities):.4f}")

# Print the confusion matrix with labels
class_names = ["MildDemented", "ModerateDemented", "NonDemented", "VeryMildDemented"]
final_cm = np.sum(all_conf_matrices, axis=0)
disp = ConfusionMatrixDisplay(confusion_matrix=final_cm, display_labels=class_names)
disp.plot(cmap=plt.cm.Blues)
plt.savefig("confusion_matrix_11.png")

plt.figure(figsize=(10, 8))
for i in range(4):
    mean_fpr = np.linspace(0, 1, 100)
    mean_tpr = np.mean([np.interp(mean_fpr, roc[0][i], roc[1][i]) for roc in all_roc_curves], axis=0)
    mean_auc = np.mean([roc[2][i] for roc in all_roc_curves])
    plt.plot(mean_fpr, mean_tpr, label=f'Class {i} (AUC = {mean_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Aggregated ROC Curves')
plt.legend()
plt.savefig("roc_curve_11.png")