# 1. Upgrade the Kaggle tool to make sure it understands the new KGAT token
!pip install -q kaggle --upgrade

# 2. Tell Colab your Kaggle API Token directly
import os
os.environ["KAGGLE_API_TOKEN"] = "Your API Token Here"

# 3. Download the Massachusetts dataset directly from Kaggle
print("Downloading dataset...")
!kaggle datasets download -d balraj98/massachusetts-buildings-dataset

# 4. Unzip the downloaded file into a folder called "dataset"
print("Unzipping dataset...")
!unzip -q massachusetts-buildings-dataset.zip -d dataset
print("Data is ready! You can now run the big training code block.")



import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import f1_score, jaccard_score
import matplotlib.pyplot as plt

# =====================================================================
# 1. ADVANCED LOSS FUNCTION (BCE + Dice Loss)
# =====================================================================
class BCEDiceLoss(nn.Module):
    def __init__(self):
        super(BCEDiceLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        
    def forward(self, logits, targets):
        # 1. Standard BCE Loss
        bce_loss = self.bce(logits, targets)
        
        # 2. Dice Loss (Forces AI to care about building shapes)
        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum(dim=(2, 3)) # Sum spatial dimensions
        union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        
        # Smooth factor (1e-5) prevents dividing by zero
        dice_score = (2. * intersection + 1e-5) / (union + 1e-5)
        dice_loss = 1.0 - dice_score.mean()
        
        # Combine them
        return bce_loss + dice_loss

# =====================================================================
# 2. DATASET LIBRARIAN
# =====================================================================
class BuildingDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        mask = (mask > 127).astype(np.float32)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        mask = mask.unsqueeze(0) 
        return image, mask

# =====================================================================
# 3. HIGH-RESOLUTION TRANSFORMS (512x512)
# =====================================================================
def get_transforms(augment=True):
    if augment:
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Resize(512, 512), # <-- UPGRADED RESOLUTION
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(512, 512), # <-- UPGRADED RESOLUTION
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
            ToTensorV2()
        ])

# =====================================================================
# 4. DEFINE THE MODEL
# =====================================================================
def get_model():
    model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)
    model.classifier[4] = nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))
    return model

# =====================================================================
# 5. METRICS & TRAINING LOOP (With Learning Rate Scheduler)
# =====================================================================
def calculate_metrics(predictions, true_masks):
    preds_flat = predictions.view(-1).numpy()
    true_flat = true_masks.view(-1).numpy()
    iou = jaccard_score(true_flat, preds_flat, average='binary', zero_division=1)
    f1 = f1_score(true_flat, preds_flat, average='binary', zero_division=1)
    return iou, f1

def train_model(model, dataloader, optimizer, criterion, scheduler, device, epochs):
    model.train()
    
    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch+1}/{epochs} ---")
        total_loss = 0
        all_preds, all_true = [], []
        
        for step, (images, masks) in enumerate(dataloader):
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)['out']
            loss = criterion(outputs, masks.float())
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            preds = torch.sigmoid(outputs) > 0.5
            all_preds.append(preds.detach().cpu())
            all_true.append(masks.cpu())
            
            if (step + 1) % 10 == 0:
                print(f"Step {step+1}/{len(dataloader)} | Loss: {loss.item():.4f}")
        
        all_preds = torch.cat(all_preds)
        all_true = torch.cat(all_true)
        iou, f1 = calculate_metrics(all_preds, all_true)
        avg_loss = total_loss / len(dataloader)
        
        # Give the loss to the scheduler so it can slow down if it gets stuck
        scheduler.step(avg_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch+1} Complete -> Avg Loss: {avg_loss:.4f} | IoU: {iou:.4f} | F1: {f1:.4f} | LR: {current_lr:.6f}")

# =====================================================================
# 6. VISUALIZATION FUNCTION
# =====================================================================
def visualize_predictions(model, dataloader, device, num_images=3):
    print("\nGenerating visual comparisons...")
    model.eval() 
    images, masks = next(iter(dataloader))
    images = images.to(device)
    
    with torch.no_grad():
        outputs = model(images)['out']
        preds = torch.sigmoid(outputs) > 0.5
    
    images, masks, preds = images.cpu(), masks.cpu(), preds.cpu()
    fig, axes = plt.subplots(num_images, 3, figsize=(12, 4 * num_images))
    mean, std = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1), torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    for i in range(num_images):
        img = np.clip((images[i] * std + mean).numpy().transpose(1, 2, 0), 0, 1) 
        axes[i, 0].imshow(img)
        axes[i, 0].set_title("Original Image"), axes[i, 0].axis('off')
        axes[i, 1].imshow(masks[i].squeeze().numpy(), cmap='gray')
        axes[i, 1].set_title("True Mask"), axes[i, 1].axis('off')
        axes[i, 2].imshow(preds[i].squeeze().numpy(), cmap='gray')
        axes[i, 2].set_title("AI Prediction"), axes[i, 2].axis('off')
        
    plt.tight_layout()
    plt.show()

# =====================================================================
# 7. HIGH-ACCURACY EXECUTION
# =====================================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running compute on: {device}")
    
    TRAIN_IMG_DIR = "/content/dataset/png/train"
    TRAIN_MASK_DIR = "/content/dataset/png/train_labels"
    
    # 50 Epochs for High Accuracy
    EPOCHS_TO_RUN = 50 
    
    # Setup Advanced Model & Loss
    model_pro = get_model().to(device)
    criterion = BCEDiceLoss() # <-- UPGRADED LOSS
    optimizer_pro = torch.optim.Adam(model_pro.parameters(), lr=0.0001)
    
    # <-- UPGRADED SCHEDULER
    # If loss doesn't improve for 3 epochs, cut learning rate by 50%
    scheduler_pro = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_pro, mode='min', patience=3, factor=0.5) 
    
    transforms_pro = get_transforms(augment=True)
    dataset_pro = BuildingDataset(image_dir=TRAIN_IMG_DIR, mask_dir=TRAIN_MASK_DIR, transform=transforms_pro)
    
    # Batch size reduced to 4 to handle 512x512 resolution safely
    loader_pro = DataLoader(dataset_pro, batch_size=4, shuffle=True, drop_last=True)
    
    print("\nStarting High-Accuracy Training Run...")
    train_model(model_pro, loader_pro, optimizer_pro, criterion, scheduler_pro, device, epochs=EPOCHS_TO_RUN)

    visualize_predictions(model_pro, loader_pro, device, num_images=3)
