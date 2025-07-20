import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models, datasets
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import argparse
import random
import numpy as np

# Calorie and macronutrient information for Food101 classes
calorie_macro_map = {
    "apple_pie": {"calories": 320, "carbs": 48, "protein": 2, "fat": 14},
    "baby_back_ribs": {"calories": 450, "carbs": 6, "protein": 35, "fat": 32},
    "baklava": {"calories": 290, "carbs": 38, "protein": 4, "fat": 15},
    "beef_carpaccio": {"calories": 210, "carbs": 1, "protein": 25, "fat": 12},
    "beef_tartare": {"calories": 250, "carbs": 1, "protein": 22, "fat": 18},
    "beet_salad": {"calories": 120, "carbs": 13, "protein": 2, "fat": 7},
    "beignets": {"calories": 290, "carbs": 32, "protein": 4, "fat": 17},
    "bibimbap": {"calories": 500, "carbs": 58, "protein": 20, "fat": 20},
    "bread_pudding": {"calories": 310, "carbs": 45, "protein": 5, "fat": 12},
    "breakfast_burrito": {"calories": 350, "carbs": 30, "protein": 18, "fat": 20},
    "bruschetta": {"calories": 180, "carbs": 22, "protein": 5, "fat": 8},
    "caesar_salad": {"calories": 180, "carbs": 10, "protein": 7, "fat": 12},
    "cannoli": {"calories": 230, "carbs": 25, "protein": 5, "fat": 12},
    "caprese_salad": {"calories": 150, "carbs": 5, "protein": 7, "fat": 10},
    "carrot_cake": {"calories": 360, "carbs": 42, "protein": 4, "fat": 20},
    "ceviche": {"calories": 140, "carbs": 6, "protein": 18, "fat": 4},
    "cheesecake": {"calories": 350, "carbs": 32, "protein": 6, "fat": 24},
    "cheese_plate": {"calories": 400, "carbs": 3, "protein": 18, "fat": 34},
    "chicken_curry": {"calories": 270, "carbs": 8, "protein": 24, "fat": 16},
    "chicken_quesadilla": {"calories": 330, "carbs": 28, "protein": 18, "fat": 18},
    "chicken_wings": {"calories": 430, "carbs": 2, "protein": 32, "fat": 34},
    "chocolate_cake": {"calories": 380, "carbs": 50, "protein": 5, "fat": 18},
    "chocolate_mousse": {"calories": 320, "carbs": 28, "protein": 4, "fat": 22},
    "churros": {"calories": 280, "carbs": 30, "protein": 4, "fat": 16},
    "clam_chowder": {"calories": 150, "carbs": 12, "protein": 6, "fat": 8},
    "club_sandwich": {"calories": 310, "carbs": 30, "protein": 16, "fat": 14},
    "crab_cakes": {"calories": 220, "carbs": 14, "protein": 12, "fat": 12},
    "creme_brulee": {"calories": 290, "carbs": 25, "protein": 5, "fat": 20},
    "croque_madame": {"calories": 330, "carbs": 20, "protein": 18, "fat": 22},
    "cup_cakes": {"calories": 230, "carbs": 30, "protein": 2, "fat": 12},
    "deviled_eggs": {"calories": 180, "carbs": 2, "protein": 8, "fat": 15},
    "donuts": {"calories": 260, "carbs": 32, "protein": 3, "fat": 14},
    "dumplings": {"calories": 210, "carbs": 28, "protein": 8, "fat": 8},
    "edamame": {"calories": 120, "carbs": 9, "protein": 11, "fat": 5},
    "eggs_benedict": {"calories": 290, "carbs": 20, "protein": 14, "fat": 18},
    "escargots": {"calories": 200, "carbs": 4, "protein": 16, "fat": 14},
    "falafel": {"calories": 330, "carbs": 26, "protein": 10, "fat": 20},
    "filet_mignon": {"calories": 500, "carbs": 0, "protein": 45, "fat": 34},
    "fish_and_chips": {"calories": 430, "carbs": 35, "protein": 18, "fat": 26},
    "foie_gras": {"calories": 450, "carbs": 3, "protein": 7, "fat": 45},
    "french_fries": {"calories": 365, "carbs": 48, "protein": 4, "fat": 18},
    "french_onion_soup": {"calories": 160, "carbs": 14, "protein": 6, "fat": 9},
    "french_toast": {"calories": 290, "carbs": 35, "protein": 8, "fat": 12},
    "fried_calamari": {"calories": 280, "carbs": 14, "protein": 20, "fat": 18},
    "fried_rice": {"calories": 340, "carbs": 45, "protein": 8, "fat": 14},
    "frozen_yogurt": {"calories": 170, "carbs": 24, "protein": 5, "fat": 4},
    "garlic_bread": {"calories": 240, "carbs": 30, "protein": 5, "fat": 10},
    "gnocchi": {"calories": 250, "carbs": 40, "protein": 6, "fat": 6},
    "greek_salad": {"calories": 140, "carbs": 7, "protein": 4, "fat": 10},
    "grilled_cheese_sandwich": {"calories": 400, "carbs": 30, "protein": 10, "fat": 24},
    "grilled_salmon": {"calories": 280, "carbs": 0, "protein": 30, "fat": 18},
    "guacamole": {"calories": 220, "carbs": 12, "protein": 3, "fat": 18},
    "gyoza": {"calories": 200, "carbs": 22, "protein": 8, "fat": 8},
    "hamburger": {"calories": 354, "carbs": 29, "protein": 17, "fat": 20},
    "hot_and_sour_soup": {"calories": 120, "carbs": 10, "protein": 6, "fat": 6},
    "hot_dog": {"calories": 290, "carbs": 24, "protein": 10, "fat": 18},
    "huevos_rancheros": {"calories": 300, "carbs": 20, "protein": 14, "fat": 18},
    "hummus": {"calories": 210, "carbs": 15, "protein": 6, "fat": 14},
    "ice_cream": {"calories": 200, "carbs": 22, "protein": 3, "fat": 11},
    "lasagna": {"calories": 350, "carbs": 28, "protein": 18, "fat": 20},
    "lobster_bisque": {"calories": 180, "carbs": 12, "protein": 8, "fat": 10},
    "lobster_roll_sandwich": {"calories": 380, "carbs": 32, "protein": 20, "fat": 22},
    "macaroni_and_cheese": {"calories": 310, "carbs": 28, "protein": 10, "fat": 18},
    "macarons": {"calories": 160, "carbs": 18, "protein": 3, "fat": 8},
    "miso_soup": {"calories": 80, "carbs": 7, "protein": 4, "fat": 3},
    "mussels": {"calories": 160, "carbs": 7, "protein": 22, "fat": 5},
    "nachos": {"calories": 420, "carbs": 36, "protein": 12, "fat": 26},
    "omelette": {"calories": 155, "carbs": 1, "protein": 11, "fat": 11},
    "onion_rings": {"calories": 320, "carbs": 35, "protein": 4, "fat": 18},
    "oysters": {"calories": 120, "carbs": 5, "protein": 14, "fat": 5},
    "pad_thai": {"calories": 430, "carbs": 50, "protein": 15, "fat": 18},
    "paella": {"calories": 350, "carbs": 40, "protein": 20, "fat": 12},
    "pancakes": {"calories": 230, "carbs": 36, "protein": 5, "fat": 7},
    "panna_cotta": {"calories": 240, "carbs": 20, "protein": 5, "fat": 16},
    "peking_duck": {"calories": 400, "carbs": 8, "protein": 30, "fat": 28},
    "pho": {"calories": 350, "carbs": 40, "protein": 20, "fat": 12},
    "pizza": {"calories": 285, "carbs": 33, "protein": 12, "fat": 12},
    "pork_chop": {"calories": 400, "carbs": 0, "protein": 35, "fat": 28},
    "poutine": {"calories": 490, "carbs": 44, "protein": 10, "fat": 30},
    "prime_rib": {"calories": 510, "carbs": 0, "protein": 40, "fat": 38},
    "pulled_pork_sandwich": {"calories": 430, "carbs": 30, "protein": 25, "fat": 26},
    "ramen": {"calories": 380, "carbs": 50, "protein": 12, "fat": 14},
    "ravioli": {"calories": 320, "carbs": 36, "protein": 12, "fat": 14},
    "red_velvet_cake": {"calories": 370, "carbs": 40, "protein": 4, "fat": 22},
    "risotto": {"calories": 310, "carbs": 36, "protein": 8, "fat": 14},
    "samosa": {"calories": 270, "carbs": 24, "protein": 6, "fat": 16},
    "sashimi": {"calories": 180, "carbs": 2, "protein": 20, "fat": 10},
    "scallops": {"calories": 170, "carbs": 4, "protein": 22, "fat": 6},
    "seaweed_salad": {"calories": 100, "carbs": 9, "protein": 2, "fat": 6},
    "shrimp_and_grits": {"calories": 350, "carbs": 28, "protein": 16, "fat": 18},
    "spaghetti_bolognese": {"calories": 320, "carbs": 35, "protein": 14, "fat": 14},
    "spaghetti_carbonara": {"calories": 360, "carbs": 30, "protein": 12, "fat": 20},
    "spring_rolls": {"calories": 210, "carbs": 28, "protein": 6, "fat": 8},
    "steak": {"calories": 480, "carbs": 0, "protein": 40, "fat": 34},
    "strawberry_shortcake": {"calories": 290, "carbs": 34, "protein": 4, "fat": 16},
    "sushi": {"calories": 200, "carbs": 28, "protein": 8, "fat": 5},
    "tacos": {"calories": 300, "carbs": 26, "protein": 12, "fat": 16},
    "takoyaki": {"calories": 310, "carbs": 32, "protein": 10, "fat": 16},
    "tiramisu": {"calories": 300, "carbs": 32, "protein": 5, "fat": 18},
    "tuna_tartare": {"calories": 200, "carbs": 2, "protein": 20, "fat": 12},
    "waffles": {"calories": 310, "carbs": 36, "protein": 6, "fat": 14}
}

class Food101Calories(Dataset):
    def __init__(self, root='./data', split='train', transform=None, download=True):
        # Initialize the base Food101 dataset
        self.dataset = datasets.Food101(root=root, split=split, download=download)
        
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        self.labels = self.dataset._labels
        self.classes = self.dataset.classes

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, class_idx = self.dataset[idx]
        label = self.classes[class_idx]  # Get the class name
        nutrition = calorie_macro_map.get(label, {"calories": 0, "carbs": 0, "protein": 0, "fat": 0})
        target = torch.tensor([
            nutrition["calories"],
            nutrition["carbs"],
            nutrition["protein"],
            nutrition["fat"]
        ], dtype=torch.float32)

        image = self.transform(image)
        return image, target

class ResNetCalories(nn.Module):
    def __init__(self):
        super(ResNetCalories, self).__init__()
        base_model = models.resnet18(pretrained=True)
        num_features = base_model.fc.in_features
        base_model.fc = nn.Linear(num_features, 4)  # Output: [calories, carbs, protein, fat]
        self.model = base_model

    def forward(self, x):
        return self.model(x)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def save_checkpoint(state, filename):
    torch.save(state, filename)
    print(f"Checkpoint saved: {filename}")

def train_model(epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed()

    BATCH_SIZE = 32
    LR = 1e-4
    
    print("Loading Food101 dataset...")
    
    # Create datasets with automatic download
    train_dataset = Food101Calories(split='train', download=True)
    val_dataset = Food101Calories(split='test', download=False)  # No need to download again

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Model, criterion, optimizer
    model = ResNetCalories().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training")

        for images, targets in train_bar:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_bar.set_postfix(loss=loss.item())

        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} - Validation")
            for images, targets in val_bar:
                images, targets = images.to(device), targets.to(device)
                outputs = model(images)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                val_bar.set_postfix(val_loss=loss.item())

        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_checkpoint({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
            }, "best_model.pth")
            print("✅ Saved new best model!")

        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_path = f"checkpoint_epoch_{epoch+1}.pth"
            save_checkpoint({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
            }, checkpoint_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train ResNetCalories model')
    parser.add_argument('--epochs', type=int, default=15, help='number of epochs to train')
    args = parser.parse_args()

    train_model(args.epochs)