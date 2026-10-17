import os
import json
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__() # initializing parent class

        self.features = nn.Sequential( # convolutional feature extractor
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))

def infer(data_dir, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    checkpoint  = torch.load(model_path, map_location=device) #loading saved model
    class_names = checkpoint["class_names"]
    img_size    = checkpoint["img_size"]
    num_classes = len(class_names)
    
    model = SimpleCNN(num_classes=num_classes) #creating model
    model.load_state_dict(checkpoint["model_state"]) #loading weights
    model.to(device)
    model.eval() #evaluation mode
    
    print(f"Model loaded from '{model_path}' | {num_classes} classes | img_size={img_size}")
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]), 
    ])

    valid_exts = {".jpg", ".jpeg", ".png"}
    image_files = []
    
    for fname in os.listdir(data_dir):
        if os.path.splitext(fname)[1].lower() in valid_exts:
            image_files.append(fname) #keeping valid images

    print(f"Found {len(image_files)} images in '{data_dir}'")
    
    results = {}
    with torch.no_grad(): #disabeling gradients for inference
        for fname in sorted(image_files):
            img_path = os.path.join(data_dir, fname)
            try:
                img    = Image.open(img_path).convert("RGB")
                tensor = transform(img).unsqueeze(0).to(device)  # add batch dim
                output = model(tensor)
                pred   = output.argmax(1).item() #predicted class index
                results[fname] = class_names[pred]
            except Exception as e:
                print(f"  Warning: could not process '{fname}': {e}")
                results[fname] = "unknown"
                
    output_path = "results.json" #results
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Done. Predictions saved to '{output_path}'")
    return results


if __name__ == "__main__":
    DATA_DIR = "data/test" #folder with test images
    MODEL_PATH = "model.pth"

    results = infer(DATA_DIR, MODEL_PATH)

    for filename, predicted_class in list(results.items())[:10]:
        print(f"  {filename:30s} → {predicted_class}")
