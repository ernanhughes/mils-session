# scorer/gram_scorer.py
# Module for scoring style similarity between images using Gram matrices

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# Use VGG19 pretrained network
vgg = models.vgg19(pretrained=True).features.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
vgg = vgg.to(device)

# Preprocessing for VGG input
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Gram Matrix helper
def gram_matrix(tensor):
    b, c, h, w = tensor.size()
    features = tensor.view(b, c, h * w)
    G = torch.bmm(features, features.transpose(1, 2))
    return G / (c * h * w)

# Extract features from an image
def get_features(image_path, layers=[0, 5, 10, 19, 28]):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    features = []
    x = image
    for i, layer in enumerate(vgg):
        x = layer(x)
        if i in layers:
            features.append(x)

    return features

# Compute style loss between two images
def score_style_similarity(image_path_1, image_path_2):
    features1 = get_features(image_path_1)
    features2 = get_features(image_path_2)

    style_loss = 0
    for f1, f2 in zip(features1, features2):
        G1 = gram_matrix(f1)
        G2 = gram_matrix(f2)
        style_loss += nn.functional.mse_loss(G1, G2)

    return style_loss.item()

if __name__ == "__main__":
    # Example usage
    img1 = "sample_image1.jpg"
    img2 = "sample_image2.jpg"

    loss = score_style_similarity(img1, img2)
    print(f"Style (Gram Matrix) difference between images: {loss:.4f}")
