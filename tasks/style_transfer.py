# tasks/style_transfer.py
# Module for neural style transfer between two images

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"

# Image preprocessing
loader = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])

def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

# Gram matrix for style representation
def gram_matrix(tensor):
    b, c, h, w = tensor.size()
    features = tensor.view(b, c, h * w)
    G = torch.bmm(features, features.transpose(1, 2))
    return G / (c * h * w)

# Content and style loss modules
class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        return nn.functional.mse_loss(input, self.target)

class StyleLoss(nn.Module):
    def __init__(self, target):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target).detach()

    def forward(self, input):
        G = gram_matrix(input)
        return nn.functional.mse_loss(G, self.target)

def style_transfer(content_path, style_path, output_path="output.jpg", num_steps=300, style_weight=1e6, content_weight=1):
    # Load images
    content_img = load_image(content_path)
    style_img = load_image(style_path)

    # Load VGG19
    cnn = models.vgg19(pretrained=True).features.to(device).eval()

    # Model setup
    normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                          std=[0.229, 0.224, 0.225])
    normalization = normalization.to(device)

    content_layers = ['conv_4']
    style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

    content_losses = []
    style_losses = []

    model = nn.Sequential(normalization)

    i = 0  # increment for each conv layer
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = f'conv_{i}'
        elif isinstance(layer, nn.ReLU):
            name = f'relu_{i}'
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = f'pool_{i}'
        elif isinstance(layer, nn.BatchNorm2d):
            name = f'bn_{i}'
        else:
            raise RuntimeError(f"Unrecognized layer: {layer.__class__.__name__}")

        model.add_module(name, layer)

        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module(f"content_loss_{i}", content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            target = model(style_img).detach()
            style_loss = StyleLoss(target)
            model.add_module(f"style_loss_{i}", style_loss)
            style_losses.append(style_loss)

    # Trim model after the last loss layer
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], (ContentLoss, StyleLoss)):
            break
    model = model[:(i + 1)]

    # Input image
    input_img = content_img.clone()
    input_img.requires_grad_(True)

    optimizer = optim.LBFGS([input_img])

    run = [0]
    while run[0] <= num_steps:

        def closure():
            input_img.data.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl(model(input_img))
            for cl in content_losses:
                content_score += cl(model(input_img))

            loss = style_weight * style_score + content_weight * content_score
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print(f"Step {run[0]}: Style Loss {style_score.item():.4f} Content Loss {content_score.item():.4f}")

            return loss

        optimizer.step(closure)

    input_img.data.clamp_(0, 1)

    # Save output
    unloader = transforms.ToPILImage()
    image = input_img.cpu().clone().squeeze(0)
    image = unloader(image)
    image.save(output_path)

    print(f"Styled image saved as {output_path}")

if __name__ == "__main__":
    # Example usage
    content = "sample_content.jpg"
    style = "sample_style.jpg"
    style_transfer(content, style, output_path="styled_output.jpg")
