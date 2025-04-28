# scorer/imagebind_scorer.py
# Module for scoring text-image similarity using Meta AI's ImageBind model

import torch
from PIL import Image
import torchvision.transforms as transforms
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load ImageBind model
model = imagebind_model.imagebind_huge(pretrained=True)
model.eval()
model.to(device)

# Image preprocessing
image_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def score_text_to_image(text_prompt, image_path):
    """
    Compute similarity between text prompt and image using ImageBind.

    Args:
        text_prompt (str): Text description.
        image_path (str): Path to image file.

    Returns:
        float: Cosine similarity score.
    """
    # Preprocess inputs
    image = Image.open(image_path).convert("RGB")
    image = image_transform(image).unsqueeze(0).to(device)

    text = [text_prompt]

    inputs = {
        ModalityType.TEXT: imagebind_model.tokenize(text).to(device),
        ModalityType.VISION: image
    }

    with torch.no_grad():
        embeddings = model(inputs)

    text_embeds = embeddings[ModalityType.TEXT]
    image_embeds = embeddings[ModalityType.VISION]

    # Normalize embeddings
    text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
    image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)

    # Cosine similarity
    similarity = (text_embeds @ image_embeds.T).item()

    return similarity

if __name__ == "__main__":
    # Example usage
    text = "A futuristic cityscape at sunset"
    image_path = "sample_image.jpg"

    score = score_text_to_image(text, image_path)
    print(f"ImageBind similarity between text and image: {score:.4f}")
