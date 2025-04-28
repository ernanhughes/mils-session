# scorer/clip_scorer.py
# Module for scoring text-image or image-image similarity using CLIP models

import torch
import clip
from PIL import Image

# Load CLIP model and preprocessing
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def score_text_to_image(text_prompt, image_path):
    """
    Compute similarity between a text prompt and an image using CLIP.

    Args:
        text_prompt (str): The text input.
        image_path (str): Path to the image file.

    Returns:
        float: Cosine similarity score (higher = more similar).
    """
    # Preprocess inputs
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    text = clip.tokenize([text_prompt]).to(device)

    # Encode with CLIP
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)

    # Normalize
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    # Compute cosine similarity
    similarity = (image_features @ text_features.T).item()

    return similarity

def score_image_to_image(image_path_1, image_path_2):
    """
    Compute similarity between two images using CLIP.

    Args:
        image_path_1 (str): Path to first image.
        image_path_2 (str): Path to second image.

    Returns:
        float: Cosine similarity score.
    """
    image1 = preprocess(Image.open(image_path_1)).unsqueeze(0).to(device)
    image2 = preprocess(Image.open(image_path_2)).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features_1 = model.encode_image(image1)
        image_features_2 = model.encode_image(image2)

    image_features_1 /= image_features_1.norm(dim=-1, keepdim=True)
    image_features_2 /= image_features_2.norm(dim=-1, keepdim=True)

    similarity = (image_features_1 @ image_features_2.T).item()

    return similarity

if __name__ == "__main__":
    # Example usage
    text = "A cozy mountain cabin"
    image_path = "sample_image.jpg"

    score = score_text_to_image(text, image_path)
    print(f"Similarity between text and image: {score:.4f}")

    # Example comparing two images
    # score2 = score_image_to_image("img1.jpg", "img2.jpg")
    # print(f"Similarity between images: {score2:.4f}")
