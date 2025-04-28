# tasks/image_captioning.py
# Module for generating text captions from images

import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

# Load BLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

def caption_image(image_path):
    """
    Generate a text caption/summary from an image.

    Args:
        image_path (str): Path to the image file.

    Returns:
        str: Generated text description.
    """
    try:
        image = Image.open(image_path).convert('RGB')
        inputs = processor(image, return_tensors="pt").to(device)

        with torch.no_grad():
            output = model.generate(**inputs)
            caption = processor.decode(output[0], skip_special_tokens=True)
        
        return caption
    except Exception as e:
        print(f"Error in image captioning: {e}")
        return None

if __name__ == "__main__":
    # Example usage
    sample_image = "sample_image.jpg"
    caption = caption_image(sample_image)
    if caption:
        print(f"Generated caption:\n{caption}")
