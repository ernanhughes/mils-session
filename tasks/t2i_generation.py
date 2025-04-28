# tasks/t2i_generation.py
# Module for generating images from text prompts using a T2I model

import os
import torch
from diffusers import StableDiffusionPipeline
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Model and API configuration
MODEL_ID = os.getenv("T2I_MODEL_ID", "runwayml/stable-diffusion-v1-5")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load Stable Diffusion pipeline
pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
)
pipe = pipe.to(DEVICE)

def generate_image_from_text(prompt, output_path="generated_image.png", guidance_scale=7.5, num_inference_steps=50):
    """
    Generate an image from a text prompt.

    Args:
        prompt (str): The text description to generate an image for.
        output_path (str): Path to save the generated image.
        guidance_scale (float): Strength of adherence to the text prompt.
        num_inference_steps (int): Number of denoising steps.

    Returns:
        str: Path to the saved image.
    """
    with torch.no_grad():
        image = pipe(prompt, guidance_scale=guidance_scale, num_inference_steps=num_inference_steps).images[0]
        image.save(output_path)
        print(f"Generated image saved to {output_path}")
        return output_path

if __name__ == "__main__":
    # Example usage
    prompt = "A serene lake surrounded by snowy mountains at sunset"
    generate_image_from_text(prompt, output_path="example_generated_image.png")
