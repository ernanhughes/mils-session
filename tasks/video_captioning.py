# tasks/video_captioning.py
# Module for generating text captions from video files

import os
import torch
import cv2
import tempfile
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

# Load BLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

def extract_frames(video_path, frame_rate=1):
    """
    Extract frames from video at a given rate.

    Args:
        video_path (str): Path to the video file.
        frame_rate (int): Number of frames per second to sample.

    Returns:
        list: List of PIL images extracted from the video.
    """
    vidcap = cv2.VideoCapture(video_path)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    interval = int(fps // frame_rate)

    frames = []
    count = 0
    success, image = vidcap.read()

    while success:
        if count % interval == 0:
            img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            frames.append(img)
        success, image = vidcap.read()
        count += 1

    vidcap.release()
    return frames

def caption_frame(image):
    """
    Generate caption for a single image frame.

    Args:
        image (PIL.Image): Input image.

    Returns:
        str: Generated text description.
    """
    inputs = processor(image, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(**inputs)
        caption = processor.decode(output[0], skip_special_tokens=True)

    return caption

def caption_video(video_path, frame_rate=1):
    """
    Generate a list of captions from frames of a video.

    Args:
        video_path (str): Path to the video file.
        frame_rate (int): Frames per second to sample.

    Returns:
        list: List of generated text captions.
    """
    frames = extract_frames(video_path, frame_rate=frame_rate)
    captions = []

    for idx, frame in enumerate(frames):
        caption = caption_frame(frame)
        captions.append((idx, caption))
        print(f"Frame {idx}: {caption}")

    return captions

if __name__ == "__main__":
    # Example usage
    video = "sample_video.mp4"
    captions = caption_video(video, frame_rate=1)
    for idx, text in captions:
        print(f"Frame {idx}: {text}")
