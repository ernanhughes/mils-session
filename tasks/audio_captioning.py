# tasks/audio_captioning.py
# Module for generating text captions from audio files

import torch
import os
import whisper

# Load Whisper model (small for quick demo; you can switch to 'base', 'large', etc.)
model = whisper.load_model("small")

def caption_audio(audio_path):
    """
    Generate a text caption/summary from an audio file.

    Args:
        audio_path (str): Path to the audio file (wav, mp3, etc.)

    Returns:
        str: Generated text description.
    """
    try:
        result = model.transcribe(audio_path)
        return result['text']
    except Exception as e:
        print(f"Error in audio captioning: {e}")
        return None

if __name__ == "__main__":
    # Example usage
    sample_audio = "sample_audio.wav"
    caption = caption_audio(sample_audio)
    if caption:
        print(f"Generated caption:\n{caption}")
    else:
        print("Failed to generate caption.")