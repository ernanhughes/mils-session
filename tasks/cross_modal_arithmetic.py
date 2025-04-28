# tasks/cross_modal_arithmetic.py
# Module for performing arithmetic operations across modality embeddings

import torch

def add_embeddings(embedding1, embedding2):
    """
    Add two embeddings element-wise.

    Args:
        embedding1 (torch.Tensor): First embedding vector.
        embedding2 (torch.Tensor): Second embedding vector.

    Returns:
        torch.Tensor: Resulting vector after addition.
    """
    return embedding1 + embedding2

def subtract_embeddings(embedding1, embedding2):
    """
    Subtract one embedding from another element-wise.

    Args:
        embedding1 (torch.Tensor): First embedding vector.
        embedding2 (torch.Tensor): Second embedding vector.

    Returns:
        torch.Tensor: Resulting vector after subtraction.
    """
    return embedding1 - embedding2

def mix_embeddings(embedding1, embedding2, alpha=0.5):
    """
    Linearly interpolate between two embeddings.

    Args:
        embedding1 (torch.Tensor): First embedding.
        embedding2 (torch.Tensor): Second embedding.
        alpha (float): Mix ratio (0.0 = all embedding1, 1.0 = all embedding2).

    Returns:
        torch.Tensor: Mixed embedding.
    """
    return (1 - alpha) * embedding1 + alpha * embedding2

def normalize_embedding(embedding):
    """
    Normalize an embedding to unit norm.

    Args:
        embedding (torch.Tensor): Input embedding.

    Returns:
        torch.Tensor: Normalized embedding.
    """
    return embedding / embedding.norm(dim=-1, keepdim=True)

if __name__ == "__main__":
    # Example usage
    emb1 = torch.randn(1, 512)  # Fake embedding (e.g., from text)
    emb2 = torch.randn(1, 512)  # Fake embedding (e.g., from image)

    added = add_embeddings(emb1, emb2)
    subtracted = subtract_embeddings(emb1, emb2)
    mixed = mix_embeddings(emb1, emb2, alpha=0.3)

    print("Added shape:", added.shape)
    print("Subtracted shape:", subtracted.shape)
    print("Mixed shape:", mixed.shape)
