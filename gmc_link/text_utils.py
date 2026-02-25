"""
TextEncoder utility module for extracting language embeddings using sentence-transformers.
"""
# pylint: disable=too-few-public-methods
from sentence_transformers import SentenceTransformer
import torch


class TextEncoder:
    """
    A simple wrapper around a pre-trained sentence transformer model to encode
    language prompts into embeddings. This can be used to convert natural language 
    descriptions into a format that the GMC-Link model can understand.
    """

    def __init__(self, model_name="all-MiniLM-L6-v2", device="mps"):
        self.model = SentenceTransformer(model_name).to(device)
        self.device = device

    def encode(self, text):
        """
        Encode a single text prompt into an embedding.
        Args:
            text: A string containing the natural language description 
                  (e.g., "The car on the right is moving fast").
        Returns:
            A tensor of shape (1, L_dim) representing the encoded language features.
        """
        with torch.no_grad():
            embedding = self.model.encode(text, convert_to_tensor=True)

        if embedding.ndim == 1:
            embedding = embedding.unsqueeze(0)  # Ensure shape is (1, L_dim)
        return embedding.to(self.device)
