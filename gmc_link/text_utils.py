"""
TextEncoder utility module for extracting language embeddings using sentence-transformers.
"""
from sentence_transformers import SentenceTransformer
import torch


class TextEncoder:
    """
    A simple wrapper around a pre-trained sentence transformer model to encode
    language prompts into embeddings. This can be used to convert natural language 
    descriptions into a format that the GMC-Link model can understand.
    """

    def __init__(self, model_name="all-MiniLM-L6-v2", device="cuda"):
        self.model = SentenceTransformer(model_name).to(device)
        self.device = device

    def encode(
        self,
        texts,
        batch_size=32,
        convert_to_tensor=True,
        show_progress_bar=False,
    ):
        """
        Encode text(s) into embeddings.

        Args:
            texts: str or List[str]
            batch_size: batch size for encoding
            convert_to_tensor: whether to return torch tensor
            show_progress_bar: whether to show progress bar

        Returns:
            Tensor of shape (N, L_dim) if batch input
            Tensor of shape (1, L_dim) if single input
        """

        with torch.no_grad():
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                convert_to_tensor=convert_to_tensor,
                show_progress_bar=show_progress_bar,
            )

        # Ensure output shape consistency
        if isinstance(texts, str):
            if embeddings.ndim == 1:
                embeddings = embeddings.unsqueeze(0)

        return embeddings.to(self.device)
