"""
TextEncoder utility module for extracting language embeddings.

Default backend: sentence-transformers (e.g., all-MiniLM-L6-v2 → 384D, BGE-base → 768D).
CLIP-text backend (Exp 40): pass model_name in form "clip:<arch>:<pretrained>",
e.g., "clip:ViT-B-32:datacomp_xl_s13b_b90k" → 512D.
"""
import torch
from sentence_transformers import SentenceTransformer


class TextEncoder:
    """
    Wrapper around a frozen text encoder. Two backends:

    - Sentence-transformers (default): any HuggingFace ST model name.
    - CLIP text tower: when `model_name` starts with "clip:" — format
      "clip:<arch>:<pretrained>" parsed and passed to open_clip.

    Both backends expose `.encode(texts)` returning an L2-friendly tensor
    of shape (N, embedding_dim) on `self.device`.
    """

    def __init__(self, model_name="all-MiniLM-L6-v2", device="cuda"):
        self.device = device
        self.model_name = model_name

        if model_name.startswith("clip:"):
            self._backend = "clip"
            parts = model_name.split(":")
            if len(parts) != 3:
                raise ValueError(
                    f"CLIP text encoder name must be 'clip:<arch>:<pretrained>', got {model_name!r}"
                )
            _, arch, pretrained = parts
            import open_clip
            model, _, _ = open_clip.create_model_and_transforms(arch, pretrained=pretrained)
            model = model.to(device).eval()
            for p in model.parameters():
                p.requires_grad = False
            self._clip_model = model
            self._clip_tokenizer = open_clip.get_tokenizer(arch)
            with torch.no_grad():
                probe = self._clip_tokenizer(["probe"]).to(device)
                feat = self._clip_model.encode_text(probe)
            self.embedding_dim = int(feat.shape[-1])
        else:
            self._backend = "st"
            self.model = SentenceTransformer(model_name).to(device)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()

    def encode(
        self,
        texts,
        batch_size=32,
        convert_to_tensor=True,
        show_progress_bar=False,
    ):
        if self._backend == "clip":
            single = isinstance(texts, str)
            text_list = [texts] if single else list(texts)
            outputs = []
            with torch.no_grad():
                for i in range(0, len(text_list), batch_size):
                    chunk = text_list[i : i + batch_size]
                    tokens = self._clip_tokenizer(chunk).to(self.device)
                    feats = self._clip_model.encode_text(tokens)
                    outputs.append(feats.float())
            embeddings = torch.cat(outputs, dim=0)
            if single and embeddings.ndim == 1:
                embeddings = embeddings.unsqueeze(0)
            return embeddings.to(self.device)

        with torch.no_grad():
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                convert_to_tensor=convert_to_tensor,
                show_progress_bar=show_progress_bar,
            )
        if isinstance(texts, str):
            if embeddings.ndim == 1:
                embeddings = embeddings.unsqueeze(0)
        return embeddings.to(self.device)
