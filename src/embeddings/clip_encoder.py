"""
CLIP encoder for image and text embeddings.
Uses openai/clip-vit-base-patch32 model.
"""

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from src.config import BATCH_SIZE, CLIP_MODEL_NAME, DEVICE, EMBEDDING_DIM
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class CLIPEncoder:
    """
    CLIP encoder for generating image and text embeddings.
    """

    def __init__(self, model_name: str = CLIP_MODEL_NAME, device: str | None = None):
        """
        Initialize CLIP encoder.

        Args:
            model_name: HuggingFace model name
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
        """
        self.model_name = model_name
        self.device = device or DEVICE

        logger.info(f"Loading CLIP model: {model_name}")
        logger.info(f"Using device: {self.device}")

        # Load model and processor
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)

        # Move to device
        self.model = self.model.to(self.device)
        self.model.eval()

        logger.info("CLIP model loaded successfully")

    def encode_images(
        self, image_paths: list[str], batch_size: int = BATCH_SIZE
    ) -> torch.Tensor:
        """
        Encode images to embeddings.

        Args:
            image_paths: List of paths to image files
            batch_size: Batch size for processing

        Returns:
            Tensor of shape [N, EMBEDDING_DIM] with normalized embeddings
        """
        if not image_paths:
            return torch.empty((0, EMBEDDING_DIM), device=self.device)

        all_embeddings = []

        with torch.no_grad():
            for i in range(0, len(image_paths), batch_size):
                batch_paths = image_paths[i : i + batch_size]

                # Load and preprocess images
                images = []
                for path in batch_paths:
                    try:
                        image = Image.open(path).convert("RGB")
                        images.append(image)
                    except Exception as e:
                        logger.warning(f"Failed to load image {path}: {e}")
                        # Use a blank image as fallback
                        images.append(Image.new("RGB", (224, 224), color="black"))

                # Process images
                inputs = self.processor(
                    images=images, return_tensors="pt", padding=True
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # Encode
                image_features = self.model.get_image_features(**inputs)

                # Normalize (L2 normalization for cosine similarity)
                image_features = torch.nn.functional.normalize(
                    image_features, p=2, dim=1
                )

                all_embeddings.append(image_features)

        # Concatenate all batches
        embeddings = torch.cat(all_embeddings, dim=0)
        return embeddings

    def encode_texts(
        self, queries: list[str], batch_size: int = BATCH_SIZE
    ) -> torch.Tensor:
        """
        Encode text queries to embeddings.

        Args:
            queries: List of text queries
            batch_size: Batch size for processing

        Returns:
            Tensor of shape [N, EMBEDDING_DIM] with normalized embeddings
        """
        if not queries:
            return torch.empty((0, EMBEDDING_DIM), device=self.device)

        all_embeddings = []

        with torch.no_grad():
            for i in range(0, len(queries), batch_size):
                batch_queries = queries[i : i + batch_size]

                # Process texts
                inputs = self.processor(
                    text=batch_queries,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # Encode
                text_features = self.model.get_text_features(**inputs)

                # Normalize (L2 normalization for cosine similarity)
                text_features = torch.nn.functional.normalize(text_features, p=2, dim=1)

                all_embeddings.append(text_features)

        # Concatenate all batches
        embeddings = torch.cat(all_embeddings, dim=0)
        return embeddings


# Convenience functions for backward compatibility
_global_encoder: CLIPEncoder | None = None


def get_encoder() -> CLIPEncoder:
    """Get or create global CLIP encoder instance."""
    global _global_encoder
    if _global_encoder is None:
        _global_encoder = CLIPEncoder()
    return _global_encoder


def encode_images(image_paths: list[str], batch_size: int = BATCH_SIZE) -> torch.Tensor:
    """
    Encode images using global encoder.

    Args:
        image_paths: List of paths to image files
        batch_size: Batch size for processing

    Returns:
        Tensor of shape [N, EMBEDDING_DIM] with normalized embeddings
    """
    encoder = get_encoder()
    return encoder.encode_images(image_paths, batch_size)


def encode_texts(queries: list[str], batch_size: int = BATCH_SIZE) -> torch.Tensor:
    """
    Encode text queries using global encoder.

    Args:
        queries: List of text queries
        batch_size: Batch size for processing

    Returns:
        Tensor of shape [N, EMBEDDING_DIM] with normalized embeddings
    """
    encoder = get_encoder()
    return encoder.encode_texts(queries, batch_size)
