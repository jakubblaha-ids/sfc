"""Utility functions for the robot localization application."""

from PIL import Image
import numpy as np


def embedding_to_image(embedding: np.ndarray, interleaved_rgb: bool) -> Image.Image:
    """
    Convert an embedding vector back to a camera view image for visualization.

    Args:
        embedding: The embedding vector to convert
        interleaved_rgb: Whether the embedding uses interleaved RGB encoding

    Returns:
        PIL Image representing the embedding as a 1-pixel tall strip
    """
    num_pixels = len(embedding) // 3

    if interleaved_rgb:
        # Interleaved: [R0, G0, B0, R1, G1, B1, ...]
        pixels = []
        for i in range(num_pixels):
            r = int(np.clip(embedding[i * 3], 0, 255))
            g = int(np.clip(embedding[i * 3 + 1], 0, 255))
            b = int(np.clip(embedding[i * 3 + 2], 0, 255))
            pixels.append((r, g, b))
    else:
        # Channel-separated: [R0, R1, ..., G0, G1, ..., B0, B1, ...]
        pixels = []
        for i in range(num_pixels):
            r = int(np.clip(embedding[i], 0, 255))
            g = int(np.clip(embedding[num_pixels + i], 0, 255))
            b = int(np.clip(embedding[2 * num_pixels + i], 0, 255))
            pixels.append((r, g, b))

    # Create a 1-pixel tall image
    img = Image.new('RGB', (num_pixels, 1))
    img.putdata(pixels)

    return img
