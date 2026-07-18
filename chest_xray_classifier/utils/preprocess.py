import numpy as np
from PIL import Image
import io
from typing import Union, Dict, Any
import os

def load_image(image_source: Union[str, io.BytesIO]) -> Image.Image:
    """
    Load image from file path or BytesIO object.

    Args:
        image_source: File path (str) or BytesIO object

    Returns:
        PIL Image object

    Raises:
        ValueError: If image cannot be loaded or is invalid
    """
    try:
        if isinstance(image_source, str):
            if not os.path.exists(image_source):
                raise FileNotFoundError(f"Image file not found: {image_source}")
            image = Image.open(image_source)
        elif isinstance(image_source, io.BytesIO):
            image = Image.open(image_source)
        else:
            raise ValueError("image_source must be a file path (str) or BytesIO object")

        # Convert to RGB if necessary
        if image.mode not in ('RGB', 'L'):
            image = image.convert('RGB')

        return image
    except Exception as e:
        raise ValueError(f"Failed to load image: {str(e)}")

def preprocess_for_model(image: Image.Image, target_size: tuple = (224, 224)) -> np.ndarray:
    """
    Preprocess image for model input: resize, convert to RGB, normalize, add batch dimension.

    Args:
        image: PIL Image object
        target_size: Target size as (height, width) tuple

    Returns:
        Preprocessed image array ready for model input (shape: (1, height, width, 3))
    """
    # Resize image
    image = image.resize(target_size)

    # Convert to RGB if not already
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Convert to numpy array
    img_array = np.array(image)

    # Normalize to [0, 1]
    img_array = img_array.astype(np.float32) / 255.0

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    return img_array

def get_image_metadata(image: Image.Image) -> Dict[str, Any]:
    """
    Get metadata for a PIL Image.

    Args:
        image: PIL Image object

    Returns:
        Dictionary with image metadata
    """
    # Get file size if image has a filename
    file_size = None
    if hasattr(image, 'filename') and image.filename:
        try:
            file_size = os.path.getsize(image.filename)
        except:
            file_size = None

    return {
        'width': image.width,
        'height': image.height,
        'mode': image.mode,
        'file_size': file_size,
        'file_size_mb': f"{file_size / (1024*1024):.2f} MB" if file_size else "Unknown"
    }