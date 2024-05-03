from PIL import Image
import io

def from_bytes_to_pil(image_bytes):
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")