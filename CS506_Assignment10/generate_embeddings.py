import numpy as np
from PIL import Image
from open_clip import create_model_and_transforms, get_tokenizer
import torch

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = create_model_and_transforms('ViT-B-32', pretrained='openai')
model = model.to(device).eval()

# Precompute text embedding
text_query = "snowy"  # Example text query
tokenizer = get_tokenizer("ViT-B-32")
text_tokens = tokenizer([text_query])
text_embedding = model.encode_text(text_tokens.to(device))
text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)
np.save("precomputed_text_embedding.npy", text_embedding.detach().cpu().numpy())

# Precompute image embedding for house.jpg
image_path = "house.jpg"  # Path to your sample image
img = Image.open(image_path).convert("RGB")
img_tensor = preprocess(img).unsqueeze(0).to(device)
image_embedding = model.encode_image(img_tensor)
image_embedding = image_embedding / image_embedding.norm(dim=-1, keepdim=True)
np.save("precomputed_image_embedding.npy", image_embedding.detach().cpu().numpy())

print("Embeddings precomputed and saved!")