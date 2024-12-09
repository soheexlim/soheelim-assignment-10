
from flask import Flask, render_template, request, jsonify
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from PIL import Image
from open_clip import create_model_and_transforms, get_tokenizer

app = Flask(__name__)

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = create_model_and_transforms('ViT-B-32', pretrained='openai')
model = model.to(device).eval()
tokenizer = get_tokenizer('ViT-B-32')

# Load image embeddings from the dataset
df = pickle.load(open("image_embeddings.pickle", "rb"))
image_embeddings = np.stack(df["embedding"].values)
image_files = df["file_name"].values

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/search", methods=["POST"])
def search():
    try:
        data = request.form
        query_type = data.get("query_type")
        weight = float(data.get("hybrid_weight", 0.5))
        query_embedding = None

        # Handle text query
        if query_type == "text_query":
            text_query = data.get("text_query")
            if not text_query:
                return jsonify({"error": "Text query is empty"}), 400

            text_tokens = tokenizer([text_query])
            query_embedding = model.encode_text(text_tokens.to(device))
            query_embedding = query_embedding / query_embedding.norm(dim=-1, keepdim=True)

        # Handle image query
        elif query_type == "image_query":
            if 'image_query' not in request.files:
                return jsonify({"error": "No image uploaded"}), 400

            image_file = request.files['image_query']
            img = Image.open(image_file).convert("RGB")
            img_tensor = preprocess(img).unsqueeze(0).to(device)
            query_embedding = model.encode_image(img_tensor)
            query_embedding = query_embedding / query_embedding.norm(dim=-1, keepdim=True)

        # Handle hybrid query
        elif query_type == "hybrid_query":
            text_query = data.get("text_query")
            if not text_query:
                return jsonify({"error": "Text query is empty for hybrid search"}), 400

            text_tokens = tokenizer([text_query])
            text_embedding = model.encode_text(text_tokens.to(device))
            text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)

            if 'image_query' not in request.files:
                return jsonify({"error": "No image uploaded for hybrid search"}), 400

            image_file = request.files['image_query']
            img = Image.open(image_file).convert("RGB")
            img_tensor = preprocess(img).unsqueeze(0).to(device)
            image_embedding = model.encode_image(img_tensor)
            image_embedding = image_embedding / image_embedding.norm(dim=-1, keepdim=True)

            # Combine text and image embeddings
            query_embedding = weight * text_embedding + (1 - weight) * image_embedding
            query_embedding = query_embedding / query_embedding.norm(dim=-1, keepdim=True)

        # Compute similarity
        similarities = cosine_similarity(query_embedding.detach().cpu().numpy(), image_embeddings)
        top_indices = np.argsort(similarities[0])[::-1][:5]
        results = [
            {"file_name": image_files[i], "similarity": float(similarities[0][i])}
            for i in top_indices
        ]

        return jsonify(results=results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)