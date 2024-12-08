import torch
import pandas as pd
import numpy as np
import torch.nn.functional as F
from flask import Flask, request, render_template
from PIL import Image
from sklearn.decomposition import PCA
from open_clip import create_model_and_transforms, get_tokenizer


app = Flask(__name__)

IMAGE_FOLDER = 'static/coco_images_resized'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "ViT-B-32"
PRETRAINED = "openai"

# Load CLIP Model
model, preprocess_train, preprocess_val = create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED)
tokenizer = get_tokenizer(MODEL_NAME)
model = model.to(DEVICE).eval()

with open('image_embeddings.pickle', 'rb') as f:
    embeddings_df = pd.read_pickle(f)

# PCA initialized
pca = PCA(n_components=50)
pca.fit(torch.stack([torch.tensor(embed) for embed in embeddings_df['embedding']]).numpy())

def calculate_results(query_embedding, df, use_pca=False):
    if use_pca:
        query_embedding = torch.tensor(
            pca.transform(query_embedding.detach().numpy().reshape(1, -1))[0], device=DEVICE
        )
        embeddings = torch.stack(
            [torch.tensor(pca.transform(embed.reshape(1, -1))[0], device=DEVICE) for embed in df['embedding']]
        )
    else:
        embeddings = torch.stack(
            [torch.tensor(embed, device=DEVICE) if isinstance(embed, (list, np.ndarray)) else embed for embed in df['embedding']]
        )

    scores = F.cosine_similarity(query_embedding, embeddings, dim=1)
    top_indices = torch.topk(scores, k=5).indices.tolist() 
    top_scores = scores[top_indices]
    top_results = [
        (df.iloc[idx]['file_name'], top_scores[i].item()) for i, idx in enumerate(top_indices)
    ]
    return top_results


@app.route('/', methods=['GET', 'POST'])
def index():
    results = None
    if request.method == 'POST':
        query_type = request.form.get('query_type')
        use_pca = 'use_pca' in request.form
        hybrid_weight = float(request.form.get('hybrid_weight', 0.5))
        
        text_query = request.form.get('text_query', '').strip()
        image_file = request.files.get('image_query')

        query_embedding = None

        if query_type == 'text' and text_query:
            # Tokenize the text query and generate a normalized text embedding
            text_tokens = tokenizer([text_query])
            query_embedding = F.normalize(model.encode_text(text_tokens.to(DEVICE)))
        
        elif query_type == 'image' and image_file:
            # Load the image, preprocess it, and generate a normalized image embedding
            image = Image.open(image_file).convert('RGB')
            image_tensor = preprocess_val(image).unsqueeze(0).to(DEVICE)
            query_embedding = F.normalize(model.encode_image(image_tensor))
        
        elif query_type == 'hybrid' and text_query and image_file:
            # Generate normalized text embedding
            text_tokens = tokenizer([text_query])
            text_embedding = F.normalize(model.encode_text(text_tokens.to(DEVICE)))
            # Generate normalized image embedding
            image = Image.open(image_file).convert('RGB')
            image_tensor = preprocess_val(image).unsqueeze(0).to(DEVICE)
            image_embedding = F.normalize(model.encode_image(image_tensor))
            # Combine text and image embeddings using the hybrid weight
            query_embedding = F.normalize(hybrid_weight * text_embedding + (1 - hybrid_weight) * image_embedding)
        # If a query embedding was successfully generated
        if query_embedding is not None:
            # Calculate results based on the query embedding
            results = calculate_results(query_embedding, embeddings_df, use_pca)

    return render_template('index.html', results=results, image_folder=IMAGE_FOLDER)

if __name__ == '__main__':
    app.run(debug=True)