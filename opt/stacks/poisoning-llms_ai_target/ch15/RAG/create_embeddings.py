import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np

# Load the dataset
df = pd.read_csv('data/ingredients_reviews.csv')  # Update the path as necessary

# Initialize the Sentence Transformer model
model = SentenceTransformer ('all-MiniLM-L6-v2')

# Prepare texts for embedding generation
texts = df['ingredient'] + " " + df['user_comments']  # Combine ingredient names and comments

# Generate embeddings
embeddings = model.encode(texts.tolist(), show_progress_bar=True)

# Save embeddings to a file
np.save('user_reviews_embeddings.npy', embeddings)
