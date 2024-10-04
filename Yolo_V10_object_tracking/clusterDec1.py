import json
import numpy as np
from sklearn.cluster import KMeans
from keras.layers import Input, Dense
from keras.models import Model
from sklearn.metrics import pairwise_distances
from sklearn.metrics import silhouette_score

# Load existing FAISS data
try:
    with open('faiss_index/faiss_index.json', 'r') as f:
        data = json.load(f)
except json.JSONDecodeError as e:
    print(f"Error reading JSON: {e}")
    exit()

# Extract encodings and IDs
encodings = []
ids = []
for person_id, details in data['person'].items():
    average_encoding = np.mean(details['encoding'], axis=0)
    
    # Normalize the average encoding
    norm = np.linalg.norm(average_encoding)
    normalized_encoding = average_encoding / norm if norm > 0 else average_encoding
    
    encodings.append(normalized_encoding)
    ids.append(person_id)

# Convert encodings to numpy array
X = np.array(encodings, dtype=np.float32)

# Define an autoencoder model for dimensionality reduction
input_dim = X.shape[1]
encoding_dim = 10  # You can adjust this

# Autoencoder architecture
input_layer = Input(shape=(input_dim,))
encoder = Dense(64, activation='relu')(input_layer)
encoder = Dense(encoding_dim, activation='relu')(encoder)

decoder = Dense(64, activation='relu')(encoder)
decoder = Dense(input_dim, activation='sigmoid')(decoder)

# Build the autoencoder model
autoencoder = Model(inputs=input_layer, outputs=decoder)
encoder_model = Model(inputs=input_layer, outputs=encoder)

# Compile the autoencoder
autoencoder.compile(optimizer='adam', loss='mse')

# Train the autoencoder
autoencoder.fit(X, X, epochs=50, batch_size=256, shuffle=True)

# Encode the embeddings (i.e., reduce dimensionality)
encoded_embeddings = encoder_model.predict(X)

# Normalize the encoded embeddings
norms = np.linalg.norm(encoded_embeddings, axis=1, keepdims=True)
normalized_embeddings = encoded_embeddings / norms

# Determine the optimal number of clusters using cosine similarity
range_n_clusters = range(2, min(len(X), 10))  # Adjust range as needed
best_n_clusters = 2
best_score = -1

for n_clusters in range_n_clusters:
    print(f'n_clusters: {n_clusters}')
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    labels = kmeans.fit_predict(normalized_embeddings)
    
    # Calculate cosine similarity matrix
    cosine_sim_matrix = 1 - pairwise_distances(normalized_embeddings, metric='cosine')
    
    # Calculate average cosine similarity for the current clustering
    cluster_similarities = []
    for cluster in range(n_clusters):
        cluster_indices = np.where(labels == cluster)[0]
        if len(cluster_indices) > 1:
            cluster_similarity = np.mean(cosine_sim_matrix[cluster_indices][:, cluster_indices])
            cluster_similarities.append(cluster_similarity)
    
    # Average similarity across clusters
    average_similarity = np.mean(cluster_similarities) if cluster_similarities else 0
    print(f'Average cosine similarity: {average_similarity}')

    # Using average similarity instead of silhouette score for the best score
    if average_similarity > best_score:
        best_score = average_similarity
        best_n_clusters = n_clusters

print(f'best_n_clusters: {best_n_clusters}')

# Apply K-Means clustering with the optimal number of clusters on the normalized embeddings
kmeans = KMeans(n_clusters=best_n_clusters, random_state=0)
labels = kmeans.fit_predict(normalized_embeddings)

# Group IDs based on cluster labels
grouped_ids = {}
for label, person_id in zip(labels, ids):
    label_str = str(label)
    if label_str not in grouped_ids:
        grouped_ids[label_str] = {'ids': []}
    grouped_ids[label_str]['ids'].append(person_id)

# Save the grouped IDs to a new JSON file
with open('faiss_index/grouped_ids_dec_1.json', 'w') as f:
    json.dump(grouped_ids, f, indent=4)

print(f"Grouped IDs saved successfully with {best_n_clusters} clusters.")
