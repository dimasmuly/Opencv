import json
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score

# Load existing FAISS data
try:
    with open('faiss_index/faiss_index.json', 'r') as f:
        data = json.load(f)
except json.JSONDecodeError as e:
    print(f"Error reading JSON: {e}")
    exit()

# Extract encodings and IDs while aggregating for duplicates
encodings = {}
for person_id, details in data['person'].items():
    average_encoding = np.mean(details['encoding'], axis=0)
    
    # Normalize the average encoding
    norm = np.linalg.norm(average_encoding)
    normalized_encoding = average_encoding / norm if norm > 0 else average_encoding
    
    # Store or aggregate encodings
    if person_id in encodings:
        encodings[person_id].append(normalized_encoding)
    else:
        encodings[person_id] = [normalized_encoding]

# Average the encodings for each unique person ID
final_encodings = []
final_ids = []
for person_id, encodings_list in encodings.items():
    final_encoding = np.mean(encodings_list, axis=0)  # Average across duplicates
    final_encodings.append(final_encoding)
    final_ids.append(person_id)

# Convert encodings to numpy array
X = np.array(final_encodings, dtype=np.float32)

# Normalize the data again before clustering
norms = np.linalg.norm(X, axis=1, keepdims=True)
normalized_encodings = X / norms

# Try K-Means clustering
best_n_clusters = 2
best_score = -1

# Determine the optimal number of clusters using silhouette score
range_n_clusters = range(2, min(len(X), 15))  # Increased upper limit for more cluster options
for n_clusters in range_n_clusters:
    print(f'K-Means n_clusters: {n_clusters}')
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, init='k-means++')
    labels = kmeans.fit_predict(normalized_encodings)
    score = silhouette_score(normalized_encodings, labels)
    print(f'K-Means score: {score}')
    
    if score > best_score:
        best_score = score
        best_n_clusters = n_clusters

print(f'Best K-Means n_clusters: {best_n_clusters}')

# Apply K-Means clustering with the optimal number of clusters on the normalized embeddings
kmeans = KMeans(n_clusters=best_n_clusters, random_state=0, init='k-means++')
kmeans_labels = kmeans.fit_predict(normalized_encodings)

# Try DBSCAN clustering as an alternative
# Adjusted parameters
dbscan_eps = 0.3  # Lowering eps for tighter clustering
dbscan_min_samples = 2  # Keeping it low for sensitive detection
dbscan = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
dbscan_labels = dbscan.fit_predict(normalized_encodings)

# Group IDs based on K-Means labels
grouped_ids_kmeans = {}
for label, person_id in zip(kmeans_labels, final_ids):
    label_str = str(label)
    if label_str not in grouped_ids_kmeans:
        grouped_ids_kmeans[label_str] = {'ids': []}
    grouped_ids_kmeans[label_str]['ids'].append(person_id)

# Group IDs based on DBSCAN labels
grouped_ids_dbscan = {}
for label, person_id in zip(dbscan_labels, final_ids):
    label_str = str(label)
    if label_str not in grouped_ids_dbscan:
        grouped_ids_dbscan[label_str] = {'ids': []}
    grouped_ids_dbscan[label_str]['ids'].append(person_id)

# Save the grouped IDs to new JSON files
with open('faiss_index/grouped_ids_kmeans.json', 'w') as f:
    json.dump(grouped_ids_kmeans, f, indent=4)

with open('faiss_index/grouped_ids_dbscan.json', 'w') as f:
    json.dump(grouped_ids_dbscan, f, indent=4)

print(f"K-Means Grouped IDs saved successfully with {best_n_clusters} clusters.")
print("DBSCAN Grouped IDs saved successfully.")
