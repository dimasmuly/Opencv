import json
import numpy as np
from sklearn.cluster import KMeans
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
    encodings.append(average_encoding)
    ids.append(person_id)

# Convert encodings to numpy array
X = np.array(encodings, dtype=np.float32)

# Determine the optimal number of clusters using silhouette score
range_n_clusters = range(2, min(len(X), 10))  # Adjust range as needed
best_n_clusters = 2
best_score = -1

for n_clusters in range_n_clusters:
    print(f'n_clusters: {n_clusters}')
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    labels = kmeans.fit_predict(X)
    print(f'labels: {labels}')
    score = silhouette_score(X, labels)
    print(f'score: {score}')
    print(f'best_score: {best_score}')
    if score > best_score:
        best_score = score
        best_n_clusters = n_clusters

print(f'best_n_clusters: {best_n_clusters}')

# Apply K-Means clustering with the optimal number of clusters
kmeans = KMeans(n_clusters=best_n_clusters, random_state=0)
labels = kmeans.fit_predict(X)

# Group IDs based on cluster labels
grouped_ids = {}
for label, person_id in zip(labels, ids):
    label_str = str(label)  # Convert label to string
    if label_str not in grouped_ids:
        grouped_ids[label_str] = {'ids': []}
    grouped_ids[label_str]['ids'].append(person_id)

# Save the grouped IDs to a new JSON file
with open('faiss_index/grouped_ids.json', 'w') as f:
    json.dump(grouped_ids, f, indent=4)

print(f"Grouped IDs saved successfully with {best_n_clusters} clusters.")