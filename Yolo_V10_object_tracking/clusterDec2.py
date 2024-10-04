import json
import numpy as np
from sklearn.metrics import silhouette_score
from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import Adam
import tensorflow as tf
from keras.initializers import VarianceScaling
import keras.backend as K
from sklearn.cluster import KMeans

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

# Define Autoencoder for dimensionality reduction
input_dim = X.shape[1]
encoding_dim = 64  # Adjust this dimension as needed for your embeddings

input_layer = Input(shape=(input_dim,))
encoder = Dense(encoding_dim, activation="relu")(input_layer)
decoder_layer = Dense(input_dim, activation="sigmoid")

# Create the autoencoder model
autoencoder = Model(inputs=input_layer, outputs=decoder_layer(encoder))
autoencoder.compile(optimizer='adam', loss='mse')

# Train the autoencoder
autoencoder.fit(X, X, epochs=100, batch_size=256, shuffle=True)

# Get the encoder model to extract lower-dimensional features
encoder_model = Model(inputs=input_layer, outputs=encoder)
X_encoded = encoder_model.predict(X)

# Define DEC Clustering Model
class ClusteringLayer(tf.keras.layers.Layer):
    def __init__(self, n_clusters, **kwargs):
        super(ClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters

    def build(self, input_shape):
        # Initialize cluster centers as trainable weights
        self.cluster_centers = self.add_weight(
            shape=(self.n_clusters, input_shape[1]),
            initializer='glorot_uniform',
            name='cluster_centers',
            trainable=True
        )
        super(ClusteringLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        # Compute soft assignments using Student's t-distribution
        q = 1.0 / (1.0 + (tf.reduce_sum(tf.square(tf.expand_dims(inputs, axis=1) - self.cluster_centers), axis=2)))
        q = q ** ((self.n_clusters + 1.0) / 2.0)
        q = q / tf.reduce_sum(q, axis=1, keepdims=True)
        return q

# Number of clusters (adjust based on your silhouette analysis)
n_clusters = 5  # Set this to the number of clusters you want
encoded_input = Input(shape=(encoding_dim,))

clustering_layer = ClusteringLayer(n_clusters, name='clustering')(encoded_input)

# Define the complete DEC model
decoder_output = decoder_layer(encoded_input)  # Use the decoder layer here
dec_model = Model(inputs=encoded_input, outputs=[clustering_layer, decoder_output])
dec_model.compile(optimizer=Adam(0.0001), loss=['kld', 'mse'])

# Pretrain the DEC model with the autoencoder weights

# Initialize the cluster centers using K-Means on the encoded data
kmeans = KMeans(n_clusters=n_clusters, n_init=20)
y_pred = kmeans.fit_predict(X_encoded)
dec_model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])

# DEC clustering training loop (self-paced learning)
max_iterations = 5000
tol = 1e-3
update_interval = 140
index_array = np.arange(X_encoded.shape[0])

for ite in range(max_iterations):
    if ite % update_interval == 0:
        q, _ = dec_model.predict(X_encoded, verbose=0)
        p = q ** 2 / q.sum(0)
        p = p / p.sum(1, keepdims=True)
        
        y_pred_last = y_pred
        y_pred = q.argmax(1)
        
        delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
        print(f'Iteration {ite}, delta_label {delta_label}')
        
        if delta_label < tol:
            print('Converged.')
            break
        
    idx = index_array[ite % X_encoded.shape[0]]
    dec_model.train_on_batch(X_encoded[idx], [p[idx], X_encoded[idx]])

# Get the final cluster assignments
final_cluster_assignments = q.argmax(1)

# Group IDs based on final cluster assignments
grouped_ids = {}
for label, person_id in zip(final_cluster_assignments, ids):
    label_str = str(label)
    if label_str not in grouped_ids:
        grouped_ids[label_str] = {'ids': []}
    grouped_ids[label_str]['ids'].append(person_id)

# Save the grouped IDs to a new JSON file
with open('faiss_index/grouped_ids_dec.json', 'w') as f:
    json.dump(grouped_ids, f, indent=4)

print(f"Grouped IDs saved successfully with {n_clusters} clusters using DEC.")