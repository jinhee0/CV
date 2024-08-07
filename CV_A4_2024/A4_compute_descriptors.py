import numpy as np
import os
import struct
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

def load_sift_features(file_path):
    with open(file_path, 'rb') as f:
        data = np.fromfile(f, dtype=np.uint8)
    num_features = len(data) // 128
    return data.reshape((num_features, 128))

def load_cnn_features(file_path):
    with open(file_path, 'rb') as f:
        data = np.fromfile(f, dtype=np.float32)
    return data.reshape((14, 14, 512))

def average_pooling(features, pool_size=2):
    pooled_features = features.reshape(features.shape[0] // pool_size, pool_size, 
                                       features.shape[1] // pool_size, pool_size, 
                                       features.shape[2]).mean(axis=(1, 3))
    return pooled_features

def max_pooling(features, pool_size=2):
    pooled_features = features.reshape(features.shape[0] // pool_size, pool_size, 
                                       features.shape[1] // pool_size, pool_size, 
                                       features.shape[2]).max(axis=(1, 3))
    return pooled_features

def compute_vlad(descriptors, kmeans):
    k = kmeans.n_clusters
    centers = kmeans.cluster_centers_
    labels = kmeans.predict(descriptors)
    vlad = np.zeros((k, descriptors.shape[1]), dtype=np.float32)
    
    for i in range(k):
        if np.sum(labels == i) > 0:
            vlad[i] = np.sum(descriptors[labels == i] - centers[i], axis=0)
    
    # Normalize intra-normalization
    vlad = normalize(vlad, norm='l2')
    # Flatten the VLAD vector
    vlad = vlad.flatten()
    # Normalize final VLAD vector
    vlad = normalize(vlad.reshape(1, -1), norm='l2')
    
    return vlad.flatten()

def compute_descriptors_with_vlad(sift_dir, cnn_dir, num_clusters=60):
    num_images = 2000
    descriptors = []

    kmeans_sift = KMeans(n_clusters=num_clusters, init='k-means++', random_state=0)
    kmeans_cnn = KMeans(n_clusters=num_clusters, init='k-means++', random_state=0)

    scaler_sift = StandardScaler()
    scaler_cnn = StandardScaler()

    pca = PCA(n_components=64)  

    all_sift_features = []
    all_cnn_features = []

    for i in range(num_images):
        sift_file = os.path.join(sift_dir, f'{i:04d}.sift')
        cnn_file = os.path.join(cnn_dir, f'{i:04d}.cnn')

        sift_features = load_sift_features(sift_file)
        cnn_features = load_cnn_features(cnn_file)

        pooled_cnn_avg = average_pooling(cnn_features)
        pooled_cnn_max = max_pooling(cnn_features)
        combined_cnn_features = np.concatenate((pooled_cnn_avg, pooled_cnn_max), axis=-1)

        if sift_features.size > 0:
            all_sift_features.append(sift_features)

        if combined_cnn_features.size > 0:
            all_cnn_features.append(combined_cnn_features.reshape(-1, 1024))

    all_sift_features = np.vstack(all_sift_features)
    all_cnn_features = np.vstack(all_cnn_features)


    all_sift_features = pca.fit_transform(all_sift_features)

    all_sift_features = scaler_sift.fit_transform(all_sift_features)
    all_cnn_features = scaler_cnn.fit_transform(all_cnn_features)

    kmeans_sift.fit(all_sift_features)
    kmeans_cnn.fit(all_cnn_features)

    for i in range(num_images):
        sift_file = os.path.join(sift_dir, f'{i:04d}.sift')
        cnn_file = os.path.join(cnn_dir, f'{i:04d}.cnn')

        sift_features = load_sift_features(sift_file)
        cnn_features = load_cnn_features(cnn_file)

        if sift_features.size == 0 or cnn_features.size == 0:
            continue

        pooled_cnn_avg = average_pooling(cnn_features)
        pooled_cnn_max = max_pooling(cnn_features)
        combined_cnn_features = np.concatenate((pooled_cnn_avg, pooled_cnn_max), axis=-1)

        avg_sift = np.mean(sift_features, axis=0)
        avg_cnn = np.mean(combined_cnn_features, axis=(0, 1))

        sift_features = scaler_sift.transform(pca.transform(sift_features))
        combined_cnn_features = scaler_cnn.transform(combined_cnn_features.reshape(-1, 1024))

        sift_vlad = compute_vlad(sift_features, kmeans_sift)
        cnn_vlad = compute_vlad(combined_cnn_features, kmeans_cnn)

        descriptor = np.concatenate((avg_sift, avg_cnn, sift_vlad, cnn_vlad))
        descriptors.append(descriptor)

    descriptors = np.array(descriptors, dtype=np.float32)
    
    final_descriptors = np.zeros((len(descriptors), 2000), dtype=np.float32)
    for i, desc in enumerate(descriptors):
        if len(desc) > 2000:
            final_descriptors[i] = desc[:2000]
        else:
            final_descriptors[i, :len(desc)] = desc

    return final_descriptors

def main():
    base_dir = './features/'
    sift_dir = os.path.join(base_dir, 'sift')
    cnn_dir = os.path.join(base_dir, 'cnn')
    output_file = 'A4_2021315782.des'

    descriptors = compute_descriptors_with_vlad(sift_dir, cnn_dir)

    num_images, dim = descriptors.shape
    with open(output_file, 'wb') as f:
        f.write(struct.pack('i', num_images))
        f.write(struct.pack('i', dim))
        for descriptor in descriptors:
            f.write(struct.pack('f' * dim, *descriptor))

if __name__ == '__main__':
    main()
