import gradio as gr
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering

def quantize_kmeans(image, n_clusters=3):
    image = np.array(image)
    (h, w, c) = image.shape
    image_2d = image.reshape(h * w, c)
    model = KMeans(n_clusters=n_clusters, random_state=42)
    labels = model.fit_predict(image_2d)
    rgb_codes = model.cluster_centers_.round(0).astype(int)
    quantized_image = np.reshape(rgb_codes[labels], (h, w, c))
    
    
    quantized_image = np.clip(quantized_image, 0, 255).astype(np.uint8)
    return Image.fromarray(quantized_image)

def quantize_dbscan(image, eps=0.3, min_samples=5):
    image = np.array(image)
    (h, w, c) = image.shape
    image_2d = image.reshape(h * w, c)
    
    
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(image_2d)
    
    
    unique_labels = np.unique(labels)
    if -1 in unique_labels:
        unique_labels = unique_labels[unique_labels != -1]
    
    colors = np.array([np.mean(image_2d[labels == i], axis=0) for i in unique_labels])
    quantized_image = np.zeros_like(image)
    for i, label in enumerate(labels):
        if label != -1:
            quantized_image.reshape(h * w, c)[i] = colors[label]
    
    
    quantized_image = np.clip(quantized_image, 0, 255).astype(np.uint8)
    return Image.fromarray(quantized_image)

def quantize_hierarchical(image, n_clusters=4):
    image = np.array(image)
    (h, w, c) = image.shape
    image_2d = image.reshape(h * w, c)
    
    
    model = AgglomerativeClustering(n_clusters=n_clusters)
    labels = model.fit_predict(image_2d)
    
    
    rgb_codes = np.array([np.mean(image_2d[labels == i], axis=0) for i in range(n_clusters)])
    
    
    quantized_image = np.zeros_like(image)
    for i, label in enumerate(labels):
        quantized_image.reshape(h * w, c)[i] = rgb_codes[label]
    
    
    quantized_image = np.clip(quantized_image, 0, 255).astype(np.uint8)
    return Image.fromarray(quantized_image)

def apply_quantization(image, algorithm="KMeans"):
    if algorithm == "KMeans":
        return quantize_kmeans(image)
    elif algorithm == "DBSCAN":
        return quantize_dbscan(image)
    elif algorithm == "Hierarchical":
        return quantize_hierarchical(image)


interface = gr.Interface(
    fn=apply_quantization,  
    inputs=[
        gr.Image(type="pil"),  
        gr.Dropdown(
            choices=["KMeans", "DBSCAN", "Hierarchical"], 
            value="KMeans",  
            label="Clustering Algorithm"
        ),
    ],
    outputs=gr.Image(type="pil"),  
    live=True,  
    title="Image Quantization with KMeans, DBSCAN, and Hierarchical Clustering",
    description="Upload an image and choose a clustering algorithm (KMeans, DBSCAN, or Hierarchical) to perform image quantization."
)


interface.launch()
