
# Image Quantization with KMeans, DBSCAN, and Hierarchical Clustering

This project implements an interactive application for image quantization using three clustering algorithms: **KMeans**, **DBSCAN**, and **Hierarchical Clustering**. The application is built using [Gradio](https://gradio.app/) for a user-friendly interface and supports real-time clustering-based image quantization.

## Features
- **KMeans Clustering**: Partition the image colors into a specified number of clusters using KMeans.
- **DBSCAN Clustering**: Apply density-based clustering for color quantization.
- **Hierarchical Clustering**: Perform agglomerative clustering to group similar colors.
- User-friendly GUI for uploading images and selecting clustering algorithms.
- Real-time quantized image generation.

## Requirements

### Python Libraries
- `gradio`
- `numpy`
- `Pillow`
- `scikit-learn`

You can install the required libraries using:
```bash
pip install gradio numpy Pillow scikit-learn
```

## File Structure
- **Code File**: Contains the implementation of the application.
- **README.md**: Documentation for understanding and running the project.

## How to Run

### Step 1: Clone the Repository
```bash
git clone <repository_url>
cd <repository_directory>
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Run the Application
Run the Python script:
```bash
python <script_name>.py
```

### Step 4: Use the Application
- The Gradio interface will launch in a browser.
- Upload an image and select a clustering algorithm from the dropdown menu.
- The quantized image will be displayed in real-time.

## Functions Overview

### 1. `quantize_kmeans(image, n_clusters=3)`
Quantizes the image using the KMeans clustering algorithm.
- Parameters:
  - `image`: Input image in PIL format.
  - `n_clusters`: Number of color clusters to form (default = 3).
- Returns: Quantized image in PIL format.

### 2. `quantize_dbscan(image, eps=0.3, min_samples=5)`
Quantizes the image using the DBSCAN clustering algorithm.
- Parameters:
  - `image`: Input image in PIL format.
  - `eps`: Maximum distance between samples for clustering.
  - `min_samples`: Minimum samples to form a cluster.
- Returns: Quantized image in PIL format.

### 3. `quantize_hierarchical(image, n_clusters=4)`
Quantizes the image using Hierarchical clustering.
- Parameters:
  - `image`: Input image in PIL format.
  - `n_clusters`: Number of clusters to form (default = 4).
- Returns: Quantized image in PIL format.

### 4. `apply_quantization(image, algorithm="KMeans")`
Dispatches the selected quantization method.
- Parameters:
  - `image`: Input image in PIL format.
  - `algorithm`: Clustering algorithm to use (default = "KMeans").
- Returns: Quantized image in PIL format.

## Gradio Interface

- **Input Components**:
  1. `gr.Image(type="pil")`: Accepts an uploaded image.
  2. `gr.Dropdown`: Allows selection of the clustering algorithm (KMeans, DBSCAN, or Hierarchical).

- **Output Component**:
  - `gr.Image(type="pil")`: Displays the quantized image.

- **Additional Options**:
  - `live=True`: Enables real-time updates for parameter changes.

## Example Usage
1. **Input**: An uploaded image.
2. **Algorithm Selection**: Select one of the clustering methods (e.g., KMeans).
3. **Output**: Quantized image displayed on the interface.

## Customization
- Modify clustering parameters (e.g., `n_clusters` for KMeans, `eps` and `min_samples` for DBSCAN) to experiment with different quantization effects.
- Add more clustering algorithms or preprocessing steps as needed.

## References
- [Gradio Documentation](https://gradio.app/)
- [Scikit-learn Clustering](https://scikit-learn.org/stable/modules/clustering.html)
- [Pillow Documentation](https://pillow.readthedocs.io/en/stable/)

## License
This project is licensed under the Apache License. See the LICENSE file for details.

---

For any issues or suggestions, feel free to open an issue on the repository or contact the developer.

