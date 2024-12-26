# Image Search: Simplified Version of Google Image Search

## Project Overview

This project implements a **simplified version of Google Image Search** with a graphical user interface (GUI). The application allows users to search for images based on text queries, image queries, or a combination of both. The goal was to return the top 5 most relevant images from a database, ranked by their similarity to the user's query, and display similarity scores. This project also allowed for the use of different embedding methods (CLIP or PCA-based embeddings) to enhance the search capabilities.

## Key Features

### 1. **Text-based Image Search**
- The user can input a **text query**.
- Upon clicking the search button, the system displays the **top 5 relevant images** from the database, along with their **similarity scores**.

### 2. **Image-based Search**
- The user can **upload an image** as a query.
- The system returns the **top 5 relevant images** based on **image similarity**, along with similarity scores.

### 3. **Combined Search (Text + Image)**
- The user can **upload both an image and a text query**.
- The system allows the user to specify a weight (between **0.0 and 1.0**) for how much to weigh the text query relative to the image query.
- The top 5 relevant images are displayed, with similarity scores adjusted according to the specified weight.

### 4. **Choice of Embeddings**
- The system supports **CLIP embeddings** (Contrastive Language-Image Pretraining), which are commonly used for semantic search.
- Alternatively, the user can choose to use embeddings based on the first **k principal components** (PCA-based embeddings), which can be used to reduce the dimensionality of image data for faster similarity matching.

### 5. **GUI Implementation**
The application features a **user-friendly graphical interface** built using a suitable Python GUI framework (e.g., Tkinter or PyQt). The GUI includes:
- A text input box for text queries.
- An option to upload an image file for image queries.
- A combined input for both text and image queries.
- A slider to adjust the weighting between text and image queries.
- A display section for showing the top 5 relevant images along with their similarity scores.

## Project Workflow

### Part 1: Preprocessing
- **Image Data:** The images used in the search database were resized and stored in a zip file (`coco_images_resized.zip`).
- **Embeddings:** The **image embeddings** were precomputed and stored in a pickle file (`image_embeddings.pickle`). These embeddings represent each image in the database as a high-dimensional vector, which is used for similarity comparison.
- **Test Image:** An example image (`house.jpg`) was provided for testing purposes.

### Part 2: GUI and Functionality
- The GUI was built to allow users to interact with the system by inputting text queries, uploading images, and adjusting query weights for combined search.
- The **search functionality** uses the precomputed embeddings to compare the query (whether text or image) to all images in the database. It then ranks the images based on similarity.

### Part 3: Similarity Calculation
- **Text Query:** Text queries are embedded using a pre-trained model (such as CLIP) to obtain a vector representation of the text.
- **Image Query:** Image queries are embedded using either the CLIP model or PCA-based embeddings.
- The similarity between the query and database images is calculated using cosine similarity or other suitable distance metrics, and the top 5 most relevant images are displayed.

### Part 4: Weighting of Queries (Text + Image)
- If both a text and image query are provided, the user can specify a weight between **0.0 and 1.0**. The weight controls how much influence the text query has relative to the image query, allowing for flexible search capabilities.

This code is available in my **GitHub repository**.

## Requirements

- **Text query input** and **image query upload** functionality.
- **Combined query search** with adjustable weights between text and image.
- Option to use **CLIP** or **PCA-based embeddings** for image search.
- **GUI** with easy-to-use controls and image display.

## Conclusion

This project allowed me to implement a fully functional image search system with a GUI. It involved:
- Using **embeddings** for efficient similarity-based image search.
- Creating a **user-friendly interface** for text and image queries.
- Allowing users to experiment with different embeddings and query weights to fine-tune search results.

This project enhanced my skills in **image processing**, **embeddings**, **similarity calculation**, and **GUI development** while providing a practical tool for searching images based on text and image queries.

