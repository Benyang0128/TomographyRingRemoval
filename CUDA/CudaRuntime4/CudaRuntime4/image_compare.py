from PIL import Image
import numpy as np
import os

def load_images(file_paths):
    return [Image.open(path) for path in file_paths]

def calculate_similarity(img1, img2):
    arr1 = np.array(img1)
    arr2 = np.array(img2)
    
    if arr1.shape != arr2.shape:
        raise ValueError("Images must have the same dimensions")
    
    diff = np.abs(arr1 - arr2)
    max_diff = np.max(arr1) * arr1.size
    actual_diff = np.sum(diff)
    
    similarity = 100 - (actual_diff / max_diff * 100)
    return similarity

def compare_image_sets(original_paths, comparison_paths):
    original_images = load_images(original_paths)
    comparison_images = load_images(comparison_paths)
    
    results = []
    for orig, comp in zip(original_images, comparison_images):
        similarity = calculate_similarity(orig, comp)
        results.append(similarity)
    
    return results

# Base directory
base_dir = r'C:\Users\PC\source\repos\CudaRuntime4\CudaRuntime4\OutputImages'

# Generate file paths using for loops
original_paths = [os.path.join(base_dir, f'out_original_{i}.tif') for i in range(1, 17)]
compare_paths = [os.path.join(base_dir, f'test_out{i}.tif') for i in range(1, 17)]

# Compare images
similarities = compare_image_sets(original_paths, compare_paths)

# Print results
for i, similarity in enumerate(similarities, 1):
    print(f"Image pair {i}: {similarity:.2f}% similar")

# Overall similarity
overall_similarity = sum(similarities) / len(similarities)
print(f"\nOverall similarity: {overall_similarity:.2f}%")