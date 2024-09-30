
from PIL import Image
import numpy as np


def load_images(file_paths):
    return [Image.open(path) for path in file_paths]

def calculate_similarity(img1, img2):
    # Convert images to numpy arrays
    arr1 = np.array(img1)
    arr2 = np.array(img2)
    
    # Ensure the arrays have the same shape
    if arr1.shape != arr2.shape:
        raise ValueError("Images must have the same dimensions")
    
    # Calculate the absolute difference
    diff = np.abs(arr1 - arr2)
    
    # Calculate the maximum possible difference
    max_diff = np.max(arr1) * arr1.size
    
    # Calculate the actual difference
    actual_diff = np.sum(diff)
    
    # Calculate similarity percentage
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

# File paths
original_paths = [
    r'C:\Users\tbeny\OneDrive\Desktop\RingRemoval_OpenCL\OutputImages\out_original_1.tif',
    r'C:\Users\tbeny\OneDrive\Desktop\RingRemoval_OpenCL\OutputImages\out_original_2.tif',
    r'C:\Users\tbeny\OneDrive\Desktop\RingRemoval_OpenCL\OutputImages\out_original_3.tif',
    r'C:\Users\tbeny\OneDrive\Desktop\RingRemoval_OpenCL\OutputImages\out_original_4.tif',
    r'C:\Users\tbeny\OneDrive\Desktop\RingRemoval_OpenCL\OutputImages\out_original_5.tif',
    r'C:\Users\tbeny\OneDrive\Desktop\RingRemoval_OpenCL\OutputImages\out_original_6.tif',
    r'C:\Users\tbeny\OneDrive\Desktop\RingRemoval_OpenCL\OutputImages\out_original_7.tif',
    r'C:\Users\tbeny\OneDrive\Desktop\RingRemoval_OpenCL\OutputImages\out_original_8.tif',
    r'C:\Users\tbeny\OneDrive\Desktop\RingRemoval_OpenCL\OutputImages\out_original_9.tif',
    r'C:\Users\tbeny\OneDrive\Desktop\RingRemoval_OpenCL\OutputImages\out_original_10.tif',
    r'C:\Users\tbeny\OneDrive\Desktop\RingRemoval_OpenCL\OutputImages\out_original_11.tif',
    r'C:\Users\tbeny\OneDrive\Desktop\RingRemoval_OpenCL\OutputImages\out_original_12.tif',
    r'C:\Users\tbeny\OneDrive\Desktop\RingRemoval_OpenCL\OutputImages\out_original_13.tif',
    r'C:\Users\tbeny\OneDrive\Desktop\RingRemoval_OpenCL\OutputImages\out_original_14.tif',
    r'C:\Users\tbeny\OneDrive\Desktop\RingRemoval_OpenCL\OutputImages\out_original_15.tif',
    r'C:\Users\tbeny\OneDrive\Desktop\RingRemoval_OpenCL\OutputImages\out_original_16.tif',
]
compare_paths = [
    r'C:\Users\tbeny\OneDrive\Desktop\RingRemoval_OpenCL\OutputImages\out_opencl_1.tif',
    r'C:\Users\tbeny\OneDrive\Desktop\RingRemoval_OpenCL\OutputImages\out_opencl_2.tif',
    r'C:\Users\tbeny\OneDrive\Desktop\RingRemoval_OpenCL\OutputImages\out_opencl_3.tif',
    r'C:\Users\tbeny\OneDrive\Desktop\RingRemoval_OpenCL\OutputImages\out_opencl_4.tif',
    r'C:\Users\tbeny\OneDrive\Desktop\RingRemoval_OpenCL\OutputImages\out_opencl_5.tif',
    r'C:\Users\tbeny\OneDrive\Desktop\RingRemoval_OpenCL\OutputImages\out_opencl_6.tif',
    r'C:\Users\tbeny\OneDrive\Desktop\RingRemoval_OpenCL\OutputImages\out_opencl_7.tif',
    r'C:\Users\tbeny\OneDrive\Desktop\RingRemoval_OpenCL\OutputImages\out_opencl_8.tif',
    r'C:\Users\tbeny\OneDrive\Desktop\RingRemoval_OpenCL\OutputImages\out_opencl_9.tif',
    r'C:\Users\tbeny\OneDrive\Desktop\RingRemoval_OpenCL\OutputImages\out_opencl_10.tif',
    r'C:\Users\tbeny\OneDrive\Desktop\RingRemoval_OpenCL\OutputImages\out_opencl_11.tif',
    r'C:\Users\tbeny\OneDrive\Desktop\RingRemoval_OpenCL\OutputImages\out_opencl_12.tif',
    r'C:\Users\tbeny\OneDrive\Desktop\RingRemoval_OpenCL\OutputImages\out_opencl_13.tif',
    r'C:\Users\tbeny\OneDrive\Desktop\RingRemoval_OpenCL\OutputImages\out_opencl_14.tif',
    r'C:\Users\tbeny\OneDrive\Desktop\RingRemoval_OpenCL\OutputImages\out_opencl_15.tif',
    r'C:\Users\tbeny\OneDrive\Desktop\RingRemoval_OpenCL\OutputImages\out_opencl_16.tif',
]

# Compare images
similarities = compare_image_sets(original_paths, compare_paths)

# Print results
for i, similarity in enumerate(similarities, 1):
    print(f"Image pair {i}: {similarity:.2f}% similar")

# Overall similarity
overall_similarity = sum(similarities) / len(similarities)
print(f"\nOverall similarity: {overall_similarity:.2f}%")