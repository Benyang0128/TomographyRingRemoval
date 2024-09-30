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
    r'output/out_original_1.tif',
    r'output/out_original_2.tif',
    r'output/out_original_3.tif',
    r'output/out_original_4.tif',
    r'output/out_original_5.tif',
    r'output/out_original_6.tif',
    r'output/out_original_7.tif',
    r'output/out_original_8.tif',
    r'output/out_original_9.tif',
    r'output/out_original_10.tif',
    r'output/out_original_11.tif',
    r'output/out_original_12.tif',
    r'output/out_original_13.tif',
    r'output/out_original_14.tif',
    r'output/out_original_15.tif',
    r'output/out_original_16.tif',
]
compare_paths = [
    r'output/out_openmp_1.tif',
    r'output/out_openmp_2.tif',
    r'output/out_openmp_3.tif',
    r'output/out_openmp_4.tif',
    r'output/out_openmp_5.tif',
    r'output/out_openmp_6.tif',
    r'output/out_openmp_7.tif',
    r'output/out_openmp_8.tif',
    r'output/out_openmp_9.tif',
    r'output/out_openmp_10.tif',
    r'output/out_openmp_11.tif',
    r'output/out_openmp_12.tif',
    r'output/out_openmp_13.tif',
    r'output/out_openmp_14.tif',
    r'output/out_openmp_15.tif',
    r'output/out_openmp_16.tif',
]

# Compare images
similarities = compare_image_sets(original_paths, compare_paths)

# Print results
for i, similarity in enumerate(similarities, 1):
    print(f"Image pair {i}: {similarity:.2f}% similar")

# Overall similarity
overall_similarity = sum(similarities) / len(similarities)
print(f"\nOverall similarity: {overall_similarity:.2f}%")
