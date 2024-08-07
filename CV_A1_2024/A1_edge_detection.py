import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
from A1_image_filtering import get_gaussian_filter_2d, cross_correlation_2d

def compute_image_gradient(filtered_img): 
    sobel_x = np.array([[1, 0, -1],
                        [2, 0, -2],
                        [1, 0, -1]])

    sobel_y = np.array([[1, 2, 1],
                        [0, 0, 0],
                        [-1, -2, -1]])
    

    gradient_x = cross_correlation_2d(filtered_img, sobel_x)
    gradient_y = cross_correlation_2d(filtered_img, sobel_y)

    magnitude = np.sqrt(gradient_x*gradient_x+ gradient_y*gradient_y)
    direction = np.arctan2(gradient_y, gradient_x)

    return magnitude , direction

def quantize_direction(angle):
    quantized_angles = [0, 45, 90, 135, 180, 225, 270, 315] # 8 angles
    closest_angle = min(quantized_angles, key=lambda x: abs(angle - x))
    return closest_angle

def non_maximum_suppression_dir(magnitude, direction):
    suppressed_magnitude = np.zeros_like(magnitude, dtype=np.float64)
    rows, cols = magnitude.shape

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            angle = np.degrees(direction[i, j]) % 360
            quantized_angle = quantize_direction(angle)
            # neighbors 중 max
            if quantized_angle == 0:
                neighbors = [(i, j-1), (i, j+1)]
            elif quantized_angle == 45:
                neighbors = [(i-1, j-1), (i+1, j+1)]
            elif quantized_angle == 90:
                neighbors = [(i-1, j), (i+1, j)]
            elif quantized_angle == 135:
                neighbors = [(i-1, j+1), (i+1, j-1)]
            elif quantized_angle == 180:
                neighbors = [(i, j-1), (i, j+1)]
            elif quantized_angle == 225:
                neighbors = [(i-1, j-1), (i+1, j+1)]
            elif quantized_angle == 270:
                neighbors = [(i-1, j), (i+1, j)]
            elif quantized_angle == 315:
                neighbors = [(i-1, j+1), (i+1, j-1)]

            center_magnitude = magnitude[i, j]
            neighbor_magnitudes = [magnitude[n[0], n[1]] for n in neighbors]
            max_neighbor_magnitude = max(neighbor_magnitudes)

            # max와 center 비교
            if center_magnitude >= max_neighbor_magnitude:
                suppressed_magnitude[i, j] = center_magnitude
            else:
                suppressed_magnitude[i, j] = 0

    return suppressed_magnitude

def main():
    img1 = cv2.imread(r'C:\2024-1\CV\A1_Images\lenna.png', cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(r'C:\2024-1\CV\A1_Images\shapes.png', cv2.IMREAD_GRAYSCALE)

    kernel = get_gaussian_filter_2d(7, 1.5)
    img1 = cross_correlation_2d(img1, kernel)
    img2 = cross_correlation_2d(img2, kernel)

    start_time = time.time()
    magnitude1, direction1 = compute_image_gradient(img1)
    end_time = time.time()
    computational_time = end_time - start_time
    print(f"Computational time taken by compute_image_gradient(lenna): {computational_time} seconds")   
    start_time = time.time()
    magnitude2, direction2 = compute_image_gradient(img2)
    end_time = time.time()
    computational_time = end_time - start_time
    print(f"Computational time taken by compute_image_gradient(shapes): {computational_time} seconds")
    
    plt.imshow(magnitude1, cmap='gray')
    plt.axis('off')
    plt.savefig('./result/part_2_edge_raw_lenna.png', bbox_inches='tight', pad_inches=0)
    plt.show()

    plt.imshow(magnitude2, cmap='gray')
    plt.axis('off')
    plt.savefig('./result/part_2_edge_raw_shapes.png', bbox_inches='tight', pad_inches=0)
    plt.show()

    start_time = time.time()
    suppressed_magnitude1 = non_maximum_suppression_dir(magnitude1, direction1)
    end_time = time.time()
    computational_time = end_time - start_time
    print(f"\nComputational time taken by  non_maximum_suppression_dir(lenna): {computational_time} seconds")
    start_time = time.time()
    suppressed_magnitude2 = non_maximum_suppression_dir(magnitude2, direction2)
    end_time = time.time()
    computational_time = end_time - start_time
    print(f"Computational time taken by  non_maximum_suppression_dir(shapes): {computational_time} seconds")

    plt.imshow(suppressed_magnitude1, cmap='gray')
    plt.axis('off')
    plt.savefig('./result/part_2_edge_sup_lenna.png', bbox_inches='tight', pad_inches=0)
    plt.show()

    plt.imshow(suppressed_magnitude2, cmap='gray')
    plt.axis('off')
    plt.savefig('./result/part_2_edge_sup_shapes.png', bbox_inches='tight', pad_inches=0)
    plt.show()

if __name__ == "__main__":
    main()