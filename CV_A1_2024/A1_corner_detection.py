import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from A1_image_filtering import get_gaussian_filter_2d, cross_correlation_2d

def compute_corner_response(img):
    mean_vector = np.mean(img)  
    img = img - mean_vector

    sobel_x = np.array([[1, 0, -1],
                        [2, 0, -2],
                        [1, 0, -1]])

    sobel_y = np.array([[1, 2, 1],
                        [0, 0, 0],
                        [-1, -2, -1]])
    
    Ix = cross_correlation_2d(img, sobel_x)
    Iy = cross_correlation_2d(img, sobel_y)
    Ix2 = Ix ** 2
    Iy2 = Iy ** 2
    Ixy = Ix * Iy

    kappa = 0.04
    height, width = img.shape
    R = np.zeros_like(img, dtype=np.float64)

    for i in range(height):
        for j in range(width):
            M = np.zeros((2, 2), dtype=np.float64)
            M[0, 0] = np.sum(Ix2[i-2:i+3, j-2:j+3])
            M[1, 1] = np.sum(Iy2[i-2:i+3, j-2:j+3])
            M[0, 1] = np.sum(Ixy[i-2:i+3, j-2:j+3])
            M[1, 0] = M[0, 1]

            eigenvalues = np.linalg.eigvals(M)
            lambda1, lambda2 = eigenvalues
            R[i, j] = lambda1 * lambda2 - kappa * (lambda1 + lambda2) ** 2

    R[R < 0] = 0
    R = (R - np.min(R)) / (np.max(R) - np.min(R))
    return R

def green_corner(img, corner_response, threshold=0.1):
    img_color = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)
    height, width = img.shape

    for i in range(height):
        for j in range(width):
            if corner_response[i][j] > threshold:
                img_color[i][j] = [0, 255, 0]

    return img_color

def green_corner_circle(img, corner_response):
    img_color = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)
    height, width = img.shape

    for i in range(height):
        for j in range(width):
            if corner_response[i][j]:
                cv2.circle(img_color, (j, i), radius=3, color=(0, 255, 0), thickness=2)  # BGR color format
    return img_color

def non_maximum_suppression_win(R, winSize=11):
    height, width = R.shape
    
    suppressed_R = np.zeros_like(R, dtype=np.float64)
    for i in range(winSize//2, height - winSize//2):
        for j in range(winSize//2, width - winSize//2):
            roi = R[i - winSize//2: i + winSize // 2 + 1, j - winSize//2: j + winSize // 2 + 1]
            max_value = np.max(roi)

            center_value = R[i, j]
            if max_value != center_value or center_value < 0.1:
                max_value = 0  

            suppressed_R[i, j] = max_value

    return suppressed_R

def main():
    img1 = cv2.imread(r'C:\2024-1\CV\A1_Images\lenna.png', cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(r'C:\2024-1\CV\A1_Images\shapes.png', cv2.IMREAD_GRAYSCALE)

    kernel = get_gaussian_filter_2d(7, 1.5)
    img1_g = cross_correlation_2d(img1, kernel)
    img2_g = cross_correlation_2d(img2, kernel)

    start_time = time.time()
    corner_response1 = compute_corner_response(img1_g)    
    end_time = time.time()
    computational_time = end_time - start_time
    print(f"Computational time taken by compute_corner_response(lenna): {computational_time} seconds")
    
    start_time = time.time()
    corner_response2 = compute_corner_response(img2_g)   
    end_time = time.time()
    computational_time = end_time - start_time
    print(f"Computational time taken by compute_corner_response(shapes): {computational_time} seconds")

    plt.imshow(corner_response1, cmap='gray')
    plt.axis('off')
    plt.savefig('./result/part_3_corner_raw_lenna.png', bbox_inches='tight', pad_inches=0)
    plt.show()   
    plt.imshow(corner_response2, cmap='gray')
    plt.axis('off')
    plt.savefig('./result/part_3_corner_raw_shapes.png', bbox_inches='tight', pad_inches=0)
    plt.show()

    img1_color = green_corner(img1, corner_response1)
    plt.imshow(img1_color)
    plt.axis('off')
    plt.savefig('./result/part_3_corner_bin_lenna.png', bbox_inches='tight', pad_inches=0)
    plt.show()
    img2_color = green_corner(img2, corner_response2)
    plt.imshow(img2_color)
    plt.axis('off')
    plt.savefig('./result/part_3_corner_bin_shapes.png', bbox_inches='tight', pad_inches=0)
    plt.show()

    start_time = time.time()
    suppressed_corner1 = non_maximum_suppression_win(corner_response1)
    end_time = time.time()
    computational_time = end_time - start_time
    print(f"\nComputational time taken by non_maximum_suppression_win(lenna): {computational_time} seconds")
    img1_sup_col = green_corner_circle(img1, suppressed_corner1)
    plt.imshow(img1_sup_col, cmap='gray')
    plt.axis('off')
    plt.savefig('./result/part_3_corner_sup_lenna.png', bbox_inches='tight', pad_inches=0)
    plt.show()

    start_time = time.time()
    suppressed_corner2 = non_maximum_suppression_win(corner_response2)
    end_time = time.time()
    computational_time = end_time - start_time
    print(f"Computational time taken by non_maximum_suppression_win(shapes): {computational_time} seconds")
    img2_sup_col = green_corner_circle(img2, suppressed_corner2)
    plt.imshow(img2_sup_col, cmap='gray')
    plt.axis('off')
    plt.savefig('./result/part_3_corner_sup_shapes.png', bbox_inches='tight', pad_inches=0)
    plt.show()



if __name__ == "__main__":
    main()