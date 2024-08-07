import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

def pad_image(img, kernel):
    pad_height = kernel.shape[0] // 2
    pad_width = kernel.shape[1] // 2
    padded_img = np.zeros((img.shape[0] + 2 * pad_height, img.shape[1] + 2 * pad_width), dtype=img.dtype)

    if kernel.shape[0] == 1:  # Horizontal
        padded_img[:, pad_width:-pad_width] = img
        padded_img[:, :pad_width] = img[:, 0][:, np.newaxis]  # Left
        padded_img[:, -pad_width:] = img[:, -1][:, np.newaxis]  # Right
    elif kernel.shape[1] == 1:  # Vertical
        padded_img[pad_height:-pad_height, :] = img
        padded_img[:pad_height, :] = img[0, :]  # Top
        padded_img[-pad_height:, :] = img[-1, :]  # Bottom
    else:
        padded_img[pad_height:-pad_height, pad_width:-pad_width] = img
        padded_img[:pad_height, pad_width:-pad_width] = img[0, :]  # Top
        padded_img[-pad_height:, pad_width:-pad_width] = img[-1, :]  # Bottom
        padded_img[:, :pad_width] = padded_img[:, pad_width:2 * pad_width][:, ::-1]  # Left
        padded_img[:, -pad_width:] = padded_img[:, -2 * pad_width:-pad_width][:, ::-1]  # Right
    
    return padded_img

def cross_correlation_1d(img , kernel):
    padded_img = pad_image(img, kernel)
    filtered_img = np.zeros_like(img, dtype=np.float64)
    if kernel.shape[0] == 1:  # Horizontal
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                filtered_img[i, j] = np.sum(padded_img[i, j:j+kernel.shape[1]] * kernel[0])
    else:  # Vertical
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                filtered_img[i, j] = np.sum(padded_img[i:i+kernel.shape[0], j] * kernel[:, 0])
    return filtered_img

def cross_correlation_2d(img, kernel):
    padded_img = pad_image(img, kernel)
    filtered_img = np.zeros_like(img, dtype=np.float64)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            filtered_img[i, j] = np.sum(padded_img[i:i+kernel.shape[0], j:j+kernel.shape[1]] * kernel)
    return filtered_img

def get_gaussian_filter_1d(size, sigma):
    kernel_1d = np.zeros(size, dtype=np.float64)
    center = size // 2
    total = 0
    for i in range(size):
        x = i - center
        kernel_1d[i] = np.exp(-x**2 / (2 * sigma**2)) / (np.sqrt(2 * np.pi) * sigma)
        total += kernel_1d[i]
    kernel_1d /= total
    kernel_1d = np.reshape(kernel_1d, (1, -1))  # Reshape to 1xsize
    return kernel_1d

def get_gaussian_filter_2d(size, sigma):
    kernel_1d = get_gaussian_filter_1d(size, sigma)
    kernel_2d = np.outer(kernel_1d, kernel_1d)
    return kernel_2d

def main():
    img1 = cv2.imread(r'C:\2024-1\CV\A1_Images\lenna.png', cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(r'C:\2024-1\CV\A1_Images\shapes.png', cv2.IMREAD_GRAYSCALE)

    size = 5
    sigma = 1
    kernel_1d = get_gaussian_filter_1d(size, sigma)
    kernel_2d = get_gaussian_filter_2d(size, sigma)

    print("1D Gaussian Filter Kernel:")
    print(kernel_1d)
    print("\n2D Gaussian Filter Kernel:")
    print(kernel_2d)

    kernel_sizes = [5, 11, 17]
    sigma_values = [1, 6, 11]

    num_rows = len(kernel_sizes)
    num_cols = len(sigma_values)

    # 1. lenna
    plt.figure(figsize=(8, 8))
    subplot_index = 1
    for size in kernel_sizes:
        for sigma in sigma_values:
            kernel_2d = get_gaussian_filter_2d(size, sigma)
            filtered_img_2d = cross_correlation_2d(img1, kernel_2d)
            
            plt.subplot(num_rows, num_cols, subplot_index)
            plt.imshow(filtered_img_2d, cmap='gray')
            plt.text(0.05, 0.95, f'{size}x{size}, s={sigma}', color='black', fontsize=8, transform=plt.gca().transAxes, ha='left', va='top')
            plt.axis('off')

            subplot_index += 1
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig('./result/part_1_gaussian_filtered_lenna.png', bbox_inches='tight', pad_inches=0)
    plt.show()

    # 2. shapes
    plt.figure(figsize=(8, 8))
    subplot_index = 1
    for size in kernel_sizes:
        for sigma in sigma_values:
            kernel_2d = get_gaussian_filter_2d(size, sigma)
            filtered_img_2d = cross_correlation_2d(img2, kernel_2d)
            
            plt.subplot(num_rows, num_cols, subplot_index)
            plt.imshow(filtered_img_2d, cmap='gray')
            plt.text(0.05, 0.95, f'{size}x{size}, s={sigma}', color='black', fontsize=8, transform=plt.gca().transAxes, ha='left', va='top')
            plt.axis('off')

            subplot_index += 1
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig('./result/part_1_gaussian_filtered_shapes.png', bbox_inches='tight', pad_inches=0)
    plt.show()

    size = 5
    sigma = 1

    # compare lenna
    kernel_1d_vertical = get_gaussian_filter_1d(size, sigma).T
    kernel_1d_horizontal = get_gaussian_filter_1d(size, sigma)
    kernel_2d = get_gaussian_filter_2d(size, sigma)

    print("======== compare lenna ========")
    start_time = time.time()
    filtered_img_1d_vertical = cross_correlation_1d(img1, kernel_1d_vertical)
    filtered_img_1d = cross_correlation_1d(filtered_img_1d_vertical, kernel_1d_horizontal)
    end_time_1d = time.time()
    print(f'Computational Time (1D Filtering): {end_time_1d - start_time}')

    start_time = time.time()
    filtered_img_2d = cross_correlation_2d(img1, kernel_2d)
    end_time_2d = time.time()
    print(f'Computational Time (2D Filtering): {end_time_2d - start_time}')

    diff_map = np.abs(filtered_img_1d - filtered_img_2d)
    sum_diff = np.sum(diff_map.astype(np.float32))
    print(f'sum of absolute intensity differences : {sum_diff}')
    cv2.imshow('Difference Map', diff_map.astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print()

    # compare shapes
    print("======== compare shapes =========")
    start_time = time.time()
    filtered_img_1d_vertical = cross_correlation_1d(img2, kernel_1d_vertical)
    filtered_img_1d = cross_correlation_1d(filtered_img_1d_vertical, kernel_1d_horizontal)
    end_time_1d = time.time()
    print(f'Computational Time (1D Filtering): {end_time_1d - start_time}')

    start_time = time.time()
    filtered_img_2d = cross_correlation_2d(img2, kernel_2d)
    end_time_2d = time.time()
    print(f'Computational Time (2D Filtering): {end_time_2d - start_time}')

    filtered_img_2d = cross_correlation_2d(img2, kernel_2d)
    diff_map = np.abs(filtered_img_1d.astype(np.float32) - filtered_img_2d.astype(np.float32))
    sum_diff = np.sum(diff_map)
    print(f'sum of absolute intensity differences : {sum_diff}')
    cv2.imshow('Difference Map', diff_map.astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print()

if __name__ == "__main__":
    main()