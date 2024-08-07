import numpy as np
import cv2
from CV_Assignment_3_Data.compute_avg_reproj_error import compute_avg_reproj_error
import random

def compute_F_raw(M):
    pts1 = np.hstack((M[:, :2], np.ones((M.shape[0], 1)))) 
    pts2 = np.hstack((M[:, 2:], np.ones((M.shape[0], 1))))

    # Construct the A matrix
    A = np.zeros((M.shape[0], 9))
    for i in range(M.shape[0]):
        A[i] = [
            pts1[i, 0] * pts2[i, 0], # x1x1'
            pts2[i, 0] * pts1[i, 1], # x1'y1
            pts2[i, 0], # x1'
            pts1[i, 0] * pts2[i, 1], # x1y1'
            pts1[i, 1] * pts2[i, 1], # y1y1'
            pts2[i, 1], # y1'
            pts1[i, 0], # x1
            pts1[i, 1], # y1
            1
        ]
    
    _, _, vh = np.linalg.svd(A)
    
    F_vec = vh[-1]

    F = F_vec.reshape(3, 3)

    return F / F[2, 2]

def compute_F_norm(M):
    mean1 = np.mean(M[:, :2], axis=0)
    mean2 = np.mean(M[:, 2:], axis=0)
    
    std1 = np.mean(np.linalg.norm(M[:, :2] - mean1, axis=1))
    std2 = np.mean(np.linalg.norm(M[:, 2:] - mean2, axis=1))

    scale_factor1 = np.sqrt(2) / std1
    scale_factor2 = np.sqrt(2) / std2
    
    # Create the transformation matrices
    T_translate1 = np.array([[1, 0, -mean1[0]],
                             [0, 1, -mean1[1]],
                             [0, 0, 1]])
    
    T_translate2 = np.array([[1, 0, -mean2[0]],
                             [0, 1, -mean2[1]],
                             [0, 0, 1]])
    
    T_scale1 = np.array([[scale_factor1, 0, 0],
                         [0, scale_factor1, 0],
                         [0, 0, 1]])
    
    T_scale2 = np.array([[scale_factor2, 0, 0],
                         [0, scale_factor2, 0],
                         [0, 0, 1]])
    
    # Combined translation and scaling transformations
    T1 = T_scale1 @ T_translate1
    T2 = T_scale2 @ T_translate2
    
    ones = np.ones((M.shape[0], 1))
    normalized_pts1 = (T1 @ np.hstack((M[:, :2], ones)).T).T
    normalized_pts2 = (T2 @ np.hstack((M[:, 2:], ones)).T).T
    
    # Remove the homogeneous coordinate
    normalized_pts1 = normalized_pts1[:, :2]
    normalized_pts2 = normalized_pts2[:, :2]

    # Construct the A matrix
    A = np.column_stack((normalized_pts1[:, 0] * normalized_pts2[:, 0],
                         normalized_pts2[:, 0] * normalized_pts1[:, 1],
                         normalized_pts2[:, 0],
                         normalized_pts1[:, 0] * normalized_pts2[:, 1],
                         normalized_pts1[:, 1] * normalized_pts2[:, 1],
                         normalized_pts2[:, 1],
                         normalized_pts1[:, 0],
                         normalized_pts1[:, 1],
                         np.ones((normalized_pts1.shape[0],))))
    
    # Perform SVD on A
    _, _, vh = np.linalg.svd(A)
    
    # Last column of V matrix gives the solution vector
    F_vec = vh[-1]
    F = F_vec.reshape(3, 3)
    
    # Enforce rank-2 constraint on F
    U, S, Vt = np.linalg.svd(F)
    S[-1] = 0
    F = U @ np.diag(S) @ Vt

    F = T2.T @ F @ T1
    
    return F / F[2, 2]

def compute_F_mine(M):
    best_F = None
    best_inliers = []
    num_matches = M.shape[0]

    # RANSAC parameters
    num_iterations = 10000
    threshold = 0.1

    for iteration in range(num_iterations):
        sample_indices = random.sample(range(num_matches), 8)
        sample_matches = M[sample_indices]

        F = compute_F_norm(sample_matches)
        
        pts1 = np.hstack((M[:, :2], np.ones((M.shape[0], 1)))) 
        pts2 = np.hstack((M[:, 2:], np.ones((M.shape[0], 1)))) 

        lines1 = (F @ pts2.T).T 
        lines1 /= np.linalg.norm(lines1[:, :2], axis=1, keepdims=True)  # Normalize lines

        distances = np.abs(np.sum(lines1 * pts1, axis=1))

        inliers = np.where(distances < threshold)[0]

        if len(inliers) > len(best_inliers):
            best_F = F
            best_inliers = inliers
            #print(f"Updated best_F at iteration {iteration} with {len(best_inliers)} inliers")

    return best_F

def draw_epipolar_lines(img1, img2, lines, pts1, pts2):
    r, c = img1.shape
    img1_color = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2_color = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

    for i, (r, pt1, pt2) in enumerate(zip(lines, pts1, pts2)):
        color = colors[i]
        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
        img1_color = cv2.line(img1_color, (x0, y0), (x1, y1), color, 1)
        img1_color = cv2.circle(img1_color, tuple(pt1), 5, color, -1)

    return img1_color, img2_color

def visualize_epipolar_lines(image1_path, image2_path, matches_path):
    image1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)
    M = np.loadtxt(matches_path)
    F = compute_F_mine(M)
    
    while True:
        indices = random.sample(range(M.shape[0]), 3)
        pts1 = M[indices, :2].astype(int)
        pts2 = M[indices, 2:].astype(int)
        
        # Compute the epipolar lines
        lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F).reshape(-1, 3)
        lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F).reshape(-1, 3)
        
        img1_with_lines, _ = draw_epipolar_lines(image1, image2, lines1, pts1, pts2)
        img2_with_lines, _ = draw_epipolar_lines(image2, image1, lines2, pts2, pts1)
        
        combined_image = np.hstack((img1_with_lines, img2_with_lines))
        cv2.imshow('Epipolar Lines', combined_image)
        
        key = cv2.waitKey(0)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()

def main():
    M_temple = np.loadtxt('./CV_Assignment_3_Data/temple_matches.txt')

    F_raw = compute_F_raw(M_temple)
    error_raw = compute_avg_reproj_error(M_temple, F_raw)
    F_norm = compute_F_norm(M_temple)
    error_norm = compute_avg_reproj_error(M_temple, F_norm)
    F_mine = compute_F_mine(M_temple)
    error_mine = compute_avg_reproj_error(M_temple, F_mine)

    print("Average reprojection error (temple1.png and temple2.png)")
    print("Raw = ", error_raw)
    print("Norm = ", error_norm)
    print("MIne = ", error_mine)

    M_house = np.loadtxt('./CV_Assignment_3_Data/house_matches.txt')

    F_raw = compute_F_raw(M_house)
    error_raw = compute_avg_reproj_error(M_house, F_raw)
    F_norm = compute_F_norm(M_house)
    error_norm = compute_avg_reproj_error(M_house, F_norm)
    F_mine = compute_F_mine(M_house)
    error_mine = compute_avg_reproj_error(M_house, F_mine)

    print("Average reprojection error (house.png and house2.png)")
    print("Raw = ", error_raw)
    print("Norm = ", error_norm)
    print("MIne = ", error_mine)

    M_library = np.loadtxt('./CV_Assignment_3_Data/library_matches.txt')

    F_raw = compute_F_raw(M_library)
    error_raw = compute_avg_reproj_error(M_library, F_raw)
    F_norm = compute_F_norm(M_library)
    error_norm = compute_avg_reproj_error(M_library, F_norm)
    F_mine = compute_F_mine(M_library)
    error_mine = compute_avg_reproj_error(M_library, F_mine)

    print("Average reprojection error (library1.png and library2.png)")
    print("Raw = ", error_raw)
    print("Norm = ", error_norm)
    print("MIne = ", error_mine)

    visualize_epipolar_lines('./CV_Assignment_3_Data/temple1.png', './CV_Assignment_3_Data/temple2.png', './CV_Assignment_3_Data/temple_matches.txt')
    visualize_epipolar_lines('./CV_Assignment_3_Data/house1.jpg', './CV_Assignment_3_Data/house2.jpg', './CV_Assignment_3_Data/house_matches.txt')
    visualize_epipolar_lines('./CV_Assignment_3_Data/library1.jpg', './CV_Assignment_3_Data/library2.jpg', './CV_Assignment_3_Data/library_matches.txt')

main()