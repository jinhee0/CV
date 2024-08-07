import cv2
import numpy as np
import time

def hamming_distance(a, b):
    return np.count_nonzero(a != b)

def normalize_points(points):
    centroid = np.mean(points, axis=0)
    translated_points = points - centroid

    max_dist = np.max(np.linalg.norm(translated_points, axis=1))
    scale_factor = np.sqrt(2) / max_dist

    T = np.array([
        [scale_factor, 0, -scale_factor * centroid[0]],
        [0, scale_factor, -scale_factor * centroid[1]],
        [0, 0, 1]
    ])

    points_homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))
    points_normalized = (T @ points_homogeneous.T).T

    return points_normalized[:, :2], T

def compute_homography(srcP, destP):
    srcP_norm, T_src = normalize_points(srcP)  # x~s, Ts
    destP_norm, T_dest = normalize_points(destP)  # x~d, Td
    A = []
    for i in range(len(srcP_norm)):
        x, y = srcP_norm[i]
        u, v = destP_norm[i]
        A.append([-x, -y, -1, 0, 0, 0, x*u, y*u, u])
        A.append([0, 0, 0, -x, -y, -1, x*v, y*v, v])

    A = np.asarray(A)
    
    _, _, Vt = np.linalg.svd(A)

    H_norm = Vt[-1].reshape(3, 3)

    # Denormalize the homography matrix: H = T_D^-1 * H_norm * T_S
    H = np.linalg.inv(T_dest) @ H_norm @ T_src

    return H

def compute_homography_ransac(srcP, destP, th, num_iterations=3000):
    max_inliers = 0
    best_H = None
    start_time = time.time()
    for i in range(num_iterations):
        rand_indices = np.random.choice(len(srcP), 4, replace=False) 
        src_subset = srcP[rand_indices]
        dest_subset = destP[rand_indices]
        
        H = compute_homography(src_subset, dest_subset)
        inliers = 0
        for i in range(len(srcP)):
            src_point = np.array([srcP[i][0], srcP[i][1], 1]).reshape(3, 1)
            estimated_dest_point = np.dot(H, src_point)
            if estimated_dest_point[2] != 0:
                estimated_dest_point /= estimated_dest_point[2]
            else:
                continue  

            dest_point = np.array([destP[i][0], destP[i][1], 1]).reshape(3, 1)
            error = np.linalg.norm(dest_point - estimated_dest_point)
            if error < th:
                inliers += 1

        if inliers > max_inliers:
            max_inliers = inliers
            best_H = H

        if time.time() - start_time > 3:
            print("3 seconds")
            break

    return best_H

def overlay_images(base_img, overlay_img):
    overlay_resized = cv2.resize(overlay_img, (base_img.shape[1], base_img.shape[0]))
    
    if len(overlay_resized.shape) == 2:
        overlay_resized = cv2.cvtColor(overlay_resized, cv2.COLOR_GRAY2BGR)

    gray_overlay = cv2.cvtColor(overlay_resized, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_overlay, 1, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    img1_bg = cv2.bitwise_and(base_img, base_img, mask=mask_inv)
    img2_fg = cv2.bitwise_and(overlay_resized, overlay_resized, mask=mask)

    result = cv2.add(img1_bg, img2_fg)

    return result

def matches_points(img1, img2):
    orb = cv2.ORB_create()
    kp1 = orb.detect(img1, None)
    kp2 = orb.detect(img2, None)
    kp1, des1 = orb.compute(img1, kp1)
    kp2, des2 = orb.compute(img2, kp2)

    matches = []
    for i in range(len(des1)):
        distances = []
        for j in range(len(des2)):
            dist = hamming_distance(des1[i], des2[j])
            distances.append((dist, j))
        
        distances = sorted(distances, key=lambda x: x[0])
        min_dist, min_j = distances[0]
        matches.append(cv2.DMatch(_queryIdx=i, _trainIdx=min_j, _imgIdx=0, _distance=min_dist))

    matches = sorted(matches, key=lambda x: x.distance)
    
    return kp1, kp2, matches

def remove_black_background(image):
    min_x = image.shape[1]  # 초기값으로 이미지의 너비로 설정
    min_y = None
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            if image[y, x] == 0 and x < min_x:  # 검은색 픽셀을 발견하고, 현재 x값이 가장 작을 때
                min_x = x

    cropped_image = image[0:image.shape[0], 0:min_x ]

    return cropped_image

def blend_images(image1, image2, blend_width):
    height, width1 = image1.shape
    _, width2 = image2.shape

    result = np.zeros((height, width1 + width2 - blend_width), dtype=image1.dtype)
    result[:, :width1] = image1
    result[:, -width2:] = image2

    # 그라디언트 생성
    gradient = np.linspace(0, 1, blend_width)

    # 이미지를 블렌딩하여 결과에 추가
    for i, alpha in enumerate(gradient):
        result[:, width1 - blend_width + i] = (1 - alpha) * image1[:, -blend_width + i] + alpha * image2[:, i]

    return result

def main():
    img_desk = cv2.imread('images/cv_desk.png', cv2.IMREAD_GRAYSCALE)
    img_cover = cv2.imread('images/cv_cover.jpg', cv2.IMREAD_GRAYSCALE)

    kp1, kp2, matches = matches_points(img_desk, img_cover)

    img_matches = cv2.drawMatches(img_desk, kp1, img_cover, kp2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow("Top 10 Matches", img_matches)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    good_matches = matches[:40]
    srcP = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)
    destP = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
    
    H = compute_homography(srcP, destP)
    print("Homography Matrix with Normalization:")
    print(H)
    h, w = img_desk.shape
    cover_warped = cv2.warpPerspective(img_cover, H, (w, h))
    cv2.imshow("Warped Image with Normalization", cover_warped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    H_ransac = compute_homography_ransac(srcP, destP, th=5)
    print("Homography Matrix with RANSAC:")
    print(H_ransac)
    cover_warped_ransac = cv2.warpPerspective(img_cover, H_ransac, (w, h))
    cv2.imshow("Warped Image with RANSAC", cover_warped_ransac)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    overlaid_image_normalization = overlay_images(cv2.cvtColor(img_desk, cv2.COLOR_GRAY2BGR), cover_warped)
    overlaid_image_ransac = overlay_images(cv2.cvtColor(img_desk, cv2.COLOR_GRAY2BGR), cover_warped_ransac)
    overlaid = np.hstack((overlaid_image_normalization, overlaid_image_ransac))
    cv2.imshow("Comparison of Overlaid Images with Normalization and RANSAC", overlaid)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    img_hp_cover = cv2.imread('images/hp_cover.jpg', cv2.IMREAD_GRAYSCALE)
    img_hp_cover = cv2.resize(img_hp_cover, (img_cover.shape[1], img_cover.shape[0]))
    cover_warped_ransac = cv2.warpPerspective(img_hp_cover, H_ransac, (w, h))
    cv2.imshow("Warped Image with RANSAC", cover_warped_ransac)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    overlaid_image_ransac = overlay_images(cv2.cvtColor(img_desk, cv2.COLOR_GRAY2BGR), cover_warped_ransac)
    cv2.imshow("Comparison of Overlaid Images with Normalization and RANSAC", overlaid_image_ransac)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # image stitching
    img1 = cv2.imread('images/diamondhead-10.png', cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread('images/diamondhead-11.png', cv2.IMREAD_GRAYSCALE)
    kp1, kp2, matches = matches_points(img1, img2)
    good_matches = matches[:50]
    src_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)
    dst_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)

    H_ransac = compute_homography_ransac(src_pts, dst_pts, th=5)
    warped_img2 = cv2.warpPerspective(img2, H_ransac, (img2.shape[1]*2, img2.shape[0]))

    result = np.zeros((img1.shape[0], img1.shape[1] + img2.shape[1]), dtype=np.uint8)
    result[:, :] = warped_img2
    result[:, :img1.shape[1]] = img1
    result = remove_black_background(result)
    cv2.imshow('Stitched Image', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    result2 = np.zeros((img1.shape[0], img1.shape[1] + img2.shape[1]), dtype=np.uint8)
    result2[:, :] = warped_img2
    result2 = result2[:, img1.shape[1]:]
    blend_result = blend_images(img1, result2,  100)
    blend_result = remove_black_background(blend_result)
    cv2.imshow('blend Image', blend_result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
