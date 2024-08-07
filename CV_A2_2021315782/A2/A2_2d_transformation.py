import cv2
import numpy as np

def get_transformed_image(img, M):
    plane_size = 801
    origin = (400, 400)
    plane = np.ones((plane_size, plane_size, 3), dtype=np.uint8) * 255

    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            homogeneous_coord = np.array([x, y, 1])
            new_coord = M @ homogeneous_coord
            new_x = int(new_coord[0] / new_coord[2])+origin[1]-img.shape[1]//2
            new_y = int(new_coord[1] / new_coord[2])+origin[0]-img.shape[0]//2
            if 0 <= new_y < plane_size and 0 <= new_x < plane_size:
                plane[new_y, new_x] = img[y, x]
                
    cv2.arrowedLine(plane, (0, origin[1]), (plane_size - 1, origin[1]), (0, 0, 0), 1)
    cv2.arrowedLine(plane, (origin[0], plane_size - 1), (origin[0], 0), (0, 0, 0), 1)
    return plane

def main():
    img = cv2.imread('images/smile.png', cv2.IMREAD_GRAYSCALE)
    origin =(400, 400)
    M = np.array([[1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]], dtype=np.float32) 
    
    while True:
        transformed_plane = get_transformed_image(img, M)
        
        cv2.imshow('Transformed Image Plane', transformed_plane)
        key = cv2.waitKey(0) & 0xFF
        
        if key == ord('a'):
            M[0, 2] -= 5  # Move left by 5 pixels
        elif key == ord('d'):
            M[0, 2] += 5  # Move right by 5 pixels
        elif key == ord('w'):
            M[1, 2] -= 5  # Move upward by 5 pixels
        elif key == ord('s'):
            M[1, 2] += 5  # Move downward by 5 pixels

        elif key == ord('r'):
            theta = np.radians(-5)  # Convert angle to radians (5 degrees)
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)
            rotation_matrix = np.array([[cos_theta, -sin_theta, 0],
                                        [sin_theta, cos_theta, 0],
                                        [0, 0, 1]])  # Rotation matrix for 5 degrees
            
            # Translate image to origin, apply rotation, then translate back
            translate_to_origin = np.array([[1, 0, -img.shape[1]//2],
                                            [0, 1, -img.shape[0]//2],
                                            [0, 0, 1]])
            
            translate_back = np.array([[1, 0, img.shape[1]//2],
                                       [0, 1, img.shape[0]//2],
                                       [0, 0, 1]])
            
            M = translate_back @ rotation_matrix @ translate_to_origin @ M

        elif key == ord('t'):
            theta = np.radians(5)  # Convert angle to radians (5 degrees)
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)
            rotation_matrix = np.array([[cos_theta, -sin_theta, 0],
                                        [sin_theta, cos_theta, 0],
                                        [0, 0, 1]])  # Rotation matrix for 5 degrees
            
            # Translate image to origin, apply rotation, then translate back
            translate_to_origin = np.array([[1, 0, -img.shape[1]//2],
                                            [0, 1, -img.shape[0]//2],
                                            [0, 0, 1]])
            translate_back = np.array([[1, 0, img.shape[1]//2],
                                       [0, 1, img.shape[0]//2],
                                       [0, 0, 1]])
            
            M = translate_back @ rotation_matrix @ translate_to_origin @ M

        # flip 구현
        elif key == ord('f'):
            M[0,2] *= -1
            M[0,2] += img.shape[1] 
            M[0,0] *= -1
            
        elif key == ord('g'):
            M[1,2] *= -1
            M[1,2] += img.shape[0] 
            M[1,1] *= -1

        elif key == ord('x'):
            M[0, 0] *= 0.95  # Shrink size by 5% along x-direction
        elif key == ord('c'):
            M[0, 0] *= 1.05  # Enlarge size by 5% along x-direction
        elif key == ord('y'):
            M[1, 1] *= 0.95 # Shrink size by 5% along y-direction
        elif key == ord('u'):
            M[1, 1] *= 1.05  # Enlarge size by 5% along y-direction

        elif key == ord('h'):
            M = np.array([[1, 0, 0],
                          [0, 1, 0],
                          [0, 0, 1]], dtype=np.float32)  
        elif key == ord('q'):
            break  # Quit the program
        
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
