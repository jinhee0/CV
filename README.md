## Assignment1
**✅ Part 1. Image Filtering**

**Cross-Correlation**
- 1D/2D 커널을 이용한 이미지 필터링 함수 구현
- 1D 함수는 주어진 kernel에 따라 수직/수평 구별
- kernel은 모두 odd sized
- 필터 크기와 이미지 크기는 동일하게 유지(입력 이미지의 가장자리 픽셀 값 사용하여 패딩)
- built-in 함수 사용 금지
  
**Gaussian filter**
- 필터 크기와 시그마 조합 총 9개를 이용해 이미지 필터링 수행 및 시각화
- 차이 맵 시각화, 픽셀 간 절댓값 합, 연산 시간 비교 출력

**✅ Part 2. Edge Detection**
- Gaussian filter 적용
- Sobel filter(edge 검출을 위해) 적용 -> x, y 방향의 gradient와 magnitude 계산
- Non-Maximum Suppression (NMS) 함수 구현(non_maximum_suppression_dir)
  - 8개 방향 중 가까운 각도로 Direction Quantization
  - 주변 픽셀과 비교하여 suppress

**✅ Part 3. Corner Detection**
- Gaussian filter 적용
- Sobel filter 적용
- Second Moment Matrix M
- Corner Response R 계산 (k=0.04), 정규화
- Non-Maximum Suppression (NMS) 함수(non_maximum_suppression_win)
  - winSize(=11) 내의 maximum 값으로 suppress
  - 0.1 이하면 무시

  
## Assignment2
**✅ Part 1. 2D Transformations**
-  Interactive 2D transformations

**✅ Part 2. Homography**
- ORB 측징 추출, Hamming 거리 기반의 ORB descriptor 유사도 측정(built-in matcher 사용 금지)
- Normalization
- RANSAC을 통한 Homography 계산
- Image wraping
- Image stitching
  - 경계 부분 그라데이션 기반 블렌딩


## Assignment3
**✅ Part 1.  Fundamental Matrix**
- Eight-Point 알고리즘, Normalization
- Fundamental Matrix
- reprojection errors
- Visualization of epipolar lines 

  
## Assignment4
**✅ Part 1.  Image Retrieval**
- Dataset: a subset of UKBench 
- computing image descriptors 

  
