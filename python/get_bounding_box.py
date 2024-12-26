import numpy as np
import cv2
import json
from matplotlib import pyplot as plt
from pycocotools import mask as maskUtils
import open3d as o3d
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


# 카메라 내재 파라미터 (Intrinsic Parameters)
intrinsic_params = {
    "fx": 3098.0475,
    "fy": 3097.4233,
    "cx": 2031.6249,
    "cy": 1524.7746
}

# 카메라 외재 파라미터 (Extrinsic Parameters)
extrinsic_params = {
    "R": np.array([
        [0.965286851297566,	0.248307782214057,	-0.0810218489295801],
        [-0.157358887774226,	0.800448714153273,	0.578377071164504],
        [0.208469362618323,	-0.545550273854123,	0.811736055345087]
    ]),
    "T": np.array([-101.197833337719,	-31.2983198623872,	356.795325964250])
}

# 1. 세그멘테이션 데이터 로드
with open('output/example4_coco.json', 'r') as f:
    coco_data = json.load(f)

# 2. 깊이 데이터 로드
depth_data = np.load('assets/example4.npz')
depth = depth_data['depth']

# 3. 원본 이미지 로드 (시각화를 위해)
image = cv2.imread('assets/example4.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 4. 마스크 초기화
height, width = depth.shape
mask = np.zeros((height, width), dtype=np.uint8)

# 5. 컵에 해당하는 세그멘테이션 마스크 생성
cup_annotation = coco_data['annotations'][0]

# 세그멘테이션 처리
segmentation = cup_annotation['segmentation']
if isinstance(segmentation, list):
    # 세그멘테이션이 [x1, y1, x2, y2, ..., xn, yn] 형식의 리스트라고 가정합니다.
    if isinstance(segmentation[0], list):
        # 여러 개의 폴리곤이 있는 경우
        for seg in segmentation:
            poly = np.array(seg).reshape((-1, 2))
            cv2.fillPoly(mask, [np.int32(poly)], color=1)
    else:
        # 단일 폴리곤인 경우
        poly = np.array(segmentation).reshape((-1, 2))
        cv2.fillPoly(mask, [np.int32(poly)], color=1)
elif isinstance(segmentation, dict):
    # RLE 형식의 세그멘테이션 처리
    rle = maskUtils.frPyObjects(segmentation, height, width)
    binary_mask = maskUtils.decode(rle)
    mask = np.maximum(mask, binary_mask)
else:
    print("알 수 없는 세그멘테이션 형식입니다.")
    exit()

# 6. 컵의 픽셀 좌표 및 깊이 값 추출
ys, xs = np.where(mask == 1)
cup_depth = depth[ys, xs]

# 유효하지 않은 깊이 값 제거
valid_indices = np.where(cup_depth > 0)
cup_depth = cup_depth[valid_indices]
xs = xs[valid_indices]
ys = ys[valid_indices]

if len(xs) == 0:
    print("유효한 깊이 값을 가진 컵의 픽셀이 없습니다.")
    exit()

# 7. 카메라 좌표계로 변환
fx = intrinsic_params['fx']
fy = intrinsic_params['fy']
cx = intrinsic_params['cx']
cy = intrinsic_params['cy']

X_c = (xs - cx) * cup_depth / fx
Y_c = (ys - cy) * cup_depth / fy
Z_c = cup_depth

points_camera = np.vstack((X_c, Y_c, Z_c))

# 8. 세계 좌표계로 변환
R = extrinsic_params['R']
T = extrinsic_params['T'].reshape(3, 1)

# 9. 깊이 값 단위
depth_dimension = 1

points_world = R.T @ (points_camera - T)

# 다운샘플링
sample_size = min(10000, points_world.shape[1])
sample_indices = np.random.choice(points_world.shape[1], size=sample_size, replace=False)
sampled_points = points_world[:, sample_indices].T  # Nx3 형식



# Open3D PointCloud 생성
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(sampled_points)

# AABB 계산
aabb = point_cloud.get_axis_aligned_bounding_box()
aabb.color = (0, 1, 0)  # 초록색

# 지면 방향 벡터 계산
ground_vector_camera = np.array([0, 0, -1])

# 포인트 클라우드 중심 계산
center = point_cloud.get_center()

# 벡터 스케일 조정 (포인트 클라우드의 크기와 비슷하게)
scale_factor = max(width, depth_dimension, height) / 2
scaled_ground_vector = center + ground_vector_camera * scale_factor

# 지면 방향 벡터 시각화
ground_line_set = o3d.geometry.LineSet(
    points=o3d.utility.Vector3dVector([center, scaled_ground_vector]),
    lines=o3d.utility.Vector2iVector([[0, 1]])
)
ground_line_set.colors = o3d.utility.Vector3dVector([[0, 0, 1]])  # 파란색

# Open3D로 시각화
o3d.visualization.draw_geometries(
    [point_cloud, aabb, ground_line_set],
    window_name="컵의 포인트 클라우드와 OBB, AABB, 지면 방향 벡터",
    width=800, height=600, left=50, top=50
)

# DBSCAN 클러스터링을 사용하여 컵의 상단과 하단을 분리
# 2D 데이터 준비 (xy_distance, z값)
xy_distances_sampled = np.sqrt(points_world[0, sample_indices]**2 + points_world[1, sample_indices]**2)
z_values_sampled = points_world[2, sample_indices]  # 실제 z값 사용
points_2d = np.vstack((xy_distances_sampled, z_values_sampled)).T  # (N, 2) 형태

# 데이터 스케일링 (DBSCAN은 거리 기반이므로 중요)
scaler = StandardScaler()
points_2d_scaled = scaler.fit_transform(points_2d)

# DBSCAN 클러스터링
# eps: 이웃 반경, min_samples: 핵심 포인트가 되기 위한 최소 이웃 수
dbscan = DBSCAN(eps=0.1, min_samples=40) # depth의 밀도에 맞게 적절히 조절해야함.
labels_dbscan = dbscan.fit_predict(points_2d_scaled)

# 클러스터 수 확인
n_clusters = len(set(labels_dbscan)) - (1 if -1 in labels_dbscan else 0)
n_noise = list(labels_dbscan).count(-1)

# 클러스터링 결과를 3D 포인트와 매핑
colors_dbscan = np.zeros((len(labels_dbscan), 3))
# 노이즈 포인트는 검정색
colors_dbscan[labels_dbscan == -1] = [0, 0, 0]
# 각 클러스터에 다른 색상 할당
unique_labels = set(labels_dbscan) - {-1}
colors_list = [[1, 0, 0], [0, 0, 1], [0, 1, 0], [1, 1, 0]]  # 빨강, 파랑, 초록, 노랑
for i, label in enumerate(unique_labels):
    colors_dbscan[labels_dbscan == label] = colors_list[i % len(colors_list)]

# 클러스터링 결과를 포인트 클라우드에 적용
clustered_cloud_dbscan = o3d.geometry.PointCloud()
clustered_cloud_dbscan.points = o3d.utility.Vector3dVector(points_world[:, sample_indices].T)
clustered_cloud_dbscan.colors = o3d.utility.Vector3dVector(colors_dbscan)

# DBSCAN 결과 시각화
o3d.visualization.draw_geometries(
    [clustered_cloud_dbscan],
    window_name="DBSCAN 클러스터링 결과",
    width=800, height=600, left=50, top=50
)

# 클러스터링 결과 정보 출력
print("\nDBSCAN 클러스터링 결과:")
print(f"클러스터 개수: {n_clusters}")

# 노이즈가 아닌 포인트들의 마스크 생성
valid_points_mask = labels_dbscan != -1

# 유효한 포인트들만 추출
valid_points = points_world[:, sample_indices][:, valid_points_mask].T
valid_colors = colors_dbscan[valid_points_mask]

# 새로운 포인트 클라우드 생성
valid_cloud = o3d.geometry.PointCloud()
valid_cloud.points = o3d.utility.Vector3dVector(valid_points)
valid_cloud.colors = o3d.utility.Vector3dVector(valid_colors)

# 바운딩 박스 계산
aabb = valid_cloud.get_axis_aligned_bounding_box()
aabb.color = (0, 1, 0)  # 초록색

# 바운딩 박스 크기 계산
min_bound = aabb.get_min_bound()
max_bound = aabb.get_max_bound()
size = max_bound - min_bound

# 시각화
o3d.visualization.draw_geometries(
    [valid_cloud, aabb],
    window_name="노이즈 제외한 클러스터링 결과",
    width=800, height=600, left=50, top=50
)

# 크기 정보 출력
print("\n노이즈 제외한 바운딩 박스 크기 (단위: mm):")
print(f"가로 (X): {size[0]:.2f}")
print(f"세로 (Y): {size[1]:.2f}")
print(f"높이 (Z): {size[2]:.2f}")
print(f"유효 포인트 개수: {len(valid_points)}")
