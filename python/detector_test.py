# Import required libraries
import io
from matplotlib import pyplot as plt
import numpy as np
import torch
import trimesh
from ultralytics import YOLO
import cv2
from PIL import Image
from depth_pro import create_model_and_transforms, load_rgb
from pycocotools import mask as maskUtils
import open3d as o3d
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
import time

# Camera intrinsic parameters
fx = 1424.3012
fy = 1412.0424
cx = 683.03484
cy = 482.65551
# Camera extrinsic parameters
R = np.array([[-0.999515407099469,0.030755588760158,0.004800492766273],
              [-0.022002558835204,-0.807136810621252,0.589954283266811],
              [0.022019045744228,0.589562772484974,0.807422379504274]])
T = np.array([19.098226514942564,-35.995804820424220,4.571113271196970e+02]).reshape(3, 1)
# Initialize segmentation and depth models
seg_model = YOLO('yolo11n-seg.pt')
device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda:0")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
depth_model, transform = create_model_and_transforms(device=device)

def calibrate_camera(image_paths):
    """Calibrate camera using checker board images"""
    # Checker board dimensions
    CHECKERBOARD = (7, 10)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Prepare object points
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1,2)

    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane

    for image_path in image_paths[:-1]:  # Process all images except the last one
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
        
        if ret:
            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            objpoints.append(objp)
            imgpoints.append(corners2)

    # Calibrate camera
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    # Get extrinsic parameters from last image
    last_img = cv2.imread(image_paths[-1])
    last_gray = cv2.cvtColor(last_img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(last_gray, CHECKERBOARD, None)
    
    if ret:
        corners2 = cv2.cornerSubPix(last_gray, corners, (11,11), (-1,-1), criteria)
        _, rvec, tvec = cv2.solvePnP(objp, corners2, mtx, dist)
        R, _ = cv2.Rodrigues(rvec)
        T = tvec

    return mtx, dist, R, T

def get_extrinsic_params(image_path):
    """Get extrinsic parameters from an image using camera intrinsic parameters"""
    # Checker board dimensions (7x10)
    CHECKERBOARD = (7, 10)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # Prepare object points
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1,2)
    # Create camera matrix from intrinsic parameters
    camera_matrix = np.array([[fx, 0, cx],
                             [0, fy, cy],
                             [0, 0, 1]], dtype=np.float32)
    
    # Process image
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
    
    if ret:
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        # Assuming no distortion
        dist_coeffs = np.zeros((4,1))
        # Get rotation and translation vectors
        _, rvec, tvec = cv2.solvePnP(objp, corners2, camera_matrix, dist_coeffs)
        R, _ = cv2.Rodrigues(rvec)
        T = tvec
        return R, T
    
    return None, None

def visualize_checkerboard_and_camera(extrinsics, checkerboard_size=(7, 10), square_size=0.020):
    """
    Visualize multiple checkerboard planes and camera positions
    Args:
        extrinsics: List of [R, T] pairs for each checkerboard position
        checkerboard_size: Tuple of (rows, cols) for checkerboard
        square_size: Size of each checkerboard square in meters
    """
    # Create coordinate frame for camera
    camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    
    # Create visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    
    # Add camera frame
    vis.add_geometry(camera_frame)
    
    # Create and add checkerboard planes for each position
    for R, T in extrinsics:
        # Create checkerboard points
        width = checkerboard_size[1] * square_size
        height = checkerboard_size[0] * square_size
        
        # Create plane vertices
        vertices = np.array([
            [0, 0, 0],
            [width, 0, 0],
            [width, height, 0],
            [0, height, 0]
        ])
        
        # Create triangles
        triangles = np.array([
            [0, 1, 2],
            [0, 2, 3]
        ])
        
        # Create mesh for checkerboard plane
        plane = o3d.geometry.TriangleMesh()
        plane.vertices = o3d.utility.Vector3dVector(vertices)
        plane.triangles = o3d.utility.Vector3iVector(triangles)
        
        # Transform plane to world coordinates
        plane.rotate(R, center=np.array([0, 0, 0]))
        plane.translate(T.reshape(3,))
        
        # Add random color to distinguish different positions
        color = np.random.rand(3)
        plane.paint_uniform_color(color)
        
        # Add plane to visualizer
        vis.add_geometry(plane)
        
        # Create coordinate frame for each checkerboard position
        board_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.05,
            origin=T.reshape(3,)
        )
        board_frame.rotate(R, center=T.reshape(3,))
        vis.add_geometry(board_frame)
    
    # Set view control
    ctr = vis.get_view_control()
    ctr.set_zoom(0.3)
    
    # Run visualizer
    vis.run()
    vis.destroy_window()

def get_extrinsic_params(image_path, checkerboard_size=(7, 10)):
    """Get extrinsic parameters from checkerboard image"""
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    # Prepare object points
    objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
    objp[:,:2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1,2)
    
    # Process image
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)
    
    if ret:
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        dist_coeffs = np.zeros((4,1))  # Assuming no distortion
        _, rvec, tvec = cv2.solvePnP(objp, corners2, [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1], dist_coeffs)
        R, _ = cv2.Rodrigues(rvec)
        return R, tvec
    
    return None, None

def generate_cylinder_obj(radius: float, height: float):
    """Generate cylinder mesh object"""
    # Create cylinder using trimesh
    cylinder = trimesh.creation.cylinder(radius=radius, height=height, sections=32)
    # Generate memory stream
    obj_stream = io.BytesIO()
    cylinder.export(obj_stream, file_type='obj')
    # Convert BytesIO to string
    obj_stream.seek(0)
    return obj_stream.read().decode('utf-8')

def generate_mask_COCO(image_path: str, visualization=False):
    """Generate COCO format segmentation mask"""
    results = seg_model.predict(source=image_path, show=False)
    result = results[0]
    if result.masks is not None and result.boxes is not None:
        mask = result.masks.xy[0]
        segmentation = [float(coord) for segment in mask.tolist() for coord in segment]
    # Visualization (optional)
        if visualization:
            # Create mask image
            image = cv2.imread(image_path)
            mask_img = np.zeros_like(image[:, :, 0], dtype=np.uint8)
            pts = np.array(mask).reshape((-1, 1, 2)).astype(np.int32)
            cv2.fillPoly(mask_img, [pts], 255)
            
            # Create colored overlay
            overlay = image.copy()
            overlay[mask_img > 0] = [0, 255, 0]  # Green overlay
            
            # Blend with original image
            alpha = 0.5
            result_image = cv2.addWeighted(image, 1-alpha, overlay, alpha, 0)
            
            # Display result
            cv2.imshow("Segmentation Result", result_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    return segmentation

def generate_depthmap(image_path: str):
    """Generate depth map from RGB image"""
    try:
        image, _, f_px = load_rgb(image_path)
    except Exception as e:
        return np.array([])
    prediction = depth_model.infer(transform(image), f_px=f_px)
    depth = prediction["depth"].detach().cpu().numpy().squeeze()
    return depth

def create_point_cloud(depth_map, mask):
    """Create point cloud from depth map and mask"""
    # Get valid points within 90th percentile depth threshold
    depth_threshold = np.percentile(depth_map[depth_map * mask > 0], 95)
    valid_indices = np.where((depth_map * mask > 0) & (depth_map <= depth_threshold))
    depths = depth_map[valid_indices]
    xs, ys = np.meshgrid(np.arange(depth_map.shape[1]), np.arange(depth_map.shape[0]))

    # Convert to camera coordinates
    points_camera = np.vstack((
        (xs[valid_indices] - cx) * depths / fx,
        (ys[valid_indices] - cy) * depths / fy,
        depths
    ))
    
    # Transform to world coordinates
    return R.T @ (points_camera - T)


def visualize_point_cloud(points, labels):
    """Visualize point cloud with clustering labels and camera position/direction"""
    # Create point cloud object
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points.T)
    
    # Assign colors based on labels
    max_label = labels.max()
    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    cloud.colors = o3d.utility.Vector3dVector(colors[:, :3])
    
    # Create camera position and direction vectors
    camera_center = -R.T @ T  # Camera position in world coordinates
    camera_direction = R.T @ np.array([0, 0, 1]).reshape(3, 1)  # Camera z-axis in world coordinates
    
    # Scale factor for visualization
    scale = np.max(points) * 0.05  # 5% of point cloud size
    camera_end = camera_center + camera_direction * scale
    
    # Create line set for camera direction
    camera_lines = o3d.geometry.LineSet()
    camera_lines.points = o3d.utility.Vector3dVector([camera_center.flatten(), camera_end.flatten()])
    camera_lines.lines = o3d.utility.Vector2iVector([[0, 1]])
    camera_lines.colors = o3d.utility.Vector3dVector([[1, 1, 0]])  # Yellow color
    
    # Create coordinate frame for camera position
    camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=scale*0.5, origin=camera_center.flatten())
    
    # Visualize
    o3d.visualization.draw_geometries(
        [cloud, camera_lines, camera_frame],
        window_name="Point Cloud with Camera Position",
        width=800, height=600
    )

def cluster_points(points, visualization=False):
    """Cluster points using DBSCAN and optionally visualize"""
    # Project to 2D space
    xy_dist = np.sqrt(points[0]**2 + points[1]**2)
    points_2d = np.vstack((xy_dist, points[2])).T
    
    # Apply DBSCAN clustering
    points_2d_scaled = StandardScaler().fit_transform(points_2d)
    labels = DBSCAN(eps=0.1, min_samples=40).fit_predict(points_2d_scaled)
    
    if visualization:
        visualize_point_cloud(points, labels)
    
    return labels != -1  # Return valid points mask

def generate_3d_AABB(segmentation, depth, visualization=False):
    """Generate 3D bounding box from segmentation and depth"""
    # Create binary mask
    mask = np.zeros_like(depth, dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(segmentation).reshape(-1, 2).astype(int)], 1)

    # Create point cloud
    points_world = create_point_cloud(depth, mask)

    # Apply downsampling
    sample_size = min(10000, points_world.shape[1])
    sample_indices = np.random.choice(points_world.shape[1], size=sample_size, replace=False)
    sampled_points = points_world[:, sample_indices]

    # Remove noise using clustering
    valid_mask = cluster_points(sampled_points, True)
    valid_points = sampled_points[:, valid_mask].T

    # Calculate bounding box
    valid_cloud = o3d.geometry.PointCloud()
    valid_cloud.points = o3d.utility.Vector3dVector(valid_points)
    aabb = valid_cloud.get_axis_aligned_bounding_box()
    
    # Calculate dimensions
    size = aabb.get_max_bound() - aabb.get_min_bound()
    # Visualization (optional)
    if visualization:  # Toggle visualization
        aabb.color = (0, 1, 0)  # Green bounding box
        valid_cloud.paint_uniform_color([1, 0, 0])  # Red points
        o3d.visualization.draw_geometries(
            [valid_cloud, aabb],
            window_name="Clustering Result without Noise",
            width=800, height=600
        )

    return size

def generate_3d_OBB(segmentation, depth, visualization=False):
    """Generate 3D bounding box from segmentation and depth"""
    # Create binary mask
    mask = np.zeros_like(depth, dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(segmentation).reshape(-1, 2).astype(int)], 1)

    # Create point cloud
    points_world = create_point_cloud(depth, mask)

    # Apply downsampling
    sample_size = min(10000, points_world.shape[1])
    sample_indices = np.random.choice(points_world.shape[1], size=sample_size, replace=False)
    sampled_points = points_world[:, sample_indices]

    # Remove noise using clustering
    valid_mask = cluster_points(sampled_points, True)
    valid_points = sampled_points[:, valid_mask].T

    # Calculate bounding box
    valid_cloud = o3d.geometry.PointCloud()
    valid_cloud.points = o3d.utility.Vector3dVector(valid_points)
    obb = valid_cloud.get_oriented_bounding_box()
    
    # Calculate dimensions
    size = obb.get_max_bound() - obb.get_min_bound()
    # Visualization (optional)
    if visualization:  # Toggle visualization
        obb.color = (0, 1, 0)  # Green bounding box
        valid_cloud.paint_uniform_color([1, 0, 0])  # Red points
        o3d.visualization.draw_geometries(
            [valid_cloud, obb],
            window_name="Clustering Result without Noise",
            width=800, height=600
        )

    return size

def generate_3d_AABB_ani(segmentation, depth, visualization=False):
    """Generate 3D bounding box from segmentation and depth with animated point cloud"""
    # Create binary mask
    mask = np.zeros_like(depth, dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(segmentation).reshape(-1, 2).astype(int)], 1)

    # Create point cloud
    points_world = create_point_cloud(depth, mask)

    # Apply downsampling
    sample_size = min(10000, points_world.shape[1])
    sample_indices = np.random.choice(points_world.shape[1], size=sample_size, replace=False)
    sampled_points = points_world[:, sample_indices]

    if visualization:
        # 각 포인트와 카메라 사이의 거리 계산
        depths = sampled_points[2, :]  # z축 값이 depth를 반영
        sorted_indices = np.argsort(depths)
        sorted_points = sampled_points[:, sorted_indices]

        # 시각화 준비
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Growing Point Cloud", width=800, height=600)
        
        # 빈 포인트 클라우드 생성
        pcd = o3d.geometry.PointCloud()
        vis.add_geometry(pcd)

        # 포인트를 점진적으로 추가
        points_per_frame = max(len(sorted_points[0]) // 100, 1)  # 100프레임으로 나누기
        
        for i in range(0, len(sorted_points[0]), points_per_frame):
            # 현재까지의 포인트 업데이트
            current_points = sorted_points[:, :i+points_per_frame].T
            pcd.points = o3d.utility.Vector3dVector(current_points)
            pcd.paint_uniform_color([1, 0, 0])
            
            # 시각화 업데이트
            vis.update_geometry(pcd)
            vis.poll_events()
            vis.update_renderer()
            
            # 약간의 딜레이 추가
            time.sleep(0.01)

        # 최종 바운딩 박스 추가
        valid_mask = cluster_points(sampled_points)
        valid_points = sampled_points[:, valid_mask].T
        aabb = o3d.geometry.AxisAlignedBoundingBox.create_from_points(
            o3d.utility.Vector3dVector(valid_points)
        )
        aabb.color = (0, 1, 0)
        vis.add_geometry(aabb)
        vis.poll_events()
        vis.update_renderer()
        
        # 잠시 대기 후 종료
        time.sleep(2)
        vis.destroy_window()

    # Remove noise using clustering
    valid_mask = cluster_points(sampled_points)
    valid_points = sampled_points[:, valid_mask].T

    # Calculate bounding box
    valid_cloud = o3d.geometry.PointCloud()
    valid_cloud.points = o3d.utility.Vector3dVector(valid_points)
    aabb = valid_cloud.get_axis_aligned_bounding_box()
    
    # Calculate dimensions
    size = aabb.get_max_bound() - aabb.get_min_bound()

    return size

    

if(__name__ == '__main__'):
    rgb_path = "assets/logi_example0.jpg"
    segmentation = generate_mask_COCO(rgb_path, True)
    depth = generate_depthmap(rgb_path)
    #size = generate_3d_AABB(segmentation, depth, True)
    generate_3d_AABB_ani(segmentation, depth, True)


