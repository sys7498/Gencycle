# Import required libraries
from rich.console import Console
from rich.logging import RichHandler
import logging
import io
from matplotlib import pyplot as plt
import numpy as np
import torch
import trimesh
from ultralytics import YOLO
import cv2
from PIL import Image, ImageDraw
from depth_pro import create_model_and_transforms, load_rgb
from pycocotools import mask as maskUtils
import open3d as o3d
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler

class Detector:
    def __init__(self, model_path):
        self.console = Console()
        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",
            handlers=[RichHandler(console=self.console, rich_tracebacks=True)]
        )
        self.logger = logging.getLogger("detector")

        # Camera intrinsic parameters
        self.fx = 1424.3012
        self.fy = 1412.0424
        self.cx = 683.03484
        self.cy = 482.65551
        # Camera extrinsic parameters
        self.R = np.array([[-0.999515407099469,0.030755588760158,0.004800492766273],
                      [-0.022002558835204,-0.807136810621252,0.589954283266811],
                      [0.022019045744228,0.589562772484974,0.807422379504274]])
        self.T = np.array([19.098226514942564,-35.995804820424220,4.571113271196970e+02]).reshape(3, 1)

        # detection model
        self.seg_model = YOLO(model_path)

        # depth estimation model
        device = torch.device("cpu")
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        self.depth_model, self.transform = create_model_and_transforms(device=device, precision=torch.half)
        self.depth_model.eval()

        self.logger.info("Models loaded successfully")
    
    # generate cylinder mesh obj
    def generate_cylinder_obj(self, radius: float, height: float):
        # generate cylinder by trimesh
        cylinder = trimesh.creation.cylinder(radius=radius, height=height, sections=32)

        # generate memory stream
        obj_stream = io.BytesIO()
        cylinder.export(obj_stream, file_type='obj')

        # convert BytesIO to string
        obj_stream.seek(0)
        return obj_stream.read().decode('utf-8')

    def generate_mask_COCO(self, image_path: str):
        """Generate COCO format segmentation mask"""
        results = self.seg_model.predict(source=image_path, show=False)
        result = results[0]
        if result.masks is not None and result.boxes is not None:
            mask = result.masks.xy[0]
            segmentation = [float(coord) for segment in mask.tolist() for coord in segment]
            return segmentation
        return []

    def generate_depthmap(self, image_path: str):
        """Generate depth map from RGB image"""
        self.logger.info("Depth prediction starts")
        try:
            image, _, f_px = load_rgb(image_path)
        except Exception as e:
            return np.array([])
        prediction = self.depth_model.infer(self.transform(image), f_px=f_px)
        depth = prediction["depth"].detach().cpu().numpy().squeeze()
        focallength_px = prediction["focallength_px"].detach().cpu().item()
        self.logger.info("Depth prediction succeded")
        return depth
        

    def create_point_cloud(self, depth_map, mask):
        """Create point cloud from depth map and mask"""
        # Get valid points within 90th percentile depth threshold
        depth_threshold = np.percentile(depth_map[depth_map * mask > 0], 95)
        valid_indices = np.where((depth_map * mask > 0) & (depth_map <= depth_threshold))
        depths = depth_map[valid_indices]
        xs, ys = np.meshgrid(np.arange(depth_map.shape[1]), np.arange(depth_map.shape[0]))

        # Convert to camera coordinates
        points_camera = np.vstack((
            (xs[valid_indices] - self.cx) * depths / self.fx,
            (ys[valid_indices] - self.cy) * depths / self.fy,
            depths
        ))

        # Transform to world coordinates
        return self.R.T @ (points_camera - self.T)

    def cluster_points(self, points):
        """Cluster points using DBSCAN and optionally visualize"""
        # Project to 2D space
        xy_dist = np.sqrt(points[0]**2 + points[1]**2)
        points_2d = np.vstack((xy_dist, points[2])).T

        # Apply DBSCAN clustering
        points_2d_scaled = StandardScaler().fit_transform(points_2d)
        labels = DBSCAN(eps=0.1, min_samples=40).fit_predict(points_2d_scaled)

        return labels != -1  # Return valid points mask

    def generate_3d_AABB(self, segmentation, depth):
        """Generate 3D bounding box from segmentation and depth"""
        self.logger.info("Generating AABB starts")
        # Create binary mask
        mask = np.zeros_like(depth, dtype=np.uint8)
        cv2.fillPoly(mask, [np.array(segmentation).reshape(-1, 2).astype(int)], 1)
        self.logger.info("Mask created")

        
        # Create point cloud
        points_world = self.create_point_cloud(depth, mask)
        self.logger.info("Point cloud created")

        # Apply downsampling
        sample_size = min(10000, points_world.shape[1])
        sample_indices = np.random.choice(points_world.shape[1], size=sample_size, replace=False)
        sampled_points = points_world[:, sample_indices]
        self.logger.info("Downsampling completed")

        # Remove noise using clustering without visualization
        valid_mask = self.cluster_points(sampled_points)
        valid_points = sampled_points[:, valid_mask].T
        self.logger.info("Noise removal completed")

        # Calculate bounding box
        valid_cloud = o3d.geometry.PointCloud()
        valid_cloud.points = o3d.utility.Vector3dVector(valid_points)
        aabb = valid_cloud.get_axis_aligned_bounding_box()

        # Calculate dimensions
        size = aabb.get_max_bound() - aabb.get_min_bound()
        self.logger.info("AABB calculation completed")

        return size

    def get_size(self, image_path):
        segmentation = self.generate_mask_COCO(image_path)
        depth = self.generate_depthmap(image_path)
        size = self.generate_3d_AABB(segmentation, depth)
        return size