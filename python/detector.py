# Import required libraries
from rich.console import Console
from rich.logging import RichHandler
import logging
import numpy as np
import torch
from ultralytics import YOLO
import cv2
from PIL import Image, ImageDraw
from depth_pro import create_model_and_transforms, load_rgb
from pycocotools import mask as maskUtils
import open3d as o3d
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm
import open3d as o3d

from PIL import Image
import torch
from torchvision import transforms
from transformers import AutoModelForImageSegmentation

from tqdm import tqdm

from sf3d.system import SF3D
from sf3d.utils import get_device, remove_background, resize_foreground
import argparse
import os
from contextlib import nullcontext

import rembg

from point_e.diffusion.configs import DIFFUSION_CONFIGS, diffusion_from_config
from point_e.diffusion.sampler import PointCloudSampler
from point_e.models.download import load_checkpoint
from point_e.models.configs import MODEL_CONFIGS, model_from_config
from point_e.util.plotting import plot_point_cloud

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
        self.R = np.array([[-0.999094531249736,	0.00190532433730880,	0.0425027924499029],
[0.0302058905613601,	-0.671762429458138,	0.740150553971214],
[0.0299620059786557,	0.740764205471397,	0.671096617552246]])
        self.T = np.array([91.2886037630424,98.4766229752523,426.684007612272]).reshape(3, 1)
        
        self.device = torch.device("cpu")
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        self.logger.info("using device: %s", self.device)
        
        # YOLO
        self.seg_model = YOLO(model_path)

        # Depth-pro
        self.depth_model, self.transform = create_model_and_transforms(device=self.device, precision=torch.half)
        self.depth_model.eval()

        # Point e
        self.logger.info("creating point model...")
        base_name = 'base40M-textvec'
        base_model = model_from_config(MODEL_CONFIGS[base_name], self.device)
        base_model.eval()
        base_diffusion = diffusion_from_config(DIFFUSION_CONFIGS[base_name])
        self.logger.info('creating upsample model...')
        upsampler_model = model_from_config(MODEL_CONFIGS['upsample'], self.device)
        upsampler_model.eval()
        upsampler_diffusion = diffusion_from_config(DIFFUSION_CONFIGS['upsample'])
        self.logger.info('downloading base checkpoint...')
        base_model.load_state_dict(load_checkpoint(base_name, self.device))
        self.logger.info('downloading upsampler checkpoint...')
        upsampler_model.load_state_dict(load_checkpoint('upsample', self.device))

        self.sampler = PointCloudSampler(
            device=self.device,
            models=[base_model, upsampler_model],
            diffusions=[base_diffusion, upsampler_diffusion],
            num_points=[1024, 4096 - 1024],
            aux_channels=['R', 'G', 'B'],
            guidance_scale=[3.0, 0.0],
            model_kwargs_key_filter=('texts', ''), # Do not condition the upsampler at all
        )

        # Background Removal
        self.logger.info("creating background removal model...")
        self.bgremoval_model = AutoModelForImageSegmentation.from_pretrained('briaai/RMBG-2.0', trust_remote_code=True)
        torch.set_float32_matmul_precision(['high', 'highest'][0])
        self.bgremoval_model.to('cuda')
        self.bgremoval_model.eval()

        self.logger.info("Models loaded successfully")
    
    def generate_mask_COCO(self, image_path: str):
        """Generate COCO format segmentation mask"""
        results = self.seg_model.predict(source=image_path, show=False)
        result = results[0]
        if result.masks is not None and result.boxes is not None:
            mask = result.masks.xy[0]
            segmentation = [float(coord) for segment in mask.tolist() for coord in segment]
            return segmentation
        return []
    
    def generate_object_mask_by_bgremoval(self, image_path: str):
        """Remove Background from RGB image"""
        self.logger.info("Background Removal starts")
        # Data settings
        image_size = (1024, 1024)
        transform_image = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        image = Image.open(image_path)
        input_images = transform_image(image).unsqueeze(0).to('cuda')

        # Prediction
        with torch.no_grad():
            preds = self.bgremoval_model(input_images)[-1].sigmoid().cpu()
        pred = preds[0].squeeze()
        pred_pil = transforms.ToPILImage()(pred)
        mask = pred_pil.resize(image.size)
        # 알파 채널 분리
        alpha = mask.split()[-1]  # 알파 채널만 가져오기
        alpha_array = np.array(alpha)  # NumPy 배열로 변환

        # 이진화 (알파 값이 0보다 크면 255로 설정)
        binary_mask = (alpha_array > 0).astype(np.uint8) * 255

        # OpenCV로 외곽선 찾기
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        polygons = []
        # 폴리곤 좌표 추출
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)  # 가장 큰 외곽선 선택
            polygons = largest_contour[:, 0, :].flatten().tolist()  # 1차원 배열로 변환
        else:
            polygons = []  # 투명하지 않은 부분이 없으면 빈 리스트
        self.logger.info("Background Removal succeded")
        return polygons

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

        #self.logger.info("Clipping points along y-axis")
    
        #y_min, y_max = 0, size[2]
        #self.logger.info(""+str(size[0])+","+str(size[1])+","+str(size[2]))
        #epsilon = 0.0001
        ## Clip points near the base (y_min ± epsilon)
        #base_indices = (valid_points[1, :] >= y_min) & (valid_points[1, :] <= y_min + epsilon)
        #base_points = valid_points[:, base_indices]
        #base_x_min, base_x_max = base_points[0, :].min(), base_points[0, :].max()
        #base_z_min, base_z_max = base_points[2, :].min(), base_points[2, :].max()

        ## Clip points near the top (y_max ± epsilon)
        #top_indices = (valid_points[1, :] >= y_max - epsilon) & (valid_points[1, :] <= y_max)
        #top_points = valid_points[:, top_indices]
        #top_x_min, top_x_max = top_points[0, :].min(), top_points[0, :].max()
        #top_z_min, top_z_max = top_points[2, :].min(), top_points[2, :].max()

        bbsize = {
            'size': size
        }
        return bbsize

    def get_size(self, image_path):
        segmentation = self.generate_object_mask_by_bgremoval(image_path)
        #segmentation = self.generate_mask_COCO(image_path)
        depth = self.generate_depthmap(image_path)
        size = self.generate_3d_AABB(segmentation, depth)
        return size
    
    def generate_pc(self, prompt):
        # Produce a sample from the model.
        samples = None
        self.logger.info("Generating point cloud")
        for x in tqdm(self.sampler.sample_batch_progressive(batch_size=1, model_kwargs=dict(texts=[prompt]))):
            samples = x
        pc = self.sampler.output_to_point_clouds(samples)[0]
        return pc.coords.tolist()
    
    