import torch
from tqdm.auto import tqdm
import numpy as np
import open3d as o3d

from point_e.diffusion.configs import DIFFUSION_CONFIGS, diffusion_from_config
from point_e.diffusion.sampler import PointCloudSampler
from point_e.models.download import load_checkpoint
from point_e.models.configs import MODEL_CONFIGS, model_from_config
from point_e.util.plotting import plot_point_cloud

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('creating base model...')
base_name = 'base40M-textvec'
base_model = model_from_config(MODEL_CONFIGS[base_name], device)
base_model.eval()
base_diffusion = diffusion_from_config(DIFFUSION_CONFIGS[base_name])

print('creating upsample model...')
upsampler_model = model_from_config(MODEL_CONFIGS['upsample'], device)
upsampler_model.eval()
upsampler_diffusion = diffusion_from_config(DIFFUSION_CONFIGS['upsample'])

print('downloading base checkpoint...')
base_model.load_state_dict(load_checkpoint(base_name, device))

print('downloading upsampler checkpoint...')
upsampler_model.load_state_dict(load_checkpoint('upsample', device))

sampler = PointCloudSampler(
    device=device,
    models=[base_model, upsampler_model],
    diffusions=[base_diffusion, upsampler_diffusion],
    num_points=[1024, 4096 - 1024],
    aux_channels=['R', 'G', 'B'],
    guidance_scale=[3.0, 0.0],
    model_kwargs_key_filter=('texts', ''), # Do not condition the upsampler at all
)

# Set a prompt to condition on.
prompt = 'a white headset'

# Produce a sample from the model.
samples = None
for x in tqdm(sampler.sample_batch_progressive(batch_size=1, model_kwargs=dict(texts=[prompt]))):
    samples = x

pc = sampler.output_to_point_clouds(samples)[0]
fig = plot_point_cloud(pc, grid_size=3, fixed_bounds=((-0.75, -0.75, -0.75),(0.75, 0.75, 0.75)))
print(pc.coords)

def compute_normals_with_open3d(points, k=10):
    """
    Open3D를 사용하여 법선 벡터를 계산하고 좌표와 결합.

    입력:
        points (np.ndarray): (N, 3) 형태의 좌표 데이터
        k (int): 법선 계산을 위한 k-최근접 이웃 수
    출력:
        augmented_points (np.ndarray): (N, 6) 형태의 데이터 (좌표 + 법선)
    """
    # Open3D PointCloud 객체 생성
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # k-최근접 이웃 기반 법선 계산
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k))
    
    # 법선 벡터 방향 정규화 (필요 시)
    pcd.orient_normals_consistent_tangent_plane(k)

    # 좌표와 법선 벡터 추출
    normals = np.asarray(pcd.normals)
    augmented_points = np.hstack((points, normals))
    
    return augmented_points

# 입력 데이터 예시
points = pc.coords  # (N, 3) 형태의 데이터

# 법선 계산 및 데이터 결합
augmented_data = compute_normals_with_open3d(points, k=10)

print(augmented_data)

# 결과 데이터를 .npy 파일로 저장
output_path = "augmented_data.npy"
np.save(output_path, augmented_data)

print(f"결과 데이터가 '{output_path}'에 저장되었습니다.")