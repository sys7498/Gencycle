import base64
import random
from flask import url_for
from rich.console import Console
from rich.logging import RichHandler
import logging
from io import BytesIO
import io
import os
from PIL import Image
from openai import OpenAI
import requests
import torch
from torchvision import transforms
from transformers import AutoModelForImageSegmentation

from tqdm import tqdm

from sf3d.system import SF3D
from sf3d.utils import get_device, remove_background, resize_foreground
import argparse
import os
from contextlib import nullcontext

from PIL import Image, ImageDraw
import json
import open3d as o3d

class Gpt:
    def __init__(self, bgremoval_model):
        self.console = Console()
        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",
            handlers=[RichHandler(console=self.console, rich_tracebacks=True)]
        )
        self.logger = logging.getLogger("gpt")

        self.bgremoval_model = bgremoval_model
        self.client = OpenAI(
            api_key="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",
        )

        self.device = torch.device("cpu")
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        self.logger.info("using device: %s", self.device)

        # Mesh generation model
        self.logger.info("creating mesh generation model...")
        self.sf3d_model = SF3D.from_pretrained(
            'stabilityai/stable-fast-3d',
            config_name="config.yaml",
            weight_name="model.safetensors",
        )
        self.sf3d_model.to(self.device)
        self.sf3d_model.eval()
    
    def download_image(self, url):
        image_response = requests.get(url)
        if image_response.status_code == 200:
            return Image.open(BytesIO(image_response.content))
        else:
            print(f"Failed to download image: {image_response.status_code}")
            return None
        
    def generate_image(self, obj):
        '''
        response = self.client.images.generate(
            prompt=prompt,
            n=1,
            size="1024x1024"
        )
         self.download_image(response.data[0].url).save("generated_img.png")
        '''
        # 폴더 경로 설정
        folder_path = os.path.join("./assets", obj)

        # 폴더 내 이미지 파일 가져오기
        image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]
        
        if not image_files:
            raise FileNotFoundError(f"No images found in folder: {folder_path}")

        # 랜덤 이미지 선택
        selected_image = random.choice(image_files)
        image_path = os.path.join(folder_path, selected_image)

        # 이미지 파일을 Base64로 인코딩
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        
        # MIME 타입 추출
        mime_type = "image/png" if selected_image.endswith(".png") else "image/jpeg"

        # Data URL 생성
        image_url = f"data:{mime_type};base64,{encoded_string}"
        
        save_path = "./generated_img.png"
        with open(save_path, "wb") as output_file:
            output_file.write(base64.b64decode(encoded_string))
        return image_url
    
    def edit_image(self, prompt):
        response = self.client.images.edit(
          model="dall-e-2",
          image=open("generated_img.png", "rb"),
          mask=open("masked_generated_img.png", "rb"),
          prompt=prompt,
          n=1,
          size="1024x1024"
        )
        self.download_image(response.data[0].url).save("generated_img.png")
        return response.data[0].url

    def generate_masked_image(self, mask):
       # 원본 이미지와 마스킹 이미지 열기
        original_image = Image.open("generated_img.png").convert("RGBA")
        mask_image = mask.convert("L")  # 마스크는 흑백 이미지로 불러옴

        # 투명도 처리: 마스크의 흰색(255)은 완전 투명, 검정색(0)은 불투명
        alpha = Image.new("L", original_image.size, 255)  # 기본적으로 불투명(255)
        alpha = Image.composite(Image.new("L", original_image.size, 0), alpha, mask_image)

        # 알파 채널을 원본 이미지에 추가
        original_image.putalpha(alpha)

        # 결과 이미지 저장
        original_image.save("masked_generated_img.png", "PNG")

    def generate_mesh(self, mode: str = "gen"):
        # Data settings
        image_path = "generated_img.png" if mode == "gen" else mode
        image = Image.open(image_path)
        image_size = image.size
        transform_image = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        input_images = transform_image(image).unsqueeze(0).to('cuda')

        # Prediction
        with torch.no_grad():
            preds = self.bgremoval_model(input_images)[-1].sigmoid().cpu()
        pred = preds[0].squeeze()
        pred_pil = transforms.ToPILImage()(pred)
        mask = pred_pil.resize(image.size)
        image.putalpha(mask)
        image.convert("RGBA")
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            with torch.autocast(
                device_type='cuda', dtype=torch.float16
            ):
                mesh, glob_dict = self.sf3d_model.run_image(
                    image,
                    bake_resolution=256,
                    remesh='triangle',
                    vertex_count=-1,
                )
        if torch.cuda.is_available():
            print("Peak Memory:", torch.cuda.max_memory_allocated() / 1024 / 1024, "MB")

        output_path = "generated_mesh.glb" if mode == "gen" else "generated_original_mesh.glb"
        out_mesh_path = os.path.join('./', output_path)
        mesh.export(out_mesh_path, include_normals=True)
        #o3d_mesh = o3d.io.read_triangle_mesh(output_path)
        # 메쉬 단순화 (간소화 비율 0.5로 설정)
        #simplified_mesh = o3d_mesh.simplify_quadric_decimation(500)

        # 결과 저장
        #o3d.io.write_triangle_mesh("generated_mesh.glb", simplified_mesh)

    def extract_model_features(self):
        mesh = o3d.io.read_triangle_mesh('generated_mesh.glb')
        mesh.compute_vertex_normals()
        num_vertices = len(mesh.vertices)
        num_triangles = len(mesh.triangles)
        bounding_box = mesh.get_axis_aligned_bounding_box()
        dimensions = bounding_box.get_extent()
        
        description = (
            f"The model has {num_vertices} vertices and {num_triangles} triangles. "
            f"It fits within a bounding box of dimensions {dimensions[0]:.2f} x {dimensions[1]:.2f} x {dimensions[2]:.2f} units. "
            "The structure appears to be a vase with curved, organic geometry. "
        )
        return description

    def generate_instruction(self):
        description = self.extract_model_features()
        prompt = (
            "Based on the following 3D model description, create detailed step-by-step instructions "
            "to design an upcycled product. The product should use recycled materials and draw inspiration "
            "from the vase's structural characteristics. Include creative assembly details and material suggestions:\n\n"
            f"Description: {description}\n\n"
            "Provide innovative upcycling instructions that transform the vase's form into a functional craft project."
        )
        
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert in 3D modeling and creative upcycling."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=750
        )
        
        instructions = response.choices[0].message.content
        print(instructions)
        return instructions
    
    def get_product_description(self, object_data):
        prompt = (
            f"Here is a detected object:\n"
            f"- Material: {object_data.get('material', 'unknown')}\n"
            f"- Name: {object_data.get('name', 'unknown')}\n\n"
            f"Write the name and material of the object without having 'This object is' and the material should be first then the name ."
        )
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7  
        )
        return response.choices[0].message.content

    
    