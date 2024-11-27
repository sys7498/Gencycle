import { Component, OnInit, HostListener } from '@angular/core';
import * as THREE from 'three';
import { HttpClient } from '@angular/common/http';
import axios from 'axios';
import { OBJLoader } from 'three/examples/jsm/loaders/OBJLoader.js';
import { MTLLoader } from 'three/examples/jsm/loaders/MTLLoader.js';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html', // 템플릿 분리
  styleUrls: ['./app.component.scss'], // 스타일 분리
})
export class AppComponent implements OnInit {
  message: string = '';
  messages: string[] = [];

  private renderer!: THREE.WebGLRenderer;
  private scene!: THREE.Scene;
  private camera!: THREE.PerspectiveCamera;
  private models: (THREE.Group | THREE.Mesh)[] = []; // 모델 배열
  private model!: THREE.Group; // OBJ 모델을 담을 변수
  private modelPositions: THREE.Vector3[] = [];

  constructor(private http: HttpClient) {
    axios.defaults.headers.post['Content-Type'] = 'application/json';
    this.modelPositions = [
      new THREE.Vector3(0, 2, 0), // 상단
      new THREE.Vector3(0, -2, 0), // 하단
      new THREE.Vector3(-2, 0, 0), // 좌측
      new THREE.Vector3(2, 0, 0), // 우측
    ];
  }

  ngOnInit() {
    this.initThreeJS(); //3
    this.animate();
  }

  // Three.js 초기화
  initThreeJS() {
    this.scene = new THREE.Scene();

    const container = document.getElementById('three-container');
    const containerHeight = container?.clientHeight || window.innerHeight;

    this.camera = new THREE.PerspectiveCamera(75, 1, 0.1, 1000);

    // 렌더러 설정
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    this.renderer.setPixelRatio(
      window.devicePixelRatio > 1 ? window.devicePixelRatio : 2
    );
    this.renderer.setSize(containerHeight, containerHeight);
    //this.renderer.outputEncoding = THREE.sRGBEncoding; // 색상 보정

    // three-container에 렌더러 추가
    container?.appendChild(this.renderer.domElement);

    // 메탈릭한 재질의 큐브 생성 및 위치 설정
    const geometry = new THREE.BoxGeometry(1, 1, 1); // 큐브 크기 조절
    const material = new THREE.MeshPhysicalMaterial({
      color: 0xcccccc,
      metalness: 0.7, // 메탈릭 속성 최대값
      roughness: 0.0, // 거칠기 최소화
      reflectivity: 1.0, // 반사율 최대화
      //clearcoat: 1.0, // 투명 코팅 최대화
      //clearcoatRoughness: 0.0, // 투명 코팅 거칠기 최소화
    });

    // 조명 추가
    const ambientLight = new THREE.AmbientLight(0xffffff, 1.5); // 환경광 강도 증가
    this.scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 10); // 방향성 조명 강도 증가
    directionalLight.position.set(0, 0, 7.5);
    this.scene.add(directionalLight);

    // 카메라 위치 설정
    this.camera.position.z = 5;
  }

  // 애니메이션 처리
  animate = () => {
    requestAnimationFrame(this.animate);

    // 각 큐브 회전
    if (this.models.length > 0) {
      this.models.forEach((model) => {
        model.rotation.z += 0.01;
      });
    }

    this.renderer.render(this.scene, this.camera);
  };

  // 화면 크기 변경에 따른 처리
  @HostListener('window:resize', ['$event'])
  onWindowResize() {
    const container = document.getElementById('three-container');
    const containerHeight = container?.clientHeight || window.innerHeight;

    this.camera.aspect = 1; // 항상 1:1 비율 유지
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(containerHeight, containerHeight);
  }

  public async getDetectionResult() {
    let response = await axios.get('http://localhost:5001/api/detect_image');
    console.log(response.data);
    return response.data;
  }

  public async getGeneratedModel() {
    // HTTP 요청을 통해 OBJ 파일 가져오기
    const response = await axios.get(
      'http://localhost:5001/api/generated_model'
    );

    const mtlData = (response.data as any).mtl;
    const mtlLoader = new MTLLoader();
    const materials = mtlLoader.parse(mtlData, '');
    materials.preload();

    const objData = (response.data as any).obj;
    const loader = new OBJLoader();
    loader.setMaterials(materials);
    this.model = loader.parse(objData);

    for (let i = 0; i < this.modelPositions.length; i++) {
      const clonedModel = this.model.clone();
      clonedModel.position.copy(this.modelPositions[i]);
      clonedModel.scale.set(0.05, 0.05, 0.05);
      clonedModel.rotation.x = -Math.PI / 2;
      this.models.push(clonedModel);
      this.scene.add(clonedModel);
    }
  }

  public async getBoundingBox() {
    const response = await axios.get('http://localhost:5001/api/size');
    const size = response.data as { x: number; y: number; z: number };

    // size.x를 10으로 설정하고 스케일 팩터 계산
    const scaleFactor = 0.5 / size.x;

    // 조정된 크기 계산
    const adjustedSize = {
      x: 0.5,
      y: size.y * scaleFactor,
      z: size.z * scaleFactor,
    };

    const geometry = new THREE.BoxGeometry(
      adjustedSize.x * 1.5,
      adjustedSize.y * 1.5,
      adjustedSize.z * 1.5
    );
    const material = new THREE.MeshPhysicalMaterial({
      color: 0xcccccc,
      metalness: 0.7,
      roughness: 0.0,
      reflectivity: 1.0,
    });

    for (let i = 0; i < this.modelPositions.length; i++) {
      const box = new THREE.Mesh(geometry, material);
      box.position.copy(this.modelPositions[i]);

      // 박스가 원점을 향하도록 회전 적용
      box.lookAt(new THREE.Vector3(0, 0, 0));

      this.models.push(box);
      this.scene.add(box);
    }
  }
}
