import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const container = document.getElementById('obj-viewer-1');
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(container.clientWidth, container.clientHeight);
container.appendChild(renderer.domElement);

renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.setClearColor(0x000000);
renderer.setPixelRatio(container.devicePixelRatio);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;

const scene = new THREE.Scene();

const camera = new THREE.PerspectiveCamera(45, container.clientWidth / container.clientHeight, 1, 1000);
camera.position.set(7, 5, 7);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.enablePan = false;
controls.minDistance = 0.12;
controls.maxDistance = 1.5;
controls.minPolarAngle = 0.0;
controls.maxPolarAngle = 1.57;
controls.autoRotate = false;
controls.target.set(0, 1, 0);
controls.update();

// 光照与地面
const groundGeometry = new THREE.PlaneGeometry(20, 20, 32, 32);
groundGeometry.rotateX(-Math.PI / 2);
const groundMaterial = new THREE.MeshStandardMaterial({ color: 0x555555, side: THREE.DoubleSide });
const groundMesh = new THREE.Mesh(groundGeometry, groundMaterial);
groundMesh.receiveShadow = true;
scene.add(groundMesh);

const lights = [
  [0, 25, 0],
  [25, 0, 0],
  [-25, 0, 0]
];
lights.forEach(pos => {
  const light = new THREE.SpotLight(0xffffff, 4, 100, 0.22, 1);
  light.position.set(...pos);
  light.castShadow = true;
  light.shadow.bias = -0.0001;
  scene.add(light);
});

window.addEventListener('resize', () => {
  camera.aspect = container.clientWidth / container.clientHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(container.clientWidth, container.clientHeight);
});

function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}
animate();

// 🔁 动态部分
const loader = new GLTFLoader().setPath('./static/model/');
let currentMesh = null;

function loadStep(stepIndex) {
  // 模型加载
  loader.load(`out${stepIndex}.gltf`, (gltf) => {
    if (currentMesh) {
      scene.remove(currentMesh);
    }

    const mesh = gltf.scene;
    mesh.traverse((child) => {
      if (child.isMesh) {
        child.castShadow = true;
        child.receiveShadow = true;
      }
    });

    mesh.position.set(0, 1, 0);
    scene.add(mesh);
    currentMesh = mesh;
  }, undefined, (error) => {
    console.error('GLTF load error:', error);
  });

  // 文本加载
  fetch(`./static/text/critic${stepIndex}.txt`)
    .then(response => response.text())
    .then(text => {
      document.getElementById('critic-text').innerText = text;
    })
    .catch(error => {
      console.error('Text load error:', error);
      document.getElementById('critic-text').innerText = '(Failed to load critic text)';
    });
}

// 默认加载 Step 1
window.onload = () => {
  loadStep(1);
};

window.loadStep = loadStep;

function selectStep(step) {
  // Remove active class from all
  for (let i = 1; i <= 3; i++) {
    document.getElementById(`step-btn-${i}`).classList.remove('active');
  }
  // Add active class to the selected
  document.getElementById(`step-btn-${step}`).classList.add('active');
  // Optionally load step content
  if (typeof loadStep === 'function') {
    loadStep(step);
  }
}

window.selectStep = selectStep;