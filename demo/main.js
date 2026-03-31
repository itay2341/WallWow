import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import GUI from 'lil-gui';

// ─── Renderer ────────────────────────────────────────────────────────────────
const canvas = document.getElementById('c');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.outputColorSpace = THREE.SRGBColorSpace;

// ─── Scene ───────────────────────────────────────────────────────────────────
const scene = new THREE.Scene();
scene.background = new THREE.Color(0xf0f0f0);

// ─── Camera ──────────────────────────────────────────────────────────────────
// Position: near the corner (0, 2, 0), looking toward room center (1, 1, 1)
const camera = new THREE.PerspectiveCamera(
  75,
  window.innerWidth / window.innerHeight,
  0.01,
  100
);
camera.position.set(0.2, 1.8, 0.2);

// ─── Orbit Controls ──────────────────────────────────────────────────────────
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(1, 1, 1);
controls.enableDamping = true;
controls.dampingFactor = 0.05;
controls.update();

// ─── Lighting ────────────────────────────────────────────────────────────────
// Soft ambient keeps the whole scene evenly lit (no harsh shadows)
scene.add(new THREE.AmbientLight(0xffffff, 0.8));
const dirLight = new THREE.DirectionalLight(0xffffff, 0.6);
dirLight.position.set(3, 4, 3);
scene.add(dirLight);

// ─── Fallback texture (shown when an asset PNG is missing) ───────────────────
function makeFallbackTexture(label) {
  const size = 128;
  const c = document.createElement('canvas');
  c.width = c.height = size;
  const ctx = c.getContext('2d');
  ctx.fillStyle = '#c8c8c8';
  ctx.fillRect(0, 0, size, size);
  ctx.strokeStyle = '#999';
  ctx.lineWidth = 3;
  ctx.strokeRect(6, 6, size - 12, size - 12);
  // diagonal cross
  ctx.beginPath();
  ctx.moveTo(6, 6);    ctx.lineTo(size - 6, size - 6);
  ctx.moveTo(size - 6, 6); ctx.lineTo(6, size - 6);
  ctx.stroke();
  ctx.fillStyle = '#666';
  ctx.font = 'bold 13px sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(label, size / 2, size / 2);
  return new THREE.CanvasTexture(c);
}

// ─── White Wall Planes ───────────────────────────────────────────────────────
// Room occupies [0, 2] on X, Y, Z.  Corner of interest at (0, 2, 0).
//
//   Left wall  → X = 0   (normal +X, rotated Y + π/2)
//   Right wall → Z = 0   (normal +Z, no rotation)
//   Ceiling    → Y = 2   (normal −Y, rotated X + π/2)

const wallMat = new THREE.MeshLambertMaterial({
  color: 0xffffff,
  side: THREE.DoubleSide,
});

function makeWall(px, py, pz, rx, ry) {
  const mesh = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), wallMat);
  mesh.position.set(px, py, pz);
  mesh.rotation.set(rx, ry, 0);
  scene.add(mesh);
  return mesh;
}

const leftWall    = makeWall(0, 1, 1,  0,              Math.PI / 2);
const rightWall   = makeWall(1, 1, 0,  0,              0);
const ceilingWall = makeWall(1, 2, 1,  Math.PI / 2,    0);

// ─── Image Planes ────────────────────────────────────────────────────────────
// Each image is placed on a thin plane positioned OFFSET metres in front of
// its wall so it floats cleanly without z-fighting.
//
// Default size: 1 m wide, height adjusted to preserve the texture's
// aspect ratio.  The GUI "Image Scale" slider multiplies this uniformly.
//
// Note: if an image with text appears horizontally mirrored on the left wall,
// set texture.repeat.set(-1, 1); texture.offset.set(1, 0) inside the loader
// callback for leftImage to correct it.

const OFFSET = 0.002;
// Image size in metres.  ~1/3 of the 2 m wall side so the white wall
// is clearly visible while the three images still touch at the corner.
const IMG_SIZE = 0.7;
const texLoader = new THREE.TextureLoader();
const imageMeshes = [];

function makeImagePlane(assetPath, fallbackLabel, px, py, pz, rx, ry) {
  const mat = new THREE.MeshBasicMaterial({
    map: makeFallbackTexture(fallbackLabel),
    side: THREE.FrontSide,
  });

  // IMG_SIZE × IMG_SIZE plane positioned so its corner-facing edges
  // sit exactly at X=0 / Z=0 / Y=2, keeping all three images in contact.
  const mesh = new THREE.Mesh(new THREE.PlaneGeometry(IMG_SIZE, IMG_SIZE), mat);
  mesh.position.set(px, py, pz);
  mesh.rotation.set(rx, ry, 0);
  scene.add(mesh);
  imageMeshes.push(mesh);

  texLoader.load(
    assetPath,
    (tex) => {
      tex.colorSpace = THREE.SRGBColorSpace;
      tex.wrapS = THREE.ClampToEdgeWrapping;
      tex.wrapT = THREE.ClampToEdgeWrapping;
      // "Cover" crop: preserve aspect ratio while filling the square IMG_SIZE plane.
      const r = tex.image.width / tex.image.height;
      if (r >= 1) {
        // Landscape / square: full height visible, crop left & right.
        tex.repeat.set(1 / r, 1);
        tex.offset.set((1 - 1 / r) / 2, 0);
      } else {
        // Portrait: full width visible, crop top & bottom.
        tex.repeat.set(1, r);
        tex.offset.set(0, (1 - r) / 2);
      }
      mat.map = tex;
      mat.needsUpdate = true;
    },
    undefined, // onProgress (not needed)
    () => console.warn(`⚠  Texture not found: "${assetPath}". Place the image in assets/.`)
  );

  return mesh;
}

// Each image is anchored so its two corner-facing edges sit exactly on
// X=0 / Z=0 / Y=2, making the three images touch each other at the seam.
// Centre = half-size away from each bounding edge.
const S = IMG_SIZE / 2; // shorthand

// Left wall image  – sits in the top-right corner of the X=0 plane
const leftImage = makeImagePlane(
  'assets/left.png', 'left.png',
  OFFSET, 2 - S, S,
  0, Math.PI / 2
);

// Right wall image – sits in the top-left corner of the Z=0 plane
const rightImage = makeImagePlane(
  'assets/right.png', 'right.png',
  S, 2 - S, OFFSET,
  0, 0
);

// Ceiling image    – sits in the corner of the Y=2 plane at X=0, Z=0
const ceilingImage = makeImagePlane(
  'assets/top.png', 'top.png',
  S, 2 - OFFSET, S,
  Math.PI / 2, 0
);

// ─── GUI Controls ────────────────────────────────────────────────────────────
const gui = new GUI({ title: 'Room Controls' });

const params = {
  imageScale:    1.0,
  showLeftWall:  true,
  showRightWall: true,
  showCeiling:   true,
};

gui.add(params, 'imageScale', 0.1, 3.0, 0.01)
  .name('Image Scale')
  .onChange((v) => imageMeshes.forEach((m) => m.scale.setScalar(v)));

const visFolder = gui.addFolder('Visibility');
visFolder.add(params, 'showLeftWall').name('Left Wall')
  .onChange((v) => { leftWall.visible = v;    leftImage.visible = v;   });
visFolder.add(params, 'showRightWall').name('Right Wall')
  .onChange((v) => { rightWall.visible = v;   rightImage.visible = v;  });
visFolder.add(params, 'showCeiling').name('Ceiling')
  .onChange((v) => { ceilingWall.visible = v; ceilingImage.visible = v; });

// ─── Resize Handler ──────────────────────────────────────────────────────────
window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

// ─── Render Loop ─────────────────────────────────────────────────────────────
(function animate() {
  requestAnimationFrame(animate);
  controls.update(); // required for damping
  renderer.render(scene, camera);
})();
