# WallWow – 3D Corner Illusion Generator

A geometry-based system that transforms a single input image into 3 aligned panels (left wall, right wall, ceiling) that create a convincing 3D illusion when physically placed at a 90° room corner.

## Overview

WallWow uses **reverse projection** and **perspective transformation** to split an image across three physical planes, ensuring visual continuity when viewed from a specific camera position.

### Key Features

- Pure geometry-based projection (no AI generation in MVP)
- Produces three printable panel images: `left.png`, `right.png`, `top.png`
- Configurable camera position, FOV, and output resolution
- Edge continuity validation

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python src/main.py <input_image> [options]
```

### Options

- `--camera-pos X Y Z` : Camera position in meters (default: 2 2 1.2)
- `--fov DEGREES` : Field of view in degrees (default: 75)
- `--output-size SIZE` : Output panel size in pixels (default: 1024)
- `--output-dir DIR` : Output directory (default: output/)

### Example

```bash
python src/main.py data/cube.png --fov 80 --output-size 2048
```

## Project Structure

```
WallWow/
├── src/
│   ├── geometry_engine.py    # 3D coordinate system, Camera, Plane classes
│   ├── math_utils.py          # Ray-plane intersection, transformations
│   ├── projection_mapper.py   # Reverse projection algorithm
│   ├── renderer.py            # Panel rendering with cv2.warpPerspective
│   ├── validator.py           # Edge continuity checker
│   └── main.py                # CLI entry point
├── data/                      # Input test images
├── output/                    # Generated panel images
├── tests/                     # Unit tests
├── requirements.txt
└── README.md
```

## How It Works

1. **Geometry Setup**: Define a 3D corner coordinate system with origin at the corner vertex
2. **Camera Configuration**: Place a virtual camera in front of the corner
3. **Reverse Projection**: For each point on each physical plane (wall/ceiling), determine which pixel from the input image should appear there
4. **Perspective Warp**: Apply transformation matrices using OpenCV to generate three panel images
5. **Validation**: Check edge continuity across adjacent panels

## Technical Details

- **Coordinate System**: Origin at corner vertex, left wall in YZ plane, right wall in XZ plane, ceiling in XY plane
- **Default Camera**: Position (2, 2, 1.2) meters, FOV 75°
- **Output**: Three 1024×1024 PNG images (configurable)
- **Continuity**: Uses ray-plane intersection to ensure seamless transitions

## Verification

Test with geometric patterns:
- Grid lines (to verify continuity)
- Diagonal lines at various angles
- Perspective corridor images
- 3D geometric shapes

## Future Extensions

- AI-based scene generation integration
- Depth-aware rendering with ControlNet
- User room photo analysis
- Web interface
- Multi-corner designs

## License

TBD
