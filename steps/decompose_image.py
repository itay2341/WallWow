"""Step 2 — Decompose an isometric 3D object image into 3 rectified face projections.

Detects the hexagonal silhouette of the isometric cube, computes the
Y-junction (where the 3 visible edges meet), and perspective-warps each
face into a square image.
"""

import logging
import os

import cv2
import numpy as np
from PIL import Image
from rembg import remove

import config

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hexagon detection
# ---------------------------------------------------------------------------

def _find_hexagon_vertices(alpha: np.ndarray) -> np.ndarray:
    """Detect the 6 corner vertices of the cube's hexagonal silhouette.

    Returns 6 vertices as float32 (6×2), sorted clockwise starting from
    the topmost vertex (smallest y):
        V0=top  V1=upper-right  V2=lower-right
        V3=bottom  V4=lower-left  V5=upper-left
    """
    _, binary = cv2.threshold(alpha, 128, 255, cv2.THRESH_BINARY)

    # Clean up mask
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No contours found in alpha mask")
    largest = max(contours, key=cv2.contourArea)
    hull_contour = cv2.convexHull(largest)
    hull = hull_contour.reshape(-1, 2).astype(np.float64)
    perimeter = cv2.arcLength(hull_contour, True)

    n = len(hull)
    if n < 5:
        raise ValueError(f"Convex hull has only {n} vertices, need ≥ 5")

    pts = None

    # Method 1: approxPolyDP — try to find epsilon giving exactly 6 vertices
    for eps_pct in np.arange(0.005, 0.15, 0.001):
        approx = cv2.approxPolyDP(hull_contour, eps_pct * perimeter, True)
        if len(approx) == 6:
            pts = approx.reshape(-1, 2).astype(np.float32)
            break

    # Method 2: greedy corner selection with minimum‐distance constraint
    if pts is None:
        angles = np.zeros(n)
        for i in range(n):
            v1 = hull[i - 1] - hull[i]
            v2 = hull[(i + 1) % n] - hull[i]
            cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
            angles[i] = np.arccos(np.clip(cos_a, -1, 1))

        min_dist = perimeter / 12          # ≈ half the average edge length
        selected: list[int] = []
        for idx in np.argsort(angles):     # sharpest corners first
            pt = hull[idx]
            if all(np.linalg.norm(pt - hull[s]) >= min_dist for s in selected):
                selected.append(int(idx))
            if len(selected) == 6:
                break

        selected.sort()
        pts = hull[np.array(selected)].astype(np.float32)

    # If we still have < 6, subdivide the longest edge(s)
    while len(pts) < 6:
        dists = np.array([np.linalg.norm(pts[(i+1) % len(pts)] - pts[i])
                          for i in range(len(pts))])
        longest = int(np.argmax(dists))
        mid = (pts[longest] + pts[(longest + 1) % len(pts)]) / 2
        pts = np.insert(pts, longest + 1, mid.reshape(1, 2), axis=0)

    # If > 6 (shouldn't happen but defensive), keep the 6 sharpest
    if len(pts) > 6:
        n2 = len(pts)
        ang = np.zeros(n2)
        for i in range(n2):
            va = pts[i - 1] - pts[i]
            vb = pts[(i + 1) % n2] - pts[i]
            cos_a = np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-10)
            ang[i] = np.arccos(np.clip(cos_a, -1, 1))
        keep = np.sort(np.argsort(ang)[:6])
        pts = pts[keep]

    # Sort clockwise in image coords (y-down) starting from topmost vertex
    centroid = pts.mean(axis=0)
    a = np.arctan2(pts[:, 1] - centroid[1], pts[:, 0] - centroid[0])
    a = (a + 2 * np.pi) % (2 * np.pi)     # [0, 2π)
    order = np.argsort(a)                  # ascending = clockwise in y-down
    pts = pts[order]

    top_idx = int(np.argmin(pts[:, 1]))
    pts = np.roll(pts, -top_idx, axis=0)
    return pts


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _line_intersection(p1, p2, p3, p4) -> np.ndarray:
    """Intersection of line(p1→p2) with line(p3→p4)."""
    x1, y1 = float(p1[0]), float(p1[1])
    x2, y2 = float(p2[0]), float(p2[1])
    x3, y3 = float(p3[0]), float(p3[1])
    x4, y4 = float(p4[0]), float(p4[1])
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-10:
        return np.array([(x1+x2+x3+x4)/4, (y1+y2+y3+y4)/4], dtype=np.float32)
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    return np.array([x1 + t*(x2-x1), y1 + t*(y2-y1)], dtype=np.float32)


# ---------------------------------------------------------------------------
# Debug overlay
# ---------------------------------------------------------------------------

def _save_debug_image(rgba, V, C, output_dir):
    """Draw hexagon, Y-junction arms, and face labels over the source."""
    debug = cv2.cvtColor(rgba[:, :, :3].copy(), cv2.COLOR_RGB2BGR)

    colors = [
        (0, 0, 255), (0, 128, 255), (0, 255, 255),
        (0, 255, 0), (255, 255, 0), (255, 0, 0),
    ]

    # Hexagon edges + vertex labels
    for i in range(6):
        pt = tuple(V[i].astype(int))
        nxt = tuple(V[(i + 1) % 6].astype(int))
        cv2.line(debug, pt, nxt, (0, 255, 0), 2)
        cv2.circle(debug, pt, 10, colors[i], -1)
        cv2.putText(debug, f"V{i}", (pt[0]+12, pt[1]-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, colors[i], 2)

    # Y-junction + arms
    c_pt = tuple(C.astype(int))
    cv2.circle(debug, c_pt, 10, (255, 255, 255), -1)
    cv2.putText(debug, "C", (c_pt[0]+12, c_pt[1]-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    for i in (1, 3, 5):
        cv2.line(debug, c_pt, tuple(V[i].astype(int)), (255, 255, 255), 2)

    # Face labels
    for label, quad in [("TOP", [V[5], V[0], V[1], C]),
                        ("RIGHT", [C, V[1], V[2], V[3]]),
                        ("LEFT", [V[5], C, V[3], V[4]])]:
        fc = np.mean(quad, axis=0).astype(int)
        cv2.putText(debug, label, tuple(fc),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

    path = os.path.join(output_dir, "debug_hexagon.png")
    cv2.imwrite(path, debug)
    log.info("Saved debug overlay → %s", path)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def decompose_image(
    image_path: str,
    output_dir: str = config.DEFAULT_OUTPUT_DIR,
    resolution: int = config.RENDER_RESOLUTION,
) -> dict[str, str]:
    """Decompose an isometric image into 3 rectified face images.

    Returns dict mapping ``"top"``/``"left"``/``"right"`` → saved file path.
    """
    log.info("Loading image: %s", image_path)
    img = Image.open(image_path).convert("RGBA")

    # --- background removal ---
    log.info("Removing background …")
    img_nobg = remove(img)
    rgba = np.array(img_nobg)          # H×W×4
    alpha = rgba[:, :, 3]
    h, w = alpha.shape

    # --- detect hexagonal outline ---
    V = _find_hexagon_vertices(alpha)
    log.info("Hexagon: %s",
             {f"V{i}": (int(V[i][0]), int(V[i][1])) for i in range(6)})

    # --- Y-junction from parallelogram constraints ---
    # Each cube face is a parallelogram. The Y-junction is the missing 4th
    # vertex for each face.  Average the 3 estimates for robustness.
    #   Top  face: V5 → V0 → V1 → F  ⇒  F = V5 + V1 − V0
    #   Right face: F → V1 → V2 → V3  ⇒  F = V1 + V3 − V2
    #   Left  face: V5 → F → V3 → V4  ⇒  F = V5 + V3 − V4
    F_top   = V[5] + V[1] - V[0]
    F_right = V[1] + V[3] - V[2]
    F_left  = V[5] + V[3] - V[4]
    C = np.mean([F_top, F_right, F_left], axis=0).astype(np.float32)
    log.info("Y-junction (parallelogram avg) at (%d, %d)  "
             "[top=(%.0f,%.0f) right=(%.0f,%.0f) left=(%.0f,%.0f)]",
             int(C[0]), int(C[1]),
             F_top[0], F_top[1], F_right[0], F_right[1],
             F_left[0], F_left[1])

    # --- debug overlay ---
    _save_debug_image(rgba, V, C, output_dir)

    # --- warp each face ---
    S = resolution
    os.makedirs(output_dir, exist_ok=True)

    # Each face is a quadrilateral mapped to [TL, TR, BR, BL]
    faces = {
        "top":   [V[5], V[0], V[1], C],      # ceiling
        "right": [C,    V[1], V[2], V[3]],    # right wall
        "left":  [V[5], C,    V[3], V[4]],    # left wall
    }

    dst = np.float32([[0, 0], [S-1, 0], [S-1, S-1], [0, S-1]])
    paths: dict[str, str] = {}

    for name, quad in faces.items():
        # Polygon mask — expand 3% outward so edges don't clip
        quad_f = np.array(quad, dtype=np.float32)
        centre = quad_f.mean(axis=0)
        expanded = centre + (quad_f - centre) * 1.03
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(mask, expanded.astype(np.int32), 255)
        mask = mask & np.where(alpha > 32, 255, 0).astype(np.uint8)

        masked = rgba.copy()
        masked[mask == 0] = (0, 0, 0, 0)

        # Perspective-warp rhombus → square
        M = cv2.getPerspectiveTransform(np.float32(quad), dst)
        warped = cv2.warpPerspective(
            masked, M, (S, S),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0),
        )

        out_path = os.path.join(output_dir, f"{name}.png")
        Image.fromarray(warped, "RGBA").save(out_path)
        paths[name] = out_path
        log.info("Saved %s → %s", name, out_path)

    return paths
