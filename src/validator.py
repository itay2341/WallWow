"""
Edge Continuity Validator for WallWow.

Checks visual continuity across panel boundaries to ensure seamless
transitions when panels are mounted at the corner.
"""

import numpy as np
from typing import Dict, Tuple, List
import cv2


class EdgeValidator:
    """
    Validates edge continuity between adjacent panels.
    
    Samples pixels along shared edges and computes color/intensity
    differences to detect discontinuities.
    """
    
    def __init__(self, panels: Dict[str, np.ndarray]):
        """
        Initialize the validator.
        
        Args:
            panels: Dictionary mapping plane names to panel images
                   Expected keys: "left_wall", "right_wall", "ceiling"
        """
        self.panels = panels
        
        # Validate that all required panels are present
        required = {"left_wall", "right_wall", "ceiling"}
        if not required.issubset(panels.keys()):
            missing = required - set(panels.keys())
            raise ValueError(f"Missing panels: {missing}")
        
        self.left = panels["left_wall"]
        self.right = panels["right_wall"]
        self.ceiling = panels["ceiling"]
        
        # Results storage
        self.edge_scores = {}
    
    def validate_left_ceiling_edge(self, num_samples: int = 100) -> Tuple[float, np.ndarray]:
        """
        Validate continuity between left wall and ceiling.
        
        The left wall's top edge should align with the ceiling's left edge.
        
        Args:
            num_samples: Number of points to sample along the edge
        
        Returns:
            Tuple of (average_difference, per_sample_differences)
        """
        # Left wall top edge (last row)
        left_edge = self.left[-1, :, :]  # Shape: (width, 3)
        
        # Ceiling left edge (first column)
        ceiling_edge = self.ceiling[:, 0, :]  # Shape: (height, 3)
        
        # Sample points uniformly
        left_samples = self._sample_uniform(left_edge, num_samples)
        ceiling_samples = self._sample_uniform(ceiling_edge, num_samples)
        
        # Compute color differences
        differences = self._color_difference(left_samples, ceiling_samples)
        avg_diff = np.mean(differences)
        
        self.edge_scores['left_ceiling'] = {
            'average': avg_diff,
            'per_sample': differences,
            'max': np.max(differences),
            'std': np.std(differences)
        }
        
        return avg_diff, differences
    
    def validate_right_ceiling_edge(self, num_samples: int = 100) -> Tuple[float, np.ndarray]:
        """
        Validate continuity between right wall and ceiling.
        
        The right wall's top edge should align with the ceiling's right edge.
        
        Args:
            num_samples: Number of points to sample along the edge
        
        Returns:
            Tuple of (average_difference, per_sample_differences)
        """
        # Right wall top edge (last row)
        right_edge = self.right[-1, :, :]  # Shape: (width, 3)
        
        # Ceiling right edge (first row)
        ceiling_edge = self.ceiling[0, :, :]  # Shape: (width, 3)
        
        # Sample points uniformly
        right_samples = self._sample_uniform(right_edge, num_samples)
        ceiling_samples = self._sample_uniform(ceiling_edge, num_samples)
        
        # Compute color differences
        differences = self._color_difference(right_samples, ceiling_samples)
        avg_diff = np.mean(differences)
        
        self.edge_scores['right_ceiling'] = {
            'average': avg_diff,
            'per_sample': differences,
            'max': np.max(differences),
            'std': np.std(differences)
        }
        
        return avg_diff, differences
    
    def validate_left_right_corner(self, num_samples: int = 100) -> Tuple[float, np.ndarray]:
        """
        Validate continuity at the vertical corner between left and right walls.
        
        The left wall's right edge should align with the right wall's left edge.
        
        Args:
            num_samples: Number of points to sample along the edge
        
        Returns:
            Tuple of (average_difference, per_sample_differences)
        """
        # Left wall right edge (last column)
        left_edge = self.left[:, -1, :]  # Shape: (height, 3)
        
        # Right wall left edge (first column)
        right_edge = self.right[:, 0, :]  # Shape: (height, 3)
        
        # Sample points uniformly
        left_samples = self._sample_uniform(left_edge, num_samples)
        right_samples = self._sample_uniform(right_edge, num_samples)
        
        # Compute color differences
        differences = self._color_difference(left_samples, right_samples)
        avg_diff = np.mean(differences)
        
        self.edge_scores['left_right_corner'] = {
            'average': avg_diff,
            'per_sample': differences,
            'max': np.max(differences),
            'std': np.std(differences)
        }
        
        return avg_diff, differences
    
    def validate_all_edges(self, num_samples: int = 100) -> Dict[str, Dict]:
        """
        Validate all edge continuities.
        
        Args:
            num_samples: Number of points to sample per edge
        
        Returns:
            Dictionary with validation results for all edges
        """
        print("Validating edge continuity...")
        
        # Validate each edge
        left_ceiling_avg, _ = self.validate_left_ceiling_edge(num_samples)
        print(f"  Left-Ceiling edge: avg difference = {left_ceiling_avg:.2f}")
        
        right_ceiling_avg, _ = self.validate_right_ceiling_edge(num_samples)
        print(f"  Right-Ceiling edge: avg difference = {right_ceiling_avg:.2f}")
        
        left_right_avg, _ = self.validate_left_right_corner(num_samples)
        print(f"  Left-Right corner: avg difference = {left_right_avg:.2f}")
        
        return self.edge_scores
    
    def _sample_uniform(self, edge_pixels: np.ndarray, num_samples: int) -> np.ndarray:
        """
        Sample uniformly along an edge.
        
        Args:
            edge_pixels: Edge pixel array (N, 3)
            num_samples: Number of samples to take
        
        Returns:
            Sampled pixels (num_samples, 3)
        """
        n = len(edge_pixels)
        if n < num_samples:
            # If edge is shorter than requested samples, use all pixels
            return edge_pixels
        
        # Sample uniformly
        indices = np.linspace(0, n - 1, num_samples, dtype=int)
        return edge_pixels[indices]
    
    def _color_difference(self, pixels1: np.ndarray, pixels2: np.ndarray) -> np.ndarray:
        """
        Compute per-pixel color difference (Euclidean distance in RGB space).
        
        Args:
            pixels1: First set of pixels (N, 3)
            pixels2: Second set of pixels (N, 3)
        
        Returns:
            Array of differences (N,)
        """
        # Euclidean distance in RGB space
        diff = np.linalg.norm(pixels1.astype(float) - pixels2.astype(float), axis=1)
        return diff
    
    def get_overall_score(self) -> float:
        """
        Compute overall continuity score (lower is better).
        
        Returns:
            Overall average difference across all edges
        """
        if not self.edge_scores:
            raise ValueError("No edge validations performed yet")
        
        avg_diffs = [score['average'] for score in self.edge_scores.values()]
        return np.mean(avg_diffs)
    
    def generate_report(self) -> str:
        """
        Generate a human-readable validation report.
        
        Returns:
            Report string
        """
        if not self.edge_scores:
            return "No validation results available"
        
        report = []
        report.append("=" * 60)
        report.append("Edge Continuity Validation Report")
        report.append("=" * 60)
        
        for edge_name, scores in self.edge_scores.items():
            report.append(f"\n{edge_name.replace('_', ' ').title()}:")
            report.append(f"  Average difference: {scores['average']:.2f}")
            report.append(f"  Maximum difference: {scores['max']:.2f}")
            report.append(f"  Std deviation:      {scores['std']:.2f}")
            
            # Interpret the score
            if scores['average'] < 5:
                status = "✓ EXCELLENT"
            elif scores['average'] < 15:
                status = "✓ GOOD"
            elif scores['average'] < 30:
                status = "⚠ FAIR"
            else:
                status = "✗ POOR"
            report.append(f"  Status: {status}")
        
        overall = self.get_overall_score()
        report.append(f"\nOverall Score: {overall:.2f}")
        
        if overall < 5:
            report.append("Status: ✓ EXCELLENT - Edges are nearly seamless")
        elif overall < 15:
            report.append("Status: ✓ GOOD - Minor discontinuities may be visible")
        elif overall < 30:
            report.append("Status: ⚠ FAIR - Noticeable discontinuities present")
        else:
            report.append("Status: ✗ POOR - Significant discontinuities, check geometry")
        
        report.append("=" * 60)
        
        return "\n".join(report)
    
    def visualize_edge_differences(self, edge_name: str) -> np.ndarray:
        """
        Create a visualization of edge differences.
        
        Args:
            edge_name: Name of the edge to visualize
        
        Returns:
            Visualization image
        """
        if edge_name not in self.edge_scores:
            raise ValueError(f"No validation for edge: {edge_name}")
        
        differences = self.edge_scores[edge_name]['per_sample']
        
        # Create a bar chart visualization
        height = 200
        width = len(differences) * 4
        vis = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Normalize differences to [0, height]
        max_diff = max(differences.max(), 1)
        normalized = (differences / max_diff * (height - 20)).astype(int)
        
        # Draw bars
        for i, val in enumerate(normalized):
            x = i * 4
            y_start = height - val - 10
            y_end = height - 10
            
            # Color based on magnitude (green = good, red = bad)
            if differences[i] < 5:
                color = (0, 255, 0)  # Green
            elif differences[i] < 15:
                color = (0, 255, 255)  # Yellow
            elif differences[i] < 30:
                color = (0, 165, 255)  # Orange
            else:
                color = (0, 0, 255)  # Red
            
            cv2.line(vis, (x, y_start), (x, y_end), color, 2)
        
        return vis


def validate_panels(panel_dir: str = "output") -> str:
    """
    Load panels from directory and validate.
    
    Args:
        panel_dir: Directory containing panel images
    
    Returns:
        Validation report string
    """
    from pathlib import Path
    
    panel_dir = Path(panel_dir)
    
    # Load panels
    left = cv2.imread(str(panel_dir / "left.png"))
    right = cv2.imread(str(panel_dir / "right.png"))
    top = cv2.imread(str(panel_dir / "top.png"))
    
    if left is None or right is None or top is None:
        raise FileNotFoundError(f"Could not load all panels from {panel_dir}")
    
    panels = {
        "left_wall": left,
        "right_wall": right,
        "ceiling": top
    }
    
    # Validate
    validator = EdgeValidator(panels)
    validator.validate_all_edges()
    
    return validator.generate_report()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        panel_dir = sys.argv[1]
    else:
        panel_dir = "output"
    
    try:
        report = validate_panels(panel_dir)
        print(report)
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)
