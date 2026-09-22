"""Matched front/side projections: show depth instead of silently dropping Z."""
import numpy as np

from ai_trainer.common_skeleton import COMMON_BONE_COLORS_BGR, COMMON_BONE_INDEX_PAIRS
from ai_trainer.render import draw_skeleton_panel, fit_transform


class SkeletonViews:
    def __init__(self, reference_coords: np.ndarray, width: int, height: int):
        self.width, self.height = width, height
        # One fixed isotropic transform for both views and both subjects.
        projections = np.concatenate((reference_coords[..., [0, 1]], reference_coords[..., [2, 1]]), axis=0)
        self.transform = fit_transform(projections, width // 2, height, margin=25, flip_y=True)

    def render(self, coords: np.ndarray, *, estimated: bool) -> np.ndarray:
        canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        for panel, (axes, title) in enumerate((([0, 1], "Front (X/Y)"), ([2, 1], "Side (Z/Y)"))):
            draw_skeleton_panel(
                canvas, (panel * (self.width // 2), 0), self.width // 2, self.height,
                self.transform(coords[:, axes]), title,
                "Estimated 3D" if estimated else "Reference 3D",
                COMMON_BONE_INDEX_PAIRS, COMMON_BONE_COLORS_BGR,
            )
        return canvas
