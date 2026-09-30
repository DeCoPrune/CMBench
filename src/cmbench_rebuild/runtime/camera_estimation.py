"""Deterministic recent-camera recovery from already observed context pixels."""
from __future__ import annotations

from typing import Any


def estimate_tail_rotation(video_cthw: Any, *, tail_frames: int = 40, stride: int = 4) -> dict[str, Any]:
    import cv2
    import numpy as np
    import torch
    from scipy.spatial.transform import Rotation

    if video_cthw.ndim != 4 or int(video_cthw.shape[0]) != 3:
        raise ValueError(f"expected [3,T,H,W], got {tuple(video_cthw.shape)}")
    if int(tail_frames) <= 0 or int(stride) <= 0:
        raise ValueError("tail_frames and stride must be positive")
    # RANSAC consumes OpenCV's global RNG. Pin it for repeatable camera input.
    cv2.setRNGSeed(0)
    rgb = (((video_cthw[:, -int(tail_frames):].clamp(-1, 1) + 1.0) * 127.5).to(torch.uint8).permute(1, 2, 3, 0).cpu().numpy())
    gray = [cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in rgb]
    height, width = gray[0].shape
    intrinsic = np.array([
        [502.9116 / 832.0 * width, 0.0, 415.77786 / 832.0 * width],
        [0.0, 503.10812 / 480.0 * height, 239.77779 / 480.0 * height],
        [0.0, 0.0, 1.0],
    ])
    detector = cv2.SIFT_create(nfeatures=4000)
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    vectors, pairs_used = [], []
    for start in range(0, len(gray) - int(stride), int(stride)):
        stop = start + int(stride)
        keypoints_a, descriptors_a = detector.detectAndCompute(gray[start], None)
        keypoints_b, descriptors_b = detector.detectAndCompute(gray[stop], None)
        if descriptors_a is None or descriptors_b is None:
            continue
        matches = matcher.knnMatch(descriptors_a, descriptors_b, k=2)
        good = [first for first, second in matches if first.distance < 0.7 * second.distance]
        if len(good) < 12:
            continue
        points_a = np.float32([keypoints_a[item.queryIdx].pt for item in good])
        points_b = np.float32([keypoints_b[item.trainIdx].pt for item in good])
        homography, mask = cv2.findHomography(points_a, points_b, cv2.RANSAC, ransacReprojThreshold=2.0)
        inliers = int(mask.sum()) if mask is not None else 0
        if homography is None or inliers < 10:
            continue
        raw_rotation = np.linalg.inv(intrinsic) @ homography @ intrinsic
        u, _, vt = np.linalg.svd(raw_rotation)
        rotation = u @ vt
        if np.linalg.det(rotation) < 0:
            u[:, -1] *= -1
            rotation = u @ vt
        vectors.append(Rotation.from_matrix(rotation.T).as_rotvec())
        pairs_used.append({"tail_pair": [start, stop], "matches": len(good), "inliers": inliers})
    if not vectors:
        return {
            "status": "static_fallback_no_robust_homography",
            "tail_frames": min(int(tail_frames), int(video_cthw.shape[1])),
            "pixel_stride": int(stride),
            "c2w_rotvec_radians_per_latent": [0.0, 0.0, 0.0],
            "robust_pairs": [],
            "opencv_rng_seed": 0,
        }
    median = np.median(np.stack(vectors), axis=0)
    return {
        "status": "estimated",
        "tail_frames": min(int(tail_frames), int(video_cthw.shape[1])),
        "pixel_stride": int(stride),
        "c2w_rotvec_radians_per_latent": median.tolist(),
        "c2w_euler_xyz_degrees_per_latent": Rotation.from_rotvec(median).as_euler("xyz", degrees=True).tolist(),
        "robust_pairs": pairs_used,
        "opencv_rng_seed": 0,
    }
