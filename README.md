# 3D-Consistency

Exploring geometry-guided diffusion models for multi-view consistent image generation.

## Overview
This project investigates the use of explicit 3D representations (point clouds with Gaussian splatting)
to guide pretrained diffusion models toward multi-view consistent image synthesis.

The core idea is to:
- Use a coarse 3D scene representation as a geometric prior
- Render depth / normal maps from novel views
- Condition a pretrained diffusion model on these geometric cues
- Enforce consistency across views with lightweight methods

## Dataset Construction

We construct a small synthetic multi-view dataset from publicly available 3D assets (Sketchfab).
Each object is converted to a point cloud and rendered from multiple camera viewpoints.

For each object:
- 4 views are rendered
- Each view includes:
  - a depth map
  - a known camera pose

This dataset is used to evaluate multi-view consistency of geometry-conditioned diffusion models.