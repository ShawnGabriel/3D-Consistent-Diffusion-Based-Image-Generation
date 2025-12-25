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

## Current Status
- Stable Diffusion sanity check completed
- Open3D point cloud loading and visualization working
- Repository structure initialized
