/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#ifndef CUDA_RASTERIZER_FORWARD_H_INCLUDED
#define CUDA_RASTERIZER_FORWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

namespace FORWARD
{
	// Perform initial steps for each Gaussian prior to rasterization.
	void preprocess(int P, int D, int M,
		const float* orig_points,
		const glm::vec3* scales,
		const float scale_modifier,
		const glm::vec4* rotations,
		const float* opacities,
		const float* shs,
		bool* clamped,
		const float* cov3D_precomp,
		const float* colors_precomp,
		const float* viewmatrix,
		const float* projmatrix,
		const glm::vec3* cam_pos,
		const int W, int H,
		const float focal_x, float focal_y,
		const float tan_fovx, float tan_fovy,
		int* radii,
		float2* points_xy_image,
		float* depths,
		float* cov3Ds,
		float* colors,
		float4* conic_opacity,
		const dim3 grid,
		uint32_t* tiles_touched,
		bool prefiltered);
	
	void preprocessPso(int P, int D, int M,
		const float* means3D,
		const glm::vec3* scales,
		const float scale_modifier,
		const glm::vec4* rotations,
		const float* opacities,
		const float* shs,
		bool* clamped,
		const float* cov3D_precomp,
		const float* colors_precomp,
		const float* viewmatrix,
		const float* projmatrix,
		const float* projmatrix_raw,
		const glm::vec3* cam_pos,
		const int width, const int height,
		const float focal_x, float focal_y,
		const float tan_fovx, float tan_fovy,
		int* radii,
		float2* means2D,
		float* depths,
		float* cov3Ds,
		float* rgb,
		float4* conic_opacity,
		const dim3 tile_grid,
		uint32_t* tiles_touched,
		bool prefiltered);
	
	void preprocessRgb(int P, int D, int M,
		const float* means3D,
		const float* shs,
		bool* clamped,
		const float* colors_precomp,
		const glm::vec3* cam_pos,
		float* rgb);
	
	void preprocessPS(int P,
		const float* viewmatrix,
		const float* projmatrix_raw,
		const float* pso,
		const int pso_idx,
		const float* search_size,
		float* viewmatrix_ps,
		float* projmatrix_ps);

	// Main rasterization method.
	void render(
		const dim3 grid, dim3 block,
		const uint2* ranges,
		const uint32_t* point_list,
		int W, int H,
		const float2* points_xy_image,
		const float* features,
		const float4* conic_opacity,
		float* final_T,
		uint32_t* n_contrib,
		const float* bg_color,
		float* out_color,
		const float* depth,
	    float* out_depth,
		float* out_opacity,
		int* n_touched);
	
	
	void renderPso(
		const dim3 grid, const dim3 block,
		int P,
		const uint2* ranges,
		const uint32_t* point_list,
		const int* num_rendered,
		int W, int H,
		const float2* points_xy_image,
		const float* features,
		const float4* conic_opacity,
		float* final_T,
		uint32_t* n_contrib,
		const float* bg_color,
		float* out_color,
		const float* depth,
	    float* out_depth,
		float* out_opacity,
		int* n_touched);
}


#endif