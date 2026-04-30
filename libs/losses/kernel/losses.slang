// Copyright (c) 2025 NVIDIA CORPORATION.  All rights reserved.

#include <slang_utils/extensions.slang>

// Losses modules
// Layer 0 as explained in the docs/architecture/modules-losses.md file.
//
// Flag values and block dimensions are injected at compile time via -D flags from nre/losses/slang_losses_constants.py
// These values are extracted from:
//   - nre/utils/types.py:RayFlags for RGB_LABEL, DROPPED, INVALID, SKY_SEMANTIC
//   - nre/losses/slang_losses.py for BLOCK_THREADS
//
// Expected defines:
//   RGB_LABEL, DROPPED, INVALID, SKY_SEMANTIC, DIFIXED, SYNTHETIC
//   GRID_NUM_ROWS, GRID_NUM_COLS, GRID_NUM_CHANNELS
//   BLOCK_THREADS
//
// N.b.: We cannot have a block definition for data with a different shape than BLOCK_THREAD (for the "marching threads" idea)
//       that would require a for-loop in the kernel able to force unroll because the kernel is differentiable.

[Differentiable]
float compute_grid_drift_loss(
    no_diff int ti,
    no_diff uint D_grid,
    no_diff uint H_grid,
    no_diff uint W_grid,
    no_diff float grid_drift_factor,
    DiffTensorView<float> grid,
) {
    float grid_drift_loss = 0.0;
    if (grid_drift_factor > 0) {
        int HW  = H_grid * W_grid;
        int DHW = D_grid * HW;

        int r = ti;
        int b = r / DHW; r -= b * DHW;
        int d = r / HW;  r -= d * HW;
        int h = r / W_grid;   r -= h * W_grid;
        int w = r;

        // Subtract from identity and accumulate squared error
        float sum = 0.0;

        // OPTME: consider writing a custom backward kernel that fuses this operation with
        // `compute_grid_drift_loss` and `compute_grid_total_variation_spatial`. This would allow 
        // accumulation to be done in-kernel and avoid the need for atomic loads.
        [ForceUnroll]
        for (int i = 0; i < GRID_NUM_ROWS; ++i) {
            [ForceUnroll]
            for (int j = 0; j < GRID_NUM_COLS; ++j) {
                int bij = b * GRID_NUM_ROWS * GRID_NUM_COLS + i * GRID_NUM_COLS + j;
                // Use `load` instead of `loadOnce` because `compute_grid_total_variation_spatial` also accesses this entry.
                float mij = grid.load(uint4(bij, d, h, w));
                float identity = (i == j ? 1.0 : 0.0);
                sum += (mij - identity) * (mij - identity);
            }
        }

        grid_drift_loss = sqrt(sum) * grid_drift_factor;
    }
    return grid_drift_loss;
}

[Differentiable]
float compute_grid_total_variation_spatial(
    no_diff int ti,
    no_diff uint D_grid,
    no_diff uint H_grid,
    no_diff uint W_grid,
    no_diff float tv_spatial_factor,
    DiffTensorView<float> grid,  // [B*12, D, H, W]
) {
    float tv_loss = 0.0;
    if (tv_spatial_factor > 0) {
        int HW  = H_grid * W_grid;
        int DHW = D_grid * HW;

        int r = ti;
        int b = r / DHW; r -= b * DHW;
        int d = r / HW;  r -= d * HW;
        int h = r / W_grid;   r -= h * W_grid;
        int w = r;
        
        [ForceUnroll]
        for (int c = 0; c < GRID_NUM_CHANNELS; ++c) {
            uint bc = b * GRID_NUM_CHANNELS + c;
            // Use `load` instead of `loadOnce` because `compute_grid_drift_loss` also accesses this entry.
            float v = grid.load(uint4(bc, d, h, w));

            // OPTME: consider using tiling with shared memory to improve memory locality and eliminate large access strides.
            if (d + 1 < D_grid) {
                // Use `load` instead of `loadOnce` because the v from another thread could be accessing the same entry.
                float u = grid.load(uint4(bc, d + 1, h, w));
                tv_loss += (u - v) * (u - v) * D_grid / (D_grid - 1) * tv_spatial_factor;
            }

            if (h + 1 < H_grid) {
                float u = grid.load(uint4(bc, d, h + 1, w));
                tv_loss += (u - v) * (u - v) * H_grid / (H_grid - 1) * tv_spatial_factor;
            }

            if (w + 1 < W_grid) {
                float u = grid.load(uint4(bc, d, h, w + 1));
                tv_loss += (u - v) * (u - v) * W_grid / (W_grid - 1) * tv_spatial_factor;
            }
        }
        tv_loss /= GRID_NUM_CHANNELS;
    }
    return tv_loss;
}

// ============================================================================
// DISPATCH 1: Camera Rendering Losses
// ============================================================================
// Combines RGB L1 loss and background MSE loss
// Grid sizing: max(rgb_blocks, bg_blocks) = ~2,025 blocks
// Data reuse: rgb_flags loaded once per thread for both losses
[CUDAKernel]
[Differentiable]
[AutoPyBindCUDA]
void camera_losses_kernel(
    // Dimensions
    no_diff uint B_rgb,
    no_diff uint H_rgb,
    no_diff uint W_rgb,
    // Factors (pre-computed 1/N_valid for mean reduction)
    no_diff float rgb_factor,      // -1 = disabled, 0 = no valid data, >0 = active
    no_diff float bg_factor,
    // Input tensors (non-differentiable)
    no_diff TensorView<int32_t> rgb_flags,  // [B_rgb, H_rgb, W_rgb, 1]
    no_diff TensorView<float> rgb_gt,       // [B_rgb, H_rgb, W_rgb, 3]
    // Input tensors (differentiable)
    DiffTensorView<float> rgb_pred,         // [B_rgb, H_rgb, W_rgb, 3]
    DiffTensorView<float> bg_pred,          // [N_rays = B_rgb*H_rgb*W_rgb]
    // Output tensors (differentiable for backprop)
    DiffTensorView<float> rgb_loss,         // [B_rgb, H_rgb, W_rgb]
    DiffTensorView<float> bg_loss,          // [N_rays]
) {
    int ti = cudaBlockIdx().x * BLOCK_THREADS + cudaThreadIdx().x;
    int HW = H_rgb * W_rgb;
    int z = ti / HW;

    // Early exit for out-of-bounds threads
    if (z >= B_rgb) return;

    int x = ti % W_rgb;
    int y = (ti / W_rgb) % H_rgb;

    // Load flags ONCE - only when needed by at least one active loss
    int rgb_f = 0;
    if (rgb_factor > 0 || bg_factor > 0) {
        rgb_f = rgb_flags.load(uint4(z, y, x, 0));
    }

    // RGB L1 Loss
    if (rgb_factor >= 0) {
        float rgb_l = 0.0f;
        if (rgb_factor > 0) {
            if ((rgb_f & RGB_LABEL) != 0 && (rgb_f & INVALID) == 0) {
                float3 p = rgb_pred.loadVecOnce<3>(uint4(z, y, x, 0));
                float3 g = no_diff(rgb_gt.loadVec<3>(uint4(z, y, x, 0)));
                rgb_l = (abs(p.x - g.x) + abs(p.y - g.y) + abs(p.z - g.z)) * rgb_factor;
            }
        }
        rgb_loss.storeOnce(uint3(z, y, x), rgb_l);
    }

    // Background MSE Loss (reuses rgb_f - no extra load!)
    // Exclude DIFIXED and SYNTHETIC rays to match Python BackgroundLoss
    if (bg_factor >= 0) {
        float bg_l = 0.0f;
        if (bg_factor > 0) {
            if ((rgb_f & INVALID) == 0 && (rgb_f & DIFIXED) == 0 && (rgb_f & SYNTHETIC) == 0) {
                float p = clamp(bg_pred.loadOnce(ti), 0, 1);
                float g = ((rgb_f & SKY_SEMANTIC) != 0) ? 0 : 1;
                bg_l = (p - g) * (p - g) * bg_factor;
            }
        }
        bg_loss.storeOnce(ti, bg_l);
    }
}

// ============================================================================
// DISPATCH 2: LiDAR Rendering Losses
// ============================================================================
// Combines LiDAR L1 loss, background LiDAR MSE loss, intensity MSE loss, and raydrop MSE loss
// Grid sizing: max(lidar_blocks, bg_lidar_blocks) = ~1,800 blocks
// Data reuse: lidar_flags loaded once per thread for all four losses
[CUDAKernel]
[Differentiable]
[AutoPyBindCUDA]
void lidar_losses_kernel(
    // Dimensions
    no_diff uint B_lidar,
    no_diff uint H_lidar,
    no_diff uint W_lidar,
    // Factors
    no_diff float lidar_factor,
    no_diff float bg_lidar_factor,
    no_diff float intensity_factor,
    no_diff float raydrop_factor,
    // Input tensors (non-differentiable)
    no_diff TensorView<int32_t> lidar_flags,  // [B_lidar, H_lidar, W_lidar, 1]
    no_diff TensorView<float> lidar_gt,       // [B_lidar, H_lidar, W_lidar, 1]
    no_diff TensorView<float> intensity_gt,   // [B_lidar, H_lidar, W_lidar, 1]
    no_diff TensorView<float> raydrop_gt,     // [B_lidar, H_lidar, W_lidar, 1]
    // Input tensors (differentiable)
    DiffTensorView<float> lidar_pred,         // [B_lidar, H_lidar, W_lidar, 1]
    DiffTensorView<float> bg_lidar_pred,      // [N_rays_lidar = B_lidar*H_lidar*W_lidar]
    DiffTensorView<float> intensity_pred,     // [B_lidar, H_lidar, W_lidar, 1]
    DiffTensorView<float> raydrop_pred,       // [B_lidar, H_lidar, W_lidar, 1]
    // Output tensors
    DiffTensorView<float> lidar_loss,         // [B_lidar, H_lidar, W_lidar]
    DiffTensorView<float> bg_lidar_loss,      // [N_rays_lidar]
    DiffTensorView<float> intensity_loss,     // [B_lidar, H_lidar, W_lidar]
    DiffTensorView<float> raydrop_loss,       // [B_lidar, H_lidar, W_lidar]
) {
    int ti = cudaBlockIdx().x * BLOCK_THREADS + cudaThreadIdx().x;
    int HW = H_lidar * W_lidar;
    int z = ti / HW;

    // Early exit for out-of-bounds threads
    if (z >= B_lidar) return;

    int x = ti % W_lidar;
    int y = (ti / W_lidar) % H_lidar;

    // Load flags ONCE - only when needed by at least one active loss
    int lidar_f = 0;
    if (lidar_factor > 0 || bg_lidar_factor > 0 || intensity_factor > 0 || raydrop_factor > 0) {
        lidar_f = lidar_flags.load(uint4(z, y, x, 0));
    }

    // LiDAR L1 Loss
    if (lidar_factor >= 0) {
        float lidar_l = 0.0f;
        if (lidar_factor > 0) {
            if ((lidar_f & INVALID) == 0 && (lidar_f & DROPPED) == 0) {
                float p = lidar_pred.loadOnce(uint4(z, y, x, 0));
                float g = no_diff(lidar_gt.load(uint4(z, y, x, 0)));
                lidar_l = abs(p - g) * lidar_factor;
            }
        }
        lidar_loss.storeOnce(uint3(z, y, x), lidar_l);
    }

    // Background LiDAR MSE Loss (reuses lidar_f - no extra load!)
    if (bg_lidar_factor >= 0) {
        float bg_lidar_l = 0.0f;
        if (bg_lidar_factor > 0) {
            if ((lidar_f & INVALID) == 0 && (lidar_f & DROPPED) == 0) {
                float p = clamp(bg_lidar_pred.loadOnce(ti), 0, 1);
                float g = ((lidar_f & SKY_SEMANTIC) != 0) ? 0 : 1;
                bg_lidar_l = (p - g) * (p - g) * bg_lidar_factor;
            }
        }
        bg_lidar_loss.storeOnce(ti, bg_lidar_l);
    }

    // Intensity MSE Loss (reuses lidar_f - no extra load!)
    if (intensity_factor >= 0) {
        float intensity_l = 0.0f;
        if (intensity_factor > 0) {
            if ((lidar_f & INVALID) == 0 && (lidar_f & DROPPED) == 0) {
                float p = intensity_pred.loadOnce(uint4(z, y, x, 0));
                float g = no_diff(intensity_gt.load(uint4(z, y, x, 0)));
                intensity_l = (p - g) * (p - g) * intensity_factor;
            }
        }
        intensity_loss.storeOnce(uint3(z, y, x), intensity_l);
    }

    // Raydrop MSE Loss (reuses lidar_f - no extra load!)
    // Note: raydrop only checks INVALID, not DROPPED -- raydrop predicts whether a ray should be dropped
    if (raydrop_factor >= 0) {
        float raydrop_l = 0.0f;
        if (raydrop_factor > 0) {
            if ((lidar_f & INVALID) == 0) {
                float p = raydrop_pred.loadOnce(uint4(z, y, x, 0));
                float g = no_diff(raydrop_gt.load(uint4(z, y, x, 0)));
                raydrop_l = (p - g) * (p - g) * raydrop_factor;
            }
        }
        raydrop_loss.storeOnce(uint3(z, y, x), raydrop_l);
    }
}

// ============================================================================
// DISPATCH 3: Gaussian Regularization Losses
// ============================================================================
// Combines scale, density, z-scale, and out-of-bound losses
// Grid sizing: max(scales_blocks, densities_blocks, z_scales_blocks, oob_blocks) = ~7,365 blocks
// Data locality: All gaussian-related parameters accessed together
[CUDAKernel]
[Differentiable]
[AutoPyBindCUDA]
void gaussian_losses_kernel(
    // Dimensions
    no_diff uint N_scales,
    no_diff uint N_densities,
    no_diff uint N_z_scales,
    no_diff uint out_of_bound_num_gaussians,
    // Factors
    no_diff float scale_factor,
    no_diff float density_factor,
    no_diff float z_scale_factor,
    no_diff float z_scale_threshold,
    no_diff float out_of_bound_factor,
    // Visibility
    no_diff TensorView<float> gaussian_visibility,  // [max(N_scales, N_densities)] per-Gaussian visibility (1 = visible, 0 = not)
    // Input tensors (differentiable)
    DiffTensorView<float> gaussian_scales,        // [N_scales, 3]
    DiffTensorView<float> gaussian_densities,     // [N_densities]
    DiffTensorView<float> gaussian_z_scales,      // [N_z_scales, 3]
    DiffTensorView<float> out_of_bound_positions, // [out_of_bound_num_gaussians, 3]
    no_diff TensorView<float> out_of_bound_cuboid_dims,  // [out_of_bound_num_gaussians, 3]
    // Output tensors
    DiffTensorView<float> scale_loss,         // [N_scales]
    DiffTensorView<float> density_loss,       // [N_densities]
    DiffTensorView<float> z_scale_loss,       // [N_z_scales]
    DiffTensorView<float> out_of_bound_loss,  // [out_of_bound_num_gaussians]
) {
    int ti = cudaBlockIdx().x * BLOCK_THREADS + cudaThreadIdx().x;

    // Scale Loss
    if (scale_factor >= 0) {
        if (ti < N_scales) {
            float scale_l = 0.0f;
            if (scale_factor > 0) {
                // Load pre-activation scales (log-space) and apply exp() activation
                float3 scales_preact = gaussian_scales.loadVecOnce<3>(uint2(ti, 0));
                float3 scales = float3(exp(scales_preact.x), exp(scales_preact.y), exp(scales_preact.z));

                // Since exp() is always positive, abs(exp(x)) = exp(x)
                float abs_sum = scales.x + scales.y + scales.z;
                scale_l = abs_sum * scale_factor;
                scale_l *= gaussian_visibility.load(uint(ti));
            }
            scale_loss.storeOnce(ti, scale_l);
        }
    }

    // Density Loss
    if (density_factor >= 0) {
        if (ti < N_densities) {
            float density_l = 0.0f;
            if (density_factor > 0) {
                float density = gaussian_densities.loadOnce(ti);
                density_l = abs(density) * density_factor;
                density_l *= gaussian_visibility.load(uint(ti));
            }
            density_loss.storeOnce(ti, density_l);
        }
    }

    // Z-Scale Loss
    if (z_scale_factor >= 0) {
        if (ti < N_z_scales) {
            float z_scale_l = 0.0f;
            if (z_scale_factor > 0) {
                float3 scales_preact = gaussian_z_scales.loadVecOnce<3>(uint2(ti, 0));
                float z_scale = exp(scales_preact.z);  // Apply exp() to z-component (column 2)
                z_scale_l = max(0.0f, z_scale - z_scale_threshold) * z_scale_factor;
            }
            z_scale_loss.storeOnce(ti, z_scale_l);
        }
    }

    // Out-of-Bound Loss
    if (out_of_bound_factor >= 0.0f) {
        if (ti < out_of_bound_num_gaussians) {
            float out_of_bound_l = 0.0f;
            if (out_of_bound_factor > 0.0f) {
                float3 position = out_of_bound_positions.loadVecOnce<3>(uint2(ti, 0));
                float3 half_dims = no_diff(out_of_bound_cuboid_dims.loadVec<3>(uint2(ti, 0))) * 0.5f;
                float3 loss = abs(position) - half_dims;
                float3 relu_loss = select(loss > float3(0.0f), loss, float3(0.0f));
                out_of_bound_l = (relu_loss.x + relu_loss.y + relu_loss.z) * out_of_bound_factor;
            }
            out_of_bound_loss.storeOnce(ti, out_of_bound_l);
        }
    }
}

// ============================================================================
// DISPATCH 4: Background Texture & Grid Regularization Losses
// ============================================================================
// Combines sky environment map TV loss and bilateral grid losses
// Grid sizing: max(bg_tex_blocks, grid_blocks) = ~18,432 blocks (dominated by bg_tex)
// Data reuse: Grid tensors accessed by both drift and TV computations
[CUDAKernel]
[Differentiable]
[AutoPyBindCUDA]
void background_grid_losses_kernel(
    // Background texture dimensions
    no_diff uint B_tex,
    no_diff uint D_tex,
    no_diff uint H_tex,
    no_diff uint W_tex,
    no_diff uint C_tex,
    // Grid dimensions (camera grids)
    no_diff uint B_gc,
    no_diff uint D_gc,
    no_diff uint H_gc,
    no_diff uint W_gc,
    // Grid dimensions (frame grids)
    no_diff uint B_gf,
    no_diff uint D_gf,
    no_diff uint H_gf,
    no_diff uint W_gf,
    // Factors
    no_diff float bg_tex_factor,
    no_diff float grid_drift_per_camera_factor,
    no_diff float grid_drift_per_frame_factor,
    no_diff float grid_camera_spatial_tv_factor,
    no_diff float grid_frame_spatial_tv_factor,
    // Input tensors
    DiffTensorView<float> bg_tex,         // [B_tex*D_tex, H_tex, W_tex, C_tex] or [B_tex, H_tex, W_tex, C_tex]
    DiffTensorView<float> grids_camera,   // [B_gc*GRID_NUM_ROWS*GRID_NUM_COLS, D_gc, H_gc, W_gc]
    DiffTensorView<float> grids_frame,    // [B_gf*GRID_NUM_ROWS*GRID_NUM_COLS, D_gf, H_gf, W_gf]
    // Output tensors
    DiffTensorView<float> bg_tex_loss,             // [B_tex*D_tex*H_tex*W_tex*C_tex]
    DiffTensorView<float> grids_loss,              // [B_gc*D_gc*H_gc*W_gc + B_gf*D_gf*H_gf*W_gf]
    DiffTensorView<float> grid_camera_spatial_tv_loss,  // [B_gc*D_gc*H_gc*W_gc]
    DiffTensorView<float> grid_frame_spatial_tv_loss,   // [B_gf*D_gf*H_gf*W_gf]
) {
    int ti = cudaBlockIdx().x * BLOCK_THREADS + cudaThreadIdx().x;

    // Precompute element counts ONCE (avoid inline multiplication in conditionals)
    int numel_gc = B_gc * D_gc * H_gc * W_gc;
    int numel_gf = B_gf * D_gf * H_gf * W_gf;
    int numel_tex = B_tex * D_tex * H_tex * W_tex * C_tex;

    // Sky env map TV loss
    if (ti < numel_tex) {
        float tv = 0.0f;
        if (bg_tex_factor > 0) {
            int WC = W_tex * C_tex;
            int HWC = H_tex * WC;
            int r = ti;
            int bd = r / HWC;  r -= bd * HWC;
            int h = r / WC;    r -= h * WC;
            int w = r / C_tex; r -= w * C_tex;
            int c = r;
            float center = bg_tex.load(uint4(bd, h, w, c));
            float delta;
            float norm;

            // Depth dimension (for CUBEMAP: D_tex == 6)
            if (D_tex > 1) {
                int b = bd / D_tex;
                int d = bd % D_tex;
                if (d < D_tex - 1) {
                    delta = center - bg_tex.load(uint4(b * D_tex + d + 1, h, w, c));
                    norm = 1.0 / (B_tex * (D_tex - 1) * H_tex * W_tex * C_tex);
                    tv += delta * delta * norm;
                }
            }

            // Height dimension
            if (H_tex > 1) {
                if (D_tex == 1) {  // EQUIRECTANGULAR
                    delta = center - bg_tex.load(uint4(bd, (h + 1) % H_tex, w, c));
                } else {  // CUBEMAP
                    delta = (h < H_tex - 1) ? center - bg_tex.load(uint4(bd, h + 1, w, c)) : 0.0;
                }
                norm = 1.0 / (B_tex * D_tex * (H_tex - 1) * W_tex * C_tex);
                tv += delta * delta * norm;
            }

            // Width dimension
            if (W_tex > 1) {
                if (D_tex == 1) {  // EQUIRECTANGULAR
                    delta = center - bg_tex.load(uint4(bd, h, (w + 1) % W_tex, c));
                } else {  // CUBEMAP
                    delta = (w < W_tex - 1) ? center - bg_tex.load(uint4(bd, h, w + 1, c)) : 0.0;
                }
                norm = 1.0 / (B_tex * D_tex * H_tex * (W_tex - 1) * C_tex);
                tv += delta * delta * norm;
            }
        }
        bg_tex_loss.storeOnce(ti, tv);
    }

    // Grid camera drift loss
    if (grid_drift_per_camera_factor > 0 && ti < numel_gc) {
        float loss = compute_grid_drift_loss(ti, D_gc, H_gc, W_gc, grid_drift_per_camera_factor, grids_camera);
        grids_loss.storeOnce(ti, loss);
    } else if (ti < numel_gc) {
        grids_loss.storeOnce(ti, 0.0f);
    }

    // Grid camera TV loss
    if (grid_camera_spatial_tv_factor > 0 && ti < numel_gc) {
        float loss = compute_grid_total_variation_spatial(ti, D_gc, H_gc, W_gc, grid_camera_spatial_tv_factor, grids_camera);
        grid_camera_spatial_tv_loss.storeOnce(ti, loss);
    } else if (ti < numel_gc) {
        grid_camera_spatial_tv_loss.storeOnce(ti, 0.0f);
    }

    // Grid frame drift loss
    if (grid_drift_per_frame_factor > 0 && ti < numel_gf) {
        float loss = compute_grid_drift_loss(ti, D_gf, H_gf, W_gf, grid_drift_per_frame_factor, grids_frame);
        grids_loss.storeOnce(ti + numel_gc, loss);
    } else if (ti < numel_gf) {
        grids_loss.storeOnce(ti + numel_gc, 0.0f);
    }

    // Grid frame TV loss
    if (grid_frame_spatial_tv_factor > 0 && ti < numel_gf) {
        float loss = compute_grid_total_variation_spatial(ti, D_gf, H_gf, W_gf, grid_frame_spatial_tv_factor, grids_frame);
        grid_frame_spatial_tv_loss.storeOnce(ti, loss);
    } else if (ti < numel_gf) {
        grid_frame_spatial_tv_loss.storeOnce(ti, 0.0f);
    }
}
