# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import json

from pathlib import Path
from typing import Any, Optional

import click
import matplotlib
import numpy as np
import torch
import yaml

from omegaconf import OmegaConf


matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt

from matplotlib.axes import Axes
from matplotlib.figure import Figure

import nre.systems

from nre.config.parse import parse_typed_config
from nre.systems.gaussians import GaussiansSystem


def computeGaussianStatistics(system: GaussiansSystem, tile_size_m: float = 10.0):
    """
    Compute comprehensive statistics about Gaussians in the model.

    Args:
        system: The GaussiansSystem containing the trained model
        tile_size_m: Size of tiles in meters for density calculations

    Returns:
        dict: Dictionary containing all computed statistics
    """
    model = system.model
    statistics: dict[str, Any] = {
        "per_layer": {},
        "overall": {},
        "density": {},
        "semantic_classes": {},
        "trajectory": None,
    }

    totalGaussians = 0
    allPositions = []

    # Try to get semantic classes map
    semanticClassesMap = system.get_ncore_semantic_classes_map()
    idxToClassName: dict[int, str] = {}
    semanticClassCounts: dict[str, int] = {}
    if semanticClassesMap:
        # Create reverse mapping: class_idx -> class_name
        idxToClassName = {idx: name for name, idx in semanticClassesMap.items()}
        semanticClassCounts = {name: 0 for name in semanticClassesMap.keys()}

    print("\n" + "=" * 80)
    print("GAUSSIAN STATISTICS")
    print("=" * 80)

    # Collect statistics per layer
    for layerId, gaussianNode in model.gaussians_nodes.items():
        numGaussians = gaussianNode.get_num_gaussians()
        totalGaussians += numGaussians

        # Get positions for this layer
        positions = gaussianNode.positions.detach().cpu()
        allPositions.append(positions)

        # Get class labels if available (convert OmegaConf to plain list)
        classLabelsRaw = model.config.layers[layerId].class_labels or []
        classLabels = OmegaConf.to_container(classLabelsRaw) if classLabelsRaw else []

        # Compute spatial extent for this layer
        if numGaussians > 0:
            minPos = positions.min(dim=0).values
            maxPos = positions.max(dim=0).values
            extent = maxPos - minPos
            volume = extent[0].item() * extent[1].item() * extent[2].item()
            areaXY = extent[0].item() * extent[1].item()
        else:
            extent = torch.zeros(3)
            volume = 0.0
            areaXY = 0.0

        layerStats = {
            "num_gaussians": numGaussians,
            "class_labels": classLabels,
            "spatial_extent_m": {"x": extent[0].item(), "y": extent[1].item(), "z": extent[2].item()},
            "volume_m3": volume,
            "area_xy_m2": areaXY,
            "density_per_m2": numGaussians / areaXY if areaXY > 0 else 0.0,
            "density_per_m3": numGaussians / volume if volume > 0 else 0.0,
        }

        # Try to get semantic class predictions for this layer
        if semanticClassesMap and numGaussians > 0:
            try:
                semanticLogits = gaussianNode.get_extra_signal_by_key("semantic_logits")
                semanticPreds = semanticLogits.argmax(dim=1).cpu()

                # Count Gaussians per semantic class for this layer
                layerSemanticCounts = {}
                for classIdx in semanticPreds.unique():
                    classIdxInt = int(classIdx.item())
                    if classIdxInt in idxToClassName:
                        className = idxToClassName[classIdxInt]
                        count = int((semanticPreds == classIdx).sum().item())
                        layerSemanticCounts[className] = count
                        semanticClassCounts[className] += count

                layerStats["semantic_classes"] = layerSemanticCounts
            except (ValueError, KeyError, RuntimeError):
                # Semantic logits not available for this layer
                pass

        statistics["per_layer"][layerId] = layerStats

        # Print per-layer statistics
        print(f"\n--- Layer: {layerId} ---")
        print(f"  Number of Gaussians: {numGaussians:,}")
        if classLabels:
            print(f"  Class Labels: {', '.join(str(label) for label in classLabels)}")
        else:
            print("  Class Labels: (none specified)")
        print(f"  Spatial Extent (m): X={extent[0]}, Y={extent[1]}, Z={extent[2]}")
        print(f"  Area (XY plane): {areaXY} m²")
        print(f"  Volume: {volume} m³")
        if areaXY > 0:
            print(f"  Density (XY plane): {numGaussians / areaXY} Gaussians/m²")
        if volume > 0:
            print(f"  Density (3D): {numGaussians / volume} Gaussians/m³")

    # Overall statistics
    if totalGaussians > 0 and len(allPositions) > 0:
        allPositionsCat = torch.cat(allPositions, dim=0)
        minPos = allPositionsCat.min(dim=0).values
        maxPos = allPositionsCat.max(dim=0).values
        overallExtent = maxPos - minPos
        overallVolume = overallExtent[0].item() * overallExtent[1].item() * overallExtent[2].item()
        overallAreaXY = overallExtent[0].item() * overallExtent[1].item()
    else:
        allPositionsCat = None
        overallExtent = torch.zeros(3)
        overallVolume = 0.0
        overallAreaXY = 0.0
        minPos = torch.zeros(3)
        maxPos = torch.zeros(3)

    statistics["overall"] = {
        "total_gaussians": totalGaussians,
        "num_layers": len(model.gaussians_nodes),
        "spatial_extent_m": {"x": overallExtent[0].item(), "y": overallExtent[1].item(), "z": overallExtent[2].item()},
        "volume_m3": overallVolume,
        "area_xy_m2": overallAreaXY,
        "density_per_m2": totalGaussians / overallAreaXY if overallAreaXY > 0 else 0.0,
        "density_per_m3": totalGaussians / overallVolume if overallVolume > 0 else 0.0,
        "scene_extent": model.scene_extent,
    }

    print(f"\n" + "=" * 80)
    print("OVERALL STATISTICS")
    print("=" * 80)
    print(f"Total Gaussians: {totalGaussians:,}")
    print(f"Number of Layers: {len(model.gaussians_nodes)}")
    print(f"Overall Spatial Extent (m): X={overallExtent[0]}, Y={overallExtent[1]}, Z={overallExtent[2]}")
    print(f"Overall Area (XY plane): {overallAreaXY} m²")
    print(f"Overall Volume: {overallVolume} m³")
    print(f"Scene Extent (from config): {model.scene_extent} m")
    if overallAreaXY > 0:
        print(f"Overall Density (XY plane): {totalGaussians / overallAreaXY} Gaussians/m²")
    if overallVolume > 0:
        print(f"Overall Density (3D): {totalGaussians / overallVolume} Gaussians/m³")

    # Tile-based density analysis
    if totalGaussians > 0 and overallAreaXY > 0 and allPositionsCat is not None:
        print(f"\n" + "=" * 80)
        print(f"TILE-BASED DENSITY ANALYSIS (Tile size: {tile_size_m}m x {tile_size_m}m)")
        print("=" * 80)

        # Create 2D grid (XY plane)
        xMin, yMin = minPos[0].item(), minPos[1].item()
        xMax, yMax = maxPos[0].item(), maxPos[1].item()

        numTilesX = int(np.ceil((xMax - xMin) / tile_size_m))
        numTilesY = int(np.ceil((yMax - yMin) / tile_size_m))
        totalTiles = numTilesX * numTilesY

        print(f"Grid dimensions: {numTilesX} x {numTilesY} tiles ({totalTiles} total)")

        # Count Gaussians per tile and compute average scale
        tileGrid = np.zeros((numTilesX, numTilesY), dtype=np.int32)
        tileScaleSum = np.zeros((numTilesX, numTilesY), dtype=np.float32)

        xIndices = ((allPositionsCat[:, 0] - xMin) / tile_size_m).long().clamp(0, numTilesX - 1)
        yIndices = ((allPositionsCat[:, 1] - yMin) / tile_size_m).long().clamp(0, numTilesY - 1)

        # Collect scales from all layers
        allScales = []
        for _, gaussianNode in model.gaussians_nodes.items():
            scales = gaussianNode.get_scales(preactivation=False).detach().cpu()
            # Use average of the 3 scale components
            avgScales = scales.mean(dim=1)
            allScales.append(avgScales)

        allScalesCat = torch.cat(allScales, dim=0) if allScales else torch.ones(totalGaussians)

        for i in range(totalGaussians):
            xIdx = int(xIndices[i].item())
            yIdx = int(yIndices[i].item())
            tileGrid[xIdx, yIdx] += 1
            tileScaleSum[xIdx, yIdx] += allScalesCat[i].item()

        # Compute tile statistics
        occupiedTiles = (tileGrid > 0).sum()
        tilesArea = tile_size_m * tile_size_m

        tileDensities = tileGrid[tileGrid > 0] / tilesArea

        tileStats = {
            "tile_size_m": tile_size_m,
            "grid_dimensions": {"x": numTilesX, "y": numTilesY},
            "total_tiles": totalTiles,
            "occupied_tiles": int(occupiedTiles),
            "empty_tiles": totalTiles - int(occupiedTiles),
            "occupancy_rate": float(occupiedTiles) / totalTiles if totalTiles > 0 else 0.0,
            "gaussians_per_tile": {
                "min": int(tileGrid.min()),
                "max": int(tileGrid.max()),
                "mean": float(tileGrid.mean()),
                "median": float(np.median(tileGrid)),
                "std": float(tileGrid.std()),
            },
            "density_per_occupied_tile": {
                "min": float(tileDensities.min()),
                "max": float(tileDensities.max()),
                "mean": float(tileDensities.mean()),
                "median": float(np.median(tileDensities)),
                "std": float(tileDensities.std()),
            },
        }

        statistics["density"]["tile_based"] = tileStats

        print(f"Occupied tiles: {occupiedTiles} ({100.0 * occupiedTiles / totalTiles:.1f}%)")
        print(f"Empty tiles: {totalTiles - occupiedTiles}")
        print(f"\nGaussians per tile:")
        print(f"  Min: {tileGrid.min()}")
        print(f"  Max: {tileGrid.max()}")
        print(f"  Mean: {tileGrid.mean()}")
        print(f"  Median: {np.median(tileGrid)}")
        print(f"  Std: {tileGrid.std()}")
        print(f"\nDensity per occupied tile (Gaussians/m²):")
        print(f"  Min: {tileDensities.min()}")
        print(f"  Max: {tileDensities.max()}")
        print(f"  Mean: {tileDensities.mean()}")
        print(f"  Median: {np.median(tileDensities)}")
        print(f"  Std: {tileDensities.std()}")

        # Compute average scale per tile
        tileAvgScale = np.zeros_like(tileGrid, dtype=np.float32)
        mask = tileGrid > 0
        tileAvgScale[mask] = tileScaleSum[mask] / tileGrid[mask]

        # Store tile grid and scale info for heatmap generation
        statistics["density"]["tile_grid"] = tileGrid
        statistics["density"]["tile_scale"] = tileAvgScale
        statistics["density"]["grid_bounds"] = {"x_min": xMin, "y_min": yMin, "x_max": xMax, "y_max": yMax}

    # Semantic class statistics
    if semanticClassesMap:
        print(f"\n" + "=" * 80)
        print("SEMANTIC CLASS STATISTICS")
        print("=" * 80)

        # Filter out classes with zero Gaussians
        nonZeroClasses = {name: count for name, count in semanticClassCounts.items() if count > 0}

        if nonZeroClasses:
            # Sort by count (descending)
            sortedClasses = sorted(nonZeroClasses.items(), key=lambda x: x[1], reverse=True)

            print(f"Total classes with Gaussians: {len(sortedClasses)}")
            print(f"\nGaussians per semantic class:")
            for className, count in sortedClasses:
                percentage = 100.0 * count / totalGaussians if totalGaussians > 0 else 0.0
                print(f"  {className:20s}: {count:8,} ({percentage:5.2f}%)")

            statistics["semantic_classes"] = {
                "total_classes_with_gaussians": len(sortedClasses),
                "gaussians_per_class": dict(sortedClasses),
            }
        else:
            print("No semantic class information available in the model.")

    # Extract vehicle trajectory data
    rigTrajectories = None
    try:
        datasource = system.datamodule.get_datasource()
        if hasattr(datasource, "get_rig_trajectories"):
            rigTrajectories = datasource.get_rig_trajectories()  # type: ignore
    except (AttributeError, ValueError, KeyError) as e:
        print(f"Warning: Could not extract trajectory data: {e}")

    if rigTrajectories:
        # Extract ego positions from all rig trajectories
        egoPositions = []
        for rigTraj in rigTrajectories.rig_trajectories:
            # T_rig_worlds is (N, 4, 4) - extract translation (X, Y, Z)
            positions = rigTraj.T_rig_worlds[:, :3, 3].cpu().numpy()
            egoPositions.append(positions)

        if egoPositions:
            # Concatenate all trajectories
            allEgoPositions = np.concatenate(egoPositions, axis=0)

            # Transform to NRE coordinates
            T_world_to_nre, S_world_to_nre = rigTrajectories.world_to_nre.get_transformation_matrices()

            # Apply transformation: (T @ [x,y,z,1] @ S)
            onesCol = np.ones((allEgoPositions.shape[0], 1), dtype=allEgoPositions.dtype)
            homogeneousPos = np.concatenate([allEgoPositions, onesCol], axis=1)  # (N, 4)

            # Apply transformation
            transformedPos = (T_world_to_nre @ homogeneousPos.T).T  # (N, 4)
            transformedPos = transformedPos @ S_world_to_nre.T  # (N, 4)

            # Extract X, Y coordinates in NRE frame
            egoXY = transformedPos[:, :2]  # (N, 2)

            # Store trajectory data
            statistics["trajectory"] = {
                "positions_xy": egoXY,  # (N, 2) array
                "num_poses": len(egoXY),
            }

            print(f"\nExtracted vehicle trajectory: {len(egoXY)} poses")

    print("\n" + "=" * 80)

    return statistics


def generateHeatmap(
    tileGrid: np.ndarray,
    tileScale: np.ndarray,
    outputPath: Path,
    tile_size_m: float,
    gridBounds: dict,
    trajectoryXY: Optional[np.ndarray] = None,
    impactFormula: str = "density",
    zoom_threshold: int = 1,
):
    """
    Generate heatmap visualizations of Gaussian distribution with impact weighting.

    Args:
        tileGrid: 2D numpy array containing Gaussian counts per tile
        tileScale: 2D numpy array containing average Gaussian scale per tile
        outputPath: Path where the heatmap image will be saved
        tile_size_m: Size of each tile in meters
        gridBounds: Dictionary with keys 'x_min', 'y_min', 'x_max', 'y_max' for grid bounds
        trajectoryXY: Optional (N, 2) array of vehicle trajectory positions in world coordinates
    """
    print(f"\nGenerating heatmaps: {outputPath}")

    # Convert trajectory to tile coordinates if provided
    trajectoryTileCoords = None
    if trajectoryXY is not None and len(trajectoryXY) > 0:
        xMin = gridBounds["x_min"]
        yMin = gridBounds["y_min"]

        # Convert world XY to tile indices
        trajX = (trajectoryXY[:, 0] - xMin) / tile_size_m
        trajY = (trajectoryXY[:, 1] - yMin) / tile_size_m
        trajectoryTileCoords = np.stack([trajX, trajY], axis=1)

        print(f"Trajectory converted to tile coordinates: {len(trajectoryTileCoords)} points")

    # Compute Gaussian impact: sqrt(2 * log(density / 0.01)) * scale
    # This represents the effective influence area of Gaussians
    tileImpact = np.zeros_like(tileGrid, dtype=np.float32)
    mask = tileGrid > 0

    # Density per tile area (Gaussians per m²)
    tileDensity = tileGrid.astype(np.float32) / (tile_size_m * tile_size_m)

    # Compute impact based on selected formula
    if impactFormula == "sqrt2log":
        # impact = sqrt(2 * log(density / 0.01)) * scale
        logArg = np.maximum(tileDensity[mask] / 0.01, 1e-10)  # Avoid log of negative/zero
        tileImpact[mask] = np.sqrt(2 * np.log(logArg)) * tileScale[mask]
        formulaDescription = "Impact = sqrt(2*log(density/0.01)) * scale"
    else:  # density
        # impact = density
        tileImpact[mask] = np.maximum(tileDensity[mask], 0.0)
        formulaDescription = "Impact = density"

    # Get statistics
    totalGaussians = int(tileGrid.sum())
    occupiedTiles = int(mask.sum())
    totalTiles = int(tileGrid.size)
    maxDensity = int(tileGrid.max())
    maxImpact = float(tileImpact.max()) if mask.any() else 0.0

    statsText = f"Total Gaussians: {totalGaussians:,}\n"
    statsText += f"Occupied Tiles: {occupiedTiles:,}/{totalTiles:,} ({100.0 * occupiedTiles / totalTiles:.2f}%)\n"
    statsText += f"Max Density: {maxDensity:,} Gaussians/tile\n"
    statsText += f"Max Impact: {maxImpact:.2f}"

    # Find bounding box of occupied tiles for zoomed view
    # tileGrid has shape (numTilesX, numTilesY), so np.where returns (x_indices, y_indices)
    occupiedX, occupiedY = np.where(tileGrid > zoom_threshold)  # exclude only few gaussians of tiles on border
    if len(occupiedX) > 0:
        xMinOcc, xMaxOcc = occupiedX.min(), occupiedX.max()
        yMinOcc, yMaxOcc = occupiedY.min(), occupiedY.max()
        # Add 5% padding
        xPad = max(1, int((xMaxOcc - xMinOcc) * 0.05))
        yPad = max(1, int((yMaxOcc - yMinOcc) * 0.05))
        xMinOcc = max(0, xMinOcc - xPad)
        xMaxOcc = min(tileGrid.shape[0] - 1, xMaxOcc + xPad)
        yMinOcc = max(0, yMinOcc - yPad)
        yMaxOcc = min(tileGrid.shape[1] - 1, yMaxOcc + yPad)
    else:
        xMinOcc, xMaxOcc = 0, tileGrid.shape[0] - 1
        yMinOcc, yMaxOcc = 0, tileGrid.shape[1] - 1

    # Extract zoomed region
    tileImpactZoomed = tileImpact[xMinOcc : xMaxOcc + 1, yMinOcc : yMaxOcc + 1]
    tileGridZoomed = tileGrid[xMinOcc : xMaxOcc + 1, yMinOcc : yMaxOcc + 1]

    # Adjust figure size based on zoomed region aspect ratio
    zoomWidth = xMaxOcc - xMinOcc + 1
    zoomHeight = yMaxOcc - yMinOcc + 1
    aspectRatio = zoomHeight / max(zoomWidth, 1)
    figWidth = min(16, max(10, zoomWidth / 100))
    figHeight = min(16, max(8, figWidth * aspectRatio))

    fig2: Figure
    ax2: Axes
    fig2, ax2 = plt.subplots(figsize=(figWidth, figHeight))  # type: ignore

    # Use log scale of impact for visualization
    tileImpactZoomedLog = np.log10(tileImpactZoomed + 1)

    im2 = ax2.imshow(tileImpactZoomedLog.T, cmap="hot", aspect="auto", origin="lower", interpolation="nearest")
    plt.colorbar(im2, ax=ax2, label=f"log10(Impact + 1)\n{formulaDescription}")

    # Plot vehicle trajectory in zoomed view if available
    if trajectoryTileCoords is not None:
        # Adjust trajectory coordinates to zoomed region
        trajZoomedX = trajectoryTileCoords[:, 0] - xMinOcc
        trajZoomedY = trajectoryTileCoords[:, 1] - yMinOcc

        # Only plot points that are within the zoomed region
        withinBounds = (trajZoomedX >= 0) & (trajZoomedX < zoomWidth) & (trajZoomedY >= 0) & (trajZoomedY < zoomHeight)

        if np.any(withinBounds):
            trajZoomedX_filtered = trajZoomedX[withinBounds]
            trajZoomedY_filtered = trajZoomedY[withinBounds]

            ax2.plot(
                trajZoomedX_filtered,
                trajZoomedY_filtered,
                color="cyan",
                linewidth=2,
                alpha=0.8,
                label="Vehicle Trajectory",
            )
            ax2.plot(
                trajZoomedX_filtered[0],
                trajZoomedY_filtered[0],
                "go",
                markersize=8,
                label="Start",
                markeredgecolor="white",
                markeredgewidth=1.5,
            )
            ax2.plot(
                trajZoomedX_filtered[-1],
                trajZoomedY_filtered[-1],
                "ro",
                markersize=8,
                label="End",
                markeredgecolor="white",
                markeredgewidth=1.5,
            )
            ax2.legend(loc="upper right", fontsize=9, framealpha=0.8)

    ax2.set_xlabel(f"X Tile Index (offset: {xMinOcc}, each tile = {tile_size_m}m)", fontsize=10)
    ax2.set_ylabel(f"Y Tile Index (offset: {yMinOcc}, each tile = {tile_size_m}m)", fontsize=10)
    ax2.set_title("Gaussian Impact Heatmap (Zoomed to Occupied Region)", fontsize=12, fontweight="bold")

    # Update stats for zoomed view
    zoomedStats = f"Zoomed Region:\n"
    zoomedStats += f"Tiles: {zoomWidth} × {zoomHeight} = {zoomWidth * zoomHeight:,}\n"
    zoomedStats += f"Gaussians: {int(tileGridZoomed.sum()):,}\n"
    zoomedStats += f"Max Impact: {tileImpactZoomed.max():.2f}"

    ax2.text(
        0.02,
        0.98,
        zoomedStats,
        transform=ax2.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    plt.tight_layout()
    plt.savefig(outputPath, dpi=150, bbox_inches="tight")
    plt.close(fig2)

    print(f"Zoomed impact heatmap saved: {outputPath}")


@click.command("gaussian-statistics")
@click.option(
    "--config-name",
    type=str,
    help="Hydra config to load - has to contain a dataset specification",
    required=True,
)
@click.option(
    "--checkpoint-name",
    type=str,
    help="Checkpoint file name to load",
    default="last.ckpt",
    required=False,
)
@click.option(
    "--tile-size",
    type=float,
    help="Tile size in meters for density analysis (default: 10.0m)",
    default=10.0,
)
@click.option(
    "--heatmap/--no-heatmap",
    type=bool,
    help="Enable or disable heatmap generation",
    default=True,
)
@click.option(
    "--zoom-threshold",
    type=int,
    help="Zoom threshold for heatmap generation (default: 1)",
    default=1,
)
@click.option(
    "--output-file",
    type=str,
    help="Optional path to save statistics (format determined by extension: .json or .yaml/.yml)",
    required=False,
)
@click.argument("hydra-args", nargs=-1)
@torch.inference_mode()
def gaussianStatistics(
    config_name: str,
    checkpoint_name: str,
    tile_size: float,
    heatmap: bool,
    zoom_threshold: int,
    output_file: Optional[str],
    hydra_args: tuple[str],
):
    """
    Compute and display statistics about Gaussians in a trained model.
    
    This tool analyzes:
    1. Number of Gaussians per class/layer
    2. Spatial density (Gaussians per square meter)
    3. Tile-based density distribution
    
    Example usage:
        python run.py gaussian-statistics \\
            --config-name=my_config \\
            --checkpoint-name=last.ckpt \\
            --tile-size=10.0 \\
            --heatmap \\
            --output-file=stats.yaml
    """
    config = parse_typed_config(config_name=config_name, hydra_args=list(hydra_args))

    # Load checkpoint
    checkpointPath = Path(config.ckpt_dir) / checkpoint_name
    config.mode = "val"
    if config.resume is None:
        config.resume = str(checkpointPath)

    print(f"Loading checkpoint: {checkpointPath}")
    system = nre.systems.make(config.system.name, config, load_from_checkpoint=str(checkpointPath))

    if not isinstance(system, GaussiansSystem):
        raise TypeError("Only GaussiansSystem is supported")

    # Compute statistics
    statistics = computeGaussianStatistics(system, tile_size_m=tile_size)

    # Save to file if requested
    if output_file:
        outputPath = Path(output_file)
        outputPath.parent.mkdir(parents=True, exist_ok=True)

        # Generate heatmap if tile grid is available
        if "tile_grid" in statistics.get("density", {}):
            tileGrid = statistics["density"]["tile_grid"]
            tileScale = statistics["density"].get("tile_scale", np.ones_like(tileGrid, dtype=np.float32))
            gridBounds = statistics["density"].get("grid_bounds", {})

            # Extract trajectory data if available
            trajectoryXY = None
            if statistics.get("trajectory") and statistics["trajectory"].get("positions_xy") is not None:
                trajectoryXY = statistics["trajectory"]["positions_xy"]

            # Remove large arrays from statistics dict before saving (don't want huge arrays in YAML)
            del statistics["density"]["tile_grid"]
            if "tile_scale" in statistics["density"]:
                del statistics["density"]["tile_scale"]
            if "trajectory" in statistics and statistics["trajectory"]:
                del statistics["trajectory"]  # Don't save trajectory in YAML

            if heatmap:
                # Generate heatmap with same base name as output file
                heatmapPath = outputPath.parent / f"{outputPath.stem}_heatmap_density.jpg"
                generateHeatmap(
                    tileGrid,
                    tileScale,
                    heatmapPath,
                    tile_size,
                    gridBounds,
                    trajectoryXY,
                    impactFormula="density",
                    zoom_threshold=zoom_threshold,
                )
                heatmapPath = outputPath.parent / f"{outputPath.stem}_heatmap_sqrt2log.jpg"
                generateHeatmap(
                    tileGrid,
                    tileScale,
                    heatmapPath,
                    tile_size,
                    gridBounds,
                    trajectoryXY,
                    impactFormula="sqrt2log",
                    zoom_threshold=zoom_threshold,
                )

        # Determine format from extension
        suffix = outputPath.suffix.lower()
        if suffix in [".yaml", ".yml"]:
            with open(outputPath, "w") as f:
                yaml.dump(statistics, f, default_flow_style=False, sort_keys=False)
            print(f"\nStatistics saved to: {outputPath} (YAML format)")
        elif suffix == ".json":
            with open(outputPath, "w") as f:
                json.dump(statistics, f, indent=2)
            print(f"\nStatistics saved to: {outputPath} (JSON format)")
        else:
            # Default to YAML
            with open(outputPath, "w") as f:
                yaml.dump(statistics, f, default_flow_style=False, sort_keys=False)
            print(f"\nStatistics saved to: {outputPath} (YAML format - default)")


if __name__ == "__main__":
    gaussianStatistics()
