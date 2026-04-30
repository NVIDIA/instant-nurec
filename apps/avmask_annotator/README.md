# Mask Annotator Tool

A tool for annotating and editing masks in camera frames, with support from Segment Anything 2 (SAM2) model.

## Features

- Browse and load images from a folder
- Use SAM2 model for automatic mask generation
- Refine/Edit masks with brush and morphological tools
- Save and export annotated masks
- Real-time mask overlay visualization

## Prerequisites

- Python 3.11 or higher
- CUDA-capable GPU (recommended for SAM2 model), works well on CPU as well.
- Sufficient disk space for model weights and temporary files

## Required Python Packages

The following packages are required:

- PySide6
- torch>=2.5.1 (for sam2 features)
- torchvision>=0.20.1 (for sam2 features)
- numpy
- opencv-python
- Pillow

## Usage

Run the mask annotator:

```bash
bazel run //apps/avmask_annotator:mask_annotator

#CPU backend
bazel run //apps/avmask_annotator:mask_annotator_cpu

#GPU backend
bazel run //apps/avmask_annotator:mask_annotator_gpu

```

### Basic Usage

1. Launch the application
2. Use the "Image Loading" tab to:

   - Browse and select a folder with images
   - Select a image from the list
   - Click "Load Selected Image for Annotation" button

3. Use the "Annotation" tab to:

   - SAM2 tools for automatic mask generation

     1. Foreground points (green) are for marking region of interest (mark points on left panel)
     2. Background points (red) are for excluding a region

   - Edit masks using the provided tools:

     - Brush tool for manual editing
     - Erode/Dilate for morphological operations (current brush size defines kernel size for these ops)

   - Left panel is for marking SAM2 prompt points, right panel is for manual mask editing. Right panel also shows the mask overlay
   - Save masks using the "Save Mask" button

## Notes

- Some features like assisted mode may be disabled if SAM2 is not properly installed.
- If sam2 is missing install it with

```bash
 pip install sam2==1.1.0
```
