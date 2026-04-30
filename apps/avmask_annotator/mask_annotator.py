# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import os
import sys

from datetime import datetime

import click
import cv2
import numpy as np
import requests

from PIL import Image
from tqdm import tqdm


HAS_QT = True
try:
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
    from PySide6.QtWidgets import (
        QApplication,
        QFileDialog,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QListWidget,
        QListWidgetItem,
        QMainWindow,
        QPushButton,
        QSlider,
        QTabWidget,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )

except Exception as e:  # noqa: BLE001 – catch *any* failure (missing libxkbcommon, etc.)
    # Importing Qt failed – most likely we are running in a headless environment
    # (e.g. CI) where the native libraries cannot be loaded.  Define **minimal**
    # stub classes/constants so that the rest of the file can be imported and
    # unit-tests that do not touch the GUI elements can still run.

    print(f"Qt import failed with error: {e}")
    HAS_QT = False

    class _Stub:
        """Very small stub object that discards all attribute access/calls."""

        def __init__(self, *_, **__):
            pass

        def __getattr__(self, _):  # noqa: D401
            return _Stub()

        def __call__(self, *_, **__):
            return _Stub()

        def __iter__(self):
            return iter(())

        def __bool__(self):  # Always false to avoid accidental trueness
            return False

    # Widgets & helpers
    QApplication = QMainWindow = QWidget = QLabel = QVBoxLayout = QHBoxLayout = QGridLayout = QPushButton = _Stub
    QSlider = QTabWidget = QTextEdit = QListWidget = QListWidgetItem = QGroupBox = QFileDialog = _Stub

    # Gui & core modules
    QPixmap = QImage = QPainter = QPen = QColor = _Stub

    class _QtConsts:  # lightweight container for Qt enum constants
        def __getattr__(self, _):
            return 0

    Qt = _QtConsts()
    QPoint = _Stub

    # NOTE: The GUI will obviously not work without real Qt; we guard against
    # accidental execution later in `run_mask_annotator`.

# Version information
__version__ = "0.1.0"
__date__ = "2025-04-17"
__description__ = "A tool for annotating and editing masks in camera frames"


# Functions for downloading model checkpoints
def download_file(url, destination):
    """Download a file with progress bar"""
    try:
        print(f"Downloading {os.path.basename(destination)} from {url}...")
        response = requests.get(url, stream=True)
        file_size = int(response.headers.get("content-length", 0))

        # Create parent directories if they don't exist
        os.makedirs(os.path.dirname(destination), exist_ok=True)

        desc = f"Downloading {os.path.basename(destination)}"
        progress = tqdm(total=file_size, unit="B", unit_scale=True, desc=desc)

        with open(destination, "wb") as f:
            for data in response.iter_content(chunk_size=1024):
                progress.update(len(data))
                f.write(data)
        progress.close()

        if file_size != 0 and progress.n != file_size:
            print(f"Download incomplete: {progress.n}/{file_size} bytes downloaded")
            return False

        print(f"Download complete: {destination}")
        return True
    except Exception as e:
        print(f"Error downloading file: {str(e)}")
        return False


def check_and_download_checkpoints():
    """Check if model checkpoints exist and download them if missing"""
    try:
        # Define base directory and checkpoint paths
        script_dir = os.path.dirname(os.path.abspath(__file__))
        checkpoints_dir = os.path.join(script_dir, "checkpoints")

        # Create checkpoints directory if it doesn't exist
        os.makedirs(checkpoints_dir, exist_ok=True)

        # Define checkpoint files and their download URLs
        checkpoints = {
            "sam2_hiera_large.pt": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt"
        }

        # Check and download each checkpoint
        for checkpoint_file, url in checkpoints.items():
            checkpoint_path = os.path.join(checkpoints_dir, checkpoint_file)

            if os.path.exists(checkpoint_path):
                print(f"✓ {checkpoint_file} already exists in {checkpoints_dir}")
            else:
                print(f"⟳ {checkpoint_file} not found. Downloading...")
                success = download_file(url, checkpoint_path)
                if success:
                    print(f"✓ Downloaded {checkpoint_file} successfully")
                else:
                    print(f"⚠ Failed to download {checkpoint_file}")

    except Exception as e:
        print(f"Error checking for checkpoints: {str(e)}")
        print("Continuing without checkpoint verification...")


# Import SAM2 if available
try:
    import sam2
    import torch

    from sam2.build_sam import build_sam2

    HAS_SAM2 = True
except ImportError:
    HAS_SAM2 = False
    print("SAM2 not available. Some features will be disabled.")


class MaskAnnotator(QMainWindow):
    def __init__(self, window_size, window_pos, device=None):
        super().__init__()
        self.device = device
        self.window_size = window_size
        self.window_pos = window_pos
        self.initUI()
        # Create initial blank images
        self.create_blank_images()

        # Initialize SAM2 model if available
        self.sam_model = None
        self.sam_predictor = None
        if HAS_SAM2:
            self.initialize_sam2()

        # Points for SAM2 prompting
        self.foreground_points = []
        self.background_points = []
        self.current_point_type = "foreground"  # or "background"

        # Initialize display overlay for showing points
        self.display_overlay = None

        # Keep track of loaded images
        self.loaded_images = []
        self.current_image_path = None

    def initialize_sam2(self):
        """Initialize the SAM2 model"""
        try:
            if not HAS_SAM2:
                self.log_status("SAM2 is not available. Please install it first.", error=True)
                return

            self.log_status("Initializing SAM2 model...")

            # Run the checkpoint check
            self.log_status("Checking for required model checkpoints...")
            check_and_download_checkpoints()

            # Use the device specified in the command line if provided
            if self.device:
                if self.device == "cuda" and not torch.cuda.is_available():
                    self.log_status("CUDA requested but not available. Falling back to CPU.", error=True)
                    device = torch.device("cpu")
                elif self.device == "mps" and not (
                    hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
                ):
                    self.log_status("MPS requested but not available. Falling back to CPU.", error=True)
                    device = torch.device("cpu")
                else:
                    device = torch.device(self.device)
                    self.log_status(f"Using device: {device} (specified by command line)")
            else:
                # Auto-detect the best available device
                if torch.cuda.is_available():
                    device = torch.device("cuda")
                    self.log_status(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    device = torch.device("mps")
                    self.log_status("Using Apple MPS (Metal Performance Shaders)")
                else:
                    device = torch.device("cpu")
                    self.log_status("CUDA not available, using CPU (this may be slower)")

            # Try to find the SAM2 config
            config_path = "sam2_hiera_l.yaml"  # None

            # Try to find the SAM2 checkpoint
            checkpoint_paths = [
                "checkpoints/sam2_hiera_large.pt",  # Local directory (preferred)
                "../checkpoints/sam2_hiera_large.pt",  # Default path
                "sam2/checkpoints/sam2_hiera_large.pt",
                os.path.expanduser("~/.cache/sam2/sam2_hiera_large.pt"),  # User cache
            ]

            checkpoint_path = None
            for path in checkpoint_paths:
                if os.path.exists(path):
                    checkpoint_path = path
                    break

            if checkpoint_path is None:
                # Try to download the checkpoint automatically
                self.log_status("SAM2 checkpoint not found. Attempting to download automatically...", error=True)
                try:
                    # Use the integrated download function
                    script_dir = os.path.dirname(os.path.abspath(__file__))
                    checkpoint_path = os.path.join(script_dir, "checkpoints", "sam2_hiera_large.pt")

                    if not os.path.exists(checkpoint_path):
                        url = "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt"
                        success = download_file(url, checkpoint_path)
                        if not success:
                            self.log_status(
                                "Automatic download failed. Please download the checkpoint manually.", error=True
                            )
                            return

                    self.log_status("Successfully found/downloaded checkpoint.")
                except Exception as e:
                    self.log_status(f"Error handling checkpoint: {str(e)}", error=True)
                    self.log_status("Please download the checkpoint manually.", error=True)
                    return

            self.log_status(f"Loading SAM2 model from {checkpoint_path}...")

            # Build the SAM2 model
            self.sam2_model = build_sam2(config_path, checkpoint_path, device=device)

            # Create the predictor
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            self.sam_predictor = SAM2ImagePredictor(self.sam2_model)

            self.log_status("SAM2 model initialized successfully.")

        except Exception as e:
            self.log_status(f"Error initializing SAM2 model: {str(e)}", error=True)
            self.sam_model = None
            self.sam_predictor = None

    def initUI(self):
        self.setWindowTitle(f"Mask Annotator v{__version__}")

        pos_x, pos_y = self.window_pos
        width, height = self.window_size
        self.setGeometry(pos_x, pos_y, width, height)

        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        # Create status panel
        self.status_panel = QTextEdit()
        self.status_panel.setReadOnly(True)
        self.status_panel.setMaximumHeight(100)
        self.status_panel.setStyleSheet("background-color: #f0f0f0; border-radius: 5px;")

        # Create tab widget
        self.tab_widget = QTabWidget()

        # Add status panel and tab widget to main layout
        layout.addWidget(self.tab_widget)
        layout.addWidget(self.status_panel)

        # Log initial status
        self.log_status("Application started. Ready.")

        # Create image loading tab
        image_loading_tab = QWidget()
        image_loading_layout = QVBoxLayout(image_loading_tab)

        # Create annotation tab
        annotation_tab = QWidget()
        annotation_layout = QVBoxLayout(annotation_tab)

        # Add tabs in the desired order (image loading first)
        self.tab_widget.addTab(image_loading_tab, "Image Loading")
        self.tab_widget.addTab(annotation_tab, "Annotation")

        # Setup image loading tab
        image_selection_group = QGroupBox("Load Images")
        image_selection_layout = QVBoxLayout()

        # Add buttons for loading images
        load_buttons_layout = QHBoxLayout()

        self.browse_button = QPushButton("Browse for Images...")
        self.browse_button.clicked.connect(self.browse_for_images)
        load_buttons_layout.addWidget(self.browse_button)

        image_selection_layout.addLayout(load_buttons_layout)

        # Add a list widget to display loaded images
        self.images_list_widget = QListWidget()
        self.images_list_widget.setMinimumWidth(400)
        self.images_list_widget.setMinimumHeight(300)
        self.images_list_widget.currentRowChanged.connect(self.update_image_info)
        image_selection_layout.addWidget(self.images_list_widget)

        # Add image preview
        self.image_preview_label = QLabel("No image selected")
        self.image_preview_label.setAlignment(Qt.AlignCenter)
        self.image_preview_label.setMinimumHeight(300)
        self.image_preview_label.setStyleSheet("background-color: #f0f0f0; border: 1px solid #ccc;")
        image_selection_layout.addWidget(self.image_preview_label)

        # Add a load button
        load_image_button = QPushButton("Load Selected Image for Annotation")
        load_image_button.clicked.connect(self.load_selected_image)
        image_selection_layout.addWidget(load_image_button)

        image_selection_group.setLayout(image_selection_layout)
        image_loading_layout.addWidget(image_selection_group)

        # Setup annotation tab
        # Add image display controls
        image_controls_layout = QHBoxLayout()

        # Add a refresh button to reload
        refresh_button = QPushButton("Reload Image")
        refresh_button.clicked.connect(self.reload_current_image)
        image_controls_layout.addWidget(refresh_button)

        image_controls_layout.addStretch()
        annotation_layout.addLayout(image_controls_layout)

        # Create image panel layout
        images_layout = QGridLayout()

        # Create labels for images
        self.overlay_image_label = QLabel()
        self.overlay_image_label.setMinimumSize(500, 500)
        self.overlay_image_label.setAlignment(Qt.AlignCenter)
        self.overlay_image_label.setStyleSheet("border: 1px solid black")

        self.editable_image_label = QLabel()
        self.editable_image_label.setMinimumSize(500, 500)
        self.editable_image_label.setAlignment(Qt.AlignCenter)
        self.editable_image_label.setStyleSheet("border: 1px solid black")

        # Add labels to grid layout
        images_layout.addWidget(self.overlay_image_label, 0, 0)
        images_layout.addWidget(self.editable_image_label, 0, 1)

        # Create drawing tools
        tools_layout = QHBoxLayout()

        # Create a group box for manual tools (renamed to Edit Tools)
        manual_tools_group = QGroupBox("Edit Tools")
        manual_tools_layout = QHBoxLayout()

        self.brush_button = QPushButton("Toggle Brush")
        self.brush_button.clicked.connect(self.toggle_brush)
        self.brush_color = Qt.white
        self.brush_size = 10
        self.is_drawing = False

        self.clear_button = QPushButton("Clear Mask")
        self.clear_button.clicked.connect(self.clear_mask)

        # Add toggle overlay button
        self.toggle_overlay_button = QPushButton("Show Overlay")
        self.toggle_overlay_button.clicked.connect(self.toggle_mask_overlay)
        self.show_overlay = False  # Track if we're showing the overlay or raw mask

        # Add save mask button
        self.save_mask_button = QPushButton("Save Mask")
        self.save_mask_button.clicked.connect(self.save_mask_image)

        # Add erode and dilate buttons
        self.erode_button = QPushButton("Erode")
        self.erode_button.clicked.connect(self.erode_mask)

        self.dilate_button = QPushButton("Dilate")
        self.dilate_button.clicked.connect(self.dilate_mask)

        # Add median filter button
        self.median_button = QPushButton("Median Filter")
        self.median_button.clicked.connect(self.median_filter)

        # Add brush size slider
        self.brush_size_slider = QSlider(Qt.Horizontal)
        self.brush_size_slider.setMinimum(1)
        self.brush_size_slider.setMaximum(50)
        self.brush_size_slider.setValue(self.brush_size)
        self.brush_size_slider.valueChanged.connect(self.update_brush_size)

        manual_tools_layout.addWidget(self.brush_button)
        manual_tools_layout.addWidget(QLabel("Brush Size:"))
        manual_tools_layout.addWidget(self.brush_size_slider)
        manual_tools_layout.addWidget(self.clear_button)
        manual_tools_layout.addWidget(self.erode_button)
        manual_tools_layout.addWidget(self.dilate_button)
        manual_tools_layout.addWidget(self.median_button)
        manual_tools_layout.addWidget(self.toggle_overlay_button)
        manual_tools_layout.addWidget(self.save_mask_button)
        manual_tools_group.setLayout(manual_tools_layout)

        # Create a group box for SAM2 tools
        sam_tools_group = QGroupBox("SAM2 Tools")
        sam_tools_layout = QHBoxLayout()

        self.point_type_button = QPushButton("Foreground Points")
        self.point_type_button.clicked.connect(self.toggle_point_type)

        self.generate_mask_button = QPushButton("Generate Mask")
        self.generate_mask_button.clicked.connect(self.generate_sam_mask)
        self.generate_mask_button.setEnabled(HAS_SAM2)

        self.clear_points_button = QPushButton("Clear Points")
        self.clear_points_button.clicked.connect(self.clear_points)

        sam_tools_layout.addWidget(self.point_type_button)
        sam_tools_layout.addWidget(self.generate_mask_button)
        sam_tools_layout.addWidget(self.clear_points_button)
        sam_tools_group.setLayout(sam_tools_layout)

        if not HAS_SAM2:
            sam_tools_group.setEnabled(False)
            sam_tools_group.setTitle("SAM2 Tools (Not Available)")

        # Add tool groups to main tools layout
        tools_layout.addWidget(manual_tools_group)
        tools_layout.addWidget(sam_tools_group)

        # Add layouts to main layout
        annotation_layout.addLayout(images_layout)
        annotation_layout.addLayout(tools_layout)

        # Enable mouse tracking for drawing
        self.editable_image_label.setMouseTracking(True)

        # Store original event handlers
        self.original_press_event = self.editable_image_label.mousePressEvent
        self.original_move_event = self.editable_image_label.mouseMoveEvent
        self.original_release_event = self.editable_image_label.mouseReleaseEvent

        # Override event handlers
        def editable_press_event(event):
            self.mousePressEvent(event)

        def editable_move_event(event):
            self.mouseMoveEvent(event)

        def editable_release_event(event):
            self.mouseReleaseEvent(event)

        self.editable_image_label.mousePressEvent = editable_press_event
        self.editable_image_label.mouseMoveEvent = editable_move_event
        self.editable_image_label.mouseReleaseEvent = editable_release_event

        # Enable mouse tracking for the overlay image (for SAM2 points)
        self.overlay_image_label.setMouseTracking(True)
        self.overlay_image_label.mousePressEvent = self.add_point

        # Initialize image arrays
        self.overlay_image = None
        self.mask_image = None
        self.original_image_np = None  # Store the original image as numpy array for SAM2

    # New methods for image loading functionality
    def browse_for_images(self):
        """Open file dialog to browse for a folder containing images"""
        folder_path = QFileDialog.getExistingDirectory(
            self, "Select Folder Containing Images", "", QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
        )

        if folder_path:
            # Define supported image extensions
            supported_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]

            # Find all supported image files in the directory
            image_files = []
            for file in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file)
                if os.path.isfile(file_path):
                    ext = os.path.splitext(file)[1].lower()
                    if ext in supported_extensions:
                        image_files.append(file_path)

            # Sort the files alphabetically for better organization
            image_files.sort()

            if image_files:
                self.add_images_to_list(image_files)
                self.log_status(f"Found {len(image_files)} images in folder: {folder_path}")
            else:
                self.log_status(f"No supported images found in folder: {folder_path}", error=True)

    def add_images_to_list(self, file_paths):
        """Add selected images to the list widget"""
        for file_path in file_paths:
            # Check if the image is already in the list
            existing_items = [
                self.images_list_widget.item(i).data(Qt.UserRole) for i in range(self.images_list_widget.count())
            ]

            if file_path not in existing_items:
                file_name = os.path.basename(file_path)
                item = QListWidgetItem(file_name)
                item.setData(Qt.UserRole, file_path)
                self.images_list_widget.addItem(item)
                self.loaded_images.append(file_path)
                self.log_status(f"Added image: {file_name}")

        # Select the first image if none is selected
        if self.images_list_widget.currentRow() == -1 and self.images_list_widget.count() > 0:
            self.images_list_widget.setCurrentRow(0)

    def toggle_brush(self):
        """Toggle brush color between white and black"""
        self.brush_color = Qt.black if self.brush_color == Qt.white else Qt.white
        brush_color_name = "White" if self.brush_color == Qt.white else "Black"
        self.brush_button.setText(f"Brush: {brush_color_name}")
        self.log_status(f"Brush color changed to {brush_color_name}")

    def update_image_info(self, index):
        """Update the image preview when an image is selected in the list"""
        if index < 0 or index >= len(self.loaded_images):
            self.image_preview_label.setText("No image selected")
            return

        file_path = self.loaded_images[index]

        try:
            # Load the image for preview
            pixmap = QPixmap(file_path)
            if pixmap.isNull():
                self.log_status(f"Failed to load image preview for: {file_path}", error=True)
                return

            # Scale the image to fit the preview area
            preview_size = self.image_preview_label.size()
            scaled_pixmap = pixmap.scaled(
                preview_size.width(), preview_size.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )

            # Display the preview
            self.image_preview_label.setPixmap(scaled_pixmap)

            # Get file info
            file_size = os.path.getsize(file_path) / 1024  # KB
            file_date = datetime.fromtimestamp(os.path.getmtime(file_path)).strftime("%Y-%m-%d %H:%M:%S")

            # Log the image info
            self.log_status(
                f"Selected image: {os.path.basename(file_path)} ({pixmap.width()}x{pixmap.height()}, {file_size:.1f} KB, {file_date})"
            )

        except Exception as e:
            self.log_status(f"Error loading image preview: {str(e)}", error=True)

    def load_selected_image(self):
        """Load the selected image for annotation"""
        index = self.images_list_widget.currentRow()
        if index < 0 or index >= len(self.loaded_images):
            self.log_status("No image selected. Please select an image to load.", error=True)
            return

        file_path = self.loaded_images[index]
        self.current_image_path = file_path

        try:
            # Load the image for annotation
            self.log_status(f"Loading image for annotation: {os.path.basename(file_path)}")

            # Load the image using PIL first to handle various formats
            pil_image = Image.open(file_path)
            np_image = np.array(pil_image.convert("RGB"))

            # Store the original image for SAM2
            self.original_image_np = np_image.copy()

            # Convert to QImage
            height, width, channel = np_image.shape
            bytes_per_line = 3 * width
            q_img = QImage(np_image.data, width, height, bytes_per_line, QImage.Format_RGB888)

            # Update the overlay image
            self.overlay_image = q_img

            # Create a new blank mask
            self.mask_image = QImage(width, height, QImage.Format_RGB32)
            self.mask_image.fill(Qt.black)

            # Clear any existing points
            self.foreground_points = []
            self.background_points = []
            self.display_overlay = None

            # Reset the overlay display state
            self.show_overlay = False
            if hasattr(self, "toggle_overlay_button"):
                self.toggle_overlay_button.setText("Show Overlay")

            # Update both displays with the same scaling
            self.update_displays_with_same_scaling()

            # Switch to annotation tab
            self.tab_widget.setCurrentIndex(1)

            self.log_status(f"Image loaded for annotation: {width}x{height}")

        except Exception as e:
            self.log_status(f"Error loading image for annotation: {str(e)}", error=True)

    def reload_current_image(self):
        """Reload the current image"""
        if not self.current_image_path or not os.path.exists(self.current_image_path):
            self.log_status("No image currently loaded to reload.", error=True)
            return

        # Store current mask if needed
        current_mask = None
        if hasattr(self, "mask_image") and self.mask_image is not None:
            current_mask = self.mask_image.copy()

        try:
            # Reload the image
            self.log_status(f"Reloading image: {os.path.basename(self.current_image_path)}")

            # Load the image using PIL first to handle various formats
            pil_image = Image.open(self.current_image_path)
            np_image = np.array(pil_image.convert("RGB"))

            # Store the original image for SAM2
            self.original_image_np = np_image.copy()

            # Convert to QImage
            height, width, channel = np_image.shape
            bytes_per_line = 3 * width
            q_img = QImage(np_image.data, width, height, bytes_per_line, QImage.Format_RGB888)

            # Update the overlay image
            self.overlay_image = q_img

            # Restore the existing mask or create a new one
            if current_mask:
                self.mask_image = current_mask
                self.log_status("Restored existing mask")
            else:
                # Create a new blank mask
                self.mask_image = QImage(width, height, QImage.Format_RGB32)
                self.mask_image.fill(Qt.black)

            # Update displays
            self.update_displays_with_same_scaling()

            self.log_status(f"Image reloaded: {width}x{height}")

        except Exception as e:
            self.log_status(f"Error reloading image: {str(e)}", error=True)

    def update_brush_size(self, value):
        self.brush_size = value

    def toggle_point_type(self):
        """Toggle between foreground and background points"""
        if self.current_point_type == "foreground":
            self.current_point_type = "background"
            self.point_type_button.setText("Background Points")
        else:
            self.current_point_type = "foreground"
            self.point_type_button.setText("Foreground Points")

        self.log_status(f"Switched to {self.current_point_type} points")

    def add_point(self, event):
        """Add a point for SAM2 prompting and visualize it on the overlay image"""
        if not hasattr(self, "original_image_np") or self.original_image_np is None:
            self.log_status("No image loaded. Please load an image first.", error=True)
            return

        # Get the position relative to the label
        label_pos = event.pos()

        # Get the pixmap size
        pixmap = self.overlay_image_label.pixmap()
        if not pixmap:
            return

        # Get the label size
        label_size = self.overlay_image_label.size()

        # Calculate the offset to center the pixmap in the label
        x_offset = (label_size.width() - pixmap.width()) / 2
        y_offset = (label_size.height() - pixmap.height()) / 2

        # Adjust the position by the offset
        adjusted_x = label_pos.x() - x_offset
        adjusted_y = label_pos.y() - y_offset

        # Check if the position is within the pixmap bounds
        if adjusted_x < 0 or adjusted_y < 0 or adjusted_x >= pixmap.width() or adjusted_y >= pixmap.height():
            return

        # Calculate the position in the original image
        image_x = int(adjusted_x * (self.original_image_np.shape[1] / pixmap.width()))
        image_y = int(adjusted_y * (self.original_image_np.shape[0] / pixmap.height()))

        # Add the point to the appropriate list
        if self.current_point_type == "foreground":
            self.foreground_points.append([image_x, image_y])
            point_color = QColor(0, 255, 0)  # Green for foreground
        else:
            self.background_points.append([image_x, image_y])
            point_color = QColor(255, 0, 0)  # Red for background

        # Create a copy of the overlay image to draw on
        if not hasattr(self, "display_overlay") or self.display_overlay is None:
            self.display_overlay = self.overlay_image.copy()

        # Draw an X marker at the point
        painter = QPainter(self.display_overlay)
        pen = QPen(point_color)
        pen.setWidth(3)
        painter.setPen(pen)

        # Draw the X (cross)
        marker_size = 10
        painter.drawLine(image_x - marker_size, image_y - marker_size, image_x + marker_size, image_y + marker_size)
        painter.drawLine(image_x - marker_size, image_y + marker_size, image_x + marker_size, image_y - marker_size)
        painter.end()

        # Update the display with the marked overlay
        pixmap = QPixmap.fromImage(self.display_overlay)
        scaled_pixmap = pixmap.scaled(
            label_size.width(), label_size.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.overlay_image_label.setPixmap(scaled_pixmap)

        self.log_status(f"Added {self.current_point_type} point at ({image_x}, {image_y})")

    def clear_points(self):
        """Clear all points for SAM2 prompting without affecting the mask image"""
        self.foreground_points = []
        self.background_points = []

        # Reset the display overlay but keep the mask image
        if hasattr(self, "display_overlay") and self.display_overlay is not None:
            # Create a fresh copy of the overlay image without points
            if hasattr(self, "overlay_image") and self.overlay_image is not None:
                self.display_overlay = self.overlay_image.copy()

                # Update the display with the clean overlay
                pixmap = QPixmap.fromImage(self.display_overlay)
                self.overlay_image_label.setPixmap(
                    pixmap.scaled(
                        self.overlay_image_label.width(),
                        self.overlay_image_label.height(),
                        Qt.KeepAspectRatio,
                        Qt.SmoothTransformation,
                    )
                )
            else:
                self.display_overlay = None

        self.log_status("Cleared all points (mask retained)")

    def generate_sam_mask(self):
        """Generate a mask using SAM2 based on the provided points"""
        if not HAS_SAM2 or not hasattr(self, "sam_predictor") or self.sam_predictor is None:
            self.log_status("SAM2 is not available or not initialized.", error=True)
            return

        if not hasattr(self, "original_image_np") or self.original_image_np is None:
            self.log_status("No image loaded. Please load an image first.", error=True)
            return

        if not self.foreground_points and not self.background_points:
            self.log_status("No points provided. Please add at least one point.", error=True)
            return

        try:
            self.log_status("Generating mask with SAM2...")

            # Set the image for the predictor
            self.sam_predictor.set_image(self.original_image_np)

            # Combine foreground and background points
            point_coords = np.array(self.foreground_points + self.background_points)
            point_labels = np.array([1] * len(self.foreground_points) + [0] * len(self.background_points))

            # Generate the mask
            masks, scores, logits = self.sam_predictor.predict(
                point_coords=point_coords, point_labels=point_labels, multimask_output=True
            )

            # Select the mask with the highest score
            mask_idx = np.argmax(scores)
            selected_mask = masks[mask_idx]

            # Convert the mask to a QImage
            height, width = selected_mask.shape
            mask_image = np.zeros((height, width, 3), dtype=np.uint8)
            mask_image[selected_mask > 0] = [255, 255, 255]  # White for the mask
            mask_image[selected_mask == 0] = [0, 0, 0]  # Black for the background

            bytes_per_line = 3 * width
            q_img = QImage(mask_image.data, width, height, bytes_per_line, QImage.Format_RGB888)

            # Update the mask image
            self.mask_image = q_img

            # Update the display
            self.update_displays_with_same_scaling()

            # Reset the overlay display state
            self.show_overlay = False

            # Update the toggle overlay button text to reflect current state
            if hasattr(self, "toggle_overlay_button"):
                self.toggle_overlay_button.setText("Show Overlay")

            self.log_status(f"Generated mask with score {scores[mask_idx]:.3f}")

        except Exception as e:
            self.log_status(f"Error generating mask: {str(e)}", error=True)

    def mousePressEvent(self, event):
        """Handle mouse press events for drawing on the mask"""
        if not hasattr(self, "mask_image") or self.mask_image is None:
            return

        # Since this is called directly from the label's event handler,
        # we know it's from the editable_image_label
        self.is_drawing = True
        self.last_point = event.pos()
        self.draw_on_mask(event.pos())

    def mouseMoveEvent(self, event):
        """Handle mouse move events for drawing on the mask"""
        # Since this is called directly from the label's event handler,
        # we know it's from the editable_image_label
        if not self.is_drawing:
            return

        self.draw_on_mask(event.pos())
        self.last_point = event.pos()

    def mouseReleaseEvent(self, event):
        """Handle mouse release events for drawing on the mask"""
        self.is_drawing = False

    def draw_on_mask(self, pos):
        """Draw on the mask image with proper coordinate mapping"""
        if not hasattr(self, "mask_image") or self.mask_image is None:
            return

        # Get the position relative to the label
        label_pos = pos

        # Get the pixmap size
        pixmap = self.editable_image_label.pixmap()
        if not pixmap:
            return

        # Get the label size
        label_size = self.editable_image_label.size()

        # Calculate the offset to center the pixmap in the label
        x_offset = (label_size.width() - pixmap.width()) / 2
        y_offset = (label_size.height() - pixmap.height()) / 2

        # Adjust the position by the offset
        adjusted_x = label_pos.x() - x_offset
        adjusted_y = label_pos.y() - y_offset

        # Check if the position is within the pixmap bounds
        if adjusted_x < 0 or adjusted_y < 0 or adjusted_x >= pixmap.width() or adjusted_y >= pixmap.height():
            return

        # Calculate the position in the original image
        image_x = int(adjusted_x * (self.mask_image.width() / pixmap.width()))
        image_y = int(adjusted_y * (self.mask_image.height() / pixmap.height()))

        # Create a painter to draw on the mask image
        painter = QPainter(self.mask_image)
        painter.setPen(QPen(self.brush_color, self.brush_size, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))

        # Draw a point at the calculated position
        painter.drawPoint(image_x, image_y)
        painter.end()

        # Update the display based on the current mode
        if self.show_overlay:
            # Regenerate the overlay
            self.toggle_mask_overlay()
            self.toggle_mask_overlay()  # Call twice to refresh the overlay
        else:
            # Update with the raw mask
            self.update_displays_with_same_scaling()

    def clear_mask(self):
        """Clear the mask image (set to black)"""
        if not hasattr(self, "mask_image") or self.mask_image is None:
            return

        # Create a new blank mask with the same dimensions
        width = self.mask_image.width()
        height = self.mask_image.height()
        self.mask_image = QImage(width, height, QImage.Format_RGB32)
        self.mask_image.fill(Qt.black)  # Fill with black instead of white

        # Update the display based on the current mode
        if self.show_overlay:
            # Reset to raw mask mode
            self.show_overlay = False
            self.toggle_overlay_button.setText("Show Overlay")
            self.update_displays_with_same_scaling()
        else:
            self.update_displays_with_same_scaling()

        self.log_status("Mask cleared (set to black)")

    def update_editable_display(self):
        """Update the editable image display with proper scaling"""
        if not hasattr(self, "mask_image") or self.mask_image is None:
            return

        # Get the label size
        label_size = self.editable_image_label.size()

        # Create a pixmap from the image
        pixmap = QPixmap.fromImage(self.mask_image)

        # Scale the pixmap to fit within the label while maintaining aspect ratio
        scaled_pixmap = pixmap.scaled(
            label_size.width(), label_size.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )

        # Set the pixmap to the label
        self.editable_image_label.setPixmap(scaled_pixmap)

        # Store the scaling factor for coordinate mapping
        self.mask_scale_factor = min(
            scaled_pixmap.width() / self.mask_image.width(), scaled_pixmap.height() / self.mask_image.height()
        )

    def create_blank_images(self):
        # Create a blank white image for the mask
        width, height = 600, 600
        self.mask_image = QImage(width, height, QImage.Format_RGB32)
        self.mask_image.fill(Qt.white)

        # Create a blank image for overlay (could be replaced with actual image later)
        self.overlay_image = QImage(width, height, QImage.Format_RGB32)
        self.overlay_image.fill(Qt.gray)

        # Update displays
        self.update_overlay_display()
        self.update_editable_display()

    def update_overlay_display(self):
        """Update the overlay image display with proper scaling"""
        if not hasattr(self, "overlay_image") or self.overlay_image is None:
            return

        # Get the label size
        label_size = self.overlay_image_label.size()

        # Create a pixmap from the image
        pixmap = QPixmap.fromImage(self.overlay_image)

        # Scale the pixmap to fit within the label while maintaining aspect ratio
        scaled_pixmap = pixmap.scaled(
            label_size.width(), label_size.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )

        # Set the pixmap to the label
        self.overlay_image_label.setPixmap(scaled_pixmap)

        # Store the scaling factor for coordinate mapping
        self.overlay_scale_factor = min(
            scaled_pixmap.width() / self.overlay_image.width(), scaled_pixmap.height() / self.overlay_image.height()
        )

    def update_displays_with_same_scaling(self):
        """Update both displays with the same scaling to maintain pixel correspondence"""
        if (
            not hasattr(self, "overlay_image")
            or self.overlay_image is None
            or not hasattr(self, "mask_image")
            or self.mask_image is None
        ):
            return

        # Get the smaller of the two label sizes to ensure both fit
        overlay_size = self.overlay_image_label.size()
        editable_size = self.editable_image_label.size()

        # Use the minimum width and height to ensure both images fit in their containers
        display_width = min(overlay_size.width(), editable_size.width())
        display_height = min(overlay_size.height(), editable_size.height())

        # Calculate the scaling factor based on the image and display sizes
        width_scale = display_width / self.overlay_image.width()
        height_scale = display_height / self.overlay_image.height()

        # Use the smaller scale to ensure the entire image is visible
        scale = min(width_scale, height_scale)

        # Calculate the new dimensions
        new_width = int(self.overlay_image.width() * scale)
        new_height = int(self.overlay_image.height() * scale)

        # Create scaled pixmaps for both images
        overlay_pixmap = QPixmap.fromImage(self.overlay_image).scaled(
            new_width, new_height, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )

        mask_pixmap = QPixmap.fromImage(self.mask_image).scaled(
            new_width, new_height, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )

        # Set the pixmaps to the labels
        self.overlay_image_label.setPixmap(overlay_pixmap)
        self.editable_image_label.setPixmap(mask_pixmap)

        # Store the scaling factor for coordinate mapping
        self.image_scale_factor = scale

        # Log the scaling information
        self.log_status(f"Images scaled by factor {scale:.2f} to maintain 1:1 pixel correspondence")

    def toggle_mask_overlay(self):
        """Toggle between showing the raw mask and the mask overlaid on the camera frame"""
        if (
            not hasattr(self, "overlay_image")
            or self.overlay_image is None
            or not hasattr(self, "mask_image")
            or self.mask_image is None
        ):
            self.log_status("No images loaded to overlay.", error=True)
            return

        self.show_overlay = not self.show_overlay

        if self.show_overlay:
            try:
                # Convert QImage to numpy array for mask
                mask_width = self.mask_image.width()
                mask_height = self.mask_image.height()

                # Create a temporary QImage with the right format
                temp_mask = self.mask_image.convertToFormat(QImage.Format_RGB888)
                mask_ptr = temp_mask.constBits()
                mask_arr = np.array(mask_ptr).reshape((mask_height, mask_width, 3))

                # Create binary mask (white = 1, black = 0)
                # Check if pixel is white (>200 for all channels)
                binary_mask = np.all(mask_arr > 200, axis=2).astype(np.uint8)

                # Convert QImage to numpy array for frame
                frame_width = self.overlay_image.width()
                frame_height = self.overlay_image.height()

                # Create a temporary QImage with the right format
                temp_frame = self.overlay_image.convertToFormat(QImage.Format_RGB888)
                frame_ptr = temp_frame.constBits()
                frame_arr = np.array(frame_ptr).reshape((frame_height, frame_width, 3))

                # Overlay the mask on the frame
                overlayed_frame = self.overlay_mask_on_frame(frame_arr, binary_mask)

                # Convert back to QImage
                height, width, channel = overlayed_frame.shape
                bytes_per_line = 3 * width
                overlayed_qimg = QImage(overlayed_frame.data, width, height, bytes_per_line, QImage.Format_RGB888)

                # Store the overlayed image
                self.overlayed_mask_image = overlayed_qimg

                # Update the display with the overlayed image
                self.toggle_overlay_button.setText("Show Raw Mask")
                self.log_status("Showing mask overlay")

                # Update the display
                self.update_displays_with_overlay()
            except Exception as e:
                self.log_status(f"Error creating overlay: {str(e)}", error=True)
                self.show_overlay = False
        else:
            # Switch back to showing the raw mask
            self.toggle_overlay_button.setText("Show Overlay")
            self.log_status("Showing raw mask")

            # Update the display
            self.update_displays_with_same_scaling()

    def update_displays_with_overlay(self):
        """Update displays with the overlay image instead of the raw mask"""
        if (
            not hasattr(self, "overlay_image")
            or self.overlay_image is None
            or not hasattr(self, "overlayed_mask_image")
            or self.overlayed_mask_image is None
        ):
            return

        # Get the smaller of the two label sizes to ensure both fit
        overlay_size = self.overlay_image_label.size()
        editable_size = self.editable_image_label.size()

        # Use the minimum width and height to ensure both images fit in their containers
        display_width = min(overlay_size.width(), editable_size.width())
        display_height = min(overlay_size.height(), editable_size.height())

        # Calculate the scaling factor based on the image and display sizes
        width_scale = display_width / self.overlay_image.width()
        height_scale = display_height / self.overlay_image.height()

        # Use the smaller scale to ensure the entire image is visible
        scale = min(width_scale, height_scale)

        # Calculate the new dimensions
        new_width = int(self.overlay_image.width() * scale)
        new_height = int(self.overlay_image.height() * scale)

        # Create scaled pixmaps for both images
        overlay_pixmap = QPixmap.fromImage(self.overlay_image).scaled(
            new_width, new_height, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )

        overlayed_mask_pixmap = QPixmap.fromImage(self.overlayed_mask_image).scaled(
            new_width, new_height, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )

        # Set the pixmaps to the labels
        self.overlay_image_label.setPixmap(overlay_pixmap)
        self.editable_image_label.setPixmap(overlayed_mask_pixmap)

        # Store the scaling factor for coordinate mapping
        self.image_scale_factor = scale

    def save_mask_image(self):
        """Save the current mask image to a file"""
        if not hasattr(self, "mask_image") or self.mask_image is None:
            self.log_status("No mask image to save.", error=True)
            return

        try:
            # Create a default filename based on the original image
            default_dir = ""
            default_filename = "mask.png"

            if self.current_image_path:
                # Get the directory of the current image
                default_dir = os.path.dirname(self.current_image_path)

                # Get the filename without extension
                base_filename = os.path.splitext(os.path.basename(self.current_image_path))[0]

                # Create the default mask filename
                default_filename = f"{base_filename}_mask.png"

            # Create the full default path
            default_path = os.path.join(default_dir, default_filename)

            # First, ask the user for the save location
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Save Mask Image", default_path, "PNG Files (*.png);;All Files (*)"
            )

            if not file_path:
                # User cancelled the dialog
                return

            # Ensure the file has a .png extension
            if not file_path.lower().endswith(".png"):
                file_path += ".png"

            # Convert the mask to a binary image (black and white only)
            # First convert to RGB888 format
            temp_mask = self.mask_image.convertToFormat(QImage.Format_RGB888)

            # Convert to numpy array
            mask_width = temp_mask.width()
            mask_height = temp_mask.height()
            mask_ptr = temp_mask.constBits()
            mask_arr = np.array(mask_ptr).reshape((mask_height, mask_width, 3))

            # Force binary mask (white = 1, black = 0)
            binary_mask = np.all(mask_arr > 200, axis=2).astype(np.uint8) * 255

            # Save the binary mask
            cv2.imwrite(file_path, binary_mask)

            self.log_status(f"Mask saved to {file_path}")

        except Exception as e:
            self.log_status(f"Error saving mask: {str(e)}", error=True)

    def erode_mask(self):
        """Apply erosion morphology operation to the mask using current brush size"""
        if not hasattr(self, "mask_image") or self.mask_image is None:
            self.log_status("No mask image to erode.", error=True)
            return

        try:
            # Convert QImage to numpy array
            mask_width = self.mask_image.width()
            mask_height = self.mask_image.height()

            # Create a temporary QImage with the right format
            temp_mask = self.mask_image.convertToFormat(QImage.Format_RGB888)
            mask_ptr = temp_mask.constBits()
            mask_arr = np.array(mask_ptr).reshape((mask_height, mask_width, 3))

            # Create binary mask (white = 1, black = 0)
            binary_mask = np.all(mask_arr > 200, axis=2).astype(np.uint8)

            # Use brush size to determine kernel size (ensure it's odd)
            kernel_size = max(3, self.brush_size)
            if kernel_size % 2 == 0:  # Make sure kernel size is odd
                kernel_size += 1

            self.log_status(f"Eroding with kernel size {kernel_size}")

            # Apply erosion
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            eroded_mask = cv2.erode(binary_mask, kernel, iterations=1)

            # Convert back to RGB
            eroded_rgb = np.zeros((mask_height, mask_width, 3), dtype=np.uint8)
            eroded_rgb[eroded_mask > 0] = [255, 255, 255]  # White for mask
            eroded_rgb[eroded_mask == 0] = [0, 0, 0]  # Black for background

            # Convert to QImage
            bytes_per_line = 3 * mask_width
            q_img = QImage(eroded_rgb.data, mask_width, mask_height, bytes_per_line, QImage.Format_RGB888)

            # Update the mask image
            self.mask_image = q_img

            # Update the display based on the current mode
            if self.show_overlay:
                # Regenerate the overlay
                self.toggle_mask_overlay()
                self.toggle_mask_overlay()  # Call twice to refresh the overlay
            else:
                # Update with the raw mask
                self.update_displays_with_same_scaling()

            self.log_status(f"Mask eroded with brush size {self.brush_size}")

        except Exception as e:
            self.log_status(f"Error eroding mask: {str(e)}", error=True)

    def dilate_mask(self):
        """Apply dilation morphology operation to the mask using current brush size"""
        if not hasattr(self, "mask_image") or self.mask_image is None:
            self.log_status("No mask image to dilate.", error=True)
            return

        try:
            # Convert QImage to numpy array
            mask_width = self.mask_image.width()
            mask_height = self.mask_image.height()

            # Create a temporary QImage with the right format
            temp_mask = self.mask_image.convertToFormat(QImage.Format_RGB888)
            mask_ptr = temp_mask.constBits()
            mask_arr = np.array(mask_ptr).reshape((mask_height, mask_width, 3))

            # Create binary mask (white = 1, black = 0)
            binary_mask = np.all(mask_arr > 200, axis=2).astype(np.uint8)

            # Use brush size to determine kernel size (ensure it's odd)
            kernel_size = max(3, self.brush_size)
            if kernel_size % 2 == 0:  # Make sure kernel size is odd
                kernel_size += 1

            self.log_status(f"Dilating with kernel size {kernel_size}")

            # Apply dilation
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            dilated_mask = cv2.dilate(binary_mask, kernel, iterations=1)

            # Convert back to RGB
            dilated_rgb = np.zeros((mask_height, mask_width, 3), dtype=np.uint8)
            dilated_rgb[dilated_mask > 0] = [255, 255, 255]  # White for mask
            dilated_rgb[dilated_mask == 0] = [0, 0, 0]  # Black for background

            # Convert to QImage
            bytes_per_line = 3 * mask_width
            q_img = QImage(dilated_rgb.data, mask_width, mask_height, bytes_per_line, QImage.Format_RGB888)

            # Update the mask image
            self.mask_image = q_img

            # Update the display based on the current mode
            if self.show_overlay:
                # Regenerate the overlay
                self.toggle_mask_overlay()
                self.toggle_mask_overlay()  # Call twice to refresh the overlay
            else:
                # Update with the raw mask
                self.update_displays_with_same_scaling()

            self.log_status(f"Mask dilated with brush size {self.brush_size}")

        except Exception as e:
            self.log_status(f"Error dilating mask: {str(e)}", error=True)

    def median_filter(self):
        """Apply median filter to the mask using current brush size as kernel size"""
        if not hasattr(self, "mask_image") or self.mask_image is None:
            self.log_status("No mask image to apply median filter.", error=True)
            return

        try:
            # Convert QImage to numpy array
            mask_width = self.mask_image.width()
            mask_height = self.mask_image.height()

            # Create a temporary QImage with the right format
            temp_mask = self.mask_image.convertToFormat(QImage.Format_RGB888)
            mask_ptr = temp_mask.constBits()
            mask_arr = np.array(mask_ptr).reshape((mask_height, mask_width, 3))

            # Create binary mask (white = 1, black = 0)
            binary_mask = np.all(mask_arr > 200, axis=2).astype(np.uint8) * 255

            # Use brush size to determine kernel size (ensure it's odd)
            kernel_size = max(3, self.brush_size)
            if kernel_size % 2 == 0:  # Make sure kernel size is odd
                kernel_size += 1

            self.log_status(f"Applying median filter with kernel size {kernel_size}")

            # Apply median filter
            filtered_mask = cv2.medianBlur(binary_mask, kernel_size)

            # Convert back to RGB
            filtered_rgb = np.zeros((mask_height, mask_width, 3), dtype=np.uint8)
            filtered_rgb[filtered_mask > 0] = [255, 255, 255]  # White for mask
            filtered_rgb[filtered_mask == 0] = [0, 0, 0]  # Black for background

            # Convert to QImage
            bytes_per_line = 3 * mask_width
            q_img = QImage(filtered_rgb.data, mask_width, mask_height, bytes_per_line, QImage.Format_RGB888)

            # Update the mask image
            self.mask_image = q_img

            # Update the display based on the current mode
            if self.show_overlay:
                # Regenerate the overlay
                self.toggle_mask_overlay()
                self.toggle_mask_overlay()  # Call twice to refresh the overlay
            else:
                # Update with the raw mask
                self.update_displays_with_same_scaling()

            self.log_status(f"Median filter applied with kernel size {kernel_size}")

        except Exception as e:
            self.log_status(f"Error applying median filter: {str(e)}", error=True)

    def log_status(self, message, error=False):
        """Display a message in the status panel instead of showing a popup"""
        timestamp = datetime.now().strftime("%H:%M:%S")

        if error:
            formatted_message = f"<span style='color: red;'>[{timestamp}] ERROR: {message}</span>"
        else:
            formatted_message = f"<span style='color: #333;'>[{timestamp}] {message}</span>"

        # Append the new message
        self.status_panel.append(formatted_message)

        # Scroll to bottom to show latest message
        self.status_panel.verticalScrollBar().setValue(self.status_panel.verticalScrollBar().maximum())

        # Also print to console for debugging
        print(f"[{timestamp}] {'ERROR: ' if error else ''}{message}")

    def overlay_mask_on_frame(self, frame, mask):
        color = np.array([0, 255, 0], dtype="uint8")

        mask_img = np.where(mask[..., None], color, frame)
        out_frame = cv2.addWeighted(frame, 0.8, mask_img, 0.2, 0)

        return out_frame


# Define Click command and options
@click.command("run-mask-annotator")
@click.option(
    "--device", type=click.Choice(["cuda", "cpu", "mps"]), help="Device to use for SAM2 model (cuda, cpu, or mps)"
)
@click.option("--window-size", default="1600x800", help="Window size in format WIDTHxHEIGHT (default: 1600x800)")
@click.option("--window-pos", default="100,100", help="Window position in format X,Y (default: 100,100)")
@click.version_option(version=f"{__version__} ({__date__})", prog_name="Mask Annotator")
def run_mask_annotator(device, window_size, window_pos):
    # ------------------------------------------------------------------
    # IMPORTANT: We only *launch* the GUI if the Qt bindings are available.
    # The argument-parsing code above runs fine without Qt, which allows our
    # unit-tests (that purposefully trigger parsing errors) to execute on a
    # headless CI.  After we know the CLI parameters are valid, we check for
    # Qt and abort gracefully if it is missing.
    # ------------------------------------------------------------------

    # Display help information in the console
    print(f"Mask Annotator v{__version__}")
    print("------------------")
    print("This tool allows you to load images from a folder, generate masks, and edit them.")
    print("Features include:")
    print("- Loading and browsing images/masks from your computer")
    print("- Editing masks with brush and morphological tools")
    print("- Using SAM2 model for automatic mask generation")
    print("- Saving and exporting annotated masks")

    # Parse window size
    try:
        width, height = map(int, window_size.lower().split("x"))
        window_size = (width, height)
    except ValueError:
        print(f"Invalid window size format: {window_size}. Expected format: WIDTHxHEIGHT (e.g., 1600x800)")
        sys.exit(1)

    # Parse window position
    try:
        pos_x, pos_y = map(int, window_pos.lower().split(","))
        window_pos = (pos_x, pos_y)
    except ValueError:
        print(f"Invalid window position format: {window_pos}. Expected format: X,Y (e.g., 100,100)")
        sys.exit(1)

    # Abort gracefully on headless systems without Qt
    if not HAS_QT:
        print("ERROR: Qt libraries (PySide6) are not available – cannot launch the GUI.")
        print(
            "If you only needed the CLI parsing (help/version) that has already been displayed. "
            "For full GUI functionality please install PySide6 and required system packages, "
            "e.g. 'libxkbcommon'."
        )
        sys.exit(1)

    app = QApplication(sys.argv)
    window = MaskAnnotator(window_size=window_size, window_pos=window_pos, device=device)
    window.show()
    sys.exit(app.exec())


# Only run the command when script is executed directly, not when imported
if __name__ == "__main__":
    run_mask_annotator()
