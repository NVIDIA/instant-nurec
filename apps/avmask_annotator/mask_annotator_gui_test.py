#!/usr/bin/env python
# mask_annotator_gui_test.py - Software Quality Assurance GUI test script for Mask Annotator

import os
import shutil
import tempfile

from pathlib import Path

import cv2
import numpy as np


# Set environment variables for headless operation before any Qt imports
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["QT_PLUGIN_PATH"] = ""  # Disable system plugin path
os.environ["QT_OPENGL"] = "software"  # Force software rendering

import ctypes

import pytest


try:
    ctypes.CDLL("libGL.so.1")
except OSError:
    pytest.skip("libGL.so.1 not found, skipping GUI tests", allow_module_level=True)

pytestmark = pytest.mark.gui
from PySide6.QtCore import QPoint, Qt
from PySide6.QtWidgets import QApplication, QFileDialog, QPushButton


# Mocks
class MockFileDialog:
    @staticmethod
    def getExistingDirectory(*args, **kwargs):
        return TEST_IMAGE_DIR

    @staticmethod
    def getSaveFileName(*args, **kwargs):
        return os.path.join(TEST_OUTPUT_DIR, "test_output_mask.png"), "PNG Files (*.png)"


# Create test directories and fixtures
@pytest.fixture(scope="session")
def test_dirs():
    # Create test directories
    test_root = tempfile.mkdtemp(prefix="mask_annotator_test_")
    test_image_dir = os.path.join(test_root, "images")
    test_output_dir = os.path.join(test_root, "output")

    os.makedirs(test_image_dir, exist_ok=True)
    os.makedirs(test_output_dir, exist_ok=True)

    # Create test images of different sizes
    create_test_images(test_image_dir)

    # Set global variables
    global TEST_ROOT, TEST_IMAGE_DIR, TEST_OUTPUT_DIR
    TEST_ROOT = test_root
    TEST_IMAGE_DIR = test_image_dir
    TEST_OUTPUT_DIR = test_output_dir

    yield test_root, test_image_dir, test_output_dir

    # Cleanup after tests
    shutil.rmtree(test_root)


def create_test_images(image_dir):
    """Create several test images of different shapes and colors"""
    # Small image (100x100)
    small_img = np.ones((100, 100, 3), dtype=np.uint8) * 255
    cv2.circle(small_img, (50, 50), 30, (0, 0, 255), -1)
    cv2.imwrite(os.path.join(image_dir, "small_circle.png"), small_img)

    # Medium image (500x500)
    medium_img = np.ones((500, 500, 3), dtype=np.uint8) * 255
    cv2.rectangle(medium_img, (100, 100), (400, 400), (0, 255, 0), -1)
    cv2.imwrite(os.path.join(image_dir, "medium_rectangle.png"), medium_img)

    # Large image (1000x1000)
    large_img = np.ones((1000, 1000, 3), dtype=np.uint8) * 255
    cv2.ellipse(large_img, (500, 500), (300, 200), 0, 0, 360, (255, 0, 0), -1)
    cv2.imwrite(os.path.join(image_dir, "large_ellipse.png"), large_img)


# Create a separate fixture for QApplication to ensure it's created with the right parameters
@pytest.fixture(scope="session")
def qapp():
    """Create a QApplication instance."""
    app = QApplication.instance()
    if app is None:
        # Create application instance with software rendering
        app = QApplication([""])
    yield app


@pytest.fixture
def app_fixture(qtbot, monkeypatch, test_dirs, qapp):
    """Create and set up the application for testing"""
    # Patch QFileDialog to use our test directories
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", MockFileDialog.getExistingDirectory)
    monkeypatch.setattr(QFileDialog, "getSaveFileName", MockFileDialog.getSaveFileName)

    # Import using the proper package path
    from apps.avmask_annotator.mask_annotator import MaskAnnotator

    # Create app instance with test parameters
    app = MaskAnnotator(window_size=(1200, 800), window_pos=(50, 50), device="cpu")
    qtbot.addWidget(app)
    app.show()

    # Return the app for testing
    yield app

    # Clean up
    app.close()


# Test cases
def test_app_initialization(app_fixture):
    """Test that the application initializes correctly"""
    # Check that main UI components are created
    assert app_fixture.browse_button is not None
    assert app_fixture.images_list_widget is not None
    assert app_fixture.tab_widget is not None
    assert app_fixture.overlay_image_label is not None
    assert app_fixture.editable_image_label is not None

    # Check that we start on the correct tab
    assert app_fixture.tab_widget.currentIndex() == 0  # Should start on Image Loading tab


def test_browse_and_load_images(app_fixture, qtbot):
    """Test browsing and loading images"""
    # Click browse button to load images from test directory
    qtbot.mouseClick(app_fixture.browse_button, Qt.LeftButton)

    # Wait for images to load
    qtbot.wait(1000)

    # Check that images were loaded into the list
    assert app_fixture.images_list_widget.count() == 3  # We created 3 test images

    # Select the first image
    app_fixture.images_list_widget.setCurrentRow(0)
    qtbot.wait(300)

    # Check that preview is updated
    assert app_fixture.image_preview_label.pixmap() is not None

    # Load the image for annotation
    load_button = None
    for button in app_fixture.findChildren(QPushButton):
        if "Load Selected Image for Annotation" in button.text():
            load_button = button
            break

    assert load_button is not None
    qtbot.mouseClick(load_button, Qt.LeftButton)
    qtbot.wait(500)

    # Check that we switched to annotation tab
    assert app_fixture.tab_widget.currentIndex() == 1  # Should now be on Annotation tab

    # Check that image is loaded in both panels
    assert app_fixture.overlay_image_label.pixmap() is not None
    assert app_fixture.editable_image_label.pixmap() is not None


def test_brush_tool(app_fixture, qtbot):
    """Test the brush drawing functionality"""
    # Make sure we're on the annotation tab
    app_fixture.tab_widget.setCurrentIndex(1)
    qtbot.wait(300)

    # Toggle brush to white
    if app_fixture.brush_color != Qt.white:
        qtbot.mouseClick(app_fixture.brush_button, Qt.LeftButton)
        qtbot.wait(200)

    # Change brush size
    app_fixture.brush_size_slider.setValue(20)
    qtbot.wait(200)
    assert app_fixture.brush_size == 20

    # Draw on the mask (simulate mouse events)
    center = app_fixture.editable_image_label.rect().center()
    qtbot.mousePress(app_fixture.editable_image_label, Qt.LeftButton, pos=center)

    # Draw a small line
    for i in range(10):
        new_pos = center + QPoint(i * 5, i * 5)
        qtbot.mouseMove(app_fixture.editable_image_label, pos=new_pos)
        qtbot.wait(10)

    qtbot.mouseRelease(app_fixture.editable_image_label, Qt.LeftButton, pos=new_pos)
    qtbot.wait(300)

    # Test toggling brush color
    qtbot.mouseClick(app_fixture.brush_button, Qt.LeftButton)
    qtbot.wait(200)
    assert app_fixture.brush_color == Qt.black


def test_mask_operations(app_fixture, qtbot):
    """Test mask morphological operations"""
    # Make sure we're on the annotation tab
    app_fixture.tab_widget.setCurrentIndex(1)
    qtbot.wait(300)

    # Test erode
    qtbot.mouseClick(app_fixture.erode_button, Qt.LeftButton)
    qtbot.wait(300)

    # Test dilate
    qtbot.mouseClick(app_fixture.dilate_button, Qt.LeftButton)
    qtbot.wait(300)

    # Test median filter
    qtbot.mouseClick(app_fixture.median_button, Qt.LeftButton)
    qtbot.wait(300)

    # Test overlay toggle
    qtbot.mouseClick(app_fixture.toggle_overlay_button, Qt.LeftButton)
    qtbot.wait(300)
    assert app_fixture.show_overlay == True

    # Toggle back
    qtbot.mouseClick(app_fixture.toggle_overlay_button, Qt.LeftButton)
    qtbot.wait(300)
    assert app_fixture.show_overlay == False


def test_sam2_points(app_fixture, qtbot):
    """Test adding SAM2 points"""

    # Make sure we're on the annotation tab
    app_fixture.tab_widget.setCurrentIndex(1)
    qtbot.wait(300)

    # Add foreground point
    center = app_fixture.overlay_image_label.rect().center()
    qtbot.mouseClick(app_fixture.overlay_image_label, Qt.LeftButton, pos=center)
    qtbot.wait(300)

    # Switch to background points
    qtbot.mouseClick(app_fixture.point_type_button, Qt.LeftButton)
    qtbot.wait(300)
    assert app_fixture.current_point_type == "background"

    # Add background point
    bg_point = center + QPoint(50, 50)
    qtbot.mouseClick(app_fixture.overlay_image_label, Qt.LeftButton, pos=bg_point)
    qtbot.wait(300)

    # Generate mask
    qtbot.mouseClick(app_fixture.generate_mask_button, Qt.LeftButton)
    qtbot.wait(2000)  # SAM2 processing might take time

    # Clear points
    qtbot.mouseClick(app_fixture.clear_points_button, Qt.LeftButton)
    qtbot.wait(300)


def test_save_mask(app_fixture, qtbot, monkeypatch):
    """Test saving mask functionality"""
    # Make sure we're on the annotation tab
    app_fixture.tab_widget.setCurrentIndex(1)
    qtbot.wait(300)

    # Prepare a mask to save
    # First clear the mask
    qtbot.mouseClick(app_fixture.clear_button, Qt.LeftButton)
    qtbot.wait(300)

    # Draw something simple on the mask
    center = app_fixture.editable_image_label.rect().center()
    qtbot.mousePress(app_fixture.editable_image_label, Qt.LeftButton, pos=center)

    # Draw a circle-like shape
    radius = 50
    for angle in range(0, 360, 10):
        x = center.x() + int(radius * np.cos(np.radians(angle)))
        y = center.y() + int(radius * np.sin(np.radians(angle)))
        new_pos = QPoint(x, y)
        qtbot.mouseMove(app_fixture.editable_image_label, pos=new_pos)
        qtbot.wait(10)

    qtbot.mouseRelease(app_fixture.editable_image_label, Qt.LeftButton, pos=center)
    qtbot.wait(300)

    # Save the mask
    qtbot.mouseClick(app_fixture.save_mask_button, Qt.LeftButton)
    qtbot.wait(500)

    # Check if the file was saved
    assert os.path.exists(os.path.join(TEST_OUTPUT_DIR, "test_output_mask.png"))


def test_resize_handling(app_fixture, qtbot):
    """Test app response to window resizing"""
    # Resize the window
    app_fixture.resize(1400, 900)
    qtbot.wait(500)

    # Resize again to smaller
    app_fixture.resize(1000, 700)
    qtbot.wait(500)


# Run the tests
if __name__ == "__main__":
    pytest.main(["-v", __file__])
