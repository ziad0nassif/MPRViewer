import sys
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib import cm
from PyQt5.QtWidgets import (QApplication, QVBoxLayout, QWidget, QHBoxLayout,
                             QLabel, QPushButton, QSpacerItem, QSizePolicy, QFileDialog,
                             QSlider, QComboBox)
from PyQt5.QtCore import Qt
import vtk
from vtk.util import numpy_support # type: ignore

class MPRViewer(QWidget):
    def __init__(self):
        super().__init__()
        self.data = None
        self.slices = [0, 0, 0]
        self.marked_points = [[], [], []]
        self.zoom_level = 1.0
        self.brightness = 0
        self.contrast = 1
        self.panning = False
        self.pan_start = None
        self.current_colormap = 'gray'

        self.initUI()

    def initUI(self):
        self.setWindowTitle('MultiPlanar Reconstruction Viewer')
        self.main_layout = QHBoxLayout()

        # Left side: Views (3/4 of the width)
        views_layout = QVBoxLayout()

        # Create a horizontal layout for the Axial and Sagittal views
        h_layout_top = QHBoxLayout()

        # Variable to track dragging status
        self.dragging = False

        # Create a vertical layout for the Axial View
        axial_layout = QVBoxLayout()
        self.axial_canvas = self.create_canvas(0)
        axial_label = QLabel("Axial View")
        axial_label.setAlignment(Qt.AlignCenter)
        axial_layout.addWidget(axial_label)
        axial_layout.addWidget(self.axial_canvas)
        
        h_layout_top.addLayout(axial_layout)

        # Create a vertical layout for the Sagittal View
        sagittal_layout = QVBoxLayout()
        self.sagittal_canvas = self.create_canvas(2)
        sagittal_label = QLabel("Sagittal View")
        sagittal_label.setAlignment(Qt.AlignCenter)
        sagittal_layout.addWidget(sagittal_label)
        sagittal_layout.addWidget(self.sagittal_canvas)
        
        h_layout_top.addLayout(sagittal_layout)

        views_layout.addLayout(h_layout_top)

        # Coronal View (below Axial and Sagittal)
        coronal_layout = QVBoxLayout()
        self.coronal_canvas = self.create_canvas(1)
        coronal_label = QLabel("Coronal View")
        coronal_label.setAlignment(Qt.AlignCenter)
        coronal_layout.addWidget(coronal_label)
        coronal_layout.addWidget(self.coronal_canvas)
        
        views_layout.addLayout(coronal_layout)

        views_widget = QWidget()
        views_widget.setLayout(views_layout)
        self.main_layout.addWidget(views_widget, 3)  # Allocate 3 parts to views

        # Right side: Controls (1/4 of the width)
        controls_layout = QVBoxLayout()

        # Button to open NIfTI file
        self.open_button = QPushButton('Open NIfTI File')
        self.open_button.clicked.connect(self.open_file)
        controls_layout.addWidget(self.open_button)

        # Button to trigger volume rendering
        self.volume_render_button = QPushButton('Show Volume Rendering')
        self.volume_render_button.clicked.connect(self.show_volume_rendering)
        controls_layout.addWidget(self.volume_render_button)

        # Add Point checkbox
        self.add_point_button = QPushButton('Add Point')
        self.add_point_button.setCheckable(True)
        self.add_point_button.setChecked(False)
        controls_layout.addWidget(self.add_point_button)

        # Create sliders for brightness and contrast
        brightness_layout = QVBoxLayout()
        brightness_layout.addWidget(QLabel("Brightness:"))
        self.brightness_slider = QSlider(Qt.Horizontal)
        self.brightness_slider.setMinimum(-100)
        self.brightness_slider.setMaximum(100)
        self.brightness_slider.setValue(0)
        self.brightness_slider.valueChanged.connect(self.update_brightness_contrast)
        brightness_layout.addWidget(self.brightness_slider)
        controls_layout.addLayout(brightness_layout)

        contrast_layout = QVBoxLayout()
        contrast_layout.addWidget(QLabel("Contrast:"))
        self.contrast_slider = QSlider(Qt.Horizontal)
        self.contrast_slider.setMinimum(1)
        self.contrast_slider.setMaximum(200)
        self.contrast_slider.setValue(100)
        self.contrast_slider.valueChanged.connect(self.update_brightness_contrast)
        contrast_layout.addWidget(self.contrast_slider)
        controls_layout.addLayout(contrast_layout)

        # Add a stretch to push controls to the top
        controls_layout.addStretch(1)

        controls_widget = QWidget()
        controls_widget.setLayout(controls_layout)
        self.main_layout.addWidget(controls_widget, 1)  # Allocate 1 part to controls

        self.setLayout(self.main_layout)
        
        # Add Colormap selection dropdown
        colormap_layout = QVBoxLayout()
        colormap_layout.addWidget(QLabel("Colormap:"))
        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems(['gray', 'viridis', 'plasma', 'inferno', 'magma', 'cividis', 'jet'])
        self.colormap_combo.currentTextChanged.connect(self.update_colormap)
        colormap_layout.addWidget(self.colormap_combo)
        controls_layout.addLayout(colormap_layout)

        # Add Pan button
        self.pan_button = QPushButton('Pan')
        self.pan_button.setCheckable(True)
        self.pan_button.setChecked(False)
        controls_layout.addWidget(self.pan_button)
        
        # Initialize slices to the center
        self.update_slices_to_center()

    def update_colormap(self, colormap_name):
        self.current_colormap = colormap_name
        self.update_views()
    
    def update_brightness_contrast(self):
        self.brightness = self.brightness_slider.value() / 100.0
        self.contrast = self.contrast_slider.value() / 100.0
        self.update_views()

    def open_file(self):
        """Open a NIfTI file and load its data."""
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Open NIfTI File", "", "NIfTI Files (*.nii *.nii.gz)", options=options)
        if file_name:
            self.load_nifti(file_name)

    def load_nifti(self, nifti_file):
        self.nifti_img = nib.load(nifti_file)
        self.data = self.nifti_img.get_fdata()
        
        # Initialize slices to the middle of each dimension
        self.slices = [dim // 2 for dim in self.data.shape]
        
        self.update_views()

    def update_slices_to_center(self):
        """Set the slices to the center of the data."""
        if self.data is not None:
            self.slices = [self.data.shape[2] // 2, self.data.shape[1] // 2, self.data.shape[0] // 2]

    def create_canvas(self, index):
        """Create a matplotlib canvas for the view."""
        fig = plt.figure(figsize=(5, 5))
        canvas = fig.canvas

        # Set up mouse events for click and drag
        canvas.mpl_connect('button_press_event', lambda event: self.on_mouse_press(event, index))
        canvas.mpl_connect('motion_notify_event', lambda event: self.on_mouse_drag(event, index))
        canvas.mpl_connect('button_release_event', lambda event: self.on_mouse_release(event))
        
        # Connect the scroll event to the wheel_zoom function
        canvas.mpl_connect('scroll_event', lambda event: self.wheel_zoom(event, index))

        return canvas


    def add_point(self, event, view_index):
        """Add a point to the marked_points list when clicked in the appropriate view."""
        if event.xdata is None or event.ydata is None:
            return  # Ensure click is inside the canvas

        # Clear previous points in all views
        self.marked_points = [[], [], []]  # Reset points for Axial, Coronal, and Sagittal views

        # Convert click coordinates to integers
        x, y = int(round(event.xdata)), int(round(event.ydata))

        # Ensure coordinates are within the image dimensions
        if view_index == 0:  # Axial view
            x = min(max(x, 0), self.data.shape[0] - 1)
            y = min(max(y, 0), self.data.shape[1] - 1)
            self.marked_points[0].append([x, y])
            # Reflect on Coronal and Sagittal views
            self.marked_points[1].append([x, self.slices[1]])
            self.marked_points[2].append([y, self.slices[2]])
        elif view_index == 1:  # Coronal view
            x = min(max(x, 0), self.data.shape[0] - 1)
            y = min(max(y, 0), self.data.shape[2] - 1)
            self.marked_points[0].append([x, self.slices[0]])
            self.marked_points[1].append([x, y])
            self.marked_points[2].append([self.slices[1], y])
        elif view_index == 2:  # Sagittal view
            x = min(max(x, 0), self.data.shape[1] - 1)
            y = min(max(y, 0), self.data.shape[2] - 1)
            self.marked_points[0].append([self.slices[0], x])
            self.marked_points[1].append([y, x])
            self.marked_points[2].append([x, y])

        # Update the views to reflect the added points
        self.update_views()




    def get_slice(self, view_index, slice_index):
        """Get the appropriate slice based on the view index and slice index."""
        if self.data is None:
            return None

        # Ensure slice_index is within valid bounds
        max_slice = self.data.shape[view_index] - 1
        slice_index = min(max(slice_index, 0), max_slice)

        if view_index == 0:  # Axial
            slice_data = self.data[:, :, slice_index]
        elif view_index == 1:  # Coronal
            slice_data = self.data[:, slice_index, :]
        elif view_index == 2:  # Sagittal
            slice_data = self.data[slice_index, :, :]
        else:
            return None

        # Apply brightness and contrast
        slice_min = np.min(slice_data)
        slice_max = np.max(slice_data)
        
        if slice_max > slice_min:
            # Normalize only if there's a difference between max and min
            slice_data = (slice_data - slice_min) / (slice_max - slice_min)
        else:
            # If all values are the same, set the entire slice to 0.5
            slice_data = np.full_like(slice_data, 0.5)
        
        # Apply contrast and brightness
        slice_data = np.clip((slice_data - 0.5) * self.contrast + 0.5 + self.brightness, 0, 1)
        
        return slice_data

    
    def on_mouse_press(self, event, view_index):
        """Handle mouse press event to start dragging the crosshairs, add a point, or start panning."""
        if event.inaxes is None:
            return

        if self.pan_button.isChecked():
            self.panning = True
            self.pan_start = (event.xdata, event.ydata)
        elif self.add_point_button.isChecked():
            self.add_point(event, view_index)
        else:
            self.dragging = True
            self.drag_start_x = event.xdata
            self.drag_start_y = event.ydata
            self.drag_view_index = view_index

    def on_mouse_drag(self, event, view_index):
        """Handle mouse drag event to update the crosshair position or pan the view."""
        if event.inaxes is None:
            return

        if self.panning and self.pan_start:
            dx = self.pan_start[0] - event.xdata
            dy = self.pan_start[1] - event.ydata
            
            ax = event.inaxes
            x_min, x_max = ax.get_xlim()
            y_min, y_max = ax.get_ylim()
            
            ax.set_xlim(x_min + dx, x_max + dx)
            ax.set_ylim(y_min + dy, y_max + dy)
            
            ax.figure.canvas.draw_idle()
        elif self.dragging:
            x, y = int(round(event.xdata)), int(round(event.ydata))

            if view_index == 0:  # Axial view
                self.slices[1] = min(max(y, 0), self.data.shape[1] - 1)
                self.slices[2] = min(max(x, 0), self.data.shape[2] - 1)
            elif view_index == 1:  # Coronal view
                self.slices[0] = min(max(y, 0), self.data.shape[0] - 1)
                self.slices[2] = min(max(x, 0), self.data.shape[2] - 1)
            elif view_index == 2:  # Sagittal view
                self.slices[0] = min(max(y, 0), self.data.shape[0] - 1)
                self.slices[1] = min(max(x, 0), self.data.shape[1] - 1)

            for i in range(3):
                self.slices[i] = min(max(self.slices[i], 0), self.data.shape[i] - 1)

            self.update_views()

    def on_mouse_release(self, event):
        """Handle mouse release event to stop dragging or panning."""
        self.dragging = False
        self.panning = False
        self.pan_start = None

    def update_views(self):
        """Update all three views."""
        if self.data is None:
            return

        views = [self.get_slice(0, self.slices[0]),
                 self.get_slice(1, self.slices[1]),
                 self.get_slice(2, self.slices[2])]

        for i, (view, canvas) in enumerate(zip(views, [self.axial_canvas, self.coronal_canvas, self.sagittal_canvas])):
            canvas.figure.clear()
            ax = canvas.figure.add_subplot(111)

            # Display the image with appropriate zoom and colormap applied
            im = ax.imshow(view.T, cmap=self.current_colormap, origin='lower', extent=[0, view.shape[1], 0, view.shape[0]])
            
            # Add colorbar
            canvas.figure.colorbar(im, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)

            # Mark points
            for point in self.marked_points[i]:
                ax.plot(point[0], point[1], 'ro', markersize=5)  # Red points for marked points

            # Crosshairs logic (now with dashed lines)
            if i == 0:  # Axial view
                ax.axhline(self.slices[1], color='yellow', linestyle='--')  # Horizontal dashed line (Y-axis)
                ax.axvline(self.slices[2], color='yellow', linestyle='--')  # Vertical dashed line (X-axis)
            elif i == 1:  # Coronal view
                ax.axhline(self.slices[0], color='yellow', linestyle='--')  # Horizontal dashed line (X-axis)
                ax.axvline(self.slices[2], color='yellow', linestyle='--')  # Vertical dashed line (Z-axis)
            elif i == 2:  # Sagittal view
                ax.axhline(self.slices[0], color='yellow', linestyle='--')  # Horizontal dashed line (X-axis)
                ax.axvline(self.slices[1], color='yellow', linestyle='--')  # Vertical dashed line (Y-axis)

            canvas.draw()


    def show_volume_rendering(self):
        """Display volume rendering of the NIfTI file data using VTK."""
        if self.data is None:
            print("No data loaded!")
            return
        
        # Create a VTK image data object
        image_data = vtk.vtkImageData()
        image_data.SetDimensions(self.data.shape[2], self.data.shape[1], self.data.shape[0])

        # Convert the NumPy array to VTK format
        vtk_data_array = numpy_support.numpy_to_vtk(self.data.ravel(), deep=True, array_type=vtk.VTK_FLOAT)
        image_data.GetPointData().SetScalars(vtk_data_array)

        # Set up volume rendering pipeline
        volume_mapper = vtk.vtkGPUVolumeRayCastMapper()
        volume_mapper.SetInputData(image_data)

        volume_property = vtk.vtkVolumeProperty()
        volume_property.ShadeOn()
        volume_property.SetInterpolationTypeToLinear()

        # Set color and opacity functions
        color_function = vtk.vtkColorTransferFunction()
        color_function.AddRGBPoint(np.min(self.data), 0.0, 0.0, 0.0)
        color_function.AddRGBPoint(np.max(self.data), 1.0, 1.0, 1.0)
        volume_property.SetColor(color_function)

        opacity_function = vtk.vtkPiecewiseFunction()
        opacity_function.AddPoint(np.min(self.data), 0.0)
        opacity_function.AddPoint(np.max(self.data), 1.0)
        volume_property.SetScalarOpacity(opacity_function)

        volume = vtk.vtkVolume()
        volume.SetMapper(volume_mapper)
        volume.SetProperty(volume_property)

        # Renderer and window
        renderer = vtk.vtkRenderer()
        renderer.AddVolume(volume)
        renderer.SetBackground(0, 0, 0)

        render_window = vtk.vtkRenderWindow()
        render_window.AddRenderer(renderer)

        render_interactor = vtk.vtkRenderWindowInteractor()
        render_interactor.SetRenderWindow(render_window)

        render_window.Render()
        render_interactor.Start()

    def wheel_zoom(self, event, index):
        """Zoom in or out based on scroll event, centered on cursor position."""
        if event.inaxes is None:
            return  # Ensure that the mouse is over an axes

        ax = event.inaxes
        
        # Determine zoom factor
        if event.button == 'up':
            scale_factor = 1.1
        elif event.button == 'down':
            scale_factor = 1 / 1.1
        else:
            return  # If it's not a scroll event, do nothing

        # Get the current x and y limits
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        
        # Get mouse location in data coordinates
        x_data = event.xdata
        y_data = event.ydata
        
        # Calculate new limits
        new_x_min = x_data - (x_data - x_min) / scale_factor
        new_x_max = x_data + (x_max - x_data) / scale_factor
        new_y_min = y_data - (y_data - y_min) / scale_factor
        new_y_max = y_data + (y_max - y_data) / scale_factor
        
        # Set new limits
        ax.set_xlim(new_x_min, new_x_max)
        ax.set_ylim(new_y_min, new_y_max)
        
        # Redraw the canvas
        ax.figure.canvas.draw_idle()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    viewer = MPRViewer()
    viewer.show()
    sys.exit(app.exec_())