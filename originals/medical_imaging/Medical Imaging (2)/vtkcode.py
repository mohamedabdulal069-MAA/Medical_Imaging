import sys
import os
import numpy as np
import pydicom
import vtk
from PyQt5 import QtWidgets, QtCore
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

class DicomViewer(QtWidgets.QWidget):
    def __init__(self, folder_path):
        super().__init__()
        self.folder_path = folder_path
        self.dicom_files = [f for f in os.listdir(folder_path) if f.endswith('.dcm')]
        self.dicom_files.sort()  # Optional: sort files alphabetically
        self.current_index = 0

        # Setup UI
        self.vl = QtWidgets.QVBoxLayout()
        self.vtkWidget = QVTKRenderWindowInteractor(self)
        self.vl.addWidget(self.vtkWidget)

        # Buttons
        self.h_layout = QtWidgets.QHBoxLayout()
        self.prev_btn = QtWidgets.QPushButton("Previous")
        self.next_btn = QtWidgets.QPushButton("Next")
        self.file_label = QtWidgets.QLabel()
        self.h_layout.addWidget(self.prev_btn)
        self.h_layout.addWidget(self.next_btn)
        self.h_layout.addWidget(self.file_label)
        self.vl.addLayout(self.h_layout)

        self.setLayout(self.vl)

        # VTK Renderer
        self.renderer = vtk.vtkRenderer()
        self.vtkWidget.GetRenderWindow().AddRenderer(self.renderer)
        self.interactor = self.vtkWidget.GetRenderWindow().GetInteractor()

        # Connect buttons
        self.prev_btn.clicked.connect(self.show_prev)
        self.next_btn.clicked.connect(self.show_next)

        # Initial image
        if self.dicom_files:
            self.show_image(self.current_index)
        else:
            self.file_label.setText("No DICOM files found.")

        self.setWindowTitle("DICOM Viewer")
        self.resize(600, 600)
        self.show()
        self.interactor.Initialize()

    def show_image(self, index):
        # Clear previous actors
        self.renderer.RemoveAllViewProps()

        file_path = os.path.join(self.folder_path, self.dicom_files[index])
        self.file_label.setText(self.dicom_files[index])

        # Load DICOM image
        dcm = pydicom.dcmread(file_path)
        img_array = dcm.pixel_array.astype(np.uint8)

        # Flip vertically to fix orientation
        img_array = np.flipud(img_array)

        height, width = img_array.shape

        # Create vtkImageData
        vtk_image = vtk.vtkImageData()
        vtk_image.SetDimensions(width, height, 1)
        vtk_image.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)

        for y in range(height):
            for x in range(width):
                vtk_image.SetScalarComponentFromFloat(x, y, 0, 0, img_array[y, x])

        # Setup actor
        actor = vtk.vtkImageActor()
        actor.GetMapper().SetInputData(vtk_image)

        self.renderer.AddActor(actor)
        self.renderer.ResetCamera()
        self.vtkWidget.GetRenderWindow().Render()

    def show_next(self):
        if not self.dicom_files:
            return
        self.current_index = (self.current_index + 1) % len(self.dicom_files)
        self.show_image(self.current_index)

    def show_prev(self):
        if not self.dicom_files:
            return
        self.current_index = (self.current_index - 1) % len(self.dicom_files)
        self.show_image(self.current_index)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    folder_path = r"C:\Users\Laiba\Images"        #use your folder path here
    viewer = DicomViewer(folder_path)
    sys.exit(app.exec_())