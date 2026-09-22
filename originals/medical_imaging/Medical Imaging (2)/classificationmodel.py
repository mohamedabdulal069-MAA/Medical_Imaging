import SimpleITK as sitk
import vtk
import joblib
import cv2
import os
import numpy as np
import tensorflow as tf

def load_dicom_slices(folder_path):
    # The files in this folder are separate series (different SeriesInstanceUIDs,
    # different sizes), so ImageSeriesReader would return only the first series.
    # Read each file on its own instead and keep them as independent 2D images.
    dicom_names = sorted(
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith(".dcm")
    )
    if not dicom_names:
        raise ValueError(f"No .dcm files found in {folder_path}")

    slices = []
    for path in dicom_names:
        reader = sitk.ImageFileReader()
        reader.SetFileName(path)
        image = reader.Execute()
        array = sitk.GetArrayFromImage(image)
        if array.ndim == 3:          # single-frame files come back as (1, y, x)
            array = array[0]
        slices.append(array)
    return slices, dicom_names

def visualize_volume(slices):
    # These slices are unrelated images of differing sizes, so resize them all to
    # the first slice's shape before stacking them into something VTK can render.
    target_h, target_w = slices[0].shape
    resized = [
        s if s.shape == (target_h, target_w)
        else cv2.resize(s, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        for s in slices
    ]
    img_array = np.ascontiguousarray(np.stack(resized, axis=0), dtype=np.uint16)  # z, y, x

    sitk_to_vtk = vtk.vtkImageImport()
    flat_array = img_array.ravel(order='C')

    sitk_to_vtk.CopyImportVoidPointer(flat_array, flat_array.nbytes)
    sitk_to_vtk.SetDataScalarTypeToUnsignedShort()
    sitk_to_vtk.SetNumberOfScalarComponents(1)
    sitk_to_vtk.SetWholeExtent(0, img_array.shape[2]-1, 0, img_array.shape[1]-1, 0, img_array.shape[0]-1)
    sitk_to_vtk.SetDataExtentToWholeExtent()
    sitk_to_vtk.Update()

    volume_mapper = vtk.vtkSmartVolumeMapper()
    volume_mapper.SetInputConnection(sitk_to_vtk.GetOutputPort())

    volume = vtk.vtkVolume()
    volume.SetMapper(volume_mapper)

    color_func = vtk.vtkColorTransferFunction()
    color_func.AddRGBPoint(0, 0.0, 0.0, 0.0)
    color_func.AddRGBPoint(1000, 1.0, 0.5, 0.3)

    opacity_func = vtk.vtkPiecewiseFunction()
    opacity_func.AddPoint(0, 0.0)
    opacity_func.AddPoint(1000, 0.5)

    volume_property = vtk.vtkVolumeProperty()
    volume_property.SetColor(color_func)
    volume_property.SetScalarOpacity(opacity_func)
    volume_property.ShadeOn()
    volume.SetProperty(volume_property)

    renderer = vtk.vtkRenderer()
    renderer.AddVolume(volume)
    renderer.SetBackground(0.1, 0.2, 0.4)

    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetSize(600, 600)

    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)
    render_window.Render()
    interactor.Start()

def extract_slices(image_3d):
    slices = sitk.GetArrayFromImage(image_3d)  # Shape: (z, y, x)
    return slices

#folder containing this script (where pca and ensemble model are present)
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load PCA model
pca = joblib.load(os.path.join(script_dir, "pca_model.pkl"))

# Load ensemble model
ensemble_model = tf.keras.models.load_model(os.path.join(script_dir, "ensemble_model.h5"))


print("PCA and ensemble model loaded successfully.")


def preprocess_slice(slice_2d):
    # Resize to match PCA input dimensions (5376 = 84 x 64)
    resized = cv2.resize(slice_2d, (84, 64))  # width=84, height=64
    flat = resized.flatten().reshape(1, -1)
    reduced = pca.transform(flat)
    return reduced


def predict_slice(slice_2d):
    features = preprocess_slice(slice_2d)
    prediction = ensemble_model.predict(features)
    return int(tf.argmax(prediction[0]))  # Convert softmax output to class label (int)

def annotate_slice(slice_2d, prediction_label):
    color_map = {
        0: (0, 255, 0),     # No hemorrhage - Green
        1: (0, 0, 255),     # Any - Red
        2: (255, 255, 0),   # Epidural - Yellow
        3: (255, 0, 255),   # Intraparenchymal - Magenta
        4: (0, 255, 255),   # Intraventricular - Cyan
        5: (255, 165, 0),   # Subarachnoid - Orange
        6: (255, 192, 203)  # Subdural - Pink
    }

    import numpy as np
    # Normalize int16 DICOM values (Hounsfield Units) to uint8 for display
    slice_norm = slice_2d.astype(np.float32)
    slice_norm = np.clip(slice_norm, -1000, 2000)  # window to brain-relevant HU range
    slice_norm = ((slice_norm + 1000) / 3000 * 255).astype(np.uint8)

    img_rgb = cv2.cvtColor(slice_norm, cv2.COLOR_GRAY2BGR)
    label_names = {
        0: "No Hemorrhage",
        1: "Any Hemorrhage",
        2: "Epidural",
        3: "Intraparenchymal",
        4: "Intraventricular",
        5: "Subarachnoid",
        6: "Subdural"
    }
    label_text = label_names.get(prediction_label, f"Class {prediction_label}")
    cv2.putText(img_rgb, label_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_map.get(prediction_label, (255,255,255)), 2)
    return img_rgb


folder = os.path.join(script_dir, "DICOM")  #change path to your DICOM Images folder
slices, dicom_names = load_dicom_slices(folder)

print(f"Loaded {len(slices)} DICOM slices. Press any key to advance, 'q' to quit.")
for name, s in zip(dicom_names, slices):
    print(f"  {os.path.basename(name)}: {s.shape} {s.dtype}")
for i, slice_2d in enumerate(slices):
    pred = predict_slice(slice_2d)
    annotated = annotate_slice(slice_2d, pred)
    # Add slice counter to the image
    cv2.putText(annotated, f"Slice {i+1}/{len(slices)}", (10, annotated.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.imshow("DICOM Classifier", annotated)  # reuse same window
    key = cv2.waitKey(0)  # wait until key is pressed
    if key == ord('q') or key == 27:  # 'q' or Esc to quit early
        break
cv2.destroyAllWindows()

# View volume
visualize_volume(image_3d)
