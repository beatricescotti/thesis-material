import numpy as np
import matplotlib.pyplot as plt
import pydicom
import SimpleITK as sitk
import glob
import os
from PIL import Image

import SimpleITK as sitk

def resample_dicom_slice(dicom_path, target_spacing_mm=[0.623, 0.623]):
    image = sitk.ReadImage(dicom_path)
    
    if image.GetDimension() == 2:
        image = sitk.JoinSeries([image]) 
    
    original_spacing = image.GetSpacing()  # pixel spacing in mm
    print(original_spacing)
    original_size = image.GetSize() 
    print(f"Original Size: {original_size}")    

    target_spacing_mm.append(original_spacing[2])
        

    # Compute new size (3D)
    new_size = [
        int(round(original_size[0] * original_spacing[0] / target_spacing_mm[0])),
        int(round(original_size[1] * original_spacing[1] / target_spacing_mm[1])),
        1  # dummy dimension
    ]

    # Resample (now with proper 3D parameters)
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(new_size)
    resampler.SetOutputSpacing(target_spacing_mm)
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetInterpolator(sitk.sitkBSplineResampler)

    resampled_image = resampler.Execute(image)
    print(f"Resampled Size: {resampled_image.GetSize()}")
    print(f"Resampled Spacing: {resampled_image.GetSpacing()}")
    
    # Convert back to 2D if needed
    # image, size (eg. (512, 512, 0) -> 512x512), starting_position (eg. (0,0,0) -> start from origin)    
    return sitk.Extract(resampled_image, (resampled_image.GetWidth(), resampled_image.GetHeight(), 0), (0,0,0)), resampled_image.GetSpacing(), original_spacing 
    

set1_path = '/gwpool/users/bscotti/tesi/dati/dati_medici/set1/slices_filtrate_1'
set2_path = '/gwpool/users/bscotti/tesi/dati/dati_medici/set2/slices_filtrate_2'

dicom_paths = glob.glob(os.path.join(set1_path, '*.dcm')) + glob.glob(os.path.join(set2_path, '*.dcm'))

output_path = '/gwpool/users/bscotti/tesi/dati/dati_medici/NLST/resampling'
os.makedirs(output_path, exist_ok=True)

for dicom_path in dicom_paths:

    resampled_image = resample_dicom_slice(dicom_path, target_spacing_mm=[0.623, 0.623])
    print("resampled_image:", resampled_image[0])

    resampled_array = sitk.GetArrayFromImage(resampled_image[0])
    print("Resampled Array Shape:", resampled_array.shape)


    print("pixel minimo:", resampled_array.min())
    print("pixel massimo:", resampled_array.max())


    window_center = -600
    window_width = 1700
    window_min = window_center - window_width // 2
    window_max = window_center + window_width // 2
    resampled_array_windowed = np.clip(resampled_array, window_min, window_max)
    hu_uint8 = ((resampled_array_windowed - window_min) / window_width * 255).astype(np.uint8)
 
    output_path = '/gwpool/users/bscotti/tesi/dati/dati_medici/NLST/resampling'
    os.makedirs(output_path, exist_ok=True)
    # Salva come PNG
    img = Image.fromarray(hu_uint8, mode='L')  # 'L' per scala di grigi
    img_name = os.path.splitext(os.path.basename(dicom_path))[0] + '.png'
    img.save(os.path.join(output_path, img_name))



   