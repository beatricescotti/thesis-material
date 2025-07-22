import os
import pydicom
from PIL import Image
import shutil

# Funzione per convertire i DICOM in PNG e salvare i metadati in .txt a parte
# def convert_dicom_to_png(dicom_folder, png_folder, metadata_folder):
def convert_dicom_to_png(dicom_folder, png_folder):

    for dicom_file in os.listdir(dicom_folder):
        if dicom_file.endswith('.dcm'):
            dicom_path = os.path.join(dicom_folder, dicom_file)

            dicom_data = pydicom.dcmread(dicom_path) # lettura dell'immagine dicom con libreria pydicom
            image_array = dicom_data.pixel_array # estraggo l'immagine come array di numpy

            # image_array = (image_array - image_array.min()) / (image_array.max() - image_array.min()) * 255

            image_array = image_array.astype('uint8')

            image = Image.fromarray(image_array)

            base_filename = os.path.splitext(dicom_file)[0] # prendo il nome senza estensione e lo userò per salvare i file png e txt
            png_filename = f"{base_filename}.png" # aggiungo le estensioni ai due nuovi file
            # txt_filename = f"{base_filename}.txt"

            # salvo immagine png
            png_path = os.path.join(png_folder, png_filename)
            image.save(png_path)

            # estrazione dei metadati
            """"
            metadata = (
                f"PatientID: {dicom_data.get('PatientID', 'N/A')}\n"
                f"StudyDate: {dicom_data.get('StudyDate', 'N/A')}\n"
                f"Modality: {dicom_data.get('Modality', 'N/A')}\n"
                f"PixelSpacing: {str(dicom_data.get('PixelSpacing', 'N/A'))}\n"
                f"Rows: {dicom_data.Rows}\n"
                f"Columns: {dicom_data.Columns}\n"
                f"ImagePositionPatient: {str(dicom_data.get('ImagePositionPatient', 'N/A'))}\n"
            )

            # salvo i metadati sul file txt creato
            txt_path = os.path.join(metadata_folder, txt_filename)
            with open(txt_path, 'w') as txt_file:
                txt_file.write(metadata) 

            """


dicom_folder = '/gwpool/users/bscotti/tesi/dati/prova'  # cartella con i file DICOM
png_folder = '/gwpool/users/bscotti/tesi/dati/prova8' # cartella di destinazione PNG
os.makedirs(png_folder, exist_ok=True)
# metadata_folder = '/content/drive/MyDrive/Tesi_magistrale/modello/metadati1'  # cartella per i metadati

convert_dicom_to_png(dicom_folder, png_folder)