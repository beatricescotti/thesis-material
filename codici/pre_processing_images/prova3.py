from PIL import Image
import os
import numpy as np

# Cartella contenente le immagini
cartella = "/gwpool/users/bscotti/tesi/dati/slices_white"

# Scansiona tutte le immagini nella cartella
for filename in os.listdir(cartella):
    filepath = os.path.join(cartella, filename)
    
    try:
        # Apri l'immagine
        img = Image.open(filepath).convert("L")  # Converti in scala di grigi
        pixels = np.array(img)

        # Se tutti i pixel hanno lo stesso valore, elimina l'immagine
        if np.all(pixels == pixels[0, 0]):
            print(f"Elimino: {filename}")
            os.remove(filepath)

    except Exception as e:
        print(f"Errore con {filename}: {e}")