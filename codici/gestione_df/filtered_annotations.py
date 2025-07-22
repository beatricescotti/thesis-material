import pandas as pd
import os

annotations_df = pd.read_csv('/gwpool/users/bscotti/tesi/csv/annotations.csv')

# qui riduco la dimensione del file annotations in modo tale da avere solo le righe che corrispondono alle immagini che ho nella cartella slice_filtrate_batch1
images_path = '/gwpool/users/bscotti/tesi/dati/slices_segmentate_png1'

nomi_immagini = [] # creazione di una lista di immagini che verrà riempita con i nomi delle immagini a cui tolgo l'estensione perchè non ci sono dentro al csv
for file in os.listdir(images_path):
    nome_immagine_senza_estensione = os.path.splitext(file)[0]  # Rimuovi l'estensione
    nomi_immagini.append(nome_immagine_senza_estensione)

righe_filtrate = [] # creo un'altra lista in cui tengo solamente le righe il cui nome = nome di un'immagine nella lista sopra
for index, row in annotations_df.iterrows():
    nome_immagine_csv = row.iloc[2]  # estraggo il nome dell'immagine dalla prima colonna del csv
    if nome_immagine_csv in nomi_immagini:
        righe_filtrate.append(row)

# creazione di un nuovo dataframe che ha come righe solo quelle per cui ha trovato la corrispondenza tra le due liste
filtered_annotations_df = pd.DataFrame(righe_filtrate)
filtered_annotations_df.to_csv('/gwpool/users/bscotti/tesi/csv/filtered_annotations.csv', index=False)