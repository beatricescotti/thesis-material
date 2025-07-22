import numpy as np 
import pandas as pd 
import os 
import matplotlib.pyplot as plt 
import shutil

folder_path = '/gwpool/users/bscotti/tesi/dati/set1/set1_batch1/sub_batch_A'
csv_path = '/gwpool/users/bscotti/tesi/csv/annotations.csv'
new_folder_path = '/gwpool/users/bscotti/tesi/dati/set1/slices_filtrate_1'
os.makedirs(new_folder_path, exist_ok=True)

annotations_df = pd.read_csv(csv_path)

prima_colonna_pid = annotations_df.columns[0]
seconda_colonna_dcm_series = annotations_df.columns[1]
terza_colonna_slices = annotations_df.columns[2]

# primo ciclo: si prende una cartella all'interno del sub_batch_A (nel codice finale si chiama così) e si selezionano del file annotations solo le righe
# in cui la prima colonna (quella del pid) è uguale al nome della cartella
for pid_folder in os.listdir(folder_path):
    pid_folder_path = os.path.join(folder_path, pid_folder)
    
    if os.path.isdir(pid_folder_path):
        df_filtrato = annotations_df[annotations_df[prima_colonna_pid] == int(pid_folder)]
        print(f"Cartella PID: {pid_folder}")

        if df_filtrato.empty:
         print(f"  Nessuna corrispondenza trovata per PID: {pid_folder}")
         continue  
        
        # secondo ciclo: nella cartella selezionata, ora che ho il file annotations ridotto apposta per lei, sono contenute diverse cartelle (Ti), all'interno delle quali 
        # ci sono delle altre cartelle il cui nome è scritto nella seconda colonna del file annotations. Quindi faccio prima un ciclo sulle cartelle Ti  
        for Ti_folder in os.listdir(pid_folder_path):
                Ti_folder_path = os.path.join(pid_folder_path, Ti_folder) 
                
                if not os.path.isdir(Ti_folder_path):
                     continue 
                     
                print(" Sottocartella T: ", Ti_folder)
                
                # terzo ciclo per le cartelle all'interno di Ti (quelle dei filtri)
                for dcm_folder in os.listdir(Ti_folder_path):
                  dcm_folder_path = os.path.join(Ti_folder_path, dcm_folder)
                  if os.path.isdir(dcm_folder_path):
                   print("   Sottocartella con filtro dcm: ", dcm_folder)
                    
                    # ultimo ciclo, quello per selezionare le immagini. ora abbiamo il file annotations filtrato con solo le righe che ci servono
                   for dcm_file in os.listdir(dcm_folder_path):
                    dcm_file_path = os.path.join(dcm_folder_path, dcm_file)
                    print(dcm_file)
                    
                    # copio le immagini che sono all'interno del file annotations filtrato
                    for dcm_file_filtered in df_filtrato[terza_colonna_slices].values:
                      if dcm_file.startswith(dcm_file_filtered) and dcm_file.endswith('.dcm'):
                        shutil.copy(dcm_file_path, new_folder_path)
                        print(f"    File copiato: {dcm_file_path} -> {new_folder_path}")  
                    
