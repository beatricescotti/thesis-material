import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def split_dataset_by_pid(df, pid_col='pid', seed=42):
    # Estrai PID unici e fai lo split 80/10/10
    pids = df[pid_col].unique()
    train_pids, test_pids = train_test_split(pids, test_size=0.2, random_state=seed)
    valid_pids, test_pids = train_test_split(test_pids, test_size=0.5, random_state=seed)
    
    return train_pids, valid_pids, test_pids

def create_steps(df_split, intensity_col='nodule_mean_intensity'):
    # Ordina per intensity decrescente (dal più alto al più basso)
    df_sorted = df_split.sort_values(intensity_col, ascending=False).reset_index(drop=True)
    n = len(df_sorted)
    
    step1_idx = int(n * 0.33)
    step2_idx = int(n * 0.66)
    
    step1 = df_sorted.iloc[:step1_idx]
    step2 = df_sorted.iloc[:step2_idx]
    step3 = df_sorted
    
    return step1, step2, step3


def main(csv_path):
    df = pd.read_csv(csv_path)
    
    # Split PID
    train_pids, valid_pids, test_pids = split_dataset_by_pid(df)
    
    # Filtra dataframe per split
    train_df = df[df['pid'].isin(train_pids)].reset_index(drop=True)
    valid_df = df[df['pid'].isin(valid_pids)].reset_index(drop=True)
    test_df = df[df['pid'].isin(test_pids)].reset_index(drop=True)
    
    # Crea steps per train
    train_step1, train_step2, train_step3 = create_steps(train_df)
    # Per validation e test, ordino ma non faccio steps (puoi adattare se vuoi)
    valid_sorted = valid_df.sort_values('nodule_mean_intensity').reset_index(drop=True)
    test_sorted = test_df.sort_values('nodule_mean_intensity').reset_index(drop=True)
    
    # Salva i CSV
    train_step1.to_csv('/gwpool/users/bscotti/tesi/train_step1.csv', index=False)
    train_step2.to_csv('/gwpool/users/bscotti/tesi/train_step2.csv', index=False)
    train_step3.to_csv('/gwpool/users/bscotti/tesi/train_step3.csv', index=False)
    valid_sorted.to_csv('/gwpool/users/bscotti/tesi/valid.csv', index=False)
    test_sorted.to_csv('/gwpool/users/bscotti/tesi/test.csv', index=False)
    
    print("Dataset suddiviso e salvato!")

if __name__ == "__main__":
    csv_path = '/gwpool/users/bscotti/tesi/2d_annotations_pruned_threshold_sorted.csv'  # modifica con il path al tuo csv
    main(csv_path)

