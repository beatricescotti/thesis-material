import optuna
from ultralytics import YOLO
import matplotlib.pyplot as plt

# importo il modello
model = YOLO('/gwpool/users/bscotti/tesi/train_6jet/train5/weights/best.pt') 

# liste per memorizzare i valori di conf e mAP50
conf_values = []
map50_values = []
# bb_loss_values=[] 

# definizione della funzione obiettivo di Optuna
def objective(trial):
    conf_threshold = trial.suggest_float("conf", 0.01, 0.9)  # cerco in questo intervallo
    
    # valutazione del modello sul validation set
    results = model.val(conf=conf_threshold)
    score = results.box.map50 # prendo la metrica mean average precision 50

    conf_values.append(conf_threshold)
    map50_values.append(score)
    
    trial.set_user_attr("map50", score)
    return score  

# crea lo studio Optuna e ottimizza
study = optuna.create_study(direction="maximize")  
study.optimize(objective, n_trials=20)  

# stampo il miglior valore di conf in base a mAP50
best_conf = study.best_params['conf']
best_map50 = study.best_value
print(f"Miglior valore di conf: {best_conf}")
print(f"Con il valore di mAP50: {best_map50}")



# PLOT

sorted_data = sorted(zip(conf_values, map50_values))
sorted_conf, sorted_map50 = zip(*sorted_data)
plt.figure(figsize=(10, 6))
plt.scatter(sorted_conf, sorted_map50, label='mAP50', color='blue', marker='o')

# evidenzio il miglior valore
plt.scatter(best_conf, best_map50, color='red', marker='*', s=200, label='Miglior mAP50')
plt.annotate(f"Best: {best_map50:.3f}", (best_conf, best_map50), 
             textcoords="offset points", xytext=(10,10), ha='center', fontsize=12, color='red')


plt.xlabel("Valore di conf")
plt.ylabel("mAP50")
plt.title("Andamento di mAP50 per diversi valori di conf")
plt.grid(True)
plt.legend()
plt.savefig("/gwpool/users/bscotti/tesi/conf_vs_map_final_jet6_bkg.png")
