
# Librerie
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt  #Librerie per la creazione di grafici e visualizzazioni

from statsmodels.distributions.empirical_distribution import ECDF #Classe ECDF per calcolare la funzione di distribuzione empirica (usata nelle analisi statistiche)

from sklearn.preprocessing import StandardScaler, LabelEncoder #Per normalizzare/standardizzare dati e convertire label testuali in numeriche
from sklearn.decomposition import PCA # Per la riduzione dimensionale secondo l’Analisi delle Componenti Principali
from sklearn.model_selection import train_test_split # Suddivisione train/test set.

from sklearn.linear_model import LinearRegression, Lasso   # Esempi di modelli di regressione
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans # Algoritmo di clustering non supervisionato

from sklearn.metrics import (ConfusionMatrixDisplay, mean_squared_error, r2_score, # Metodi di valutazione dei modelli, metriche di regressione e clustering
                             confusion_matrix, silhouette_score, davies_bouldin_score, 
                             calinski_harabasz_score)
from sklearn.utils import resample # Per campionare sottodataset (usato ad esempio nelle analisi silhouette)
from scipy.spatial.distance import cdist # Per calcolare la matrice di distanza (euclidea in questo caso) tra vettori (ad esempio i centroidi)

# =============================================================================
#                      FLAG DI CONFIGURAZIONE
# =============================================================================
# Sezione dei flag: consente di abilitare/disabilitare blocchi di codice/funzionalità come per esempio outlier filtering, scomposizione IP, 
# PCA, regressione, clustering, plot, ecc...
RUN_OUTLIER_FILTER = True        # Se True, abilita il filtraggio degli outlier in base a min_rtt
USE_IP = True                    # Se True, scompone gli IP in ottetti
RUN_PCA = True                   # Se True, esegue PCA

# Nuovo flag: se True, la PCA userà solo [min_rtt, min_ttl, uniq_byte]
# altrimenti userà l'insieme “esteso” (IP scomposti, timestamp, throughput, ecc.).
PCA_ONLY_CORE = False        

# --- Machine Learning (Regressione Throughput) ---
RUN_ML = True                  
RUN_LINEAR_REGRESSION = True     
RUN_KNN = True                   
RUN_RANDOM_FOREST = False         # Probabilmente RF non è la scelta migliore per questo task   
RUN_LASSO = True                 
TRAIN_ON_WEEK1_VALIDATE_OTHERS = True  
RUN_TEST_WEEKS = True 

# --- Clustering (K-Means) ---
RUN_CLUSTERING = True            
KMEANS_AUTO = True              
KMEANS_N_CLUSTERS = 12     
CLUSTER_OTHER_WEEKS = True       

# --- Plot e Visualizzazioni ---
RUN_PLOTS = True                
RUN_PLOTS_ECDF = True           
RUN_PLOTS_SCATTER = True        
RUN_PLOTS_CLUSTER = True        
RUN_PLOTS_IP_2D = True          
RUN_PLOTS_PCA_3D = True         
RUN_PLOTS_PCA_3D_EDGENODE = True

# --- Altri Parametri ---
PARALLEL_LEVEL = 8       
MODEL_COMPLEXITY = 2    
TARGET_COL = 'throughput'

# =============================================================================
#                  FUNZIONI DI SUPPORTO
# =============================================================================
# definiscono comportamenti comuni (come get_n_jobs, parametri dei modelli in base a MODEL_COMPLEXITY, funzioni di plot e così via)
def get_n_jobs(parallel_level: int) -> int:
    """
    Mappa [1..10] su n_jobs di scikit-learn.
    Restituisce -1 (tutti i core) se parallel_level>=10, altrimenti parallel_level.
    """
    if parallel_level < 1:
        return 1
    elif parallel_level < 10:
        return parallel_level
    else:
        return -1

n_jobs_value = get_n_jobs(PARALLEL_LEVEL)

def get_knn_neighbors(complexity: int) -> int:
    """
    Restituisce il numero di vicini K in base al livello di under/overfitting
    per il KNeighborsRegressor:
      - complexity=1 => 50 vicini (underfitting)
      - complexity=2 => 5 vicini (default)
      - complexity=3 => 1 vicino (overfitting)
    """
    if complexity == 1:
        return 50
    elif complexity == 2:
        return 5
    else:
        return 1

def get_rf_params(complexity: int):
    """
    Restituisce (n_estimators, max_depth) per RandomForest in base al livello di fitting:
      - complexity=1 => pochi alberi e max_depth ridotta (underfitting)
      - complexity=2 => configurazione default
      - complexity=3 => molti alberi e nessuna limitazione di profondità (overfitting)
    """
    if complexity == 1:
        return 10, 5
    elif complexity == 2:
        return 100, None
    else:
        return 300, None

def get_lasso_alpha(complexity: int) -> float:
    """
    Restituisce il valore di alpha per Lasso in base al livello di fitting:
      - complexity=1 => alpha più alta => regolarizzazione forte => underfit
      - complexity=2 => alpha=1.0 => default
      - complexity=3 => alpha più bassa => regolarizzazione più debole => overfit
    """
    if complexity == 1:
        return 10.0
    elif complexity == 2:
        return 1.0
    else:
        return 0.01

def elbow_method(X, max_k=10):
    """
    Semplice Elbow Method per individuare un numero di cluster k.
    Calcola la WCSS per k in [2..max_k], quindi restituisce un k_suggerito (euristico).
    """
    wcss = []
    K_range = range(2, max_k+1)
    for k in K_range:
        kmeans_tmp = KMeans(n_clusters=k, random_state=42)
        kmeans_tmp.fit(X)
        wcss.append(kmeans_tmp.inertia_)
    diffs = np.diff(wcss)
    idx_min = np.argmin(diffs)
    k_suggested = 2 + idx_min
    return k_suggested

def ip_to_num(o1, o2, o3, o4) -> int:
    """
    Concatena i 4 ottetti in un unico intero (es: 123.123.123.132 -> 123123123132).
    Utile per rappresentare IP su un asse numerico.
    """
    ip_str = f"{o1}{o2}{o3}{o4}"
    return int(ip_str)

def plot_predictions(y_true, y_pred, model_name, week_name, phase_name, output_folder):
    """
    Scatter plot (y_true vs y_pred) con linea y=x per valutare la qualità del modello di regressione.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.6, c='blue', edgecolors='k')
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.title(f"{model_name} - {phase_name} su {week_name}", fontsize=14)
    plt.xlabel("Valore Reale (y_true)", fontsize=12)
    plt.ylabel("Valore Predetto (y_pred)", fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    file_name = f"{model_name}_{phase_name}_{week_name}.png"
    save_path = os.path.join(output_folder, file_name)
    plt.savefig(save_path, format="png")
    plt.close()

def plot_ecdf(data, column_name, model_name, week_name, output_folder, phase_name):
    """
    Genera e salva il grafico ECDF per un dato array (colonna del dataset).
    ECDF = empirical cumulative distribution function.
    """
    ecdf = ECDF(data)
    plt.figure(figsize=(8, 6))
    plt.step(ecdf.x, ecdf.y, label=f"ECDF di {column_name}")
    plt.title(f"ECDF: {column_name} ({model_name} - {week_name}, {phase_name})")
    plt.xlabel(column_name)
    plt.ylabel("Cumulative Probability")
    plt.grid(True)
    plt.tight_layout()
    file_name = f"ECDF_{model_name}_{column_name}_{week_name}_{phase_name}.png"
    save_path = os.path.join(output_folder, file_name)
    plt.savefig(save_path, format="png")
    plt.close()


def create_correlation_matrix(df, output_folder="RISULTATI_COMPLETI"):
    """
    Genera la matrice di correlazione per il DataFrame 'df' (solo sulle colonne numeriche)
    e la salva come immagine PNG dentro 'output_folder'.
    """

    # Creiamo la cartella di output se non esiste
    os.makedirs(output_folder, exist_ok=True)

    # Selezioniamo solo le colonne numeriche
    numeric_cols = df.select_dtypes(include=["number"]).columns

    # Calcoliamo la matrice di correlazione
    corr_matrix = df[numeric_cols].corr()

    # Creiamo la figura
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        corr_matrix,
        annot=True,        # Mostra il valore della correlazione
        cmap="coolwarm",   # Mappa di colori
        fmt=".2f",         # Formato numerico
        square=True,
        linewidths=.5
    )
    plt.title("Correlation Matrix")
    plt.tight_layout()

    # Salviamo il grafico in PNG
    output_path = os.path.join(output_folder, "correlation_matrix.png")
    plt.savefig(output_path, format="png", dpi=150)
    plt.close()

    print(f"[OK] Correlation matrix salvata in: {output_path}")

# =============================================================================
#                  PERCORSI E NOMI COLONNE
# =============================================================================
# Directory di output e percorsi relativi ai file CSV
OUTPUT_FOLDER = "RISULTATI_COMPLETI"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

TRAIN_FILE = "./dataset_Project6/Week_1.csv"
TEST_FILES = [
    "./dataset_Project6/Week_2.csv",
    "./dataset_Project6/Week_3.csv",
    "./dataset_Project6/Week_4.csv",
    "./dataset_Project6/Week_5.csv",
    "./dataset_Project6/Week_6.csv"
]

# Nomi delle colonne con cui carichiamo i CSV
COLUMN_NAMES = [
    'client_ip', 'server_ip', 'min_rtt', 'min_ttl', 'uniq_byte',
    'timestamp', 'resp_time', 'throughput', 'retransmission', 'edge_node'
]

# Cartelle per i plot generati
PLOTS_FOLDER = os.path.join(OUTPUT_FOLDER, "plots")
ECDF_FOLDER = os.path.join(PLOTS_FOLDER, "ecdf_plots")
SCATTER_FOLDER = os.path.join(PLOTS_FOLDER, "scatter_plots")
CLUSTER_FOLDER = os.path.join(PLOTS_FOLDER, "cluster_plots")

os.makedirs(PLOTS_FOLDER, exist_ok=True)
os.makedirs(ECDF_FOLDER, exist_ok=True)
os.makedirs(SCATTER_FOLDER, exist_ok=True)
os.makedirs(CLUSTER_FOLDER, exist_ok=True)

# =============================================================================
#                  SECTION 1 - Data exploration and pre-processing
# =============================================================================
# In this first section we investigated the provided dataset, in particular the first week indicated 
# as baseline. We analyzed the behaviour of some specific features (the ones playing significant roles)
# on different flow levels. We used data visualization techniques and statistical analysis to understand
# the behaviour of the features and then we filtered out the useless information (outliers).
# First we load the dataset and apply the filter. For each edge node we focused on RTT and we  
# computed statistic values like mean, median and std. The filter drops out the values that are +/- 2*std  
# higher or lower than the mean/median value of RTT (for each edge node). 

# In questa sezione carichiamo il dataset della Week_1, effettuiamo il filtraggio
# degli outlier (opzionale) e, se necessario, scomponiamo gli IP e prepariamo le colonne
# da utilizzare successivamente nella PCA.

# 1) CARICAMENTO WEEK_1 E (OPZIONALE) FILTRAGGIO OUTLIER
print("=== CARICAMENTO WEEK_1 ===")
df_week1 = pd.read_csv(TRAIN_FILE, header=0, names=COLUMN_NAMES)
print(f"Week_1 CSV: {len(df_week1)} righe lette.")

if RUN_OUTLIER_FILTER:
    print("Filtraggio outlier basato su min_rtt (Week_1).")

    # Calcoliamo statistiche (media, std, mediana) per ogni edge_node
    stats_by_node = df_week1.groupby('edge_node')['min_rtt'].agg(['mean', 'std', 'median'])
    # Costruiamo bounds (lower_bound e upper_bound) su base mean±2*std e median±2*std
    stats_by_node['mean_lower_bound'] = stats_by_node['mean'] - 2 * stats_by_node['std']
    stats_by_node['mean_upper_bound'] = stats_by_node['mean'] + 2 * stats_by_node['std']
    stats_by_node['median_lower_bound'] = stats_by_node['median'] - 2 * stats_by_node['std']
    stats_by_node['median_upper_bound'] = stats_by_node['median'] + 2 * stats_by_node['std']

    def filter_by_group(df, stats, method='mean'):
        """
        Funzione che filtra i valori di min_rtt in base a method='mean' o 'median',
        limitandosi a 2*std dal valore medio o mediano di ogni edge_node.
        """
        filtered = []
        for edge_node, group in df.groupby('edge_node'):
            if edge_node not in stats.index:
                filtered.append(group)
                continue
            bounds = stats.loc[edge_node]
            if method == 'mean':
                lb, ub = bounds['mean_lower_bound'], bounds['mean_upper_bound']
            else:
                lb, ub = bounds['median_lower_bound'], bounds['median_upper_bound']
            sub = group[(group['min_rtt'] >= lb) & (group['min_rtt'] <= ub)]
            filtered.append(sub)
        return pd.concat(filtered)

    # Filtro su base mean e median, poi unisco i risultati
    mean_f = filter_by_group(df_week1, stats_by_node, 'mean')
    median_f = filter_by_group(df_week1, stats_by_node, 'median')
    df_week1_filtered = pd.concat([mean_f, median_f]).drop_duplicates()
    print(f"Week_1 originale: {len(df_week1)}")
    print(f"Week_1 filtrata (mean): {len(mean_f)}")
    print(f"Week_1 filtrata (median): {len(median_f)}")
    print(f"Week_1 combinata outlier-free: {len(df_week1_filtered)}")

else:
    df_week1_filtered = df_week1.copy()
    stats_by_node = None
    
create_correlation_matrix(df_week1_filtered, "RISULTATI_COMPLETI")
#==============================

# ------------------ FUNZIONE DI SUPPORTO PER LA STATISTICA E IL PLOT ---------------
def plot_throughput_stats(df, scenario_name, output_folder, order_stats=None):
    """
    Calcola (per edge_node) le statistiche 'order_stats' su 'throughput',
    genera e salva il grafico a barre in un file PNG.
    """
    if order_stats is None:
        order_stats = ['mean','std','median','max','min']

    # Calcolo statistiche (nell'ordine desiderato) e converto in "long format"
    stats_df = df.groupby("edge_node")['throughput'].agg(order_stats).reset_index()
    stats_df_long = stats_df.melt(
        id_vars="edge_node",
        value_vars=order_stats,
        var_name="Statistica",
        value_name="Valore"
    )

    # Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=stats_df_long,
        x="edge_node",
        y="Valore",
        hue="Statistica",
        palette="tab10"
    )
    plt.title(f"Statistiche Throughput - {scenario_name}", fontsize=14)
    plt.xlabel("Edge Node", fontsize=12)
    plt.ylabel("Throughput", fontsize=12)
    plt.legend(title="Statistica", fontsize=10)
    plt.tight_layout()

    # Salvataggio
    file_name = f"throughput_{scenario_name.replace(' ', '_').lower()}.png"
    file_path = os.path.join(output_folder, file_name)
    plt.savefig(file_path, dpi=150, format="png")
    plt.close()
    print(f"[OK] Grafico '{scenario_name}' salvato in: {file_path}")

# ------------------ FILTRAGGIO OUTLIER (mean ± 2*std) O (median ± 2*std) ------------------
def filter_by_throughput(df, stats, method='mean'):
    """
    Filtra i record di 'df' (colonne: edge_node, throughput)
    mantenendo solo quelli che rientrano in [method_lower_bound, method_upper_bound].
    """
    filtered = []
    for edge_node, group in df.groupby('edge_node'):
        if edge_node not in stats.index:
            filtered.append(group)
            continue
        lb = stats.loc[edge_node, f"{method}_lower_bound"]
        ub = stats.loc[edge_node, f"{method}_upper_bound"]
        sub = group[(group['throughput'] >= lb) & (group['throughput'] <= ub)]
        filtered.append(sub)
    return pd.concat(filtered)

# ------------------  NESSUN FILTRAGGIO ------------------
print("\n=== 1) Nessun Filtraggio ===")
plot_throughput_stats(df_week1, "No Filter", OUTPUT_FOLDER,
                      order_stats=['min','max','mean','median','std'])

# ------------------ Calcolo dei bound per mean±2*std e median±2*std ------------------
stats_by_node = df_week1.groupby('edge_node')['throughput'].agg(['mean','std','median'])
stats_by_node['mean_lower_bound']   = stats_by_node['mean']   - 2*stats_by_node['std']
stats_by_node['mean_upper_bound']   = stats_by_node['mean']   + 2*stats_by_node['std']
stats_by_node['median_lower_bound'] = stats_by_node['median'] - 2*stats_by_node['std']
stats_by_node['median_upper_bound'] = stats_by_node['median'] + 2*stats_by_node['std']

# ------------------  FILTRO mean ± 2*std ------------------
print("\n=== 2) Filtraggio (mean ± 2*std) ===")
df_mean_f = filter_by_throughput(df_week1, stats_by_node, method='mean')
print(f"   Righe originali: {len(df_week1)}, Righe dopo (mean±2*std): {len(df_mean_f)}")
plot_throughput_stats(df_mean_f, "Mean Filter", OUTPUT_FOLDER,
                      order_stats=['min','max','mean','median','std'])

# ------------------  FILTRO median ± 2*std ------------------
print("\n=== 3) Filtraggio (median ± 2*std) ===")
df_median_f = filter_by_throughput(df_week1, stats_by_node, method='median')
print(f"   Righe originali: {len(df_week1)}, Righe dopo (median±2*std): {len(df_median_f)}")
plot_throughput_stats(df_median_f, "Median Filter", OUTPUT_FOLDER,
                      order_stats=['min','max','mean','median','std'])

# ------------------  FILTRO COMBINATO (mean ± 2*std) O (median ± 2*std) ------------------
print("\n=== 4) Filtraggio Combinato (mean ± 2*std) O (median ± 2*std) ===")
df_combined = pd.concat([df_mean_f, df_median_f]).drop_duplicates()
print(f"   Righe originali: {len(df_week1)}, Righe dopo (combined): {len(df_combined)}")
plot_throughput_stats(df_combined, "Combined Filter", OUTPUT_FOLDER,
                      order_stats=['min','max','mean','median','std'])

#=============================
def filter_by_group(df, stats, method='mean'):
    filtered = []
    for edge_node, group in df.groupby('edge_node'):
        if edge_node not in stats.index:
            filtered.append(group)
            continue
        bounds = stats.loc[edge_node]
        lb, ub = (bounds[f"{method}_lower_bound"], bounds[f"{method}_upper_bound"])
        sub = group[(group['throughput'] >= lb) & (group['throughput'] <= ub)]
        filtered.append(sub)
    return pd.concat(filtered)

mean_f = filter_by_group(df_week1, stats_by_node, 'mean')
median_f = filter_by_group(df_week1, stats_by_node, 'median')
df_week1_filtered = pd.concat([mean_f, median_f]).drop_duplicates()

print(f"Week_1 originale: {len(df_week1)} righe")
print(f"Week_1 filtrata (mean): {len(mean_f)} righe")
print(f"Week_1 filtrata (median): {len(median_f)} righe")
print(f"Week_1 combinata outlier-free: {len(df_week1_filtered)} righe")

# =============================================================================
#                      3. VISUALIZZAZIONE: THROUGHPUT DOPO IL FILTRAGGIO
# =============================================================================

print("\n=== VISUALIZZAZIONE: THROUGHPUT DOPO IL FILTRAGGIO ===")
throughput_stats_post = df_week1_filtered.groupby("edge_node")['throughput'].agg(['min', 'max', 'mean', 'median', 'std']).reset_index()
throughput_stats_post_long = throughput_stats_post.melt(
    id_vars="edge_node",
    value_vars=["min", "max", "mean", "median", "std"],
    var_name="Statistica",
    value_name="Valore"
)

plt.figure(figsize=(10, 6))
sns.barplot(
    data=throughput_stats_post_long,
    x="edge_node",
    y="Valore",
    hue="Statistica",
    palette="tab10"
)
plt.title("Statistiche del Throughput per Edge Node (Dopo il Filtraggio)", fontsize=16)
plt.xlabel("Edge Node", fontsize=12)
plt.ylabel("Throughput", fontsize=12)
plt.legend(title="Statistiche", fontsize=10)
plt.tight_layout()

plot_path_post = os.path.join(OUTPUT_FOLDER, "throughput_post_filtraggio.png")
plt.savefig(plot_path_post, format="png", dpi=150)
plt.close()
print(f"[OK] Grafico Throughput (dopo il filtraggio) salvato in: {plot_path_post}")



# 2) AGGIUNTA COLONNE IP (SE USE_IP = True) E DEFINIZIONE FEATURE PER PCA
def split_ip(ip):
    """
    Suddivide l'indirizzo IP in 4 ottetti numerici.
    In caso di errore, restituisce [0,0,0,0].
    """
    try:
        return list(map(int, ip.split('.')))
    except:
        return [0, 0, 0, 0]

# Se vogliamo gestire gli IP come feature, scompongo negli ottetti
if USE_IP:
    df_week1_filtered[['s_ip1','s_ip2','s_ip3','s_ip4']] = df_week1_filtered['server_ip'].apply(split_ip).tolist()
    df_week1_filtered[['c_ip1','c_ip2','c_ip3','c_ip4']] = df_week1_filtered['client_ip'].apply(split_ip).tolist()

# Queste colonne verranno utilizzate per la PCA, se abilitata
pca_columns = [
    'min_rtt','min_ttl','uniq_byte',
    'timestamp','resp_time','throughput',
    'retransmission'
]

# Se vogliamo anche gli IP fra le feature da passare in PCA
if USE_IP:
    pca_columns = [
        's_ip1','s_ip2','s_ip3','s_ip4',
        'c_ip1','c_ip2','c_ip3','c_ip4',
        'min_rtt','min_ttl','uniq_byte',
        'timestamp','resp_time','throughput',
        'retransmission'
    ]




# 3) PCA
pca = None
scaler_pca = None

if RUN_PCA:
    print("Eseguo PCA (90% varianza) su Week_1 filtrata.")
    # Standardizzo le colonne selezionate per la PCA
    scaler_pca = StandardScaler()
    X_w1_pca = df_week1_filtered[pca_columns].values
    X_w1_pca_scaled = scaler_pca.fit_transform(X_w1_pca)

    # PCA con n_components=0.90 => mantiene componenti finché non superiamo il 90% di varianza spiegata
    pca = PCA(n_components=0.90)
    pca_features_w1 = pca.fit_transform(X_w1_pca_scaled)

    # Aggiungo le nuove colonne PCA (PCA1, PCA2, ecc.) nel DataFrame
    for i in range(pca_features_w1.shape[1]):
        df_week1_filtered[f'PCA{i+1}'] = pca_features_w1[:, i]

    explained_var = pca.explained_variance_ratio_ * 100
    print("Varianza spiegata (Week_1):")
    for i, ev in enumerate(explained_var, 1):
        print(f"  PCA{i}: {ev:.2f}%")
    
    # ===================== ECDF PLOTTING FOR ALL PCA FEATURES ======================
    print("\nGenerazione degli ECDF per tutte le feature PCA...")

    # Crea la cartella "ECDF_PCA_PLOT" se non esiste
    PLOTS_PCA_FOLDER = os.path.join(OUTPUT_FOLDER, "ECDF_PCA_PLOT")
    os.makedirs(PLOTS_PCA_FOLDER, exist_ok=True)

    # Identifica tutte le colonne che iniziano con 'PCA'
    pca_col_names = [col for col in df_week1_filtered.columns if col.startswith('PCA')]

    # Per ciascuna nuova feature PCA, genera l'ECDF e salva il plot
    for col_name in pca_col_names:
        data_for_ecdf = df_week1_filtered[col_name].values
        ecdf = ECDF(data_for_ecdf)

        # Crea il plot dell'ECDF
        plt.figure(figsize=(8, 6))
        plt.step(ecdf.x, ecdf.y, label=f"ECDF di {col_name}")
        plt.title(f"ECDF: {col_name} (PCA Features - Week_1)")
        plt.xlabel(col_name)
        plt.ylabel("Cumulative Probability")
        plt.grid(True)
        plt.tight_layout()

        # Salvataggio del file
        file_name = f"ECDF_{col_name}_Week_1_PCA.png"
        save_path = os.path.join(PLOTS_PCA_FOLDER, file_name)
        plt.savefig(save_path, format="png")
        plt.close()

    print("ECDF plots per le componenti PCA generati con successo.")

# =============================================================================
#   SECTION 2 - Supervised learning - regression - estimate throughput of the flow
# =============================================================================
# In questa sezione, se RUN_ML=True, addestriamo diversi modelli di regressione
# sulla Week_1, e (facoltativamente) li validiamo o testiamo sulle altre settimane.

# In this section we performed regression for each flow for Throughput. Once we standardized our dataset 
# we considered the baseline week as training set and the other weeks as validation set. We chose 4 ML 
# methods in order to perform the model training: Linear regressor, KNN regressor, Random Forest regressor and Lasso regressor.
#

# Scegliamo quali colonne utilizzare come feature: se abbiamo PCA, usiamo le PCA
if RUN_PCA and pca is not None:
    feature_cols = [c for c in df_week1_filtered.columns if c.startswith('PCA')]
else:
    feature_cols = pca_columns.copy()

# Dividiamo il dataset in X (feature) e y (target)
X_w1_full = df_week1_filtered[feature_cols].values
y_w1_full = df_week1_filtered[TARGET_COL].values

if RUN_ML:
    # Scenario 1: uso tutta Week_1 come training e validazione sulle altre week
    # Scenario 2: splitto Week_1 in train/validation 80/20
    if TRAIN_ON_WEEK1_VALIDATE_OTHERS:
        print("\n[TRAINING] Uso l'INTERA Week_1 per allenare i modelli (niente split interno).")
        X_train_w1 = X_w1_full
        y_train_w1 = y_w1_full
        X_val_w1, y_val_w1 = None, None
    else:
        print("\n[TRAINING+VALIDATION] Faccio train/val split su Week_1 (80/20).")
        X_train_w1, X_val_w1, y_train_w1, y_val_w1 = train_test_split(
            X_w1_full, y_w1_full, test_size=0.2, random_state=42
        )
        print(f"Train set size: {len(X_train_w1)}, Validation set size: {len(X_val_w1)}")

    # Standardizzo (per la regressione) i dati di training, e (se presente) di validation
    scaler_reg = StandardScaler()
    X_train_w1_scaled = scaler_reg.fit_transform(X_train_w1)
    if y_val_w1 is not None:
        X_val_w1_scaled = scaler_reg.transform(X_val_w1)

    # Inizializzo le istanze dei modelli di regressione in base ai relativi FLAG e model complexity
    linreg = None
    knn    = None
    rf     = None
    lasso  = None

    # ------------------ LINEAR REGRESSION ------------------------------------------
    if RUN_LINEAR_REGRESSION:
        # Stampo informazioni sul MODEL_COMPLEXITY (ma la LinearRegression non ha parametri specifici)
        if MODEL_COMPLEXITY == 1:
            print("\n[TRAIN] LinearRegression su Week_1 (UNDERFITTING)")
        elif MODEL_COMPLEXITY == 2:
            print("\n[TRAIN] LinearRegression su Week_1 (DEFAULT)")
        else:
            print("\n[TRAIN] LinearRegression su Week_1 (OVERFITTING)")

        # Costruisco l'oggetto LinearRegression
        try:
            linreg = LinearRegression(n_jobs=n_jobs_value)
        except TypeError:
            # Per versioni sklearn <0.23 potremmo non avere il parametro n_jobs
            linreg = LinearRegression()

        # Fitting
        linreg.fit(X_train_w1_scaled, y_train_w1)

        if RUN_LINEAR_REGRESSION and linreg is not None:
            y_pred_lin_train = linreg.predict(X_train_w1_scaled)
            mse_lin_train = mean_squared_error(y_train_w1, y_pred_lin_train)
            r2_lin_train = r2_score(y_train_w1, y_pred_lin_train)
            print(f"LinearRegression -> MSE={mse_lin_train:.3f}, R2={r2_lin_train:.3f}")

        # Se (not TRAIN_ON_WEEK1_VALIDATE_OTHERS), ho X_val_w1 e y_val_w1 per validazione interna
        if ((TRAIN_ON_WEEK1_VALIDATE_OTHERS) and (y_val_w1 is not None)):
            y_val_pred_linreg = linreg.predict(X_val_w1_scaled)
            mse_lin_val = mean_squared_error(y_val_w1, y_val_pred_linreg)
            r2_lin_val = r2_score(y_val_w1, y_val_pred_linreg)
            print(f"  Validation L.Reg -> MSE: {mse_lin_val:.3f}, R2: {r2_lin_val:.3f}")

    # ------------------ KNN REGRESSOR ----------------------------------------------
    if RUN_KNN:
        n_neighbors = get_knn_neighbors(MODEL_COMPLEXITY)
        print(f"\n[TRAIN] KNeighborsRegressor su Week_1 con n_neighbors={n_neighbors}")
        knn = KNeighborsRegressor(n_neighbors=n_neighbors, n_jobs=n_jobs_value)
        knn.fit(X_train_w1_scaled, y_train_w1)

        if RUN_KNN and knn is not None:
            y_pred_knn_train = knn.predict(X_train_w1_scaled)
            mse_knn_train = mean_squared_error(y_train_w1, y_pred_knn_train)
            r2_knn_train = r2_score(y_train_w1, y_pred_knn_train)
            print(f"KNN Regressor    -> MSE={mse_knn_train:.3f}, R2={r2_knn_train:.3f}")

        if (TRAIN_ON_WEEK1_VALIDATE_OTHERS) and (y_val_w1 is not None):
            y_val_pred_knn = knn.predict(X_val_w1_scaled)
            mse_knn_val = mean_squared_error(y_val_w1, y_val_pred_knn)
            r2_knn_val = r2_score(y_val_w1, y_val_pred_knn)
            print(f"  Validation KNN  -> MSE: {mse_knn_val:.3f}, R2: {r2_knn_val:.3f}")

    # ------------------ RANDOM FOREST REGRESSOR ------------------------------------
    if RUN_RANDOM_FOREST:
        n_estimators, max_depth = get_rf_params(MODEL_COMPLEXITY)
        print(f"\n[TRAIN] RandomForestRegressor su Week_1 con n_estimators={n_estimators}, max_depth={max_depth}")
        rf = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=n_jobs_value
        )
        rf.fit(X_train_w1_scaled, y_train_w1)

        if (TRAIN_ON_WEEK1_VALIDATE_OTHERS) and (y_val_w1 is not None):
            y_val_pred_rf = rf.predict(X_val_w1_scaled)
            mse_rf_val = mean_squared_error(y_val_w1, y_val_pred_rf)
            r2_rf_val = r2_score(y_val_w1, y_val_pred_rf)
            print(f"  Validation RF   -> MSE: {mse_rf_val:.3f}, R2: {r2_rf_val:.3f}")

    # ------------------ LASSO REGRESSOR --------------------------------------------
    if RUN_LASSO:
        alpha_val = get_lasso_alpha(MODEL_COMPLEXITY)
        print(f"\n[TRAIN] Lasso Regressor su Week_1 con alpha={alpha_val}")
        lasso = Lasso(alpha=alpha_val, random_state=42)
        lasso.fit(X_train_w1_scaled, y_train_w1)

        if RUN_LASSO and lasso is not None:
            y_pred_lasso_train = lasso.predict(X_train_w1_scaled)
            mse_lasso_train = mean_squared_error(y_train_w1, y_pred_lasso_train)
            r2_lasso_train = r2_score(y_train_w1, y_pred_lasso_train)
            print(f"Lasso           -> MSE={mse_lasso_train:.3f}, R2={r2_lasso_train:.3f}")

        if (TRAIN_ON_WEEK1_VALIDATE_OTHERS) and (y_val_w1 is not None):
            y_val_pred_lasso = lasso.predict(X_val_w1_scaled)
            mse_lasso_val = mean_squared_error(y_val_w1, y_val_pred_lasso)
            r2_lasso_val = r2_score(y_val_w1, y_val_pred_lasso)
            print(f"  Validation Lasso -> MSE: {mse_lasso_val:.3f}, R2: {r2_lasso_val:.3f}")

    # 2. TESTING/VALIDATION SU WEEK_2..WEEK_6 (SE RUN_TEST_WEEKS = True)
    if RUN_TEST_WEEKS:
        for test_file in TEST_FILES:
            if not os.path.exists(test_file):
                print(f"\nFile non trovato: {test_file}")
                continue

            df_test = pd.read_csv(test_file, header=0, names=COLUMN_NAMES)
            week_name = os.path.splitext(os.path.basename(test_file))[0]

            # Se ho filtrato outlier in Week_1, applico la stessa logica anche sulle altre week
            if RUN_OUTLIER_FILTER and stats_by_node is not None:
                m_f = filter_by_group(df_test, stats_by_node, 'mean')
                md_f = filter_by_group(df_test, stats_by_node, 'median')
                df_test_filtered = pd.concat([m_f, md_f]).drop_duplicates()
            else:
                df_test_filtered = df_test.copy()

            # Scompongo IP se USE_IP è True
            if USE_IP:
                df_test_filtered[['s_ip1','s_ip2','s_ip3','s_ip4']] = df_test_filtered['server_ip'].apply(split_ip).tolist()
                df_test_filtered[['c_ip1','c_ip2','c_ip3','c_ip4']] = df_test_filtered['client_ip'].apply(split_ip).tolist()

            # Se ho PCA, trasformo con gli stessi scaler/PC usati in training
            if RUN_PCA and pca is not None and scaler_pca is not None:
                X_test_raw = df_test_filtered[pca_columns].values
                X_test_scaled = scaler_pca.transform(X_test_raw)
                pca_features_test = pca.transform(X_test_scaled)
                for i in range(pca_features_test.shape[1]):
                    df_test_filtered[f'PCA{i+1}'] = pca_features_test[:, i]
                X_test_final = df_test_filtered[[c for c in df_test_filtered.columns if c.startswith('PCA')]].values
            else:
                X_test_final = df_test_filtered[pca_columns].values

            y_test_final = df_test_filtered[TARGET_COL].values

            phase_name = "Validation" if TRAIN_ON_WEEK1_VALIDATE_OTHERS else "Test"
            print(f"\n=== {phase_name} su {week_name} ===")
            print(f"Dati originali: {len(df_test)}, filtrati: {len(df_test_filtered)}")

            # Applico lo stesso scaler usato in training
            X_test_final_scaled = scaler_reg.transform(X_test_final)

            # Valuto i modelli attivati e genero i plot (se RUN_PLOTS=True)
            if RUN_LINEAR_REGRESSION and linreg is not None:
                y_pred_lin = linreg.predict(X_test_final_scaled)
                mse_lin = mean_squared_error(y_test_final, y_pred_lin)
                r2_lin = r2_score(y_test_final, y_pred_lin)
                print(f"  LinearRegression -> MSE={mse_lin:.3f}, R2={r2_lin:.3f}")
                if RUN_PLOTS:
                    plot_predictions(y_test_final, y_pred_lin, "LinearRegression", week_name, phase_name, PLOTS_FOLDER)

            if RUN_KNN and knn is not None:
                y_pred_knn_ = knn.predict(X_test_final_scaled)
                mse_knn_ = mean_squared_error(y_test_final, y_pred_knn_)
                r2_knn_ = r2_score(y_test_final, y_pred_knn_)
                print(f"  KNN Regressor    -> MSE={mse_knn_:.3f}, R2={r2_knn_:.3f}")
                if RUN_PLOTS:
                    plot_predictions(y_test_final, y_pred_knn_, "KNN", week_name, phase_name, PLOTS_FOLDER)

            if RUN_RANDOM_FOREST and rf is not None:
                y_pred_rf_ = rf.predict(X_test_final_scaled)
                mse_rf_ = mean_squared_error(y_test_final, y_pred_rf_)
                r2_rf_ = r2_score(y_test_final, y_pred_rf_)
                print(f"  RandomForest     -> MSE={mse_rf_:.3f}, R2={r2_rf_:.3f}")
                if RUN_PLOTS:
                    plot_predictions(y_test_final, y_pred_rf_, "RandomForest", week_name, phase_name, PLOTS_FOLDER)

            if RUN_LASSO and lasso is not None:
                y_pred_lasso_ = lasso.predict(X_test_final_scaled)
                mse_lasso_ = mean_squared_error(y_test_final, y_pred_lasso_)
                r2_lasso_ = r2_score(y_test_final, y_pred_lasso_)
                print(f"  Lasso            -> MSE={mse_lasso_:.3f}, R2={r2_lasso_:.3f}")
                if RUN_PLOTS:
                    plot_predictions(y_test_final, y_pred_lasso_, "Lasso", week_name, phase_name, PLOTS_FOLDER)

    print("\n=== Fine script ML ===")

# =============================================================================
#                     SECTION 3 - Unsupervised learning - clustering
# =============================================================================

# We grouped the caches into clusters representing the edge nodes and we implemented a clustering 
# algorithm to group caches into edge nodes for identifying different edge-node configurations. In order to
# represent cache by means of features we aggregated them for each unique server IP. To identify the edge-nodes we implemented 
# a clustering algorithm that used 3 numeric features (min_rtt, min_ttl and uniq_byte) instead of the textual code.  

# Qui ci occupiamo di tecniche di clustering (K-Means) su Week_1, con possibilità
# di calcolare automaticamente il numero di cluster via Elbow e Silhouette. 
# Generiamo inoltre grafici 2D/3D e valutiamo le performance del clustering.

def elbow_method(X, max_k=15):
    """
    Calcola WCSS per determinare il numero ideale di cluster usando l'Elbow Method
    e salva il plot in CLUSTER_FOLDER. Stampa inoltre il k suggerito.
    """
    wcss = []
    for k in range(2, max_k+1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)

    plt.figure(figsize=(8, 6))
    plt.plot(range(2, max_k+1), wcss, marker='o', linestyle='--')
    plt.title('Elbow Method: Determinazione del numero ideale di cluster')
    plt.xlabel('Numero di cluster (k)')
    plt.ylabel('WCSS')
    plt.grid(True)
    elbow_plot_path = os.path.join(CLUSTER_FOLDER, "elbow_method.png")
    plt.savefig(elbow_plot_path, format="png")
    plt.close()
    print(f"\nElbow Method plot salvato in: {elbow_plot_path}")

    diffs = np.diff(wcss)
    second_diffs = np.diff(diffs)
    suggested_k = np.argmax(second_diffs) + 2
    print(f"Numero suggerito di cluster (Elbow): {suggested_k}")
    return suggested_k

def silhouette_analysis(X, max_k=15):
    """
    Determina il numero ideale di cluster usando l'analisi Silhouette.
    Campiona fino a 10000 punti (se il dataset è più grande) per velocizzare i calcoli.
    Stampa il k con il punteggio Silhouette più alto.
    """
    silhouette_scores = []
    k_range = range(2, max_k+1)
    X_sample = resample(X, n_samples=10000, random_state=42)

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X_sample)
        score = silhouette_score(X_sample, labels)
        silhouette_scores.append(score)

    plt.figure(figsize=(8, 6))
    plt.plot(k_range, silhouette_scores, marker='o', linestyle='--')
    plt.title('Silhouette Analysis: Determinazione del numero ideale di cluster')
    plt.xlabel('Numero di cluster (k)')
    plt.ylabel('Silhouette Score')
    plt.grid(True)
    silhouette_plot_path = os.path.join(CLUSTER_FOLDER, "silhouette_analysis.png")
    plt.savefig(silhouette_plot_path, format="png")
    plt.close()
    print(f"\nSilhouette Analysis plot salvato in: {silhouette_plot_path}")

    suggested_k = np.argmax(silhouette_scores) + 2
    print(f"Numero suggerito di cluster (Silhouette): {suggested_k}")
    return suggested_k

def evaluate_clustering_performance(X1, labels1, method_name):
    """
    Valuta le performance del clustering usando:
     - Silhouette Score (più alto => meglio)
     - Davies-Bouldin Index (più basso => meglio)
     - Calinski-Harabasz Index (più alto => meglio)
    """
    X = resample(X1, n_samples=10000, random_state=42)
    labels = resample(labels1, n_samples=10000, random_state=42)

    silhouette = silhouette_score(X, labels)
    davies_bouldin = davies_bouldin_score(X, labels)
    calinski_harabasz = calinski_harabasz_score(X, labels)

    print(f"\n=== Performance del clustering ({method_name}) ===")
    print(f"Silhouette Score: {silhouette:.4f}")
    print(f"Davies-Bouldin Index: {davies_bouldin:.4f} (meno è meglio)")
    print(f"Calinski-Harabasz Index: {calinski_harabasz:.4f} (più è meglio)")

def plot_clusters_ip_2d(df, x_col, y_col, cluster_col, output_path="cluster_ip_plot.png"):
    """
    Plot 2D (Server IP vs Client IP numerici) colorando i punti in base al cluster.
    """
    plt.figure(figsize=(8,6))
    sns.scatterplot(
        data=df,
        x=x_col,
        y=y_col,
        hue=cluster_col,
        palette='viridis',
        alpha=0.8,
        edgecolor='k'
    )
    plt.title("Clusters su IP numerici (Server vs Client)")
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.legend(title=cluster_col, loc='best')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, format="png")
    plt.close()

def plot_clusters_pca_3d(df, cluster_col, pca_cols=('PCA1','PCA2','PCA3'), output_path="cluster_pca_3d.png"):
    """
    Plot 3D delle prime tre componenti PCA, colorando i punti in base al cluster.
    """
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(
        df[pca_cols[0]],
        df[pca_cols[1]],
        df[pca_cols[2]],
        c=df[cluster_col],
        cmap='viridis',
        edgecolor='k',
        alpha=0.8
    )
    ax.set_title("Cluster su prime 3 PCA (3D)")
    ax.set_xlabel(pca_cols[0])
    ax.set_ylabel(pca_cols[1])
    ax.set_zlabel(pca_cols[2])
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label(cluster_col)
    plt.tight_layout()
    plt.savefig(output_path, format="png")
    plt.close()

def plot_pca_3d_edge_node(df, edge_node_col='edge_node', pca_cols=('PCA1','PCA2','PCA3'), output_path="pca_3d_edgenode.png"):
    """
    Plot 3D delle prime tre PCA, colorando i punti in base all'edge_node effettivo
    (ricodificato con un LabelEncoder interno).
    """
    from mpl_toolkits.mplot3d import Axes3D
    le_edge = LabelEncoder()
    df['edge_node_label'] = le_edge.fit_transform(df[edge_node_col])

    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(
        df[pca_cols[0]],
        df[pca_cols[1]],
        df[pca_cols[2]],
        c=df['edge_node_label'],
        cmap='viridis',
        edgecolor='k',
        alpha=0.8
    )
    ax.set_title("Prime 3 PCA (3D) - Edge Node Reale")
    ax.set_xlabel(pca_cols[0])
    ax.set_ylabel(pca_cols[1])
    ax.set_zlabel(pca_cols[2])
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label(edge_node_col)
    plt.tight_layout()
    plt.savefig(output_path, format="png")
    plt.close()


# =============================================================================
# This function forcibly applies K-Means clustering with k=5 to the given dataset,
# focusing on the 'retransmission' variable. For each of the five clusters, it creates
# an Empirical Cumulative Distribution Function (ECDF) plot depicting the distribution
# of retransmission values. These plots are saved into a folder named "3.7 COARSE."
# By examining the ECDF plots, one can identify clusters with unusually high or low
# retransmissions and thus gain insight into potentially problematic network flows.
# =============================================================================

def cluster_and_ecdf_by_retransmission(cluster_df, X_cluster, output_main_folder):

    # Create the folder "3.7 COARSE" if it doesn't exist
    coarse_folder = os.path.join(output_main_folder, "3.7 COARSE")
    os.makedirs(coarse_folder, exist_ok=True)

    # Force K-Means clustering with k=5 that is the value predticted by Silhouette_analysis
    k = 5
    print(f"\n=== Forcing K-Means with k={k} for additional analysis ===")
    kmeans_5 = KMeans(n_clusters=k, random_state=42)
    cluster_labels_5 = kmeans_5.fit_predict(X_cluster)

    # Add the new labels to the DataFrame
    cluster_df['cluster_5'] = cluster_labels_5

    # For each cluster, build an ECDF of 'retransmission' and save the plot
    for cluster_id in range(k):
        subset = cluster_df[cluster_df['cluster_5'] == cluster_id]

        # Extract the 'retransmission' values for the flows in this cluster
        data_retrans = subset['retransmission'].values

        # Compute the Empirical Cumulative Distribution Function for retransmission
        ecdf = ECDF(data_retrans)

        # Plot the ECDF
        plt.figure(figsize=(8, 6))
        plt.step(ecdf.x, ecdf.y, label=f"Cluster {cluster_id} - Retransmission ECDF", where='post')
        plt.title(f"ECDF of Retransmission - Cluster {cluster_id}")
        plt.xlabel("retransmission")
        plt.ylabel("Cumulative Probability")
        plt.grid(True)
        plt.tight_layout()

        # Save the plot
        plot_filename = f"ECDF_cluster_{cluster_id}_retransmission.png"
        plot_path = os.path.join(coarse_folder, plot_filename)
        plt.savefig(plot_path, format="png")
        plt.close()

    print(f"\nECDF plots for each cluster (grouped by retransmission) have been saved in: {coarse_folder}")

# Se RUN_CLUSTERING=True, eseguiamo K-Means sulla Week_1.
if RUN_CLUSTERING:
    print("\n=== CLUSTERING K-MEANS (PCA + edge_node) ===")
    cluster_df = df_week1_filtered.copy()
    le = LabelEncoder()
    cluster_df['edge_node_label'] = le.fit_transform(cluster_df['edge_node'].astype(str))

    # Se ho PCA, definisco pca_cols e uso quelle come feature
    if RUN_PCA and pca is not None and pca_features_w1 is not None:
        pca_cols = [c for c in cluster_df.columns if c.startswith('PCA')]
        cluster_features = cluster_df[pca_cols].copy()
        cluster_features['edge_node_label'] = cluster_df['edge_node_label']
    else:
        # Se PCA_ONLY_CORE => useremo solo [min_rtt,min_ttl,uniq_byte] + edge_node_label
        # altrimenti useremo la logica estesa, eventualmente con IP
        if PCA_ONLY_CORE:
            cluster_features = cluster_df[['min_rtt','min_ttl','uniq_byte']].copy()
            cluster_features['edge_node_label'] = cluster_df['edge_node_label']
        else:
            if USE_IP:
                cluster_features = cluster_df[[
                    's_ip1','s_ip2','s_ip3','s_ip4',
                    'c_ip1','c_ip2','c_ip3','c_ip4',
                    'min_rtt','min_ttl','uniq_byte',
                    'timestamp','resp_time','throughput','retransmission'
                ]].copy()
                cluster_features['edge_node_label'] = cluster_df['edge_node_label']
            else:
                cluster_features = cluster_df[[
                    'min_rtt','min_ttl','uniq_byte',
                    'timestamp','resp_time','throughput','retransmission'
                ]].copy()
                cluster_features['edge_node_label'] = cluster_df['edge_node_label']

    # Standardizzazione prima di KMeans
    scaler_cluster = StandardScaler()
    X_cluster = scaler_cluster.fit_transform(cluster_features)

    # KMEANS_AUTO => determina automaticamente il k ideale con Elbow e Silhouette
    if KMEANS_AUTO:
        print("\n=== CLUSTERING K-MEANS ===")
        cluster_df = df_week1_filtered.copy()

        scaler_cluster = StandardScaler()
        X_cluster = scaler_cluster.fit_transform(cluster_df[feature_cols])

        print("\n=== Determinazione del numero di cluster ===")
        k_elbow = elbow_method(X_cluster, max_k=10)
        k_silhouette = silhouette_analysis(X_cluster, max_k=10)

        final_k = max(k_elbow, k_silhouette)
        print(f"\nNumero finale di cluster scelto: {final_k}")
        n_clusters = final_k
    else:
        n_clusters = KMEANS_N_CLUSTERS

    # Applico K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(X_cluster)
    cluster_df['cluster'] = cluster_labels

    # Valuto il clustering
    evaluate_clustering_performance(X_cluster, cluster_labels, "KMeans")
    cluster_and_ecdf_by_retransmission(cluster_df, X_cluster, OUTPUT_FOLDER)


    # Se voglio concatenare i 4 ottetti come IP numerici
    def ip_concat_srv(row):
        return ip_to_num(row['s_ip1'], row['s_ip2'], row['s_ip3'], row['s_ip4'])
    def ip_concat_cli(row):
        return ip_to_num(row['c_ip1'], row['c_ip2'], row['c_ip3'], row['c_ip4'])

    if USE_IP:
        cluster_df['server_ip_num'] = cluster_df.apply(ip_concat_srv, axis=1)
        cluster_df['client_ip_num'] = cluster_df.apply(ip_concat_cli, axis=1)

    # Plot 2D IP numerici
    if RUN_PLOTS and RUN_PLOTS_CLUSTER and RUN_PLOTS_IP_2D and USE_IP:
        plot_clusters_ip_2d(
            cluster_df,
            x_col='server_ip_num',
            y_col='client_ip_num',
            cluster_col='cluster',
            output_path=os.path.join(CLUSTER_FOLDER, "cluster_ip_plot.png")
        )

    # Plot 3D su prime 3 PCA -> cluster
    if RUN_PLOTS and RUN_PLOTS_CLUSTER and RUN_PLOTS_PCA_3D and RUN_PCA:
        if 'PCA3' in cluster_df.columns:
            plot_clusters_pca_3d(
                df=cluster_df,
                cluster_col='cluster',
                pca_cols=('PCA1','PCA2','PCA3'),
                output_path=os.path.join(CLUSTER_FOLDER, "cluster_pca_3d.png")
            )

    # Plot 3D su prime 3 PCA -> edge_node reale
    if RUN_PLOTS and RUN_PLOTS_CLUSTER and RUN_PLOTS_PCA_3D_EDGENODE and RUN_PCA:
        if 'PCA3' in cluster_df.columns:
            plot_pca_3d_edge_node(
                df=cluster_df,
                edge_node_col='edge_node',
                pca_cols=('PCA1','PCA2','PCA3'),
                output_path=os.path.join(CLUSTER_FOLDER, "pca_3d_edgenode.png")
            )

print("\n=== Fine script con PCA_ONLY_CORE completato ===")


# 2. CONFUSION MATRIX FOR WEEK_1
# Generiamo una confusion matrix per la Week_1 confrontando le label edge_node
# con i cluster ottenuti via KMeans (n_clusters=4).
print("\nCreazione della Confusion Matrix per Week 1...")
data_week1 = df_week1_filtered.copy()
features_week1 = data_week1.select_dtypes(include=[np.number])
kmeans_week1 = KMeans(n_clusters=4, random_state=42)
clusters_week1 = kmeans_week1.fit_predict(features_week1)

true_labels = data_week1['edge_node'].values
predicted_labels = clusters_week1
le_node = LabelEncoder()
true_labels_encoded = le_node.fit_transform(true_labels)

cm = confusion_matrix(true_labels_encoded, predicted_labels)
display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le_node.classes_)

plt.figure(figsize=(10, 8))
display.plot()
plt.title('Confusion Matrix per Week 1')
plt.savefig('RISULTATI_COMPLETI/confusion_matrix_week1.png')
plt.close()
print("Confusion Matrix salvata con successo.")

# =============================================================================
#                     SECTION 4 - Clustering evolution
# =============================================================================

# To analyze how clusters evolve over time, we start by comparing the clustering results of week 2 (C2) to cluster C1 (related to week 1), 
# which serves as the baseline. We calculate the similarity between these results and sum all the distances to estimate the overall difference.
# This process is repeated for every pair of weeks, resulting in a matrix that contains all the distances. The matrix is then visualized using a heatmap, 
# allowing us to observe how the distances between weeks vary over time. Finally, we identify the week most different from the baseline by finding the one 
# with the highest distance from the baseline week in the matrix.

# In questa sezione applichiamo il clustering a tutte le settimane (week_1..week_6)
# e generiamo i corrispondenti plot e confusion matrix.

#      1. CLUSTERING DELLE ALTRE SETTIMANE (SE CLUSTER_OTHER_WEEKS = True)
if CLUSTER_OTHER_WEEKS:
    # Caricamento dei dati per le diverse settimane in un dizionario
    data_weeks = {
        'week_1': pd.read_csv('./dataset_Project6/Week_1.csv'),
        'week_2': pd.read_csv('./dataset_Project6/Week_2.csv'),
        'week_3': pd.read_csv('./dataset_Project6/Week_3.csv'),
        'week_4': pd.read_csv('./dataset_Project6/Week_4.csv'),
        'week_5': pd.read_csv('./dataset_Project6/Week_5.csv'),
        'week_6': pd.read_csv('./dataset_Project6/Week_6.csv')
    }

    n_clusters = 5  # Esempio: numero di cluster determinato con metodo del gomito o silhouette
    random_state = 42

    def apply_clustering(data, week, n_clusters, random_state):
        """
        Esegue un KMeans(n_clusters, random_state) su 'data',
        produce due plot (2D e 3D) e restituisce data con colonna 'cluster'.
        """
        features = data.select_dtypes(include=[np.number])
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        clusters = kmeans.fit_predict(features)
        data['cluster'] = clusters 

        # Plot 2D (usando come x e y le prime due colonne numeriche, es. RTT e TTL)
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(features.iloc[:, 0], features.iloc[:, 1], 
                              c=clusters, cmap='viridis', s=50, edgecolor='k')
        plt.title(f'Resultati del Clustering - Settimana {week}', fontsize=16)
        plt.xlabel("RTT", fontsize=12)
        plt.ylabel("TTL", fontsize=12)
        plt.colorbar(scatter, label='Cluster')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(f'RISULTATI_COMPLETI/cluster_week_{week}.png')
        plt.close()

        # Plot 3D (se ci sono almeno 3 colonne numeriche)
        if features.shape[1] >= 3:
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')
            scatter_3d = ax.scatter(
                features.iloc[:, 0],
                features.iloc[:, 1],
                features.iloc[:, 2],
                c=clusters, cmap='viridis', s=50, edgecolor='k'
            )
            ax.set_title(f'Resultati del Clustering - Settimana {week} (3D)', fontsize=16)
            plt.xlabel("RTT", fontsize=12)
            plt.ylabel("TTL", fontsize=12)
            ax.set_zlabel("UNIQ BYTE", fontsize=12)
            fig.colorbar(scatter_3d, label='Cluster')
            plt.savefig(f'RISULTATI_COMPLETI/cluster_week_{week}_3D.png')
            plt.close()
        return data

    # Clusterizzo ciascuna settimana, memorizzando i risultati in clustering_results
    clustering_results = {}
    for week in range(1, 7):
        print(f'Processing clustering for week {week}...')
        clustering_results[f'week_{week}'] = apply_clustering(
            data_weeks[f'week_{week}'], week, n_clusters, random_state)
        
    print("Clustering completato per tutte le settimane.")

# 2) CREAZIONE DELLA CONFUSION MATRIX PER OGNI WEEK
def create_and_save_confusion_matrix(df, week_label, n_clusters=4):
    """
    Esegue un KMeans su 'df' (solo colonne numeriche), crea la confusion matrix 
    confrontando 'edge_node' vs cluster, e salva il grafico come PNG.
    """
    # Seleziona colonne numeriche
    features = df.select_dtypes(include=[np.number])
    
    # Esegui KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    predicted_labels = kmeans.fit_predict(features)

    # True labels = edge_node
    true_labels = df['edge_node'].values
    le_node = LabelEncoder()
    true_labels_encoded = le_node.fit_transform(true_labels)

    # Creazione della Confusion Matrix
    cm = confusion_matrix(true_labels_encoded, predicted_labels)
    display_cm = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le_node.classes_)

    # Plot e salvataggio
    plt.figure(figsize=(10, 8))
    display_cm.plot()
    plt.title(f'Confusion Matrix per Week {week_label}')
    filename = f'confusion_matrix_week{week_label}.png'
    save_path = os.path.join(OUTPUT_FOLDER, filename)
    plt.savefig(save_path)
    plt.close()
    print(f"[OK] Confusion Matrix salvata per Week {week_label}: {save_path}")



# -------------- COSTRUZIONE DELLA CONFUSION MATRIX PER WEEK_1..WEEK_6 --------------
# Abbiamo già la conf. matrix per Week_1 (nel tuo codice originale).
# Ora la generiamo anche per week_2..week_6 (nel dizionario 'clustering_results' o dove preferisci).

# Esempio: supponiamo che:
#  - df_week1_filtered sia il DF della week_1
#  - clustering_results['week_2'] ... 'week_6'] siano i DF delle altre settimane.
#  - Usa n_clusters = 4 se vuoi la stessa configurazione di week_1 (oppure 5, come hai fatto in Part4)

# Ricreo la Confusion Matrix per Week_1 (n_clusters=4) e poi per Weeks 2..6
create_and_save_confusion_matrix(df_week1_filtered, "1", n_clusters=4)

for w in range(2, 7):
    df_w = clustering_results.get(f'week_{w}')
    if df_w is not None:
        create_and_save_confusion_matrix(df_w, str(w), n_clusters=4)
    else:
        print(f"[ATTENZIONE] Nessun DataFrame per week_{w}")

# 3) FUNZIONE PER CALCOLARE DISTANZA FRA DUE CLUSTERING E MATRICE DELLE DISTANZE
def compute_cluster_centroids(df, cluster_col="cluster", sample_size=100, feature_cols=None):
    """
    Per ciascun cluster in df[cluster_col], campiona fino a sample_size righe
    e calcola la media (centroide) sulle feature in feature_cols (o su tutte le numeric).
    Restituisce un dizionario {cluster_id -> centroid_array}.
    """
    centroids_dict = {}
    if feature_cols is None:
        # se non specificato, estraggo le colonne numeriche, escludendo il cluster_col
        feature_cols = df.select_dtypes(include=[np.number]).columns
        feature_cols = [c for c in feature_cols if c != cluster_col]

    for cl_id, group in df.groupby(cluster_col):
        # Eseguo un resample limitando a sample_size per velocizzare
        if len(group) > sample_size:
            group_sampled = resample(group, n_samples=sample_size, random_state=42)
        else:
            group_sampled = group

        centroid = group_sampled[feature_cols].mean().values
        centroids_dict[cl_id] = centroid
    
    return centroids_dict

def clustering_distance(df1, df2, cluster_col="cluster", sample_size=100, feature_cols=None):
    """
    Distanza fra due clustering (df1 e df2), definita come:
      - per ogni cluster in df1, cerco il cluster in df2 più vicino (centroide)
      - sommo queste distanze
      - ripeto invertendo df1 e df2
      - restituisco la somma delle due parti
    """
    centroids_1 = compute_cluster_centroids(df1, cluster_col, sample_size, feature_cols)
    centroids_2 = compute_cluster_centroids(df2, cluster_col, sample_size, feature_cols)
    
    if not centroids_1 or not centroids_2:
        return 0.0

    # Estraggo le coordinate in array (c1_points, c2_points)
    c1_ids, c1_points = zip(*centroids_1.items())
    c2_ids, c2_points = zip(*centroids_2.items())
    c1_points = np.vstack(c1_points)
    c2_points = np.vstack(c2_points)

    # Matrice delle distanze euclidee fra centroidi
    dist_matrix = cdist(c1_points, c2_points, metric='euclidean')
    
    # min_dist_1to2 = sommo la minima distanza di ogni cluster di df1 verso i cluster di df2
    min_dist_1to2 = dist_matrix.min(axis=1).sum()
    # min_dist_2to1 = analogo, invertendo df1 e df2
    min_dist_2to1 = dist_matrix.min(axis=0).sum()

    return min_dist_1to2 + min_dist_2to1

def compute_distance_matrix(clustering_dict, cluster_col="cluster", sample_size=100, feature_cols=None):
    """
    Costruisce la matrice di distanza fra tutti i pair di week (week_1..week_6),
    data un dizionario di DataFrame clusterizzati (clustering_dict).
    'clustering_dict' è un dizionario es. {'week_2': df_week2, ...} 
    con colonna cluster_col = cluster.
    """
    weeks = list(clustering_dict.keys())
    dist_mat = pd.DataFrame(
        data=np.zeros((len(weeks), len(weeks))),
        index=weeks,
        columns=weeks
    )
    
    for i, w_i in enumerate(weeks):
        for j, w_j in enumerate(weeks):
            if i == j:
                dist_mat.loc[w_i, w_j] = 0.0
            elif i < j:
                d = clustering_distance(
                    clustering_dict[w_i],
                    clustering_dict[w_j],
                    cluster_col=cluster_col,
                    sample_size=sample_size,
                    feature_cols=feature_cols
                )
                dist_mat.loc[w_i, w_j] = d
                dist_mat.loc[w_j, w_i] = d
    
    return dist_mat

def plot_distance_matrix(distance_matrix, title="Clustering Distance Matrix"):
    """
    Disegna e salva la heatmap della matrice di distanza generata.
    """
    plt.figure(figsize=(8,6))
    sns.heatmap(distance_matrix, annot=True, cmap='viridis')
    plt.title(title)
    plt.tight_layout()
    
    fname = "distance_matrix.png"
    outpath = os.path.join(OUTPUT_FOLDER, fname)
    plt.savefig(outpath)
    plt.close()
    print(f"[OK] Distance matrix salvata in {outpath}")

###############################################################################
#  CREAZIONE DISTANCE MATRIX
###############################################################################
if __name__ == "__main__":

  # Costruiamo una matrice di distanza
  print("\n[INFO] Calcolo matrice di distanza tra i clustering (weeks 1..6)")
  distance_mat = compute_distance_matrix(
      clustering_dict=clustering_results,  # ad es. con chiavi 'week_2'..'week_6'
      cluster_col='cluster', 
      sample_size=100,
      feature_cols=None
  )

  print(distance_mat)
  plot_distance_matrix(distance_mat, title="Distance Matrix (All Weeks)")

  print("\n=== Fine delle funzioni aggiuntive ===")
