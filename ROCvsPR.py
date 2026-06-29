import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

# --- Configuración de tipografía estilo LaTeX para la Tesis ---
plt.rcParams.update({
    "text.usetex": False,            # Cambiar a True si tienes LaTeX instalado en el sistema
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "DejaVu Serif", "Times New Roman"],
    "font.size": 11,                # Ajustado para que luzca bien en un texto de 12pt
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10
})

# 1. Dataset sintético binario
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 2. Pipeline básico
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression())
])
pipeline.fit(X_train, y_train)

# 3. Probabilidades
y_scores = pipeline.predict_proba(X_test)[:, 1]

# 4. Cálculo de métricas
fpr, tpr, _ = roc_curve(y_test, y_scores)
roc_auc = auc(fpr, tpr)

precision, recall, _ = precision_recall_curve(y_test, y_scores)
ap_score = average_precision_score(y_test, y_scores)

# 5. Graficar ambas curvas en los mismos ejes (No partida)
# Usamos subplots con un solo panel para controlar el facecolor limpiamente
fig, ax = plt.subplots(figsize=(6.5, 5), facecolor='white')

# Curva ROC
ax.plot(fpr, tpr, color='#1f77b4', lw=2, label=f'Curva ROC (AUC = {roc_auc:.2f})')

# Curva Precision-Recall
ax.plot(recall, precision, color='#2ca02c', lw=2, label=f'Curva Precision-Recall (AP = {ap_score:.2f})')

# Línea de referencia aleatoria para ROC
ax.plot([0, 1], [0, 1], color='gray', linestyle='--', alpha=0.7, label='Clasificador aleatorio (ROC)')

# Configuración de los ejes
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('Ratio Falsos Positivos (FPR) / Recall (Exhaustividad)')
ax.set_ylabel('Ratio Verdaderos Positivos (TPR) / Precision (Precisión)')
ax.set_title('Comparación de Curvas ROC y Precision-Recall')
ax.legend(loc="lower left", frameon=True, facecolor='white', edgecolor='gainsboro')
ax.grid(True, linestyle=':', alpha=0.6)
ax.set_facecolor('white')

plt.tight_layout()

# 6. Guardar en PDF con fondo blanco absoluto
plt.savefig('curva_combinada_tesis.pdf', facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')