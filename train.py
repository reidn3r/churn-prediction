# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import mlflow

import os
from dotenv import load_dotenv

load_dotenv()

mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")
mlflow_xp_id = 1

mlflow.set_tracking_uri(mlflow_uri)
mlflow.set_experiment(experiment_id=mlflow_xp_id)

sns.set_style("whitegrid")
sns.set_palette("husl")

pd.options.display.max_columns = 100
pd.options.display.max_rows = 100
# %%
path = os.getenv("DATA_PATH")
df = pd.read_csv(path)

print(f'df shape: {df.shape}')
df.head()
# %%
cols = df.columns
for c in cols:
  print(f'{c}: {df[c].dtype}')

df.describe()
# %%
print(df['dtRef'].value_counts().sort_index())

dates = df['dtRef'].value_counts().sort_index()
oot = df[df['dtRef'] == dates.index[-1]].copy()

print(f'oot shape: {oot.shape}')
oot.head()
# %%
df_train = df[df['dtRef'] != dates.index[-1]].copy()
print(f'train shape: {df_train.shape}')
# %%
feature_cols, target_col = df_train.columns[2: -1], df_train.columns[-1]
X, y = df_train[feature_cols], df_train[target_col]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, 
  test_size=0.25,
  stratify=y,
  shuffle=True,
  random_state=42, 
  )

print(f'xtrain shape: {X_train.shape}')
print(f'xtest shape: {X_test.shape}')

'''
  - Verifica se as distriuições das variáveis objetivos 
    em treino e teste são semelhanets
  
  - stratify: Garante que as duas amostras tenham a mesma taxa da variáveil resposta
    Operação custosa, fazer com cautela
'''
print(f'train churn rate: {y_train.mean()}')
print(f'test churn rate: {y_test.mean()}')
# %%
nrows, ncols = 10, 4

fig, axes = plt.subplots(nrows, ncols, figsize=(20, 25))
axes_flat = axes.flatten()

for i in range(nrows):
  for j in range(ncols):
    idx = i * ncols + j
    sns.kdeplot(data=df_train, x=feature_cols[idx], hue=target_col, 
                fill=True, alpha=0.5, ax=axes[i,j])

plt.tight_layout()
plt.show()
# %%
X_train.isna().sum().sort_values(ascending=True)
# %%
df_bivar = X_train.copy()
df_bivar[target_col] = y_train

agg = df_bivar.groupby(by=target_col).agg(['median', 'mean']).T
agg['diff'] = agg.iloc[:, 0] / agg.iloc[:, 1]
agg.sort_values(by='diff', ascending=False)

agg
# %%
from sklearn.tree import DecisionTreeClassifier, plot_tree

dtf = DecisionTreeClassifier(random_state=42, max_depth=5)
dtf.fit(X_train, y_train)

plt.figure(dpi=800, figsize=(10, 10))
_ = plot_tree(
  dtf,
  feature_names=X_train.columns,
  class_names = ["Nao Churn", "Churn"],
  filled=True,
  )
# %%
#Feature Importance
dtf = DecisionTreeClassifier(random_state=42)
dtf.fit(X_train, y_train)

importance = pd.Series(dtf.feature_importances_, index=X_train.columns)

fig, ax = plt.subplots(figsize=(8, 12))
sns.barplot(
  x=importance.sort_values(ascending=False).values,
  y=importance.sort_values(ascending=False).index,
  ax=ax
)
ax.set_title("Feature Importance")
ax.set_xlabel("Importance")
plt.tight_layout()
plt.show()
# %%
importance_sum = (
    importance
    .sort_values(ascending=False)
    .reset_index()
    .rename(columns={'index': 'feature', 0: 'importance'})
)
importance_sum['acumulada'] = importance_sum['importance'].cumsum()
importance_sum
# %%
selected_features = importance_sum[importance_sum['acumulada'] < 0.95]
feature_list = selected_features['feature'].to_list()

X_train = X_train[feature_list].copy()
X_test = X_test[feature_list].copy()
X_oot = oot[feature_list].copy()
# %%
from sklearn import pipeline
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from feature_engine import discretisation, encoding
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV

tree_discretisation = discretisation.DecisionTreeDiscretiser(
  variables=feature_list, 
  regression=False, #classificacao
  bin_output='bin_number',
  cv=3,
)

ohe = encoding.OneHotEncoder(
  variables=feature_list,
  ignore_format=True,
  )

model = RandomForestClassifier(random_state=42,)

# model = AdaBoostClassifier(
#   random_state=42,
#   learning_rate=1e-3,
# )

grid_params = {
  "n_estimators": [400, 600],
  "max_depth": [50, 200, 600],
  "min_samples_leaf": [20, 50, 100]
}

grid = GridSearchCV(
  estimator=model,
  param_grid=grid_params,
  cv=3,
  scoring="roc_auc",
  verbose=3
)

model_name = model.__class__.__name__
ohe_name = "One Hot"
discretizer_name = "Discretizer"
gs_name = grid.__class__.__name__

# model = LogisticRegression(random_state=42)
model_pipeline = pipeline.Pipeline(
  steps=[
    (discretizer_name, tree_discretisation),
    (ohe_name, ohe),
    (gs_name, grid),
  ],
)

with mlflow.start_run(run_name=model_name):
  mlflow.sklearn.autolog()
  model_pipeline.fit(X_train, y_train)

  predict_proba = model_pipeline.predict_proba(X_train)[:, 1]
  predict = model_pipeline.predict(X_train)

  acc_train = accuracy_score(y_train, predict)
  roc = roc_auc_score(y_train, predict_proba)
  train_curve = roc_curve(y_train, predict_proba)

  test_predict_proba = model_pipeline.predict_proba(X_test)[:, 1]
  test_predict = model_pipeline.predict(X_test)

  acc_test = accuracy_score(y_test, test_predict)
  roc_test = roc_auc_score(y_test, test_predict_proba)
  test_curve = roc_curve(y_test, test_predict_proba)

  oot_predict_proba = model_pipeline.predict_proba(X_oot)[:, 1]
  oot_predict = model_pipeline.predict(X_oot)

  acc_oot = accuracy_score(oot[target_col], oot_predict)
  roc_oot = roc_auc_score(oot[target_col], oot_predict_proba)
  ooc_curve = roc_curve(oot[target_col], oot_predict_proba)

  mlflow.log_metrics({
    "acc_train": acc_train,
    "auc_train": roc,
    "acc_test": acc_test,
    "auc_test": roc_test,
    "acc_oot": acc_oot,
    "auc_oot": roc_oot,
  })
 
# %%
gs_results = (
  pd
    .DataFrame(grid.cv_results_)
    .sort_values(by="rank_test_score", ascending=True)
  )

print(f"Melhor params: {grid.best_params_}")
print(f"Melhor AUC: {grid.best_score_:.4f}")

gs_results.head()
# %%
plt.figure(figsize=(10, 8))
plt.plot(train_curve[0], train_curve[1], 
  label=f'Train ROC (AUC = {roc:.3f})', 
  linewidth=2)

plt.plot(test_curve[0], test_curve[1], 
  label=f'Test ROC (AUC = {roc_test:.3f})', 
  linewidth=2)

plt.plot(ooc_curve[0], ooc_curve[1], 
  label=f'OOT ROC (AUC = {roc_oot:.3f})', 
  linewidth=2)

plt.plot([0,1], [0,1], '--', color="black")

plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curves - Train vs Test vs OOT', fontsize=14)
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %%
from sklearn.metrics import precision_recall_curve,  classification_report

probs = model_pipeline.predict_proba(X_test)[:, 1]

precisions, recalls, thresholds = precision_recall_curve(y_test, probs)
f1s = 2*(precisions * recalls)/(precisions + recalls)

best_f1_idx = np.argmax(f1s)
best_threshold = thresholds[best_f1_idx]
print(f'Optimal threshold: {best_threshold}')
# %%
plt.figure(figsize=(10, 6))
plt.plot(thresholds, precisions[:-1], label="Precisions")
plt.plot(thresholds, recalls[:-1], label="Recalls")
plt.plot(thresholds, f1s[:-1], label="F1 Score")
plt.axvline(x=best_threshold, color="red", label="Best Threshold")

plt.legend()
plt.xlabel("Thresholds")
plt.title("Precision x Recall x F1")

# %%
oot_proba = model_pipeline.predict_proba(X_oot)[:, 1]
oot_pred_threshold = (oot_proba >= best_threshold).astype(int)
print(classification_report(oot[target_col], oot_pred_threshold))
