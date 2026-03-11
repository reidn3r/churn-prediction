# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style("whitegrid")
sns.set_palette("husl")

pd.options.display.max_columns = 100
pd.options.display.max_rows = 100

# %%
path = "/home/reidner/dev/portfolio/ml@tmw_churn/data/abt_churn.csv"
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

# %%
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
# %%
from sklearn.linear_model import LogisticRegression
from feature_engine import discretisation, encoding
from sklearn import pipeline

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

lr = LogisticRegression(random_state=42)

# %%
model_pipeline = pipeline.Pipeline(
  steps=[
    ('Discretizar', tree_discretisation),
    ('One Hot', ohe),
    ('Reg. Logistica', lr),
  ],
  )

model_pipeline.fit(X_train, y_train)
# %%
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

predict_proba = model_pipeline.predict_proba(X_train)[:, 1]
predict = model_pipeline.predict(X_train)

acc_train = accuracy_score(y_train, predict)
roc = roc_auc_score(y_train, predict_proba)
train_curve = roc_curve(y_train, predict_proba)

print(f'train acc: {acc_train}')
print(f'train AUC: {roc}')

# %%

test_predict_proba = model_pipeline.predict_proba(X_test)[:, 1]
test_predict = model_pipeline.predict(X_test)

acc_test = accuracy_score(y_test, test_predict)
roc_test = roc_auc_score(y_test, test_predict_proba)
test_curve = roc_curve(y_test, test_predict_proba)

print(f'test acc: {acc_test}')
print(f'test AUC: {roc_test}')
# %%
X_oot = oot[feature_list].copy()
oot_predict_proba = model_pipeline.predict_proba(X_oot)[:, 1]
oot_predict = model_pipeline.predict(X_oot)

acc_oot = accuracy_score(oot[target_col], oot_predict)
roc_oot = roc_auc_score(oot[target_col], oot_predict_proba)
ooc_curve = roc_curve(oot[target_col], oot_predict_proba)

print(f'oot acc: {acc_oot}')
print(f'oot AUC: {roc_oot}')
 