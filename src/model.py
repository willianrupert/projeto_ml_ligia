import xgboost as xgb
import optuna
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

def objective(trial, X, y):
    """
    Função que o Optuna usa para 'aprender' a melhor configuração.
    """
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'booster': 'gbtree',
        'tree_method': 'hist', # Otimizado para velocidade
        'random_state': 42,    # Reprodutibilidade (Exigência do Edital)
        
        # Espaço de busca (Hiperparâmetros)
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 100.0), # Peso para classes desbalanceadas
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'lambda': trial.suggest_float('lambda', 1e-3, 10.0, log=True),
        'alpha': trial.suggest_float('alpha', 1e-3, 10.0, log=True)
    }

    # Validação Cruzada Estratificada (5 Folds)
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auc_scores = []
    
    for train_idx, val_idx in kf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model = xgb.XGBClassifier(**params, n_estimators=1000, early_stopping_rounds=50)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        
        preds = model.predict_proba(X_val)[:, 1]
        auc_scores.append(roc_auc_score(y_val, preds))
    
    return np.mean(auc_scores)

def train_final_model(X, y, n_trials=20):
    print("🚀 Iniciando Otimização Bayesiana com Optuna...")
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, X, y), n_trials=n_trials)
    
    print(f"✅ Melhores parâmetros: {study.best_params}")
    
    # Treina o modelo final com TODOS os dados e os melhores parâmetros
    best_params = study.best_params
    best_params['random_state'] = 42
    final_model = xgb.XGBClassifier(**best_params, n_estimators=1000)
    final_model.fit(X, y)
    
    return final_model