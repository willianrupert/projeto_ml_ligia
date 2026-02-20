from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
from catboost import CatBoostClassifier

def train_stacking_ensemble(X, y):
    """
    Constrói e treina um Stacking Ensemble de Estado da Arte.
    Nível 0: XGBoost, LightGBM, CatBoost
    Nível 1: Regressão Logística (Meta-Modelo)
    """
    print("🏗️ A montar a arquitetura do Stacking Ensemble...")
    
    # 1. Modelos Base (Nível 0)
    # Colocamos os hiperparâmetros excelentes que o Optuna já lhe tinha dado antes
    # Isto poupa o tempo de ter de otimizar tudo de novo!
    xgb_model = xgb.XGBClassifier(
        scale_pos_weight=89.8, 
        learning_rate=0.09, 
        max_depth=4,
        subsample=0.84, 
        colsample_bytree=0.87, 
        random_state=42,
        eval_metric='logloss'
    )
    
    # LightGBM (muito rápido e lida nativamente com o desbalanceamento)
    lgb_model = lgb.LGBMClassifier(
        scale_pos_weight=89.8, # Usamos o mesmo peso descoberto pelo Optuna
        random_state=42, 
        n_estimators=300,
        verbose=-1
    )
    
    # CatBoost (Excecional para evitar overfitting em dados complexos)
    cat_model = CatBoostClassifier(
        auto_class_weights='Balanced', # Ele balança o peso automaticamente
        random_state=42, 
        verbose=0, 
        iterations=300
    )
    
    estimators = [
        ('xgb', xgb_model),
        ('lgb', lgb_model),
        ('cat', cat_model)
    ]
    
    # 2. Meta-Modelo (Nível 1)
    # Uma regressão logística simples para aprender a ponderar as previsões dos 3 acima
    meta_model = LogisticRegression(class_weight='balanced', random_state=42)
    
    # 3. Construção do Stacking
    # O cv=5 usa Validação Cruzada interna para treinar a regressão logística, evitando vazamento!
    stacking_clf = StackingClassifier(
        estimators=estimators,
        final_estimator=meta_model,
        cv=5,
        n_jobs=-1 # Usa todos os núcleos do processador do seu Mac
    )
    
    print("🚀 A treinar o Stacking (Pode demorar uns minutos, os modelos estão a treinar e a votar)...")
    stacking_clf.fit(X, y)
    print("✅ Stacking treinado com sucesso!")
    
    return stacking_clf