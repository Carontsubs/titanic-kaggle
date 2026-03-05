"""
Pipeline complet per competicions Kaggle (classificació binària)
Exemple basat en el problema del Titanic
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. CÀRREGA DE DADES
# ============================================================

def load_data():
    """Carrega train i test, els combina per fer feature engineering conjunt."""
    train = pd.read_csv('train.csv')
    test  = pd.read_csv('test.csv')

    print(f"Train: {train.shape}, Test: {test.shape}")
    print(f"\nValors nuls (train):\n{train.isnull().sum()[train.isnull().sum() > 0]}")

    return train, test


# ============================================================
# 2. FEATURE ENGINEERING
# ============================================================

def feature_engineering(df):
    """
    Transforma les dades crues en features útils.
    Aquí és on es guanyen les competicions.
    """

    df = df.copy()

    # --- Titanic específic ---
    # Extreure títol del nom (Mr, Mrs, Miss, Master...)
    df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    df['Title'] = df['Title'].replace(
        ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr',
         'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare'
    )
    df['Title'] = df['Title'].replace({'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'})

    # Mida de la família
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone']    = (df['FamilySize'] == 1).astype(int)

    # Categoria de preu del bitllet
    df['FareBand'] = pd.qcut(df['Fare'].fillna(df['Fare'].median()), 4, labels=False)

    # Categoria d'edat
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['AgeBand'] = pd.cut(df['Age'], bins=[0, 12, 18, 35, 60, 100], labels=False)

    # Coberta del vaixell (lletra de la cabina)
    df['Deck'] = df['Cabin'].str[0].fillna('U')

    return df


# ============================================================
# 3. PREPROCESSAMENT
# ============================================================

def preprocess(train, test, target_col='Survived'):
    """Codifica variables categòriques i escala numèriques."""

    # Combinar per codificar igual train i test
    combined = pd.concat([train, test], axis=0, ignore_index=True)
    combined = feature_engineering(combined)

    # Columnes a usar
    features = ['Pclass', 'Sex', 'AgeBand', 'FareBand',
                'FamilySize', 'IsAlone', 'Title', 'Deck', 'Embarked']

    # Codificar categòriques
    le = LabelEncoder()
    for col in ['Sex', 'Title', 'Deck', 'Embarked']:
        combined[col] = le.fit_transform(combined[col].fillna('U').astype(str))

    # Separar de nou
    n_train = len(train)
    X      = combined[features].iloc[:n_train]
    y      = train[target_col]
    X_test = combined[features].iloc[n_train:]

    print(f"\nFeatures finals: {features}")
    print(f"X shape: {X.shape}, X_test shape: {X_test.shape}")

    return X, y, X_test, features


# ============================================================
# 4. VALIDACIÓ CREUADA
# ============================================================

def cross_validate_model(model, X, y, n_folds=5):
    """
    Validació creuada estratificada.
    CRÍTIC: el leaderboard públic de Kaggle només és el 30% de les dades.
    Confiar en CV local és més fiable que el leaderboard públic.
    """
    skf    = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=skf, scoring='roc_auc')

    print(f"  CV AUC: {scores.mean():.4f} ± {scores.std():.4f}")
    return scores.mean()


# ============================================================
# 5. ENTRENAMENT DE MODELS
# ============================================================

def train_models(X, y):
    """Entrena múltiples models per fer ensemble."""

    models = {
        'RandomForest': RandomForestClassifier(
            n_estimators=300, max_depth=6,
            min_samples_leaf=4, random_state=42
        ),
        'GradientBoosting': GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05,
            max_depth=4, random_state=42
        ),
        'LogisticRegression': LogisticRegression(
            C=0.1, max_iter=1000, random_state=42
        ),
    }

    results = {}
    print("\n--- Validació creuada per model ---")
    for name, model in models.items():
        print(f"\n{name}:")
        score = cross_validate_model(model, X, y)
        model.fit(X, y)
        results[name] = {'model': model, 'score': score}

    return results


# ============================================================
# 6. ENSEMBLE (combinació de models)
# ============================================================

def ensemble_predict(models_dict, X_test):
    """
    Ensemble per mitjana de probabilitats.
    Quasi sempre millora qualsevol model individual.
    Pots ponderar per CV score si vols.
    """
    preds = []
    weights = []

    for name, info in models_dict.items():
        prob = info['model'].predict_proba(X_test)[:, 1]
        preds.append(prob)
        weights.append(info['score'])
        print(f"  {name}: score={info['score']:.4f}")

    # Mitjana ponderada per CV score
    weights = np.array(weights) / sum(weights)
    final_pred = np.average(preds, axis=0, weights=weights)

    return final_pred


# ============================================================
# 7. GENERAR SUBMISSION
# ============================================================

def generate_submission(test_df, predictions, threshold=0.5, filename='submission.csv'):
    """Genera el fitxer per pujar a Kaggle."""

    submission = pd.DataFrame({
        'PassengerId': test_df['PassengerId'],
        'Survived':   (predictions >= threshold).astype(int)
    })

    submission.to_csv(filename, index=False)
    print(f"\nSubmission guardada: {filename}")
    print(f"Distribució prediccions: {submission['Survived'].value_counts().to_dict()}")

    return submission


# ============================================================
# 8. PIPELINE PRINCIPAL
# ============================================================

def main():
    print("=" * 50)
    print("KAGGLE PIPELINE - Titanic")
    print("=" * 50)

    # Pas 1: Càrrega
    train, test = load_data()

    # Pas 2 & 3: Feature engineering + preprocessament
    X, y, X_test, features = preprocess(train, test)

    # Pas 4 & 5: Entrenament amb validació creuada
    models_dict = train_models(X, y)

    # Pas 6: Ensemble
    print("\n--- Ensemble final ---")
    final_predictions = ensemble_predict(models_dict, X_test)

    # Pas 7: Submission
    submission = generate_submission(test, final_predictions)

    print("\n✅ Pipeline completat!")
    print("\nPropers passos per millorar:")
    print("  1. Afegir més features creatives")
    print("  2. Provar XGBoost / LightGBM")
    print("  3. Hyperparameter tuning amb Optuna")
    print("  4. Stacking (usar prediccions com a features)")

    return submission


if __name__ == "__main__":
    main()


# ============================================================
# BONUS: Kelly Criterion per prediction markets
# (connecta amb la conversa anterior)
# ============================================================

def kelly_criterion(prob_teva, prob_mercat):
    """
    Calcula quina fracció del capital apostar.
    
    prob_teva   : la teva estimació (ex: 0.70)
    prob_mercat : el preu del mercat (ex: 0.60)
    """
    if prob_mercat >= 1 or prob_mercat <= 0:
        return 0

    # Quota implícita
    b = (1 - prob_mercat) / prob_mercat  # guany net per €1 apostada
    p = prob_teva
    q = 1 - p

    kelly = (p * b - q) / b

    # Kelly fraccionari (25%) per ser conservador
    kelly_fraccional = kelly * 0.25

    print(f"\nKelly Criterion:")
    print(f"  Prob teva:    {prob_teva:.0%}")
    print(f"  Prob mercat:  {prob_mercat:.0%}")
    print(f"  Edge:         {prob_teva - prob_mercat:+.0%}")
    print(f"  Kelly pur:    {kelly:.1%} del capital")
    print(f"  Kelly 25%:    {kelly_fraccional:.1%} del capital (recomanat)")

    if kelly <= 0:
        print("  ⚠️  No hi ha edge, NO apostis")
    else:
        print(f"  ✅ Amb 1.000€ apostaries: {kelly_fraccional * 1000:.0f}€")

    return kelly_fraccional


# Exemple
kelly_criterion(prob_teva=0.70, prob_mercat=0.58)
