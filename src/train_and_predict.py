"""Train a classification model for the startup success dataset and generate a submission."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingClassifier


def load_data(base_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_path = base_path / "train.csv"
    test_path = base_path / "test.csv"
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


def build_pipeline(train_df: pd.DataFrame) -> Pipeline:
    target_col = "labels"
    id_col = "id"
    categorical_cols = ["category_code"]
    numeric_cols = [
        col
        for col in train_df.columns
        if col not in {target_col, id_col, *categorical_cols}
    ]

    numeric_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median", add_indicator=True),
            )
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    model = GradientBoostingClassifier(
        random_state=42,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    return pipeline


def evaluate_model(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series) -> float:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
    return float(scores.mean())


def tune_hyperparameters(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series) -> Pipeline:
    param_distributions = {
        "model__n_estimators": np.arange(150, 301, 15),
        "model__learning_rate": np.linspace(0.02, 0.2, num=20),
        "model__max_depth": [2, 3, 4, 5],
        "model__min_samples_leaf": [1, 2, 4, 6, 8],
        "model__subsample": np.linspace(0.6, 1.0, num=9),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_distributions,
        n_iter=60,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
        random_state=42,
        refit=True,
    )

    search.fit(X, y)
    print("Melhores hiperparâmetros:", search.best_params_)
    print("Acurácia média em validação cruzada:", search.best_score_)
    return search.best_estimator_


def train_and_predict(base_path: Path) -> tuple[pd.DataFrame, float]:
    train_df, test_df = load_data(base_path)
    target_col = "labels"
    id_col = "id"

    X = train_df.drop(columns=[target_col])
    y = train_df[target_col]

    pipeline = build_pipeline(train_df)

    baseline_score = evaluate_model(pipeline, X, y)
    print(f"Acurácia média (pipeline base): {baseline_score:.4f}")

    tuned_pipeline = tune_hyperparameters(pipeline, X, y)

    tuned_pipeline.fit(X, y)
    final_cv = evaluate_model(tuned_pipeline, X, y)
    print(f"Acurácia média (pipeline ajustado): {final_cv:.4f}")

    test_predictions = tuned_pipeline.predict(test_df)
    submission = pd.DataFrame({id_col: test_df[id_col], "labels": test_predictions})

    return submission, final_cv


def main() -> None:
    base_path = Path(__file__).resolve().parent.parent
    submission, cv_score = train_and_predict(base_path)

    output_path = base_path / "submission.csv"
    submission.to_csv(output_path, index=False)
    print(f"Arquivo de submissão salvo em: {output_path}")
    print(f"Acurácia média estimada em validação cruzada: {cv_score:.4f}")


if __name__ == "__main__":
    main()
