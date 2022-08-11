from xgboost import XGBClassifier
from optuna.samplers import TPESampler
from optuna.integration import XGBoostPruningCallback
from sklearn.model_selection import KFold
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from optuna import create_study
from gpt3forchem.baseline import BaseLineModel
from sklearn.preprocessing import LabelEncoder

from wandb.xgboost import WandbCallback


class XGBClassificationBaseline(BaseLineModel):
    def __init__(self, seed, num_trials=100) -> None:
        self.seed = seed
        self.num_trials = num_trials
        self.model = XGBClassifier()

        self.label_encoder = LabelEncoder()

    def tune(self, X_train, y_train):
        y_train = self.label_encoder.fit_transform(y_train)

        def objective(
            trial,
            X,
            y,
            random_state=22,
            n_splits=3,
            n_jobs=1,
            early_stopping_rounds=100,
        ):
            # XGBoost parameters
            params = {
                "verbosity": 0,  # 0 (silent) - 3 (debug)
                "n_estimators": trial.suggest_int("n_estimators", 4, 10_000),
                "max_depth": trial.suggest_int("max_depth", 4, 100),
                "learning_rate": trial.suggest_loguniform("learning_rate", 0.001, 0.05),
                "colsample_bytree": trial.suggest_loguniform("colsample_bytree", 0.2, 1),
                "subsample": trial.suggest_loguniform("subsample", 0.00001, 1),
                "alpha": trial.suggest_loguniform("alpha", 1e-8, 10.0),
                "lambda": trial.suggest_loguniform("lambda", 1e-8, 10.0),
                "seed": random_state,
                "n_jobs": n_jobs,
            }

            model = XGBClassifier(**params)
            pruning_callback = XGBoostPruningCallback(trial, "validation_0-mlogloss")
            kf = KFold(n_splits=n_splits)
            X_values = X.values
            y_values = y

            scores = []
            for train_index, test_index in kf.split(X_values):
                X_A, X_B = X_values[train_index, :], X_values[test_index, :]
                y_A, y_B = y_values[train_index], y_values[test_index]

                model.fit(
                    X_A,
                    y_A,
                    eval_set=[(X_B, y_B)],
                    eval_metric="mlogloss",
                    verbose=0,
                    callbacks=[pruning_callback],
                    early_stopping_rounds=early_stopping_rounds,
                )
                y_pred = model.predict(X_B)
                scores.append(f1_score(y_pred, y_B, average="macro"))
            return np.mean(scores)

        sampler = TPESampler(seed=self.seed)
        study = create_study(direction="maximize", sampler=sampler)
        study.optimize(
            lambda trial: objective(
                trial,
                X_train,
                y_train,
                random_state=self.seed,
                n_splits=5,
                n_jobs=-1,
                early_stopping_rounds=100,
            ),
            n_trials=self.num_trials,
            n_jobs=1,
        )

        self.model = XGBClassifier(**study.best_params, callbacks=[WandbCallback()])

    def fit(self, X_train, y_train):
        y_train = self.label_encoder.fit_transform(y_train)
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.label_encoder.inverse_transform(self.model.predict(X))
