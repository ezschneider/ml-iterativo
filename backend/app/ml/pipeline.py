import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

class MLPipeline:
    def __init__(self, df: pd.DataFrame, target_column: str):
        self.df = df
        self.target_column = target_column
        self.X = df.drop(columns=[target_column])
        self.y = df[target_column]
        self.problem_type = self._detect_problem_type()
        self.models = self._define_models()
        self.params = self._define_param_grid()

    def _detect_problem_type(self):
        if self.y.nunique() <= 10 and self.y.dtype != 'float':
            return 'classification'
        else:
            raise ValueError("Apenas problemas de classificação com até 10 classes são suportados nesta versão.")

    def _define_models(self):
        return {
            "LogisticRegression": LogisticRegression(max_iter=1000),
            "RandomForest": RandomForestClassifier(),
            "SVC": SVC()
        }

    def _define_param_grid(self):
        return {
            "LogisticRegression": {"classifier__C": [0.1, 1, 10]},
            "RandomForest": {"classifier__n_estimators": [50, 100]},
            "SVC": {"classifier__C": [0.1, 1], "classifier__kernel": ["linear", "rbf"]},
        }

    def _preprocess(self):
        num_cols = self.X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        cat_cols = self.X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

        numeric_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, num_cols),
                ("cat", categorical_transformer, cat_cols)
            ]
        )

        return preprocessor

    def run(self):
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, stratify=self.y, test_size=0.2, random_state=42
        )

        preprocessor = self._preprocess()

        best_score = 0
        best_model_name = None
        best_report = None
        best_pipeline = None
        best_predictions = None

        for name, model in self.models.items():
            pipe = Pipeline(steps=[
                ("preprocessor", preprocessor),
                ("classifier", model)
            ])

            grid = GridSearchCV(pipe, param_grid=self.params[name], cv=3, scoring="accuracy")
            grid.fit(X_train, y_train)

            score = grid.score(X_test, y_test)

            if score > best_score:
                best_score = score
                best_model_name = name
                y_pred = grid.predict(X_test)
                best_report = classification_report(y_test, y_pred, output_dict=True)
                best_pipeline = grid.best_estimator_
                best_predictions = pd.DataFrame({"actual": y_test, "predicted": y_pred})

        # Salvar modelo e previsões
        model_path = os.path.join(RESULTS_DIR, f"{best_model_name}_model.joblib")
        preds_path = os.path.join(RESULTS_DIR, f"{best_model_name}_predictions.csv")

        joblib.dump(best_pipeline, model_path)
        best_predictions.to_csv(preds_path, index=False)

        return {
            "problem_type": self.problem_type,
            "best_model": best_model_name,
            "accuracy": best_score,
            "report": best_report,
            "model_path": model_path,
            "predictions_path": preds_path
        }
