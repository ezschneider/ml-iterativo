import matplotlib.pyplot as plt
import pandas as pd
import base64
from io import BytesIO
import shap
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


def generate_feature_importance(model, feature_names):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = model.coef_[0]
    else:
        return None

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values(by="importance", ascending=False).head(10)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(importance_df["feature"], importance_df["importance"])
    ax.set_xlabel("Importância")
    ax.set_title("Top 10 Features Mais Relevantes")
    plt.gca().invert_yaxis()

    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def generate_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax)
    buf = BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def generate_shap_summary_plot(pipeline, X_sample):
    try:
        model = pipeline.named_steps["classifier"]
        preprocessor = pipeline.named_steps["preprocessor"]
        X_transformed = preprocessor.transform(X_sample)

        explainer = shap.Explainer(model, X_transformed)
        shap_values = explainer(X_transformed)

        fig = plt.figure(figsize=(8, 6))
        shap.plots.beeswarm(shap_values, show=False)

        buf = BytesIO()
        plt.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")
    except Exception:
        return None