import pandas as pd
from src.features.cve_features import build_cve_features
from src.models.ransomware_risk import RansomwareRiskModel

cve = pd.read_csv("data/raw/2_cve_vulnerabilities.csv")
cve["dateAdded"] = pd.to_datetime(cve["dateAdded"], errors="coerce")
cve["dueDate"]   = pd.to_datetime(cve["dueDate"],   errors="coerce")
features = build_cve_features(cve)

model = RansomwareRiskModel.load("models/ransomware_risk.joblib")
print(model.shap_ranking(features, top_n=15).to_string(index=False))