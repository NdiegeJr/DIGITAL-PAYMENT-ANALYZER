"""
train_models.py
================
Loads Digital_Payments_Data.csv, cleans it, engineers features, trains the
three ML models used by the dashboard, and saves everything needed at runtime
(models + pre-aggregated tables) into the models/ and data/ folders.

Run this once (from PyCharm or terminal) before starting the dashboard:
    python train_models.py
"""

import warnings
warnings.filterwarnings("ignore")

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "Digital_Payments_Data.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. LOAD & CLEAN
# ---------------------------------------------------------------------------
def load_and_clean(path=DATA_PATH):
    df = pd.read_csv(path)

    # Parse timestamp (day-first, e.g. 21-08-2025 12:37:21)
    df["Transaction_Timestamp"] = pd.to_datetime(
        df["Transaction_Timestamp"], dayfirst=True, errors="coerce"
    )
    df = df.dropna(subset=["Transaction_Timestamp"])

    # Fill categorical NAs sensibly
    for col in ["Customer_Segment", "Device_Type", "Promo_Code", "App_Version"]:
        df[col] = df[col].fillna("Unknown")

    df["Card_Type"] = df["Card_Type"].fillna("None")
    df["Card_Category"] = df["Card_Category"].fillna("None")
    df["Wallet_Provider"] = df["Wallet_Provider"].fillna("None")
    df["Crypto_Type"] = df["Crypto_Type"].fillna("None")

    # Keep only successful transactions for trend/volume analysis
    # (a separate flag is kept so the dashboard can still show success rate)
    df["Is_Successful"] = (df["Payment_Status"] == "Successful").astype(int)

    # Time features
    df["Year"] = df["Transaction_Timestamp"].dt.year
    df["Month"] = df["Transaction_Timestamp"].dt.month
    df["MonthPeriod"] = df["Transaction_Timestamp"].dt.to_period("M").astype(str)
    df["DayOfWeek"] = df["Transaction_Timestamp"].dt.day_name()
    df["DayOfWeekNum"] = df["Transaction_Timestamp"].dt.dayofweek
    df["Hour"] = df["Transaction_Timestamp"].dt.hour
    df["Quarter"] = df["Transaction_Timestamp"].dt.quarter

    # Broad channel grouping: Mobile-style (Digital_Wallet/QR/Virtual_Card on
    # Mobile/In_App) vs Bank/Card-style payments. This answers the
    # "mobile payments vs bank payments" preference question explicitly.
    mobile_methods = {"Digital_Wallet", "QR_Payment", "Virtual_Card", "Cryptocurrency"}
    df["Payment_Group"] = np.where(
        df["Payment_Method"].isin(mobile_methods), "Mobile/Digital Payment", "Card/Bank Payment"
    )

    return df


# ---------------------------------------------------------------------------
# 2. AGGREGATIONS used by the dashboard (saved as parquet/csv for fast load)
# ---------------------------------------------------------------------------
def build_aggregates(df):
    monthly = (
        df.groupby("MonthPeriod")
        .agg(
            Transaction_Count=("Transaction_ID", "count"),
            Total_Value=("Transaction_Amount_GBP", "sum"),
            Avg_Value=("Transaction_Amount_GBP", "mean"),
            Success_Rate=("Is_Successful", "mean"),
        )
        .reset_index()
        .sort_values("MonthPeriod")
    )

    channel_monthly = (
        df.groupby(["MonthPeriod", "Transaction_Channel"])
        .size()
        .reset_index(name="Count")
    )

    method_monthly = (
        df.groupby(["MonthPeriod", "Payment_Method"])
        .size()
        .reset_index(name="Count")
    )

    group_monthly = (
        df.groupby(["MonthPeriod", "Payment_Group"])
        .size()
        .reset_index(name="Count")
    )

    dow_volume = (
        df.groupby(["DayOfWeek", "DayOfWeekNum"])
        .size()
        .reset_index(name="Count")
        .sort_values("DayOfWeekNum")
    )

    hour_volume = df.groupby("Hour").size().reset_index(name="Count")

    sector_pref = (
        df.groupby(["Sector", "Payment_Group"]).size().reset_index(name="Count")
    )

    segment_pref = (
        df.groupby(["Customer_Segment", "Payment_Group"]).size().reset_index(name="Count")
    )

    monthly.to_csv(os.path.join(BASE_DIR, "data", "agg_monthly.csv"), index=False)
    channel_monthly.to_csv(os.path.join(BASE_DIR, "data", "agg_channel_monthly.csv"), index=False)
    method_monthly.to_csv(os.path.join(BASE_DIR, "data", "agg_method_monthly.csv"), index=False)
    group_monthly.to_csv(os.path.join(BASE_DIR, "data", "agg_group_monthly.csv"), index=False)
    dow_volume.to_csv(os.path.join(BASE_DIR, "data", "agg_dow.csv"), index=False)
    hour_volume.to_csv(os.path.join(BASE_DIR, "data", "agg_hour.csv"), index=False)
    sector_pref.to_csv(os.path.join(BASE_DIR, "data", "agg_sector_pref.csv"), index=False)
    segment_pref.to_csv(os.path.join(BASE_DIR, "data", "agg_segment_pref.csv"), index=False)

    return monthly


# ---------------------------------------------------------------------------
# 3. MODEL A — Future volume forecaster (RandomForestRegressor on monthly series)
# ---------------------------------------------------------------------------
def train_forecast_model(monthly):
    m = monthly.reset_index(drop=True).copy()
    m["t"] = np.arange(len(m))
    m["month_num"] = pd.PeriodIndex(m["MonthPeriod"], freq="M").month
    m["lag_1"] = m["Transaction_Count"].shift(1).bfill()
    m["lag_2"] = m["Transaction_Count"].shift(2).bfill()

    X = m[["t", "month_num", "lag_1", "lag_2"]]
    y = m["Transaction_Count"]

    model = RandomForestRegressor(n_estimators=300, max_depth=5, random_state=42)
    model.fit(X, y)

    preds = model.predict(X)
    r2 = r2_score(y, preds)
    mae = mean_absolute_error(y, preds)
    print(f"[Forecast model]  R^2={r2:.3f}  MAE={mae:.1f}")

    joblib.dump(model, os.path.join(MODELS_DIR, "forecast_model.pkl"))
    joblib.dump(m, os.path.join(MODELS_DIR, "forecast_history.pkl"))
    return model, m


# ---------------------------------------------------------------------------
# 4. MODEL B — Payment channel / method preference classifier
# ---------------------------------------------------------------------------
def train_channel_model(df):
    features = [
        "Sector",
        "Customer_Segment",
        "Device_Type",
        "Transaction_Amount_GBP",
        "DayOfWeekNum",
        "Hour",
        "Quarter",
    ]
    target = "Payment_Group"

    data = df[features + [target]].dropna()

    encoders = {}
    X = data[features].copy()
    for col in ["Sector", "Customer_Segment", "Device_Type"]:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le

    y_le = LabelEncoder()
    y = y_le.fit_transform(data[target])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42, class_weight="balanced")
    clf.fit(X_train, y_train)

    acc = accuracy_score(y_test, clf.predict(X_test))
    print(f"[Channel preference model] Accuracy={acc:.3f}  Classes={list(y_le.classes_)}")

    joblib.dump(clf, os.path.join(MODELS_DIR, "channel_model.pkl"))
    joblib.dump(encoders, os.path.join(MODELS_DIR, "channel_encoders.pkl"))
    joblib.dump(y_le, os.path.join(MODELS_DIR, "channel_target_encoder.pkl"))
    joblib.dump(features, os.path.join(MODELS_DIR, "channel_features.pkl"))
    return clf


# ---------------------------------------------------------------------------
# 5. MODEL C — Specific payment-method predictor (Card / Digital_Wallet / etc.)
# ---------------------------------------------------------------------------
def train_method_model(df):
    features = [
        "Sector",
        "Customer_Segment",
        "Device_Type",
        "Transaction_Amount_GBP",
        "DayOfWeekNum",
        "Hour",
    ]
    target = "Payment_Method"

    data = df[features + [target]].dropna()

    encoders = {}
    X = data[features].copy()
    for col in ["Sector", "Customer_Segment", "Device_Type"]:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le

    y_le = LabelEncoder()
    y = y_le.fit_transform(data[target])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42, class_weight="balanced")
    clf.fit(X_train, y_train)

    acc = accuracy_score(y_test, clf.predict(X_test))
    print(f"[Payment method model] Accuracy={acc:.3f}  Classes={list(y_le.classes_)}")

    joblib.dump(clf, os.path.join(MODELS_DIR, "method_model.pkl"))
    joblib.dump(encoders, os.path.join(MODELS_DIR, "method_encoders.pkl"))
    joblib.dump(y_le, os.path.join(MODELS_DIR, "method_target_encoder.pkl"))
    joblib.dump(features, os.path.join(MODELS_DIR, "method_features.pkl"))
    return clf


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Loading and cleaning data...")
    df = load_and_clean()
    print(f"Rows after cleaning: {len(df)}")

    print("\nBuilding aggregate tables for dashboard...")
    monthly = build_aggregates(df)

    print("\nTraining forecast model...")
    train_forecast_model(monthly)

    print("\nTraining channel preference model...")
    train_channel_model(df)

    print("\nTraining payment method model...")
    train_method_model(df)

    # Save cleaned dataframe (lightweight columns only) for the dashboard
    keep_cols = [
        "Transaction_Timestamp", "MonthPeriod", "Sector", "Payment_Method",
        "Transaction_Channel", "Payment_Group", "Transaction_Amount_GBP",
        "Payment_Status", "Customer_Segment", "Device_Type", "DayOfWeek",
        "DayOfWeekNum", "Hour", "Quarter", "Location",
    ]
    df[keep_cols].to_csv(os.path.join(BASE_DIR, "data", "cleaned_transactions.csv"), index=False)

    print("\nAll models and aggregates saved to /models and /data. Done.")
