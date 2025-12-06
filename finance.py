# ============================================================
# 0. IMPORTS & CONFIG
# ============================================================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error,
)

plt.style.use("seaborn-v0_8")


# ============================================================
# 1. LOAD DATA FROM EXCEL FILES
# ============================================================

# Folder where your files are stored
BASE_PATH = r"C:\Users\ashwi\OneDrive\Documents\Resumes\Projects"

# Exact filenames (change here if they differ)
path_is = os.path.join(BASE_PATH, "Income Statement.xlsx")
path_bs = os.path.join(BASE_PATH, "Balance Sheet.xlsx")
path_cf = os.path.join(BASE_PATH, "Cash Flow Statement.xlsx")

# --- Income Statement ---
is_raw = pd.read_excel(path_is)
is_raw.rename(columns={is_raw.columns[0]: "Metric"}, inplace=True)

# --- Balance Sheet ---
bs_raw = pd.read_excel(path_bs)
bs_raw.rename(columns={bs_raw.columns[0]: "Metric"}, inplace=True)

# --- Cash Flow Statement ---
cf_raw = pd.read_excel(path_cf)
cf_raw.rename(columns={cf_raw.columns[0]: "Metric"}, inplace=True)


# ============================================================
# 2. TRANSFORM TABLES (METRIC AS ROW, DATES AS INDEX)
# ============================================================

is_df = is_raw.set_index("Metric").T
bs_df = bs_raw.set_index("Metric").T
cf_df = cf_raw.set_index("Metric").T

# Dates like 30-09-2025 (day-first)
is_df.index = pd.to_datetime(is_df.index, dayfirst=True, errors="coerce")
bs_df.index = pd.to_datetime(bs_df.index, dayfirst=True, errors="coerce")
cf_df.index = pd.to_datetime(cf_df.index, dayfirst=True, errors="coerce")

# Ensure numeric
is_df = is_df.apply(pd.to_numeric, errors="coerce")
bs_df = bs_df.apply(pd.to_numeric, errors="coerce")
cf_df = cf_df.apply(pd.to_numeric, errors="coerce")


# ============================================================
# 3. BUILD MASTER QUARTERLY DATAFRAME
# ============================================================

income_sel = is_df[["Revenue", "Net Income"]].rename(
    columns={"Revenue": "revenue", "Net Income": "net_income"}
)

balance_sel = bs_df[
    ["Total Assets", "Total Liabilities", "Shareholder Equity"]
].rename(
    columns={
        "Total Assets": "total_assets",
        "Total Liabilities": "total_liabilities",
        "Shareholder Equity": "equity",
    }
)

cashflow_sel = cf_df[
    [
        "Cash Flow From Operating Activities",
        "Net Change In Property Plant And Equipment",
        "Net Cash Flow",
    ]
].rename(
    columns={
        "Cash Flow From Operating Activities": "ocf",
        "Net Change In Property Plant And Equipment": "capex",
        "Net Cash Flow": "net_cash_flow",
    }
)

master = (
    income_sel
    .join(balance_sel, how="inner")
    .join(cashflow_sel, how="inner")
    .sort_index()
)

master["FCF"] = master["ocf"] + master["capex"]

print("=== MASTER DATAFRAME (HEAD) ===")
print(master.head())
print("\nColumns:", master.columns.tolist())


# ============================================================
# 4. EDA (OPTIONAL PLOTS)
# ============================================================

print("\n=== MASTER INFO ===")
print(master.info())

print("\n=== SUMMARY STATISTICS ===")
print(master.describe().T)

# Comment this block out if you don't want plots
fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
master["revenue"].plot(ax=axes[0], marker="o", title="Revenue (Quarterly)")
master["net_income"].plot(ax=axes[1], marker="o", title="Net Income (Quarterly)")
master["FCF"].plot(ax=axes[2], marker="o", title="Free Cash Flow (Quarterly)")
plt.tight_layout()
plt.show()


# ============================================================
# 5. FEATURE ENGINEERING (LAGS + QOQ GROWTH)
# ============================================================

df_ml = master.copy()
targets = ["revenue", "net_income", "FCF"]

# Lag features
for col in targets:
    df_ml[f"{col}_lag1"] = df_ml[col].shift(1)
    df_ml[f"{col}_lag2"] = df_ml[col].shift(2)
    df_ml[f"{col}_lag4"] = df_ml[col].shift(4)

# QoQ growth
for col in targets:
    df_ml[f"{col}_qoq"] = df_ml[col].pct_change()

df_ml = df_ml.dropna().copy()

print("\n=== ML DATASET SHAPE ===", df_ml.shape)
print("ML columns:", df_ml.columns.tolist())


# ============================================================
# 6. TRAIN / TEST SPLIT (TIME-BASED)
# ============================================================

test_horizon = 4
train = df_ml.iloc[:-test_horizon, :]
test = df_ml.iloc[-test_horizon:, :]

print("\nTrain period:", train.index.min().date(), "to", train.index.max().date())
print("Test period :", test.index.min().date(), "to", test.index.max().date())

feature_cols = sorted(
    [f"{col}_lag1" for col in targets]
    + [f"{col}_lag2" for col in targets]
    + [f"{col}_lag4" for col in targets]
    + [f"{col}_qoq" for col in targets]
)

print("\nFeature columns used:")
print(feature_cols)


# ============================================================
# 7. MODEL TRAINING & EVALUATION (USING NUMPY ARRAYS)
# ============================================================

models = {}
metrics = {}

for target in targets:
    X_train = train[feature_cols].values
    y_train = train[target].values
    X_test = test[feature_cols].values
    y_test = test[target].values

    model = RandomForestRegressor(
        n_estimators=500,
        random_state=42,
        max_depth=None,
        min_samples_leaf=1,
    )
    model.fit(X_train, y_train)
    models[target] = model

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    metrics[target] = {"MAE": mae, "MAPE": mape, "R2": r2}

    print(f"\n=== MODEL EVALUATION: {target.upper()} ===")
    print("MAE :", mae)
    print("MAPE:", mape)
    print("R²  :", r2)

    # Plot actual vs predicted
    plt.figure(figsize=(8, 4))
    plt.plot(test.index, y_test, marker="o", label="Actual")
    plt.plot(test.index, y_pred, marker="o", label="Predicted")
    plt.title(f"{target} – Actual vs Predicted (Test)")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

print("\n=== ALL METRICS ===")
for t, m in metrics.items():
    print(t, ":", m)


# ============================================================
# 8. FORECAST NEXT 4 QUARTERS (RECURSIVE, USING ARRAYS)
# ============================================================

future_dates = pd.date_range(
    start=df_ml.index[-1] + pd.offsets.QuarterEnd(1),
    periods=4,
    freq="QE",  # use QE instead of deprecated Q
)

forecast_rows = []
history = df_ml.copy()

for date in future_dates:
    row_feats = {}

    for col in targets:
        row_feats[f"{col}_lag1"] = history[col].iloc[-1]
        row_feats[f"{col}_lag2"] = history[col].iloc[-2]
        row_feats[f"{col}_lag4"] = history[col].iloc[-4]
        row_feats[f"{col}_qoq"] = history[f"{col}_qoq"].iloc[-1]

    # Build feature vector in the SAME order as feature_cols
    X_new = np.array([[row_feats[c] for c in feature_cols]])

    preds = {}
    for col in targets:
        preds[col] = models[col].predict(X_new)[0]

    forecast_row = {**{c: row_feats[c] for c in feature_cols}, **preds}
    forecast_rows.append(forecast_row)

    # Extend history with predicted values (for recursive forecasting)
    new_hist_row = history.iloc[-1:].copy()
    for col in targets:
        last_val = history[col].iloc[-1]
        new_val = preds[col]
        new_hist_row[col] = new_val
        new_hist_row[f"{col}_qoq"] = (
            (new_val - last_val) / last_val if last_val != 0 else 0
        )
    new_hist_row.index = [date]
    history = pd.concat([history, new_hist_row])

forecast_df = pd.DataFrame(forecast_rows, index=future_dates)
print("\n=== NEXT 4 QUARTERS FORECAST ===")
print(forecast_df[["revenue", "net_income", "FCF"]])

# Plot revenue actual vs forecast
plt.figure(figsize=(10, 5))
plt.plot(master.index, master["revenue"], marker="o", label="Revenue (actual)")
plt.plot(forecast_df.index, forecast_df["revenue"], marker="o", linestyle="--", label="Revenue (forecast)")
plt.title("Revenue – Actual vs ML Forecast")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# ============================================================
# 9. SIMPLE DCF VALUATION USING FCF FORECAST
# ============================================================

year1_fcf = forecast_df["FCF"].sum()

wacc = 0.09        # 9% discount rate
g_terminal = 0.025 # 2.5% perpetual growth

pv_year1 = year1_fcf / (1 + wacc)

terminal_fcf = year1_fcf * (1 + g_terminal)
terminal_value = terminal_fcf / (wacc - g_terminal)
pv_terminal = terminal_value / (1 + wacc)

enterprise_value = pv_year1 + pv_terminal

print("\n=== SIMPLE DCF BASED ON ML FCF FORECAST ===")
print("Year 1 FCF forecast :", year1_fcf)
print("PV of Year 1 FCF    :", pv_year1)
print("PV of Terminal Value:", pv_terminal)
print("Enterprise Value    :", enterprise_value)
master.to_csv("amazon_master_financials.csv")
forecast_df[["revenue", "net_income", "FCF"]].to_csv("amazon_forecast.csv")

# ============================================================
# END
# ============================================================