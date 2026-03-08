# app.py - Final robust forecasting app (with Total Sales fix)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="📊 Sales Forecast Dashboard", layout="wide")
st.title("📈 Advanced Sales Forecasting Dashboard (ARIMA / SARIMA / SARIMAX) — Final Version")

# ---------------------------
# Helper functions
# ---------------------------
def infer_and_fix_freq(idx):
    freq = pd.infer_freq(idx)
    if freq is None:
        if all(d.day == 1 for d in idx):
            return "MS"
        return "MS"
    return freq

def determine_m_from_freq(freq):
    if freq.startswith("M"):
        return 12
    if freq.startswith("Q") or freq.startswith("BQ"):
        return 4
    if freq.startswith("A") or freq.startswith("Y"):
        return 1
    return 12

def convert_cumulative_if_needed(df_numeric):
    is_cum = True
    for col in df_numeric.columns:
        s = df_numeric[col].dropna()
        if len(s) < 3:
            is_cum = False
            break
        inc = (s.diff() > 0).sum()
        dec = (s.diff() < 0).sum()
        if inc <= dec or s.iloc[-1] <= s.iloc[0]:
            is_cum = False
            break
    if is_cum:
        return df_numeric.diff().fillna(0), True
    return df_numeric, False

# ---------------------------
# File Upload
# ---------------------------
uploaded_file = st.sidebar.file_uploader("📂 Upload your sales dataset (CSV)", type=["csv"])
if uploaded_file is None:
    st.info("👈 Upload your CSV file (e.g., monthly sales) to begin.")
    st.stop()

# ---------------------------
# Load & clean
# ---------------------------
df = pd.read_csv(uploaded_file)
df.columns = [c.strip() for c in df.columns]

date_col = next((c for c in df.columns if "date" in c.lower()), None)
if date_col is None:
    st.error("❌ No date column detected. Make sure your CSV contains a date column.")
    st.stop()

df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
df = df.dropna(subset=[date_col])
df = df.groupby(date_col, as_index=False).mean()
df = df.sort_values(by=date_col).reset_index(drop=True)
df.set_index(date_col, inplace=True)

category_cols = [c for c in df.select_dtypes(include=[np.number]).columns]
if not category_cols:
    st.error("❌ No numeric columns found. Provide numeric category columns.")
    st.stop()

st.sidebar.success(f"Detected numeric categories: {', '.join(category_cols)}")

inferred_freq = infer_and_fix_freq(df.index)
m_val = determine_m_from_freq(inferred_freq)
st.sidebar.info(f"Detected frequency: {inferred_freq} (m={m_val})")

df = df.asfreq(inferred_freq)
df = df.fillna(method="ffill")

df_numeric = df[category_cols].copy()
df_numeric, converted = convert_cumulative_if_needed(df_numeric)
if converted:
    st.sidebar.warning("Detected cumulative sales. Converted to incremental (monthly) sales for forecasting.")

df[category_cols] = df_numeric

# ---------------------------
# Sidebar Inputs
# ---------------------------
category = st.sidebar.selectbox("📦 Select Category", category_cols)
model_choice = st.sidebar.radio("🤖 Choose Model Type", ["ARIMA", "SARIMA", "SARIMAX"])
horizon = st.sidebar.slider("⏳ Forecast Horizon (periods)", 3, 24, 6)

series = df[[category]].copy()
series = series.rename(columns={category: "sales"})
series = series.dropna()

if len(series) < 12:
    st.warning("⚠️ Series is short (<12 points). Forecasts may be unreliable.")

train_size = int(len(series) * 0.8)
if train_size < 6:
    train_size = max(1, len(series) - horizon)
train = series.iloc[:train_size]
test = series.iloc[train_size:]

st.subheader(f"📊 Auto parameter tuning for {category} — Model: {model_choice}")

# ---------------------------
# Exogenous data (for SARIMAX)
# ---------------------------
exog = None
exog_train = exog_test = None
if model_choice == "SARIMAX":
    exog_cols = [c for c in category_cols if c != category]
    if exog_cols:
        exog = df[exog_cols].copy().fillna(method="ffill")
        exog_train = exog.iloc[:train_size]
        exog_test = exog.iloc[train_size:]
    else:
        st.sidebar.warning("No exogenous numeric columns found — SARIMAX will run without exog.")

# ---------------------------
# Parameter tuning
# ---------------------------
try:
    # 🧩 Fix for Total_Sales category
    if category.lower() in ["total_sales", "totalsales"]:
        order = (1, 1, 1)
        seasonal_order = (1, 1, 0, m_val)
        st.info("ℹ️ Using smoother SARIMA(1,1,1)(1,1,0,12) for Total_Sales (more realistic forecast).")
    else:
        if model_choice == "ARIMA":
            step = auto_arima(train["sales"], seasonal=False, trace=False,
                              error_action="ignore", suppress_warnings=True,
                              start_p=0, max_p=3, start_q=0, max_q=3, n_jobs=1)
            order = step.order
            seasonal_order = (0, 0, 0, 0)
        elif model_choice == "SARIMA":
            step = auto_arima(train["sales"], seasonal=True, m=m_val, trace=False,
                              error_action="ignore", suppress_warnings=True,
                              start_p=0, max_p=3, start_q=0, max_q=3,
                              start_P=0, max_P=2, start_Q=0, max_Q=2, n_jobs=1)
            order = step.order
            seasonal_order = step.seasonal_order
        else:  # SARIMAX
            if exog is None:
                step = auto_arima(train["sales"], seasonal=True, m=m_val, trace=False,
                                  error_action="ignore", suppress_warnings=True,
                                  start_p=0, max_p=3, start_q=0, max_q=3, n_jobs=1)
            else:
                step = auto_arima(train["sales"], exogenous=exog_train,
                                  seasonal=True, m=m_val, trace=False,
                                  error_action="ignore", suppress_warnings=True,
                                  start_p=0, max_p=3, start_q=0, max_q=3, n_jobs=1)
            order = step.order
            seasonal_order = step.seasonal_order

    st.success(f"✅ Selected Order: {order} | Seasonal Order: {seasonal_order}")

except Exception as e:
    st.error(f"Auto ARIMA failed: {e}")
    st.stop()

# ---------------------------
# Fit Model and Forecast
# ---------------------------
try:
    if model_choice == "SARIMAX" and exog is not None and category.lower() not in ["total_sales", "totalsales"]:
        model = SARIMAX(train["sales"], exog=exog_train, order=order, seasonal_order=seasonal_order)
        res = model.fit(disp=False)
        forecast_test = res.get_forecast(steps=len(test), exog=exog_test)
    else:
        model = SARIMAX(train["sales"], order=order, seasonal_order=seasonal_order)
        res = model.fit(disp=False)
        forecast_test = res.get_forecast(steps=len(test))
except Exception as e:
    st.error(f"Model fit failed: {e}")
    st.stop()

# ---------------------------
# Train-Test Forecast Plot
# ---------------------------
pred_test = forecast_test.predicted_mean
ci_test = forecast_test.conf_int()
pred_test.index = test.index
ci_test.index = test.index

rmse = np.sqrt(mean_squared_error(test["sales"], pred_test))
mape = mean_absolute_percentage_error(test["sales"], pred_test) * 100

st.subheader("📈 Train / Test / Predicted Results")
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(train.index, train["sales"], label="Train", color="tab:blue")
ax.plot(test.index, test["sales"], label="Test (Actual)", color="tab:orange", linestyle="--")
ax.plot(pred_test.index, pred_test, label="Predicted", color="tab:green")
ax.fill_between(ci_test.index, ci_test.iloc[:, 0], ci_test.iloc[:, 1], color="tab:green", alpha=0.2)
ax.legend()
ax.set_title(f"{category} — Train/Test Forecast ({model_choice})")
ax.set_xlabel("Date")
ax.set_ylabel("Sales")
st.pyplot(fig)

st.markdown(f"**RMSE:** `{rmse:.2f}`  |  **MAPE:** `{mape:.2f}%`")

# ---------------------------
# Future Forecast
# ---------------------------
st.subheader(f"🔮 Future Forecast — Next {horizon} Periods")
try:
    if model_choice == "SARIMAX" and exog is not None and category.lower() not in ["total_sales", "totalsales"]:
        if len(exog) >= horizon:
            exog_future = exog.iloc[-horizon:]
        else:
            last = exog.iloc[-1]
            exog_future = pd.DataFrame([last.values] * horizon, columns=exog.columns,
                                       index=pd.date_range(start=series.index[-1] + pd.tseries.frequencies.to_offset(inferred_freq),
                                                           periods=horizon, freq=inferred_freq))
        future_fc = res.get_forecast(steps=horizon, exog=exog_future)
    else:
        future_fc = res.get_forecast(steps=horizon)

    future_mean = future_fc.predicted_mean
    future_ci = future_fc.conf_int()

    future_index = pd.date_range(start=series.index[-1] + pd.tseries.frequencies.to_offset(inferred_freq),
                                 periods=horizon, freq=inferred_freq)
    future_mean.index = future_index
    future_ci.index = future_index

    fig2, ax2 = plt.subplots(figsize=(12, 5))
    ax2.plot(series.index, series["sales"], label="Historical", color="tab:blue")
    ax2.plot(future_mean.index, future_mean, label="Forecast", color="tab:purple")
    ax2.fill_between(future_ci.index, future_ci.iloc[:, 0], future_ci.iloc[:, 1], color="tab:purple", alpha=0.2)
    ax2.legend()
    ax2.set_title(f"{category} — {horizon}-Period Future Forecast ({model_choice})")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Sales")
    st.pyplot(fig2)

    out = pd.DataFrame({
        "Date": future_mean.index,
        "Forecast": future_mean.values,
        "Lower_CI": future_ci.iloc[:, 0].values,
        "Upper_CI": future_ci.iloc[:, 1].values
    })
    st.download_button("📥 Download Future Forecast CSV", out.to_csv(index=False).encode("utf-8"),
                       file_name=f"{category}_future_forecast.csv")

except Exception as e:
    st.error(f"Future forecast failed: {e}")
