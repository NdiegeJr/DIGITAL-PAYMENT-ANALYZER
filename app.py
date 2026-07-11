"""
app.py
======
Digital Payments Trend Analyzer — Dash dashboard (Velonic-style light admin theme).

Tabs:
  1. Trends           — monthly volume & value charts
  2. Peak Seasons     — seasonality by month / day / hour
  3. Forecast         — RandomForest future volume prediction
  4. Channel Preference — mobile vs bank analysis
  5. ML Predictor     — live payment channel prediction form
  6. Upload & Analyze — upload ANY csv and get instant analysis charts

Run:  python app.py   then open http://127.0.0.1:8050
Pre-requisite: python train_models.py  (once, to build models & aggregates)
"""

import base64
import io
import json
import os

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, callback_context, dcc, html, dash_table
import dash

from report_generator import generate_report

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# ---------------------------------------------------------------------------
# LOAD PRE-COMPUTED DATA  (built by train_models.py)
# ---------------------------------------------------------------------------
def _read(name):
    path = os.path.join(DATA_DIR, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {path}. Run `python train_models.py` first.")
    return pd.read_csv(path)

monthly        = _read("agg_monthly.csv")
channel_monthly = _read("agg_channel_monthly.csv")
method_monthly  = _read("agg_method_monthly.csv")
group_monthly   = _read("agg_group_monthly.csv")
dow_volume      = _read("agg_dow.csv")
hour_volume     = _read("agg_hour.csv")
sector_pref     = _read("agg_sector_pref.csv")
segment_pref    = _read("agg_segment_pref.csv")
transactions    = _read("cleaned_transactions.csv")

forecast_model        = joblib.load(os.path.join(MODELS_DIR, "forecast_model.pkl"))
forecast_history      = joblib.load(os.path.join(MODELS_DIR, "forecast_history.pkl"))
channel_model         = joblib.load(os.path.join(MODELS_DIR, "channel_model.pkl"))
channel_encoders      = joblib.load(os.path.join(MODELS_DIR, "channel_encoders.pkl"))
channel_target_encoder = joblib.load(os.path.join(MODELS_DIR, "channel_target_encoder.pkl"))
channel_features      = joblib.load(os.path.join(MODELS_DIR, "channel_features.pkl"))
method_model          = joblib.load(os.path.join(MODELS_DIR, "method_model.pkl"))
method_encoders       = joblib.load(os.path.join(MODELS_DIR, "method_encoders.pkl"))
method_target_encoder = joblib.load(os.path.join(MODELS_DIR, "method_target_encoder.pkl"))
method_features       = joblib.load(os.path.join(MODELS_DIR, "method_features.pkl"))

DOW_ORDER = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]

# ---------------------------------------------------------------------------
# CHART STYLE
# ---------------------------------------------------------------------------
CHART_COLORS = ["#2196f3","#7b6ef6","#2bc9b6","#ff4f81","#ffb703","#0b1a30"]
PLOTLY_LAYOUT = dict(
    template="plotly_white",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#1c2b3a", size=12),
    margin=dict(t=50, l=10, r=10, b=10),
    colorway=CHART_COLORS,
    title_font=dict(size=14, color="#1c2b3a"),
    legend=dict(orientation="h", y=-0.2),
)

def style_fig(fig, **kwargs):
    fig.update_layout(**{**PLOTLY_LAYOUT, **kwargs})
    return fig

# ---------------------------------------------------------------------------
# FORECAST HELPER
# ---------------------------------------------------------------------------
def forecast_future(n_months=6):
    hist = forecast_history.copy()
    last_t      = hist["t"].iloc[-1]
    last_period = pd.Period(hist["MonthPeriod"].iloc[-1], freq="M")
    lag1 = hist["Transaction_Count"].iloc[-1]
    lag2 = hist["Transaction_Count"].iloc[-2]
    rows = []
    for i in range(1, n_months + 1):
        period = last_period + i
        X = pd.DataFrame([[last_t + i, period.month, lag1, lag2]],
                          columns=["t","month_num","lag_1","lag_2"])
        pred = forecast_model.predict(X)[0]
        rows.append({"MonthPeriod": str(period), "Transaction_Count": pred})
        lag2, lag1 = lag1, pred
    return pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# SMART CSV ANALYSER  (works with ANY uploaded CSV)
# ---------------------------------------------------------------------------
def parse_uploaded_csv(contents, filename):
    """Decode base64 upload → DataFrame, return (df, error_msg)."""
    try:
        _, content_string = contents.split(",")
        decoded = base64.b64decode(content_string)
        if filename.lower().endswith(".csv"):
            df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
        else:
            return None, "Only CSV files are supported. Please upload a .csv file."
        if df.empty:
            return None, "The uploaded file is empty."
        return df, None
    except Exception as e:
        return None, f"Could not read file: {e}"


def detect_columns(df):
    """Return (date_col, num_cols, cat_cols) for any DataFrame."""
    date_col = None
    # Try to find a datetime column
    for col in df.columns:
        if df[col].dtype == object or str(df[col].dtype).startswith("datetime"):
            try:
                parsed = pd.to_datetime(df[col], dayfirst=True, errors="coerce")
                valid_ratio = parsed.notna().sum() / max(len(df), 1)
                if valid_ratio > 0.7:
                    date_col = col
                    df[col] = parsed
                    break
            except Exception:
                pass

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Categorical: object columns with reasonable cardinality (2-50 unique values)
    cat_cols = [
        c for c in df.columns
        if c != date_col
        and df[c].dtype == object
        and 2 <= df[c].nunique() <= 50
    ]
    return date_col, num_cols, cat_cols


def build_upload_figures(df):
    """Generate a list of (title, figure) tuples from any uploaded DataFrame."""
    date_col, num_cols, cat_cols = detect_columns(df)
    figures = []

    # ── 1. Time-series charts (if a date column was found) ──────────────────
    if date_col and num_cols:
        df["_period"] = df[date_col].dt.to_period("M").astype(str)
        period_df = df.groupby("_period")[num_cols[:4]].sum().reset_index()

        for col in num_cols[:3]:
            fig = px.line(period_df, x="_period", y=col, markers=True,
                          title=f"Monthly {col} Over Time",
                          labels={"_period": "Month", col: col})
            style_fig(fig)
            figures.append((f"Monthly {col} Over Time", fig))

    # ── 2. Numeric distributions ─────────────────────────────────────────────
    for col in num_cols[:4]:
        fig = px.histogram(df, x=col, nbins=30,
                           title=f"Distribution of {col}",
                           color_discrete_sequence=["#2196f3"])
        style_fig(fig)
        figures.append((f"Distribution of {col}", fig))

    # ── 3. Categorical bar charts ────────────────────────────────────────────
    for col in cat_cols[:4]:
        vc = df[col].value_counts().reset_index()
        vc.columns = [col, "Count"]
        fig = px.bar(vc, x=col, y="Count",
                     title=f"Count by {col}",
                     color="Count",
                     color_continuous_scale=["#c7e2ff","#2196f3","#0b1a30"])
        style_fig(fig)
        figures.append((f"Count by {col}", fig))

    # ── 4. Cross-analysis: first categorical vs first numeric ────────────────
    if cat_cols and num_cols:
        cat, num = cat_cols[0], num_cols[0]
        grp = (df.groupby(cat)[num].mean()
                 .reset_index()
                 .sort_values(num, ascending=False))
        fig = px.bar(grp, x=cat, y=num,
                     title=f"Average {num} by {cat}",
                     color=num,
                     color_continuous_scale=["#c3f4ef","#2bc9b6","#0b1a30"])
        style_fig(fig)
        figures.append((f"Average {num} by {cat}", fig))

    # ── 5. Pie chart for first categorical ───────────────────────────────────
    if cat_cols:
        col = cat_cols[0]
        vc = df[col].value_counts().head(10).reset_index()
        vc.columns = [col, "Count"]
        fig = px.pie(vc, names=col, values="Count", hole=0.45,
                     title=f"Share by {col}",
                     color_discrete_sequence=CHART_COLORS)
        style_fig(fig)
        figures.append((f"Share by {col}", fig))

    # ── 6. Second categorical vs first numeric (if both exist) ───────────────
    if len(cat_cols) >= 2 and num_cols:
        cat, num = cat_cols[1], num_cols[0]
        grp = (df.groupby(cat)[num].sum()
                 .reset_index()
                 .sort_values(num, ascending=True))
        fig = px.bar(grp, x=num, y=cat, orientation="h",
                     title=f"Total {num} by {cat}",
                     color=num,
                     color_continuous_scale=["#ede0fd","#7b6ef6","#0b1a30"])
        style_fig(fig)
        figures.append((f"Total {num} by {cat}", fig))

    return figures


def build_overview_stats(df):
    """Return (overview_rows, describe_df) for the uploaded DataFrame."""
    date_col, num_cols, cat_cols = detect_columns(df)

    date_range = "N/A"
    if date_col:
        mn = df[date_col].min()
        mx = df[date_col].max()
        date_range = f"{mn.strftime('%d %b %Y')} → {mx.strftime('%d %b %Y')}"

    overview = [
        ("Total Records", f"{len(df):,}"),
        ("Total Columns", str(len(df.columns))),
        ("Numeric Columns", str(len(num_cols))),
        ("Categorical Columns", str(len(cat_cols))),
        ("Date Column Detected", date_col if date_col else "None"),
        ("Date Range", date_range),
        ("Missing Values", str(int(df.isna().sum().sum()))),
        ("Duplicate Rows", str(int(df.duplicated().sum()))),
    ]

    desc = None
    if num_cols:
        desc = df[num_cols].describe().round(2).reset_index()
        desc.rename(columns={"index": "Statistic"}, inplace=True)

    return overview, desc

# ---------------------------------------------------------------------------
# APP SETUP
# ---------------------------------------------------------------------------
app = Dash(__name__, suppress_callback_exceptions=True)
app.title = "Digital Payments Trend Analyzer"

app.index_string = """
<!DOCTYPE html>
<html>
<head>
{%metas%}
<title>{%title%}</title>
{%favicon%}
{%css%}
<link rel="stylesheet"
  href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css">
</head>
<body>
{%app_entry%}
<footer>{%config%}{%scripts%}{%renderer%}</footer>
</body>
</html>
"""

# ── KPI values ──────────────────────────────────────────────────────────────
total_tx    = int(monthly["Transaction_Count"].sum())
total_val   = monthly["Total_Value"].sum()
avg_success = monthly["Success_Rate"].mean()
mobile_share = (
    group_monthly[group_monthly["Payment_Group"] == "Mobile/Digital Payment"]["Count"].sum()
    / group_monthly["Count"].sum()
)
last2 = monthly.tail(2)
tx_delta  = (last2["Transaction_Count"].iloc[1] / last2["Transaction_Count"].iloc[0] - 1) * 100 if len(last2) == 2 else 0.0
val_delta = (last2["Total_Value"].iloc[1] / last2["Total_Value"].iloc[0] - 1) * 100           if len(last2) == 2 else 0.0


def kpi_card(icon, label, value, delta, css_class):
    return html.Div([
        html.Div([
            html.Div(label, className="kpi-label"),
            html.Div(value, className="kpi-value"),
            html.Div([
                html.Span(f"{delta:+.2f}%", className="badge"),
                html.Span("Since last month"),
            ], className="kpi-delta"),
        ], className="kpi-main"),
        html.Div(html.I(className=f"fa-solid {icon}"), className="kpi-icon"),
    ], className=f"kpi-card {css_class}")


kpi_row = html.Div([
    kpi_card("fa-chart-line",   "Total Transactions",   f"{total_tx:,}",                    tx_delta,  "kpi-pink"),
    kpi_card("fa-sterling-sign","Total Value",           f"TSH{total_val/1_000_000:.2f}M",     val_delta, "kpi-purple"),
    kpi_card("fa-circle-check", "Avg Success Rate",     f"{avg_success*100:.1f}%",           0.0,       "kpi-blue"),
    kpi_card("fa-mobile-screen","Mobile/Digital Share", f"{mobile_share*100:.1f}%",          0.0,       "kpi-teal"),
], className="kpi-row")


def panel(title, children, extra_class=""):
    return html.Div(
        [html.Div(title, className="panel-title") if title else None, children],
        className=f"panel {extra_class}",
    )


def field(label, comp):
    return html.Div(
        [html.Label(label, className="field-label"), comp],
        style={"flex": "1", "marginRight": "14px"},
    )

# ---------------------------------------------------------------------------
# TAB 1 — TRENDS
# ---------------------------------------------------------------------------
def trends_tab():
    fig_vol      = px.line(monthly, x="MonthPeriod", y="Transaction_Count", markers=True, title="Monthly Transaction Volume")
    fig_val      = px.bar(monthly,  x="MonthPeriod", y="Total_Value",       title="Monthly Transaction Value (£)")
    fig_methods  = px.area(method_monthly,  x="MonthPeriod", y="Count", color="Payment_Method",       title="Payment Method Volume Over Time")
    fig_channels = px.area(channel_monthly, x="MonthPeriod", y="Count", color="Transaction_Channel",  title="Transaction Channel Over Time (Online / In-Store / In-App)")
    for f in [fig_vol, fig_val, fig_methods, fig_channels]:
        style_fig(f)
    return html.Div([
        panel(None, dcc.Graph(figure=fig_vol,      config={"displayModeBar": False})),
        panel(None, dcc.Graph(figure=fig_val,      config={"displayModeBar": False})),
        panel(None, dcc.Graph(figure=fig_methods,  config={"displayModeBar": False})),
        panel(None, dcc.Graph(figure=fig_channels, config={"displayModeBar": False})),
    ])

# ---------------------------------------------------------------------------
# TAB 2 — PEAK SEASONS
# ---------------------------------------------------------------------------
def peak_tab():
    monthly2 = monthly.copy()
    monthly2["month_num"] = pd.PeriodIndex(monthly2["MonthPeriod"], freq="M").month
    seasonal_avg = monthly2.groupby("month_num")["Transaction_Count"].mean().reset_index()
    month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    seasonal_avg["Month"] = seasonal_avg["month_num"].apply(lambda m: month_names[m - 1])

    fig_season = px.bar(seasonal_avg, x="Month", y="Transaction_Count",
                         category_orders={"Month": month_names},
                         title="Average Transactions by Calendar Month (Seasonality)")
    dow = dow_volume.copy()
    dow["DayOfWeek"] = pd.Categorical(dow["DayOfWeek"], categories=DOW_ORDER, ordered=True)
    dow = dow.sort_values("DayOfWeek")
    fig_dow  = px.bar(dow, x="DayOfWeek", y="Count",  title="Transaction Volume by Day of Week")
    fig_hour = px.line(hour_volume, x="Hour", y="Count", markers=True, title="Transaction Volume by Hour of Day")
    for f in [fig_season, fig_dow, fig_hour]:
        style_fig(f)

    top_month = monthly.loc[monthly["Transaction_Count"].idxmax()]
    low_month = monthly.loc[monthly["Transaction_Count"].idxmin()]

    insight = panel(None, html.Div([
        html.Div([
            html.I(className="fa-solid fa-arrow-trend-up", style={"color": "#2bc9b6", "marginRight": "8px"}),
            html.Span("Peak month: "),
            html.B(f"{top_month['MonthPeriod']} ({int(top_month['Transaction_Count'])} txns)"),
        ], className="insight-item"),
        html.Div([
            html.I(className="fa-solid fa-arrow-trend-down", style={"color": "#ff4f81", "marginRight": "8px"}),
            html.Span("Lowest month: "),
            html.B(f"{low_month['MonthPeriod']} ({int(low_month['Transaction_Count'])} txns)"),
        ], className="insight-item"),
    ], className="insight-row"))

    return html.Div([
        insight,
        panel(None, dcc.Graph(figure=fig_season, config={"displayModeBar": False})),
        panel(None, dcc.Graph(figure=fig_dow,    config={"displayModeBar": False})),
        panel(None, dcc.Graph(figure=fig_hour,   config={"displayModeBar": False})),
    ])

# ---------------------------------------------------------------------------
# TAB 3 — FORECAST
# ---------------------------------------------------------------------------
def forecast_tab():
    return html.Div([
        panel("Forecast Horizon", html.Div([
            dcc.Slider(id="forecast-months", min=1, max=12, step=1, value=6,
                       marks={i: str(i) for i in [1, 3, 6, 9, 12]}),
        ], style={"padding": "10px 4px"})),
        panel(None, dcc.Graph(id="forecast-graph", config={"displayModeBar": False})),
        panel("Predicted Values", html.Div(id="forecast-table")),
    ])

# ---------------------------------------------------------------------------
# TAB 4 — CHANNEL PREFERENCE
# ---------------------------------------------------------------------------
def preference_tab():
    fig_group = px.pie(
        group_monthly.groupby("Payment_Group")["Count"].sum().reset_index(),
        names="Payment_Group", values="Count", hole=0.55,
        title="Mobile/Digital vs Card/Bank Payments",
        color_discrete_sequence=["#2196f3","#ff4f81"])
    fig_method = px.bar(
        method_monthly.groupby("Payment_Method")["Count"].sum().reset_index().sort_values("Count"),
        x="Count", y="Payment_Method", orientation="h",
        title="Total Volume by Specific Payment Method")
    fig_sector = px.bar(sector_pref, x="Sector", y="Count", color="Payment_Group", barmode="group",
                         title="Payment Preference by Business Sector",
                         color_discrete_sequence=["#2196f3","#ff4f81"])
    fig_segment = px.bar(segment_pref, x="Customer_Segment", y="Count", color="Payment_Group", barmode="group",
                          title="Payment Preference by Customer Segment",
                          category_orders={"Customer_Segment": ["Low","Medium","High"]},
                          color_discrete_sequence=["#2196f3","#ff4f81"])
    for f in [fig_group, fig_method, fig_sector, fig_segment]:
        style_fig(f)
    return html.Div([
        html.Div([
            panel(None, dcc.Graph(figure=fig_group,  config={"displayModeBar": False})),
            panel(None, dcc.Graph(figure=fig_method, config={"displayModeBar": False})),
        ], className="panel-row"),
        panel(None, dcc.Graph(figure=fig_sector,  config={"displayModeBar": False})),
        panel(None, dcc.Graph(figure=fig_segment, config={"displayModeBar": False})),
    ])

# ---------------------------------------------------------------------------
# TAB 5 — ML PREDICTOR
# ---------------------------------------------------------------------------
sectors  = sorted(transactions["Sector"].unique())
segments = sorted(transactions["Customer_Segment"].unique())
devices  = sorted(transactions["Device_Type"].unique())

def predictor_tab():
    return panel("Predict Preferred Payment Channel", html.Div([
        html.Div([
            field("Sector",           dcc.Dropdown(id="in-sector",   options=[{"label": s, "value": s} for s in sectors],  value=sectors[0],  clearable=False)),
            field("Customer Segment", dcc.Dropdown(id="in-segment",  options=[{"label": s, "value": s} for s in segments], value=segments[0], clearable=False)),
            field("Device Type",      dcc.Dropdown(id="in-device",   options=[{"label": d, "value": d} for d in devices],  value=devices[0],  clearable=False)),
        ], style={"display": "flex", "marginBottom": "16px"}),
        html.Div([
            field("Transaction Amount (£)", dcc.Input(id="in-amount", type="number", value=100,
                  style={"width":"100%","padding":"8px","borderRadius":"6px","border":"1px solid #d8dee8"})),
            field("Day of Week", dcc.Dropdown(id="in-dow",
                  options=[{"label": d, "value": i} for i, d in enumerate(DOW_ORDER)], value=0, clearable=False)),
            field("Hour of Day (0-23)", dcc.Input(id="in-hour", type="number", value=12, min=0, max=23,
                  style={"width":"100%","padding":"8px","borderRadius":"6px","border":"1px solid #d8dee8"})),
        ], style={"display": "flex", "marginBottom": "20px"}),
        html.Button("Predict", id="predict-btn", n_clicks=0, className="predict-btn"),
        html.Div(id="prediction-output", style={"marginTop": "18px"}),
    ]))

# ---------------------------------------------------------------------------
# TAB 6 — UPLOAD & ANALYZE
# ---------------------------------------------------------------------------
def upload_tab():
    return html.Div([

        # ── Drop zone ───────────────────────────────────────────────────────
        panel("Upload a CSV Dataset", html.Div([
            dcc.Upload(
                id="upload-csv",
                children=html.Div([
                    html.Div([
                        html.I(className="fa-solid fa-cloud-arrow-up",
                               style={"fontSize":"40px","color":"#2196f3","marginBottom":"12px","display":"block"}),
                        html.Div("Drag & drop a CSV file here, or",
                                 style={"fontSize":"15px","color":"#374151","marginBottom":"6px"}),
                        html.Div("click to browse", style={"fontSize":"13px","color":"#8a94a6"}),
                    ], style={"textAlign":"center"}),
                ]),
                style={
                    "border": "2px dashed #2196f3",
                    "borderRadius": "12px",
                    "padding": "40px 20px",
                    "cursor": "pointer",
                    "background": "#f0f7ff",
                    "transition": "background 0.2s",
                },
                multiple=False,
            ),
            html.Div(id="upload-status", style={"marginTop":"14px"}),
        ])),

        # ── Overview stats + analysis output (hidden until file uploaded) ──
        html.Div(id="upload-overview"),
        html.Div(id="upload-charts"),

        # ── PDF Report section (shown after analysis is ready) ─────────────
        html.Div(id="report-section"),

    ])

# ---------------------------------------------------------------------------
# LAYOUT
# ---------------------------------------------------------------------------
NAV_ITEMS = [
    ("trends",     "Trends"),
    ("peaks",      "Peak Seasons"),
    ("forecast",   "Forecast"),
    ("preference", "Channel Preference"),
    ("predictor",  "ML Predictor"),
    ("upload",     "Upload & Analyze"),
]

sidebar = html.Div([
    html.Div([
        html.I(className="fa-solid fa-layer-group",
               style={"marginRight":"10px","color":"#4cc9c0"}),
        "PayAnalytics",
    ], className="sidebar-logo"),
    html.Div("Main", className="sidebar-section-label"),
    dcc.Tabs(
        id="tabs", value="trends", vertical=True,
        className="sidebar-tabs",
        children=[dcc.Tab(label=label, value=val) for val, label in NAV_ITEMS],
    ),
    html.Div("Data", className="sidebar-section-label"),
    html.Div([
        html.I(className="fa-solid fa-file-csv",
               style={"marginRight":"8px","color":"#2bc9b6"}),
        "Upload & Analyze",
    ], style={"padding":"10px 24px","fontSize":"13.5px","color":"#b9c2d0",
               "cursor":"pointer"},
       id="sidebar-upload-hint"),
], className="sidebar")

topbar = html.Div([
    html.Div([
        html.Div("Digital Payments Trend Analyzer", className="topbar-title"),
        html.Div("Transaction analysis · seasonality · forecasting · channel preference · CSV upload",
                 className="topbar-sub"),
    ]),
], className="topbar")

app.layout = html.Div([
    # Global persistent components — live outside tabs
    dcc.Store(id="uploaded-df-store"),
    dcc.Store(id="uploaded-filename-store"),
    dcc.Download(id="download-pdf-report"),
    sidebar,
    html.Div([
        topbar,
        html.Div([
            kpi_row,
            html.Div(id="tab-content"),
        ], className="content-pad"),
    ], className="main-area"),
], className="app-shell")


# ---------------------------------------------------------------------------
# CALLBACKS
# ---------------------------------------------------------------------------

# ── Tab router ──────────────────────────────────────────────────────────────
@app.callback(Output("tab-content", "children"), Input("tabs", "value"))
def render_tab(tab):
    if tab == "trends":     return trends_tab()
    if tab == "peaks":      return peak_tab()
    if tab == "forecast":   return forecast_tab()
    if tab == "preference": return preference_tab()
    if tab == "predictor":  return predictor_tab()
    if tab == "upload":     return upload_tab()
    return html.Div("Unknown tab")


# ── Forecast slider ──────────────────────────────────────────────────────────
@app.callback(
    Output("forecast-graph", "figure"),
    Output("forecast-table", "children"),
    Input("forecast-months", "value"),
)
def update_forecast(n_months):
    fut  = forecast_future(n_months)
    hist = forecast_history[["MonthPeriod","Transaction_Count"]].copy()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist["MonthPeriod"], y=hist["Transaction_Count"],
                              mode="lines+markers", name="Historical",
                              line=dict(color="#2196f3")))
    fig.add_trace(go.Scatter(x=fut["MonthPeriod"], y=fut["Transaction_Count"],
                              mode="lines+markers", name="Forecast",
                              line=dict(color="#ff4f81", dash="dash")))
    style_fig(fig, title=f"Transaction Volume Forecast — Next {n_months} Month(s)")

    rows = [html.Tr([
        html.Th("Month",                   style={"textAlign":"left","padding":"8px","color":"#5d6b82"}),
        html.Th("Predicted Transactions",  style={"textAlign":"left","padding":"8px","color":"#5d6b82"}),
    ])]
    for _, r in fut.iterrows():
        rows.append(html.Tr([
            html.Td(r["MonthPeriod"],              style={"padding":"8px","borderTop":"1px solid #eef1f6"}),
            html.Td(f"{r['Transaction_Count']:.0f}", style={"padding":"8px","borderTop":"1px solid #eef1f6"}),
        ]))
    table = html.Table(rows, style={"width":"100%","borderCollapse":"collapse",
                                     "fontSize":"13.5px","color":"#1c2b3a"})
    return fig, table


# ── ML Predictor ─────────────────────────────────────────────────────────────
@app.callback(
    Output("prediction-output", "children"),
    Input("predict-btn", "n_clicks"),
    State("in-sector",  "value"),
    State("in-segment", "value"),
    State("in-device",  "value"),
    State("in-amount",  "value"),
    State("in-dow",     "value"),
    State("in-hour",    "value"),
)
def predict_channel(n_clicks, sector, segment, device, amount, dow, hour):
    if n_clicks == 0:
        return html.Div("Set the inputs above and click Predict.", style={"color":"#8a94a6"})

    quarter = (pd.Timestamp.now().month - 1) // 3 + 1

    row = pd.DataFrame([{
        "Sector":                  channel_encoders["Sector"].transform([sector])[0],
        "Customer_Segment":        channel_encoders["Customer_Segment"].transform([segment])[0],
        "Device_Type":             channel_encoders["Device_Type"].transform([device])[0],
        "Transaction_Amount_GBP":  amount,
        "DayOfWeekNum":            dow,
        "Hour":                    hour,
        "Quarter":                 quarter,
    }])[channel_features]
    group_pred   = channel_target_encoder.inverse_transform(channel_model.predict(row))[0]
    group_proba  = channel_model.predict_proba(row)[0]
    group_labels = channel_target_encoder.classes_

    row2 = pd.DataFrame([{
        "Sector":                 method_encoders["Sector"].transform([sector])[0],
        "Customer_Segment":       method_encoders["Customer_Segment"].transform([segment])[0],
        "Device_Type":            method_encoders["Device_Type"].transform([device])[0],
        "Transaction_Amount_GBP": amount,
        "DayOfWeekNum":           dow,
        "Hour":                   hour,
    }])[method_features]
    method_pred = method_target_encoder.inverse_transform(method_model.predict(row2))[0]
    proba_text  = ", ".join(f"{lbl}: {p*100:.1f}%" for lbl, p in zip(group_labels, group_proba))

    return html.Div([
        html.Div([
            html.I(className="fa-solid fa-circle-check",
                   style={"color":"#2bc9b6","marginRight":"10px","fontSize":"20px"}),
            html.Span("Prediction Result",
                      style={"fontWeight":"700","fontSize":"15px","color":"#1c2b3a"}),
        ], style={"marginBottom":"10px"}),
        html.P(["Most likely payment group: ",   html.B(group_pred)],  style={"color":"#374151"}),
        html.P(["Most likely specific method: ", html.B(method_pred)], style={"color":"#374151"}),
        html.P(f"Confidence breakdown — {proba_text}",
               style={"color":"#8a94a6","fontSize":"12.5px"}),
        html.P("Note: specific-method prediction accuracy is low (~21%) because this dataset's "
               "Payment_Method field shows little real correlation with customer/transaction features. "
               "The Mobile-vs-Card/Bank group prediction is more reliable (~60% accuracy vs ~41% baseline).",
               style={"color":"#a0aec0","fontSize":"11.5px","marginTop":"10px"}),
    ], style={"background":"#f7faf9","border":"1px solid #e3f3ef","borderRadius":"8px","padding":"16px"})


# ── UPLOAD — Step 1: parse CSV → store data + filename ──────────────────────
@app.callback(
    Output("uploaded-df-store",       "data"),
    Output("uploaded-filename-store", "data"),
    Output("upload-status",           "children"),
    Input("upload-csv", "contents"),
    State("upload-csv", "filename"),
    prevent_initial_call=True,
)
def store_uploaded_csv(contents, filename):
    if contents is None:
        return None, None, ""

    df, err = parse_uploaded_csv(contents, filename)
    if err:
        msg = html.Div([
            html.I(className="fa-solid fa-circle-xmark",
                   style={"color":"#ff4f81","marginRight":"8px"}),
            err,
        ], style={"color":"#ff4f81","fontWeight":"600","padding":"10px",
                   "background":"#fff0f4","borderRadius":"8px","border":"1px solid #ffcdd8"})
        return None, None, msg

    success_msg = html.Div([
        html.I(className="fa-solid fa-circle-check",
               style={"color":"#2bc9b6","marginRight":"8px"}),
        html.B(f"'{filename}'"),
        f" uploaded successfully — {len(df):,} rows × {len(df.columns)} columns. Analysing…",
    ], style={"color":"#059669","fontWeight":"500","padding":"12px",
               "background":"#f0fdf4","borderRadius":"8px","border":"1px solid #bbf7d0"})

    return df.to_json(date_format="iso", orient="split"), filename, success_msg


# ── UPLOAD — Step 2: JSON → overview stats + charts + PDF button ─────────────
@app.callback(
    Output("upload-overview",  "children"),
    Output("upload-charts",    "children"),
    Output("report-section",   "children"),
    Input("uploaded-df-store", "data"),
    prevent_initial_call=True,
)
def render_upload_analysis(data):
    if data is None:
        return "", "", ""

    df = pd.read_json(io.StringIO(data), orient="split")

    # ── Overview stats ───────────────────────────────────────────────────────
    overview_rows, desc_df = build_overview_stats(df)

    stat_cards = html.Div([
        html.Div([
            html.Div(val, style={"fontSize":"22px","fontWeight":"700","color":"#1c2b3a"}),
            html.Div(lbl, style={"fontSize":"11.5px","color":"#8a94a6","textTransform":"uppercase",
                                   "letterSpacing":".04em","marginTop":"2px"}),
        ], style={
            "background":"#f8fafc","borderRadius":"10px","padding":"16px 20px",
            "border":"1px solid #e8edf4","flex":"1","minWidth":"140px",
        })
        for lbl, val in overview_rows
    ], style={"display":"flex","gap":"12px","flexWrap":"wrap","marginBottom":"4px"})

    overview_section = panel("Dataset Overview", stat_cards)

    # ── Descriptive statistics table ─────────────────────────────────────────
    desc_section = ""
    if desc_df is not None:
        tbl = dash_table.DataTable(
            data=desc_df.to_dict("records"),
            columns=[{"name": c, "id": c} for c in desc_df.columns],
            style_table={"overflowX": "auto"},
            style_header={"backgroundColor":"#0b1a30","color":"#ffffff",
                          "fontWeight":"700","fontSize":"12px","padding":"10px"},
            style_cell={"padding":"8px 12px","fontSize":"12px",
                        "color":"#374151","border":"1px solid #eef1f6"},
            style_data_conditional=[{"if":{"row_index":"odd"},"backgroundColor":"#f9fafb"}],
        )
        desc_section = panel("Statistical Summary (Numeric Columns)", tbl)

    # ── Column info table ─────────────────────────────────────────────────────
    col_info = []
    for c in df.columns:
        col_info.append({
            "Column":       c,
            "Data Type":    str(df[c].dtype),
            "Non-Null":     int(df[c].notna().sum()),
            "Missing":      int(df[c].isna().sum()),
            "Unique Values":int(df[c].nunique()),
            "Sample Value": str(df[c].dropna().iloc[0]) if df[c].notna().any() else "N/A",
        })
    col_tbl = dash_table.DataTable(
        data=col_info,
        columns=[{"name": k, "id": k} for k in col_info[0].keys()],
        style_table={"overflowX":"auto"},
        style_header={"backgroundColor":"#2196f3","color":"#ffffff",
                      "fontWeight":"700","fontSize":"12px","padding":"10px"},
        style_cell={"padding":"8px 12px","fontSize":"12px","color":"#374151",
                    "border":"1px solid #eef1f6","maxWidth":"200px",
                    "overflow":"hidden","textOverflow":"ellipsis"},
        style_data_conditional=[{"if":{"row_index":"odd"},"backgroundColor":"#f9fafb"}],
        page_size=15,
    )
    col_section = panel("Column Details", col_tbl)

    # ── Analysis charts ──────────────────────────────────────────────────────
    figs = build_upload_figures(df)

    if not figs:
        charts_out = panel("Analysis", html.Div(
            "No analysable columns detected. The file needs at least one numeric or categorical column.",
            style={"color":"#8a94a6","padding":"20px","textAlign":"center"},
        ))
        report_section = ""
    else:
        chart_divs = []
        for i in range(0, len(figs), 2):
            pair = figs[i: i + 2]
            if len(pair) == 2:
                chart_divs.append(html.Div([
                    panel(pair[0][0], dcc.Graph(figure=pair[0][1], config={"displayModeBar": False})),
                    panel(pair[1][0], dcc.Graph(figure=pair[1][1], config={"displayModeBar": False})),
                ], className="panel-row"))
            else:
                chart_divs.append(
                    panel(pair[0][0], dcc.Graph(figure=pair[0][1], config={"displayModeBar": False}))
                )

        charts_out = html.Div([
            html.Div(
                f"Analysis complete — {len(figs)} charts generated from your dataset",
                style={"fontWeight":"600","color":"#059669","marginBottom":"12px",
                        "padding":"10px 16px","background":"#f0fdf4",
                        "borderRadius":"8px","border":"1px solid #bbf7d0","fontSize":"14px"},
            ),
        ] + chart_divs)

        # ── PDF Report button panel ──────────────────────────────────────────
        report_section = panel("Generate PDF Report", html.Div([
            html.Div(
                "All charts, statistics, dataset overview, and key insights will be "
                "included in the PDF report.",
                style={"color":"#5d6b82","fontSize":"13.5px","marginBottom":"18px"},
            ),
            html.Div([
                html.Button([
                    html.I(className="fa-solid fa-file-pdf",
                           style={"marginRight":"8px","fontSize":"15px"}),
                    "Download PDF Report",
                ], id="generate-report-btn", n_clicks=0,
                   className="predict-btn",
                   style={"background":"linear-gradient(135deg,#ff4f81,#7b6ef6)",
                          "fontSize":"14px","padding":"11px 26px"}),

                html.Div(id="report-progress",
                         style={"marginLeft":"18px","fontSize":"13.5px",
                                "color":"#5d6b82","alignSelf":"center"}),
            ], style={"display":"flex","alignItems":"center"}),
        ]))

    return (
        html.Div([overview_section, desc_section, col_section]),
        charts_out,
        report_section,
    )


# ── UPLOAD — Step 3: Generate PDF and serve as download ──────────────────────
@app.callback(
    Output("download-pdf-report", "data"),
    Output("report-progress",     "children"),
    Input("generate-report-btn",  "n_clicks"),
    State("uploaded-df-store",       "data"),
    State("uploaded-filename-store", "data"),
    prevent_initial_call=True,
)
def download_pdf(n_clicks, store_data, filename):
    if n_clicks == 0 or store_data is None:
        return dash.no_update, ""

    try:
        df = pd.read_json(io.StringIO(store_data), orient="split")
        dataset_name = filename if filename else "Uploaded Dataset"

        # Re-generate the same figures used in the dashboard view
        figs = build_upload_figures(df)

        # Build PDF
        pdf_bytes = generate_report(df, figs, dataset_name=dataset_name)

        # Filename for the download
        safe_name = dataset_name.replace(".csv","").replace(" ","_")
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M")
        pdf_filename = f"PayAnalytics_Report_{safe_name}_{ts}.pdf"

        return (
            dcc.send_bytes(pdf_bytes, pdf_filename),
            html.Span([
                html.I(className="fa-solid fa-circle-check",
                       style={"color":"#2bc9b6","marginRight":"6px"}),
                "Report downloaded successfully!",
            ], style={"color":"#059669","fontWeight":"600"}),
        )

    except Exception as e:
        return (
            dash.no_update,
            html.Span([
                html.I(className="fa-solid fa-triangle-exclamation",
                       style={"color":"#ff4f81","marginRight":"6px"}),
                f"Error generating report: {str(e)[:80]}",
            ], style={"color":"#ff4f81"}),
        )


if __name__ == "__main__":
    app.run(debug=True, port=8050)

