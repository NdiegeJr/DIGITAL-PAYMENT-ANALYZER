"""
app.py
======
Digital Payments Trend Analyzer — Dash dashboard (Velonic-style light admin theme).

Run from PyCharm: right-click app.py -> Run 'app'
Or from terminal:  python app.py
Then open http://127.0.0.1:8050 in your browser.

IMPORTANT: run `python train_models.py` once first so the models/ and
data/agg_*.csv files this app reads actually exist.
"""

import os
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from dash import Dash, dcc, html, Input, Output, State

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# ---------------------------------------------------------------------------
# LOAD PRE-COMPUTED DATA (built by train_models.py)
# ---------------------------------------------------------------------------
def _read(name):
    path = os.path.join(DATA_DIR, name)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing {path}. Run `python train_models.py` first to generate it."
        )
    return pd.read_csv(path)

monthly = _read("agg_monthly.csv")
channel_monthly = _read("agg_channel_monthly.csv")
method_monthly = _read("agg_method_monthly.csv")
group_monthly = _read("agg_group_monthly.csv")
dow_volume = _read("agg_dow.csv")
hour_volume = _read("agg_hour.csv")
sector_pref = _read("agg_sector_pref.csv")
segment_pref = _read("agg_segment_pref.csv")
transactions = _read("cleaned_transactions.csv")

forecast_model = joblib.load(os.path.join(MODELS_DIR, "forecast_model.pkl"))
forecast_history = joblib.load(os.path.join(MODELS_DIR, "forecast_history.pkl"))

channel_model = joblib.load(os.path.join(MODELS_DIR, "channel_model.pkl"))
channel_encoders = joblib.load(os.path.join(MODELS_DIR, "channel_encoders.pkl"))
channel_target_encoder = joblib.load(os.path.join(MODELS_DIR, "channel_target_encoder.pkl"))
channel_features = joblib.load(os.path.join(MODELS_DIR, "channel_features.pkl"))

method_model = joblib.load(os.path.join(MODELS_DIR, "method_model.pkl"))
method_encoders = joblib.load(os.path.join(MODELS_DIR, "method_encoders.pkl"))
method_target_encoder = joblib.load(os.path.join(MODELS_DIR, "method_target_encoder.pkl"))
method_features = joblib.load(os.path.join(MODELS_DIR, "method_features.pkl"))

DOW_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

CHART_COLORS = ["#2196f3", "#7b6ef6", "#2bc9b6", "#ff4f81", "#ffb703", "#0b1a30"]
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
    layout = {**PLOTLY_LAYOUT, **kwargs}
    fig.update_layout(**layout)
    return fig


# ---------------------------------------------------------------------------
# FORECAST HELPER — recursively predicts the next N months
# ---------------------------------------------------------------------------
def forecast_future(n_months=6):
    hist = forecast_history.copy()
    last_t = hist["t"].iloc[-1]
    last_period = pd.Period(hist["MonthPeriod"].iloc[-1], freq="M")
    lag1 = hist["Transaction_Count"].iloc[-1]
    lag2 = hist["Transaction_Count"].iloc[-2]

    rows = []
    for i in range(1, n_months + 1):
        period = last_period + i
        t = last_t + i
        month_num = period.month
        X = pd.DataFrame([[t, month_num, lag1, lag2]],
                          columns=["t", "month_num", "lag_1", "lag_2"])
        pred = forecast_model.predict(X)[0]
        rows.append({"MonthPeriod": str(period), "Transaction_Count": pred, "type": "Forecast"})
        lag2 = lag1
        lag1 = pred

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# APP SETUP
# ---------------------------------------------------------------------------
app = Dash(__name__)
app.title = "Digital Payments Trend Analyzer"

app.index_string = """
<!DOCTYPE html>
<html>
<head>
{%metas%}
<title>{%title%}</title>
{%favicon%}
{%css%}
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css">
</head>
<body>
{%app_entry%}
<footer>{%config%}{%scripts%}{%renderer%}</footer>
</body>
</html>
"""


# ---- KPIs ----
total_tx = int(monthly["Transaction_Count"].sum())
total_val = monthly["Total_Value"].sum()
avg_success = monthly["Success_Rate"].mean()
mobile_share = group_monthly[group_monthly["Payment_Group"] == "Mobile/Digital Payment"]["Count"].sum() / group_monthly["Count"].sum()

last2 = monthly.tail(2)
if len(last2) == 2:
    tx_delta = (last2["Transaction_Count"].iloc[1] / last2["Transaction_Count"].iloc[0] - 1) * 100
    val_delta = (last2["Total_Value"].iloc[1] / last2["Total_Value"].iloc[0] - 1) * 100
else:
    tx_delta = val_delta = 0.0


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
    kpi_card("fa-chart-line", "Total Transactions", f"{total_tx:,}", tx_delta, "kpi-pink"),
    kpi_card("fa-sterling-sign", "Total Value", f"£{total_val/1_000_000:.2f}M", val_delta, "kpi-purple"),
    kpi_card("fa-circle-check", "Avg Success Rate", f"{avg_success*100:.1f}%", 0.0, "kpi-blue"),
    kpi_card("fa-mobile-screen", "Mobile/Digital Share", f"{mobile_share*100:.1f}%", 0.0, "kpi-teal"),
], className="kpi-row")


def panel(title, children, extra_class=""):
    return html.Div([
        html.Div(title, className="panel-title") if title else None,
        children,
    ], className=f"panel {extra_class}")


# ---------------------------------------------------------------------------
# TAB 1 — TRENDS
# ---------------------------------------------------------------------------
def trends_tab():
    fig_vol = px.line(monthly, x="MonthPeriod", y="Transaction_Count", markers=True,
                       title="Monthly Transaction Volume")
    fig_val = px.bar(monthly, x="MonthPeriod", y="Total_Value", title="Monthly Transaction Value (£)")
    fig_methods = px.area(method_monthly, x="MonthPeriod", y="Count", color="Payment_Method",
                           title="Payment Method Volume Over Time")
    fig_channels = px.area(channel_monthly, x="MonthPeriod", y="Count", color="Transaction_Channel",
                            title="Transaction Channel Over Time (Online / In-Store / In-App)")

    for f in [fig_vol, fig_val, fig_methods, fig_channels]:
        style_fig(f)

    return html.Div([
        panel(None, dcc.Graph(figure=fig_vol, config={"displayModeBar": False})),
        panel(None, dcc.Graph(figure=fig_val, config={"displayModeBar": False})),
        panel(None, dcc.Graph(figure=fig_methods, config={"displayModeBar": False})),
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
    seasonal_avg["Month"] = seasonal_avg["month_num"].apply(lambda m: month_names[m-1])

    fig_season = px.bar(seasonal_avg, x="Month", y="Transaction_Count", category_orders={"Month": month_names},
                         title="Average Transactions by Calendar Month (Seasonality)")

    dow = dow_volume.copy()
    dow["DayOfWeek"] = pd.Categorical(dow["DayOfWeek"], categories=DOW_ORDER, ordered=True)
    dow = dow.sort_values("DayOfWeek")
    fig_dow = px.bar(dow, x="DayOfWeek", y="Count", title="Transaction Volume by Day of Week")

    fig_hour = px.line(hour_volume, x="Hour", y="Count", markers=True, title="Transaction Volume by Hour of Day")

    for f in [fig_season, fig_dow, fig_hour]:
        style_fig(f)

    top_month = monthly.loc[monthly["Transaction_Count"].idxmax()]
    low_month = monthly.loc[monthly["Transaction_Count"].idxmin()]

    insight = panel(None, html.Div([
        html.Div([html.I(className="fa-solid fa-arrow-trend-up", style={"color": "#2bc9b6", "marginRight": "8px"}),
                  html.Span("Peak month: "),
                  html.B(f"{top_month['MonthPeriod']} ({int(top_month['Transaction_Count'])} txns)")], className="insight-item"),
        html.Div([html.I(className="fa-solid fa-arrow-trend-down", style={"color": "#ff4f81", "marginRight": "8px"}),
                  html.Span("Lowest month: "),
                  html.B(f"{low_month['MonthPeriod']} ({int(low_month['Transaction_Count'])} txns)")], className="insight-item"),
    ], className="insight-row"))

    return html.Div([
        insight,
        panel(None, dcc.Graph(figure=fig_season, config={"displayModeBar": False})),
        panel(None, dcc.Graph(figure=fig_dow, config={"displayModeBar": False})),
        panel(None, dcc.Graph(figure=fig_hour, config={"displayModeBar": False})),
    ])


# ---------------------------------------------------------------------------
# TAB 3 — FORECAST
# ---------------------------------------------------------------------------
def forecast_tab():
    return html.Div([
        panel("Forecast Horizon", html.Div([
            dcc.Slider(id="forecast-months", min=1, max=12, step=1, value=6,
                       marks={i: str(i) for i in [1,3,6,9,12]}),
        ], style={"padding": "10px 4px"})),
        panel(None, dcc.Graph(id="forecast-graph", config={"displayModeBar": False})),
        panel("Predicted Values", html.Div(id="forecast-table")),
    ])


# ---------------------------------------------------------------------------
# TAB 4 — CHANNEL / METHOD PREFERENCE
# ---------------------------------------------------------------------------
def preference_tab():
    fig_group = px.pie(group_monthly.groupby("Payment_Group")["Count"].sum().reset_index(),
                        names="Payment_Group", values="Count", hole=0.55,
                        title="Mobile/Digital vs Card/Bank Payments",
                        color_discrete_sequence=["#2196f3", "#ff4f81"])

    fig_method = px.bar(method_monthly.groupby("Payment_Method")["Count"].sum().reset_index().sort_values("Count"),
                         x="Count", y="Payment_Method", orientation="h",
                         title="Total Volume by Specific Payment Method")

    fig_sector = px.bar(sector_pref, x="Sector", y="Count", color="Payment_Group", barmode="group",
                         title="Payment Preference by Business Sector",
                         color_discrete_sequence=["#2196f3", "#ff4f81"])

    fig_segment = px.bar(segment_pref, x="Customer_Segment", y="Count", color="Payment_Group", barmode="group",
                          title="Payment Preference by Customer Segment",
                          category_orders={"Customer_Segment": ["Low", "Medium", "High"]},
                          color_discrete_sequence=["#2196f3", "#ff4f81"])

    for f in [fig_group, fig_method, fig_sector, fig_segment]:
        style_fig(f)

    return html.Div([
        html.Div([
            panel(None, dcc.Graph(figure=fig_group, config={"displayModeBar": False})),
            panel(None, dcc.Graph(figure=fig_method, config={"displayModeBar": False})),
        ], className="panel-row"),
        panel(None, dcc.Graph(figure=fig_sector, config={"displayModeBar": False})),
        panel(None, dcc.Graph(figure=fig_segment, config={"displayModeBar": False})),
    ])


# ---------------------------------------------------------------------------
# TAB 5 — ML PREDICTOR (live, interactive)
# ---------------------------------------------------------------------------
sectors = sorted(transactions["Sector"].unique())
segments = sorted(transactions["Customer_Segment"].unique())
devices = sorted(transactions["Device_Type"].unique())

def field(label, comp):
    return html.Div([html.Label(label, className="field-label"), comp], style={"flex": "1", "marginRight": "14px"})

def predictor_tab():
    return panel("Predict Preferred Payment Channel", html.Div([
        html.Div([
            field("Sector", dcc.Dropdown(id="in-sector", options=[{"label": s, "value": s} for s in sectors], value=sectors[0], clearable=False)),
            field("Customer Segment", dcc.Dropdown(id="in-segment", options=[{"label": s, "value": s} for s in segments], value=segments[0], clearable=False)),
            field("Device Type", dcc.Dropdown(id="in-device", options=[{"label": d, "value": d} for d in devices], value=devices[0], clearable=False)),
        ], style={"display": "flex", "marginBottom": "16px"}),

        html.Div([
            field("Transaction Amount (£)", dcc.Input(id="in-amount", type="number", value=100, style={"width": "100%", "padding": "8px", "borderRadius": "6px", "border": "1px solid #d8dee8"})),
            field("Day of Week", dcc.Dropdown(id="in-dow", options=[{"label": d, "value": i} for i, d in enumerate(DOW_ORDER)], value=0, clearable=False)),
            field("Hour of Day (0-23)", dcc.Input(id="in-hour", type="number", value=12, min=0, max=23, style={"width": "100%", "padding": "8px", "borderRadius": "6px", "border": "1px solid #d8dee8"})),
        ], style={"display": "flex", "marginBottom": "20px"}),

        html.Button("Predict", id="predict-btn", n_clicks=0, className="predict-btn"),
        html.Div(id="prediction-output", style={"marginTop": "18px"}),
    ]))


# ---------------------------------------------------------------------------
# LAYOUT
# ---------------------------------------------------------------------------
NAV_ITEMS = [
    ("trends", "Trends"),
    ("peaks", "Peak Seasons"),
    ("forecast", "Forecast"),
    ("preference", "Channel Preference"),
    ("predictor", "ML Predictor"),
]

sidebar = html.Div([
    html.Div([html.I(className="fa-solid fa-layer-group", style={"marginRight": "10px", "color": "#4cc9c0"}), "PayAnalytics"], className="sidebar-logo"),
    html.Div("Main", className="sidebar-section-label"),
    dcc.Tabs(
        id="tabs", value="trends", vertical=True,
        className="sidebar-tabs",
        children=[dcc.Tab(label=label, value=val) for val, label in NAV_ITEMS],
    ),
], className="sidebar")

topbar = html.Div([
    html.Div([
        html.Div("Digital Payments Trend Analyzer", className="topbar-title"),
        html.Div("Transaction analysis · seasonality · forecasting · channel preference", className="topbar-sub"),
    ]),
    html.Div([
        html.I(className="fa-regular fa-bell"),
        html.I(className="fa-regular fa-envelope"),
        html.I(className="fa-solid fa-gear"),
        html.Span("Thomson", style={"fontWeight": "600", "color": "#1c2b3a"}),
    ], className="topbar-right"),
], className="topbar")

app.layout = html.Div([
    sidebar,
    html.Div([
        topbar,
        html.Div([
            kpi_row,
            html.Div(id="tab-content"),
        ], className="content-pad"),
    ], className="main-area"),
], className="app-shell")


@app.callback(Output("tab-content", "children"), Input("tabs", "value"))
def render_tab(tab):
    if tab == "trends":
        return trends_tab()
    elif tab == "peaks":
        return peak_tab()
    elif tab == "forecast":
        return forecast_tab()
    elif tab == "preference":
        return preference_tab()
    elif tab == "predictor":
        return predictor_tab()
    return html.Div("Unknown tab")


@app.callback(
    Output("forecast-graph", "figure"),
    Output("forecast-table", "children"),
    Input("forecast-months", "value"),
)
def update_forecast(n_months):
    fut = forecast_future(n_months)
    hist = forecast_history[["MonthPeriod", "Transaction_Count"]].copy()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist["MonthPeriod"], y=hist["Transaction_Count"], mode="lines+markers",
                              name="Historical", line=dict(color="#2196f3")))
    fig.add_trace(go.Scatter(x=fut["MonthPeriod"], y=fut["Transaction_Count"], mode="lines+markers",
                              name="Forecast", line=dict(color="#ff4f81", dash="dash")))
    style_fig(fig, title=f"Transaction Volume Forecast — Next {n_months} Month(s)")

    rows = [html.Tr([html.Th("Month", style={"textAlign": "left", "padding": "8px", "color": "#5d6b82"}),
                      html.Th("Predicted Transactions", style={"textAlign": "left", "padding": "8px", "color": "#5d6b82"})])]
    for _, r in fut.iterrows():
        rows.append(html.Tr([
            html.Td(r["MonthPeriod"], style={"padding": "8px", "borderTop": "1px solid #eef1f6"}),
            html.Td(f"{r['Transaction_Count']:.0f}", style={"padding": "8px", "borderTop": "1px solid #eef1f6"}),
        ]))
    table = html.Table(rows, style={"width": "100%", "borderCollapse": "collapse", "fontSize": "13.5px", "color": "#1c2b3a"})

    return fig, table


@app.callback(
    Output("prediction-output", "children"),
    Input("predict-btn", "n_clicks"),
    State("in-sector", "value"), State("in-segment", "value"), State("in-device", "value"),
    State("in-amount", "value"), State("in-dow", "value"), State("in-hour", "value"),
)
def predict_channel(n_clicks, sector, segment, device, amount, dow, hour):
    if n_clicks == 0:
        return html.Div("Set the inputs above and click Predict.", style={"color": "#8a94a6"})

    quarter = (pd.Timestamp.now().month - 1) // 3 + 1

    row = pd.DataFrame([{
        "Sector": channel_encoders["Sector"].transform([sector])[0],
        "Customer_Segment": channel_encoders["Customer_Segment"].transform([segment])[0],
        "Device_Type": channel_encoders["Device_Type"].transform([device])[0],
        "Transaction_Amount_GBP": amount,
        "DayOfWeekNum": dow,
        "Hour": hour,
        "Quarter": quarter,
    }])[channel_features]
    group_pred = channel_target_encoder.inverse_transform(channel_model.predict(row))[0]
    group_proba = channel_model.predict_proba(row)[0]
    group_labels = channel_target_encoder.classes_

    row2 = pd.DataFrame([{
        "Sector": method_encoders["Sector"].transform([sector])[0],
        "Customer_Segment": method_encoders["Customer_Segment"].transform([segment])[0],
        "Device_Type": method_encoders["Device_Type"].transform([device])[0],
        "Transaction_Amount_GBP": amount,
        "DayOfWeekNum": dow,
        "Hour": hour,
    }])[method_features]
    method_pred = method_target_encoder.inverse_transform(method_model.predict(row2))[0]

    proba_text = ", ".join(f"{lbl}: {p*100:.1f}%" for lbl, p in zip(group_labels, group_proba))

    return html.Div([
        html.Div([
            html.I(className="fa-solid fa-circle-check", style={"color": "#2bc9b6", "marginRight": "10px", "fontSize": "20px"}),
            html.Span("Prediction Result", style={"fontWeight": "700", "fontSize": "15px", "color": "#1c2b3a"}),
        ], style={"marginBottom": "10px"}),
        html.P(["Most likely payment group: ", html.B(group_pred)], style={"color": "#374151"}),
        html.P(["Most likely specific method: ", html.B(method_pred)], style={"color": "#374151"}),
        html.P(f"Confidence breakdown — {proba_text}", style={"color": "#8a94a6", "fontSize": "12.5px"}),
        html.P("Note: specific-method prediction accuracy is low (~21%) because this dataset's "
               "Payment_Method field shows little real correlation with customer/transaction features. "
               "The Mobile-vs-Card/Bank group prediction is more reliable (~60% accuracy vs ~41% baseline).",
               style={"color": "#a0aec0", "fontSize": "11.5px", "marginTop": "10px"}),
    ], style={"background": "#f7faf9", "border": "1px solid #e3f3ef", "borderRadius": "8px", "padding": "16px"})


if __name__ == "__main__":
    app.run(debug=True, port=8050)
