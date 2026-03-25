"""
╔══════════════════════════════════════════════════════════════════╗
║   BIOMEDICAL SIGNAL PROCESSING & TOPOLOGICAL ANALYSIS ENGINE    ║
║   Takens' Attractor Dashboard  ·  v2.0                          ║
╚══════════════════════════════════════════════════════════════════╝

SUPPORTED FILE FORMATS:
  · EDF / BDF  — European Data Format (clinical EEG standard)
  · CSV / TXT  — Tabular signal data (auto-detect time/channel columns)
  · FIF        — MNE/Neuromag format
  · SET        — EEGLAB format
  · NPY        — NumPy array (1D or 2D, first column used)

INSTALL:
  pip install dash dash-bootstrap-components plotly numpy scipy
              pywt mne pandas dash-uploader

RUN:
  python takens_attractor_dashboard.py
  Then open → http://127.0.0.1:8050
"""

import io
import base64
import warnings
import traceback
import tempfile
import os
import pathlib

import numpy as np
import pandas as pd
import scipy.signal as sp_signal
from numpy.lib.stride_tricks import as_strided

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

import dash
from dash import dcc, html, Input, Output, State, callback_context, no_update
import dash_bootstrap_components as dbc

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS & EEG BAND DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────

EEG_BANDS = {
    "δ Delta":   (0.5,  4.0,  "#4361ee"),
    "θ Theta":   (4.0,  8.0,  "#7209b7"),
    "α Alpha":   (8.0,  13.0, "#f72585"),
    "β Beta":    (13.0, 30.0, "#fb8500"),
    "γ Gamma":   (30.0, 45.0, "#06d6a0"),
}

COLORSCALE = "Turbo"
DARK_BG    = "#0a0c10"
CARD_BG    = "#0f1318"
ACCENT     = "#00d4ff"
BORDER     = "#1e2530"

ACCEPTED_EXTENSIONS = {".edf", ".bdf", ".fif", ".set", ".csv", ".txt", ".tsv", ".npy"}

# ─────────────────────────────────────────────────────────────────────────────
# 1. MULTI-FORMAT FILE INGESTION
# ─────────────────────────────────────────────────────────────────────────────

def ingest_file(content_bytes: bytes, filename: str):
    """
    Universal file ingestion.  Returns:
        channels   : dict[name → np.ndarray]
        fs         : float  (sampling frequency)
        format_tag : str
    """
    ext = pathlib.Path(filename).suffix.lower()

    # ── EDF / BDF ──────────────────────────────────────────────────────────
    if ext in (".edf", ".bdf"):
        try:
            import mne
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                tmp.write(content_bytes)
                tmp_path = tmp.name
            raw = mne.io.read_raw_edf(tmp_path, preload=True, verbose=False)
            raw.filter(l_freq=0.5, h_freq=None, verbose=False)
            fs = raw.info["sfreq"]
            channels = {ch: raw.get_data(picks=[ch])[0] for ch in raw.ch_names[:32]}
            os.unlink(tmp_path)
            return channels, fs, ext.upper()
        except ImportError:
            raise RuntimeError("MNE-Python is required for EDF/BDF files.  pip install mne")

    # ── FIF ────────────────────────────────────────────────────────────────
    elif ext == ".fif":
        try:
            import mne
            with tempfile.NamedTemporaryFile(suffix=".fif", delete=False) as tmp:
                tmp.write(content_bytes)
                tmp_path = tmp.name
            raw = mne.io.read_raw_fif(tmp_path, preload=True, verbose=False)
            fs = raw.info["sfreq"]
            channels = {ch: raw.get_data(picks=[ch])[0] for ch in raw.ch_names[:32]}
            os.unlink(tmp_path)
            return channels, fs, "FIF"
        except ImportError:
            raise RuntimeError("MNE-Python is required for FIF files.  pip install mne")

    # ── SET (EEGLAB) ────────────────────────────────────────────────────────
    elif ext == ".set":
        try:
            import mne
            with tempfile.NamedTemporaryFile(suffix=".set", delete=False) as tmp:
                tmp.write(content_bytes)
                tmp_path = tmp.name
            raw = mne.io.read_raw_eeglab(tmp_path, preload=True, verbose=False)
            fs = raw.info["sfreq"]
            channels = {ch: raw.get_data(picks=[ch])[0] for ch in raw.ch_names[:32]}
            os.unlink(tmp_path)
            return channels, fs, "EEGLAB/.SET"
        except ImportError:
            raise RuntimeError("MNE-Python is required for SET files.  pip install mne")

    # ── CSV / TSV / TXT ────────────────────────────────────────────────────
    elif ext in (".csv", ".txt", ".tsv"):
        sep = "\t" if ext == ".tsv" else None
        df = pd.read_csv(io.BytesIO(content_bytes), sep=sep, engine="python",
                         on_bad_lines="skip")
        df = df.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")

        # Heuristic: drop obvious time columns (monotone-increasing)
        time_col = None
        for col in df.columns:
            diffs = np.diff(df[col].dropna().values)
            if np.all(diffs > 0) and diffs.std() < diffs.mean() * 0.05:
                time_col = col
                break

        sig_cols = [c for c in df.columns if c != time_col]
        if not sig_cols:
            raise ValueError("No numeric signal columns detected in CSV.")

        # Estimate fs from time column or assume 256 Hz
        if time_col is not None:
            dt = np.median(np.diff(df[time_col].dropna().values))
            fs = float(1.0 / dt) if dt > 0 else 256.0
        else:
            fs = 256.0

        channels = {str(c): df[c].dropna().values.astype(np.float64)
                    for c in sig_cols[:32]}
        return channels, fs, "CSV"

    # ── NPY ────────────────────────────────────────────────────────────────
    elif ext == ".npy":
        arr = np.load(io.BytesIO(content_bytes), allow_pickle=False)
        if arr.ndim == 1:
            channels = {"CH_0": arr.astype(np.float64)}
        elif arr.ndim == 2:
            # rows = samples, cols = channels  (or transpose if needed)
            if arr.shape[0] < arr.shape[1]:
                arr = arr.T
            channels = {f"CH_{i}": arr[:, i].astype(np.float64)
                        for i in range(min(32, arr.shape[1]))}
        else:
            raise ValueError("NPY array must be 1D or 2D.")
        return channels, 256.0, "NumPy/.NPY"

    else:
        raise ValueError(
            f"Unsupported format: '{ext}'.  "
            f"Accepted: {', '.join(sorted(ACCEPTED_EXTENSIONS))}"
        )

# ─────────────────────────────────────────────────────────────────────────────
# 2. SIGNAL ANALYSIS UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def bandpass(x: np.ndarray, fs: float, lo: float, hi: float) -> np.ndarray:
    nyq = fs / 2.0
    lo_n, hi_n = max(lo / nyq, 1e-4), min(hi / nyq, 0.999)
    if lo_n >= hi_n:
        return np.zeros_like(x)
    b, a = sp_signal.butter(4, [lo_n, hi_n], btype="band")
    return sp_signal.filtfilt(b, a, x)


def compute_band_powers(x: np.ndarray, fs: float) -> dict:
    """Welch PSD → relative band powers (%)."""
    nperseg = min(len(x), int(fs * 4))
    freqs, psd = sp_signal.welch(x, fs=fs, nperseg=nperseg)
    total = np.trapz(psd, freqs) + 1e-12
    powers = {}
    for name, (lo, hi, _) in EEG_BANDS.items():
        mask = (freqs >= lo) & (freqs <= hi)
        powers[name] = float(np.trapz(psd[mask], freqs[mask]) / total * 100)
    return powers, freqs, psd


def compute_optimal_tau(x: np.ndarray, fs: float, max_lag_sec: float = 0.5) -> int:
    x_c = x - np.mean(x)
    max_lag = int(max_lag_sec * fs)
    acf = sp_signal.correlate(x_c, x_c, mode="full", method="fft")
    acf = acf[len(acf) // 2:][:max_lag]
    acf /= acf[0] + 1e-12
    zc = np.where(np.diff(np.sign(acf)))[0]
    return int(zc[0]) if len(zc) > 0 else max(1, int(fs * 0.05))


def phase_space(x: np.ndarray, tau: int, dim: int = 3) -> np.ndarray:
    N = len(x)
    M = N - (dim - 1) * tau
    if M <= 0:
        raise ValueError("Signal too short for chosen tau/dimension.")
    sz = x.itemsize
    return as_strided(x, shape=(M, dim), strides=(sz, sz * tau))


def signal_quality(x: np.ndarray, fs: float) -> dict:
    """Compute basic signal quality metrics."""
    rms   = float(np.sqrt(np.mean(x ** 2)))
    snr   = float(10 * np.log10(np.var(x) / (np.var(np.diff(x)) + 1e-12)))
    kurt  = float(pd.Series(x).kurtosis())
    hurst = _hurst_exponent(x)
    # Artifact score: z-score spikes
    z     = np.abs((x - np.mean(x)) / (np.std(x) + 1e-12))
    artifact_pct = float(np.mean(z > 4) * 100)
    return {
        "RMS Amplitude (μV)": f"{rms:.3f}",
        "Signal SNR (dB)":    f"{snr:.1f}",
        "Kurtosis":           f"{kurt:.2f}",
        "Hurst Exponent":     f"{hurst:.3f}",
        "Artifact %":         f"{artifact_pct:.2f}%",
    }


def _hurst_exponent(x: np.ndarray, max_lag: int = 100) -> float:
    """Simplified R/S Hurst exponent."""
    lags = range(2, min(max_lag, len(x) // 4))
    rs_list = []
    for lag in lags:
        s = x[:lag]
        r = np.ptp(np.cumsum(s - np.mean(s)))
        sd = np.std(s) + 1e-12
        rs_list.append(r / sd)
    if len(rs_list) < 2:
        return 0.5
    log_lags = np.log(list(lags))
    log_rs   = np.log(rs_list)
    poly     = np.polyfit(log_lags, log_rs, 1)
    return float(np.clip(poly[0], 0, 1))


def spectral_energy_gradient(x: np.ndarray, fs: float, n_points: int) -> np.ndarray:
    """Fast short-time spectral energy via STFT (replaces heavy CWT for speed)."""
    try:
        import pywt
        freqs_cwt = np.linspace(1, min(45, fs / 2 - 1), 30)
        scales    = pywt.frequency2scale("morl", freqs_cwt / fs)
        cwt_mat, _= pywt.cwt(x, scales, "morl", sampling_period=1 / fs)
        energy    = np.sum(np.abs(cwt_mat) ** 2, axis=0)
    except Exception:
        # Fallback: STFT-based energy
        nperseg = min(256, len(x) // 4)
        _, _, Zxx = sp_signal.stft(x, fs=fs, nperseg=nperseg,
                                   noverlap=nperseg - 1, boundary=None)
        energy = np.sum(np.abs(Zxx) ** 2, axis=0)
        if len(energy) > len(x):
            energy = energy[:len(x)]
        elif len(energy) < len(x):
            energy = np.interp(np.linspace(0, 1, len(x)),
                               np.linspace(0, 1, len(energy)), energy)
    mn, mx = energy.min(), energy.max()
    energy = (energy - mn) / (mx - mn + 1e-12)
    return energy[:n_points]

# ─────────────────────────────────────────────────────────────────────────────
# 3. FIGURE BUILDERS
# ─────────────────────────────────────────────────────────────────────────────

TRANSPARENT = "rgba(0,0,0,0)"

def _layout_base(title=""):
    return dict(
        template="plotly_dark",
        paper_bgcolor=TRANSPARENT,
        plot_bgcolor=TRANSPARENT,
        title_text=title,
        title_font=dict(size=13, color="#a0aec0", family="'JetBrains Mono', monospace"),
        margin=dict(l=40, r=20, t=40, b=40),
        font=dict(family="'JetBrains Mono', monospace", color="#a0aec0"),
    )


def fig_timeseries(t, x, energy, channel, tau):
    M = len(energy)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t[:M], y=x[:M],
        mode="lines",
        line=dict(color="rgba(0,212,255,0.25)", width=1),
        showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=t[:M], y=x[:M],
        mode="markers",
        marker=dict(size=2.5, color=energy, colorscale=COLORSCALE,
                    showscale=True,
                    colorbar=dict(title="Energy", thickness=10, x=1.02,
                                  tickfont=dict(size=9))),
        name="EEG",
        hovertemplate="t=%{x:.3f}s<br>μV=%{y:.2f}<extra></extra>",
    ))
    fig.update_layout(**_layout_base(f"Time Domain  ·  {channel}  ·  τ={tau} smp"),
                      height=280,
                      xaxis=dict(title="Time (s)", gridcolor=BORDER, zeroline=False),
                      yaxis=dict(title="μV", gridcolor=BORDER, zeroline=False))
    return fig


def fig_phase_space(E, energy, tau):
    fig = go.Figure(go.Scatter3d(
        x=E[:, 0], y=E[:, 1], z=E[:, 2],
        mode="lines",
        line=dict(width=2.5, color=energy, colorscale=COLORSCALE,
                  cmin=0, cmax=1,
                  colorbar=dict(title="Energy", thickness=10,
                                tickfont=dict(size=9))),
        hovertemplate="x(t)=%{x:.2f}<br>x(t+τ)=%{y:.2f}<br>x(t+2τ)=%{z:.2f}<extra></extra>",
    ))
    fig.update_layout(
        **_layout_base(f"Takens' Attractor  ·  dim=3  ·  τ={tau}"),
        height=480,
        scene=dict(
            xaxis=dict(title="x(t)",     backgroundcolor=DARK_BG,
                       gridcolor=BORDER, zerolinecolor=BORDER),
            yaxis=dict(title="x(t+τ)",   backgroundcolor=DARK_BG,
                       gridcolor=BORDER, zerolinecolor=BORDER),
            zaxis=dict(title="x(t+2τ)",  backgroundcolor=DARK_BG,
                       gridcolor=BORDER, zerolinecolor=BORDER),
            bgcolor=DARK_BG,
        ),
        scene_camera=dict(eye=dict(x=1.6, y=1.6, z=0.9)),
    )
    return fig


def fig_psd(freqs, psd):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=freqs, y=10 * np.log10(psd + 1e-30),
                             mode="lines", line=dict(color=ACCENT, width=1.5),
                             fill="tozeroy",
                             fillcolor="rgba(0,212,255,0.06)",
                             name="PSD",
                             hovertemplate="%{x:.1f} Hz : %{y:.1f} dB<extra></extra>"))
    # Band overlays
    for name, (lo, hi, col) in EEG_BANDS.items():
        fig.add_vrect(x0=lo, x1=hi,
                      fillcolor=col, opacity=0.08,
                      line_width=0, annotation_text=name.split()[0],
                      annotation_position="top",
                      annotation_font=dict(size=8, color=col))
    fig.update_layout(**_layout_base("Power Spectral Density"),
                      height=240,
                      xaxis=dict(title="Frequency (Hz)", gridcolor=BORDER,
                                 range=[0, 50]),
                      yaxis=dict(title="dB/Hz", gridcolor=BORDER))
    return fig


def fig_band_radar(powers: dict):
    names  = list(powers.keys())
    values = list(powers.values())
    colors = [EEG_BANDS[n][2] for n in names]
    fig = go.Figure(go.Barpolar(
        r=values, theta=names,
        marker=dict(color=colors, line_color=DARK_BG, line_width=1),
        opacity=0.85,
        hovertemplate="%{theta}: %{r:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        **_layout_base("Band Power Distribution"),
        height=260,
        polar=dict(
            bgcolor=TRANSPARENT,
            radialaxis=dict(visible=True, gridcolor=BORDER,
                            tickfont=dict(size=8),
                            ticksuffix="%"),
            angularaxis=dict(gridcolor=BORDER),
        ),
        showlegend=False,
    )
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# 4. DASH APPLICATION
# ─────────────────────────────────────────────────────────────────────────────

GOOGLE_FONTS = (
    "https://fonts.googleapis.com/css2?"
    "family=JetBrains+Mono:wght@300;400;600&"
    "family=Space+Grotesk:wght@300;400;700&display=swap"
)

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.CYBORG, GOOGLE_FONTS],
    suppress_callback_exceptions=True,
)
app.title = "Takens Attractor · EEG Dashboard"

# ── CSS overrides ─────────────────────────────────────────────────────────────
EXTRA_CSS = f"""
:root {{
  --bg:      {DARK_BG};
  --card:    {CARD_BG};
  --accent:  {ACCENT};
  --border:  {BORDER};
  --text:    #c8d6e5;
  --muted:   #566573;
  --mono:    'JetBrains Mono', monospace;
  --sans:    'Space Grotesk', sans-serif;
}}
body, .container-fluid {{ background: var(--bg) !important; color: var(--text); }}
.card {{ background: var(--card) !important; border: 1px solid var(--border) !important;
         border-radius: 10px !important; }}
.card-header {{ background: rgba(255,255,255,0.03) !important;
                border-bottom: 1px solid var(--border) !important;
                font-family: var(--mono); font-size: 11px; letter-spacing: 0.1em;
                color: var(--accent) !important; text-transform: uppercase; }}
.metric-card {{ background: var(--card); border: 1px solid var(--border);
                border-radius: 8px; padding: 14px 18px; }}
.metric-label {{ font-family: var(--mono); font-size: 10px; color: var(--muted);
                 letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 4px; }}
.metric-value {{ font-family: var(--mono); font-size: 18px; color: var(--accent);
                 font-weight: 600; }}
.insight-row {{ display: flex; justify-content: space-between; padding: 6px 0;
                border-bottom: 1px solid var(--border); font-family: var(--mono);
                font-size: 11px; }}
.insight-label {{ color: var(--muted); }}
.insight-value {{ color: var(--text); font-weight: 600; }}
.drop-zone {{
  border: 2px dashed var(--border);
  border-radius: 12px;
  background: rgba(0,212,255,0.02);
  padding: 40px;
  text-align: center;
  cursor: pointer;
  transition: all 0.2s ease;
  font-family: var(--mono);
}}
.drop-zone:hover, .drop-zone.dragover {{
  border-color: var(--accent);
  background: rgba(0,212,255,0.06);
}}
.drop-icon {{ font-size: 40px; margin-bottom: 12px; opacity: 0.5; }}
.drop-title {{ color: var(--text); font-size: 14px; margin-bottom: 6px; }}
.drop-sub {{ color: var(--muted); font-size: 11px; }}
.badge-fmt {{ background: var(--accent); color: var(--bg); padding: 2px 8px;
              border-radius: 4px; font-family: var(--mono); font-size: 10px;
              font-weight: 600; }}
Select, select {{ background: var(--card) !important; color: var(--text) !important;
                  border-color: var(--border) !important; font-family: var(--mono) !important;
                  font-size: 12px !important; }}
.Select-control {{ background: var(--card) !important; border-color: var(--border) !important; }}
.Select-menu-outer {{ background: var(--card) !important; border-color: var(--border) !important; }}
.Select-option {{ background: var(--card) !important; color: var(--text) !important; }}
.Select-value-label {{ color: var(--text) !important; font-family: var(--mono) !important;
                        font-size: 12px !important; }}
.btn-run {{ background: linear-gradient(135deg, var(--accent), #0080ff);
            border: none; color: var(--bg); font-family: var(--mono);
            font-size: 12px; font-weight: 700; letter-spacing: 0.05em;
            padding: 10px 28px; border-radius: 6px; cursor: pointer;
            transition: opacity 0.2s; width: 100%; margin-top: 12px; }}
.btn-run:hover {{ opacity: 0.85; }}
.rc-slider-track {{ background-color: var(--accent) !important; }}
.rc-slider-handle {{ border-color: var(--accent) !important; background: var(--accent) !important; }}
#status-bar {{ font-family: var(--mono); font-size: 11px; padding: 8px 14px;
               border-radius: 6px; margin-top: 10px; }}
"""
import pathlib
pathlib.Path("assets").mkdir(exist_ok=True)
with open("assets/custom.css", "w") as _f:
    _f.write(EXTRA_CSS)
# ── Layout ─────────────────────────────────────────────────────────────────────

def make_metric_card(label, value_id, default="—"):
    return html.Div([
        html.Div(label, className="metric-label"),
        html.Div(default, id=value_id, className="metric-value"),
    ], className="metric-card")


sidebar = html.Div([
    # Header
    html.Div([
        html.Div("⬡", style={"fontSize": "24px", "color": ACCENT, "marginBottom": "4px"}),
        html.Div("TAKENS ENGINE", style={
            "fontFamily": "'JetBrains Mono', monospace",
            "fontSize": "13px", "fontWeight": "600",
            "color": ACCENT, "letterSpacing": "0.12em",
        }),
        html.Div("v2.0 · Topological EEG Analysis", style={
            "fontFamily": "'JetBrains Mono', monospace",
            "fontSize": "10px", "color": "#566573", "marginTop": "2px",
        }),
    ], style={"textAlign": "center", "padding": "20px 10px 16px"}),

    html.Hr(style={"borderColor": BORDER, "margin": "0 0 16px"}),

    # Upload
    dbc.Card([
        dbc.CardHeader("01 · Upload Signal File"),
        dbc.CardBody([
            dcc.Upload(
                id="upload-data",
                children=html.Div([
                    html.Div("⬆", className="drop-icon"),
                    html.Div("Drop file here or click to browse", className="drop-title"),
                    html.Div("EDF · BDF · CSV · FIF · SET · NPY · TXT", className="drop-sub"),
                ]),
                className="drop-zone",
                multiple=False,
                style={"outline": "none"},
            ),
            html.Div(id="file-info", style={"marginTop": "10px"}),
        ])
    ], style={"marginBottom": "12px"}),

    # Channel selector
    dbc.Card([
        dbc.CardHeader("02 · Select Channel"),
        dbc.CardBody([
            dcc.Dropdown(id="channel-select",
                         placeholder="— load a file first —",
                         clearable=False,
                         style={"fontFamily": "'JetBrains Mono',monospace",
                                "fontSize": "12px"}),
        ])
    ], style={"marginBottom": "12px"}),

    # Epoch
    dbc.Card([
        dbc.CardHeader("03 · Epoch Window (seconds)"),
        dbc.CardBody([
            dcc.RangeSlider(id="epoch-slider", min=0, max=60, step=0.5,
                            value=[0, 15], allowCross=False,
                            tooltip={"placement": "bottom", "always_visible": True},
                            marks={i: {"label": str(i),
                                       "style": {"fontSize": "10px", "color": "#566573"}}
                                   for i in range(0, 61, 10)}),
        ])
    ], style={"marginBottom": "12px"}),

    # Parameters
    dbc.Card([
        dbc.CardHeader("04 · Parameters"),
        dbc.CardBody([
            html.Div("Embedding Dimension", style={
                "fontFamily": "'JetBrains Mono',monospace",
                "fontSize": "10px", "color": "#566573",
                "textTransform": "uppercase", "marginBottom": "4px",
            }),
            dcc.Slider(id="dim-slider", min=3, max=5, step=1, value=3,
                       marks={3: "3", 4: "4", 5: "5 (slow)"},
                       tooltip={"always_visible": False}),
            html.Div("Max Tau Search (ms)", style={
                "fontFamily": "'JetBrains Mono',monospace",
                "fontSize": "10px", "color": "#566573",
                "textTransform": "uppercase",
                "marginTop": "14px", "marginBottom": "4px",
            }),
            dcc.Slider(id="tau-slider", min=10, max=500, step=10, value=150,
                       tooltip={"placement": "bottom", "always_visible": True}),
        ])
    ], style={"marginBottom": "12px"}),

    # Run button
    html.Button("▶  RUN ANALYSIS", id="run-btn", n_clicks=0,
                className="btn-run"),

    html.Div(id="status-bar", style={"display": "none"}),

], style={
    "width": "280px", "minWidth": "280px",
    "background": CARD_BG,
    "borderRight": f"1px solid {BORDER}",
    "height": "100vh",
    "overflowY": "auto",
    "padding": "0",
    "display": "flex",
    "flexDirection": "column",
})

main_area = html.Div([
    # Top metrics row
    dbc.Row([
        dbc.Col(make_metric_card("Sampling Rate", "m-fs"),    md=2),
        dbc.Col(make_metric_card("Duration",       "m-dur"),  md=2),
        dbc.Col(make_metric_card("Optimal τ",      "m-tau"),  md=2),
        dbc.Col(make_metric_card("Samples",        "m-smp"),  md=2),
        dbc.Col(make_metric_card("RMS (μV)",        "m-rms"),  md=2),
        dbc.Col(make_metric_card("SNR (dB)",        "m-snr"),  md=2),
    ], className="g-2", style={"marginBottom": "16px"}),

    # Main plots
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Time Domain Signal"),
                dbc.CardBody([dcc.Graph(id="fig-ts",
                                       config={"displayModeBar": True,
                                               "modeBarButtonsToRemove": ["select2d", "lasso2d"]},
                                       style={"height": "280px"})],
                             style={"padding": "0"}),
            ]),
        ], md=12, style={"marginBottom": "12px"}),
    ]),

    dbc.Row([
        # Attractor
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("3D State-Space Attractor"),
                dbc.CardBody([dcc.Graph(id="fig-3d",
                                       config={"displayModeBar": True},
                                       style={"height": "480px"})],
                             style={"padding": "0"}),
            ]),
        ], md=7),

        # Right panel
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Power Spectral Density"),
                dbc.CardBody([dcc.Graph(id="fig-psd",
                                       config={"displayModeBar": False},
                                       style={"height": "240px"})],
                             style={"padding": "0"}),
            ], style={"marginBottom": "12px"}),

            dbc.Card([
                dbc.CardHeader("EEG Band Distribution"),
                dbc.CardBody([dcc.Graph(id="fig-bands",
                                       config={"displayModeBar": False},
                                       style={"height": "260px"})],
                             style={"padding": "0"}),
            ]),
        ], md=5),
    ], className="g-3", style={"marginBottom": "12px"}),

    # Insights row
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Signal Quality Metrics"),
                dbc.CardBody(id="quality-body", style={"padding": "10px 16px"}),
            ]),
        ], md=4),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Spectral Insights"),
                dbc.CardBody(id="spectral-body", style={"padding": "10px 16px"}),
            ]),
        ], md=4),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Dynamics Interpretation"),
                dbc.CardBody(id="dynamics-body", style={"padding": "10px 16px"}),
            ]),
        ], md=4),
    ], className="g-3"),

], style={"flex": "1", "overflowY": "auto", "padding": "20px"})

app.layout = html.Div([
    # Hidden stores
    dcc.Store(id="store-channels"),   # {name: list_of_floats}
    dcc.Store(id="store-fs"),         # float
    dcc.Store(id="store-fmt"),        # str
    # Layout
    html.Div([sidebar, main_area],
             style={"display": "flex", "height": "100vh", "overflow": "hidden"}),
], style={"background": DARK_BG})

# ─────────────────────────────────────────────────────────────────────────────
# 5. CALLBACKS
# ─────────────────────────────────────────────────────────────────────────────

@app.callback(
    Output("store-channels", "data"),
    Output("store-fs",       "data"),
    Output("store-fmt",      "data"),
    Output("channel-select", "options"),
    Output("channel-select", "value"),
    Output("epoch-slider",   "max"),
    Output("epoch-slider",   "value"),
    Output("file-info",      "children"),
    Input("upload-data", "contents"),
    State("upload-data", "filename"),
    prevent_initial_call=True,
)
def on_upload(contents, filename):
    if contents is None:
        return no_update

    _, b64 = contents.split(",", 1)
    raw_bytes = base64.b64decode(b64)

    try:
        channels, fs, fmt = ingest_file(raw_bytes, filename)
    except Exception as e:
        err = html.Div([
            html.Span("⚠ ", style={"color": "#e74c3c"}),
            html.Span(str(e), style={"fontFamily": "'JetBrains Mono',monospace",
                                     "fontSize": "11px", "color": "#e74c3c"}),
        ])
        return (no_update,) * 7 + (err,)

    # Determine max epoch from shortest channel
    max_dur = float(min(len(v) for v in channels.values()) / fs)
    default_end = min(15.0, max_dur)

    # Serialise (JSON-friendly)
    ch_data = {k: v.tolist() for k, v in channels.items()}
    ch_options = [{"label": k, "value": k} for k in channels.keys()]
    first_ch   = ch_options[0]["value"]

    # File info badge
    info = html.Div([
        html.Span(fmt, className="badge-fmt"),
        html.Span(f"  {filename}", style={
            "fontFamily": "'JetBrains Mono',monospace",
            "fontSize": "10px", "color": "#a0aec0", "marginLeft": "8px",
        }),
        html.Br(),
        html.Span(f"{len(channels)} ch · {fs:.0f} Hz · {max_dur:.1f}s", style={
            "fontFamily": "'JetBrains Mono',monospace",
            "fontSize": "10px", "color": "#566573",
        }),
    ], style={"marginTop": "8px"})

    return ch_data, fs, fmt, ch_options, first_ch, max_dur, [0, default_end], info


@app.callback(
    # Figures
    Output("fig-ts",    "figure"),
    Output("fig-3d",    "figure"),
    Output("fig-psd",   "figure"),
    Output("fig-bands", "figure"),
    # Top metrics
    Output("m-fs",  "children"),
    Output("m-dur", "children"),
    Output("m-tau", "children"),
    Output("m-smp", "children"),
    Output("m-rms", "children"),
    Output("m-snr", "children"),
    # Insight panels
    Output("quality-body",  "children"),
    Output("spectral-body", "children"),
    Output("dynamics-body", "children"),
    # Status
    Output("status-bar", "children"),
    Output("status-bar", "style"),
    # ── Inputs ──
    Input("run-btn", "n_clicks"),
    State("store-channels",  "data"),
    State("store-fs",        "data"),
    State("channel-select",  "value"),
    State("epoch-slider",    "value"),
    State("dim-slider",      "value"),
    State("tau-slider",      "value"),
    prevent_initial_call=True,
)
def run_analysis(n_clicks, ch_data, fs, channel, epoch, dim, max_tau_ms):
    empty_fig = go.Figure().update_layout(**_layout_base(), height=300,
                                          paper_bgcolor=TRANSPARENT,
                                          plot_bgcolor=TRANSPARENT)

    def fail(msg):
        status_style = {
            "display": "block", "background": "rgba(231,76,60,0.1)",
            "border": "1px solid #e74c3c", "color": "#e74c3c",
        }
        return ([empty_fig] * 4 + ["—"] * 6 + [html.Div()] * 3
                + [f"⚠ {msg}", status_style])

    if not ch_data or not channel or channel not in ch_data:
        return fail("No data loaded.  Upload a file and select a channel.")

    try:
        x_full = np.array(ch_data[channel], dtype=np.float64)
        fs     = float(fs)

        t0, t1   = float(epoch[0]), float(epoch[1])
        i0, i1   = int(t0 * fs), min(int(t1 * fs), len(x_full))
        x        = x_full[i0:i1]
        t        = np.linspace(t0, t1, len(x))

        if len(x) < 50:
            return fail("Epoch too short (< 50 samples).  Expand the time window.")

        # ── Tau ─────────────────────────────────────────────────────────────
        max_tau_sec = max_tau_ms / 1000.0
        tau = compute_optimal_tau(x, fs, max_lag_sec=max_tau_sec)
        tau = max(1, tau)

        # ── Phase Space ──────────────────────────────────────────────────────
        E = phase_space(x, tau, dim=min(int(dim), 3))  # render always 3D

        # ── Spectral Energy for color ────────────────────────────────────────
        M      = E.shape[0]
        energy = spectral_energy_gradient(x, fs, M)

        # ── PSD + Band Powers ───────────────────────────────────────────────
        powers, freqs, psd = compute_band_powers(x, fs)

        # ── Signal Quality ──────────────────────────────────────────────────
        quality = signal_quality(x, fs)

        # ── Figures ─────────────────────────────────────────────────────────
        f_ts    = fig_timeseries(t, x, energy, channel, tau)
        f_3d    = fig_phase_space(E, energy, tau)
        f_psd   = fig_psd(freqs, psd)
        f_bands = fig_band_radar(powers)

        # ── Top metrics ─────────────────────────────────────────────────────
        rms = float(np.sqrt(np.mean(x ** 2)))
        snr = float(10 * np.log10(np.var(x) / (np.var(np.diff(x)) + 1e-12)))

        # ── Insight panels ───────────────────────────────────────────────────
        def insight_row(label, value):
            return html.Div([
                html.Span(label, className="insight-label"),
                html.Span(value, className="insight-value"),
            ], className="insight-row")

        quality_panel = html.Div([
            insight_row(k, v) for k, v in quality.items()
        ])

        dominant_band = max(powers, key=powers.get)
        spectral_panel = html.Div([
            insight_row(k, f"{v:.1f}%") for k, v in powers.items()
        ] + [
            html.Div(style={"height": "8px"}),
            html.Div([
                html.Span("Dominant Band: ", className="insight-label"),
                html.Span(dominant_band, style={"color": ACCENT,
                                                 "fontFamily": "'JetBrains Mono',monospace",
                                                 "fontSize": "12px",
                                                 "fontWeight": "600"}),
            ], className="insight-row"),
        ])

        hurst_val = float(quality["Hurst Exponent"])
        kurtosis  = float(quality["Kurtosis"])
        artifact  = quality["Artifact %"]

        def interpret_hurst(h):
            if h > 0.6:  return ("Persistent (long-range dependence)", "#06d6a0")
            if h < 0.4:  return ("Anti-persistent (mean-reverting)", "#f72585")
            return ("Random walk (near Brownian)", "#fb8500")

        h_text, h_color = interpret_hurst(hurst_val)

        alpha_pct = powers.get("α Alpha", 0)
        delta_pct = powers.get("δ Delta", 0)
        theta_pct = powers.get("θ Theta", 0)

        state_hints = []
        if alpha_pct > 30:
            state_hints.append("↑ Alpha  →  relaxed / eyes-closed")
        if delta_pct > 40:
            state_hints.append("↑ Delta  →  deep sleep / pathology")
        if theta_pct > 25:
            state_hints.append("↑ Theta  →  drowsiness / memory encoding")
        if kurtosis > 10:
            state_hints.append("High kurtosis  →  possible spike artifacts")
        if not state_hints:
            state_hints = ["Normal waking spectrum pattern"]

        dynamics_panel = html.Div([
            insight_row("Hurst Exponent", f"{hurst_val:.3f}"),
            html.Div([
                html.Span(h_text, style={"fontFamily": "'JetBrains Mono',monospace",
                                          "fontSize": "10px", "color": h_color}),
            ], style={"padding": "4px 0 8px"}),
            insight_row("Kurtosis", quality["Kurtosis"]),
            insight_row("Artifact Rate", artifact),
            html.Div(style={"height": "6px"}),
            *[html.Div(h, style={
                "fontFamily": "'JetBrains Mono',monospace",
                "fontSize": "10px", "color": "#a0aec0",
                "padding": "3px 0",
                "borderBottom": f"1px solid {BORDER}",
              }) for h in state_hints],
        ])

        status_style = {
            "display": "block",
            "background": "rgba(0,212,255,0.06)",
            "border": f"1px solid {ACCENT}",
            "color": ACCENT,
        }
        status_msg = (
            f"✓  Analysis complete  ·  {channel}  ·  "
            f"{len(x)} samples  ·  τ={tau} smp ({tau/fs*1000:.1f}ms)  ·  "
            f"Manifold shape: {E.shape}"
        )

        return (
            f_ts, f_3d, f_psd, f_bands,
            f"{fs:.0f} Hz",
            f"{(t1-t0):.1f}s",
            f"{tau} smp",
            f"{len(x):,}",
            f"{rms:.2f}",
            f"{snr:.1f}",
            quality_panel,
            spectral_panel,
            dynamics_panel,
            status_msg,
            status_style,
        )

    except Exception as e:
        tb = traceback.format_exc()
        print(tb)
        return fail(str(e))

# ─────────────────────────────────────────────────────────────────────────────
# 6. ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print(" TAKENS ATTRACTOR DASHBOARD  v2.0")
    print("=" * 60)
    print(" Open  →  http://127.0.0.1:8050")
    print(" Formats: EDF, BDF, CSV, TXT, FIF, SET, NPY")
    print("=" * 60)
    app.run(debug=False, port=8050)
