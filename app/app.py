"""
Product Demand Forecasting - Desktop Application
PySide6 + TensorFlow + LightGBM + pyqtgraph
"""
import sys, os, json, pickle, re, warnings
from datetime import timedelta

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from PySide6.QtCore import (Qt, QTimer, QPropertyAnimation, QEasingCurve,
                             QPoint, QThread, Signal, QObject)
from PySide6.QtGui import QColor, QFont, QPainter, QLinearGradient, QBrush, QPen
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel,
                                QPushButton, QVBoxLayout, QHBoxLayout, QGridLayout,
                                QComboBox, QRadioButton, QSlider, QFrame,
                                QGraphicsDropShadowEffect, QStackedWidget,
                                QSizePolicy, QButtonGroup, QSpacerItem,
                                QScrollArea)
import pyqtgraph as pg

# ============================================================
# PATHS
# ============================================================
APP_DIR    = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.dirname(APP_DIR)
MODELS_DIR = os.path.join(ROOT_DIR, "models")
DATA_DIR   = os.path.join(ROOT_DIR, "data")

MODEL_LSTM     = os.path.join(MODELS_DIR, "Product_Demand_LSTM_v3.keras")
MODEL_SCALER   = os.path.join(MODELS_DIR, "Product_Demand_LSTM_v3_scaler.pkl")
MODEL_VOCAB    = os.path.join(MODELS_DIR, "Product_Demand_LSTM_v3_vocabularies.pkl")
MODEL_META     = os.path.join(MODELS_DIR, "Product_Demand_LSTM_v3_metadata.json")
MODEL_LGBM     = os.path.join(MODELS_DIR, "Product_Demand_LGBM_v4.pkl")
MODEL_ENSEMBLE = os.path.join(MODELS_DIR, "Product_Demand_Ensemble_v4.pkl")
MODEL_LABELS   = os.path.join(MODELS_DIR, "Product_Demand_labels.pkl")
DATA_PATH      = os.path.join(DATA_DIR,   "master_data.parquet")

# ============================================================
# COLORS
# ============================================================
BG        = "#08040f"
PANEL     = "#1c0e3a"
PANEL_2   = "#150929"
BORDER    = "#2f1d5c"
BORDER_LT = "#4c1d95"
PURPLE    = "#7c3aed"
PURPLE_2  = "#8b5cf6"
PURPLE_3  = "#a78bfa"
PURPLE_4  = "#c4b5fd"
ACCENT    = "#d946ef"
ACCENT_2  = "#e879f9"
TEXT      = "#f5f3ff"
MUTED     = "#a89ec7"
DIM       = "#5b4f7d"

QSS = f"""
* {{
    font-family: 'Segoe UI', 'Inter', -apple-system, sans-serif;
    color: {TEXT};
    outline: none;
}}
QMainWindow, QWidget#Root {{
    background: {BG};
}}
QFrame#Hero {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
        stop:0 #4c1d95, stop:0.5 #6d28d9, stop:1 #a855f7);
    border-radius: 20px;
}}
QLabel#HeroTitle {{ color: white; font-size: 28px; font-weight: 900; }}
QLabel#HeroSub   {{ color: rgba(255,255,255,0.85); font-size: 13px; }}
QLabel#HeroK {{ color: rgba(255,255,255,0.55); font-size: 10px; font-weight: 700; letter-spacing: 1.5px; }}
QLabel#HeroV {{ color: white; font-size: 22px; font-weight: 800; font-family: 'Consolas', monospace; }}

QFrame#Card {{
    background: {PANEL};
    border: 1px solid {BORDER};
    border-radius: 16px;
}}
QLabel#SectionLabel {{
    color: {PURPLE_4};
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1.2px;
}}
QLabel#Title {{ color: {TEXT}; font-size: 22px; font-weight: 800; }}
QLabel#Hint  {{ color: {MUTED}; font-size: 12px; }}

QComboBox {{
    background: {PANEL_2};
    color: {TEXT};
    border: 1.5px solid {BORDER};
    border-radius: 10px;
    padding: 10px 14px;
    font-size: 13.5px;
    min-height: 20px;
}}
QComboBox:hover {{ border-color: {BORDER_LT}; }}
QComboBox:focus {{ border-color: {PURPLE_2}; }}
QComboBox::drop-down {{ border: 0; width: 26px; }}
QComboBox QAbstractItemView {{
    background: {PANEL};
    border: 1px solid {BORDER_LT};
    border-radius: 10px;
    selection-background-color: {PURPLE};
    padding: 4px;
    color: {TEXT};
    outline: 0;
}}

QRadioButton {{
    color: {MUTED};
    font-size: 12.5px;
    font-weight: 600;
    padding: 8px 14px;
    border: 1.5px solid {BORDER};
    border-radius: 10px;
    background: {PANEL_2};
    spacing: 8px;
}}
QRadioButton:hover {{ border-color: {BORDER_LT}; color: {PURPLE_4}; }}
QRadioButton:checked {{
    background: {PURPLE};
    border-color: {PURPLE_2};
    color: white;
}}
QRadioButton::indicator {{ width: 0px; height: 0px; }}

QSlider::groove:horizontal {{
    background: {PANEL_2};
    height: 6px;
    border-radius: 3px;
}}
QSlider::sub-page:horizontal {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 {PURPLE}, stop:1 {ACCENT});
    border-radius: 3px;
}}
QSlider::handle:horizontal {{
    background: white;
    width: 18px;
    height: 18px;
    margin: -7px 0;
    border-radius: 9px;
    border: 2px solid {PURPLE_2};
}}

QPushButton#Primary {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 {PURPLE}, stop:0.5 {PURPLE_2}, stop:1 {ACCENT});
    color: white;
    border: 0;
    border-radius: 14px;
    padding: 16px 24px;
    font-size: 15px;
    font-weight: 800;
    letter-spacing: 0.4px;
}}
QPushButton#Primary:hover {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 {PURPLE_2}, stop:1 {ACCENT_2});
}}
QPushButton#Primary:pressed {{ padding-top: 17px; padding-bottom: 15px; }}
QPushButton#Primary:disabled {{ background: #4c1d95; color: #a89ec7; }}

QFrame#ResultCard {{
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 {PANEL}, stop:1 #120626);
    border: 1.5px solid {BORDER_LT};
    border-radius: 20px;
}}
QLabel#ResultDate {{ color: {MUTED}; font-size: 11px; font-weight: 700; letter-spacing: 2px; }}
QLabel#ResultNum {{
    color: {PURPLE_4};
    font-size: 68px;
    font-weight: 900;
    font-family: 'Consolas', monospace;
}}
QLabel#ResultUnit {{ color: {MUTED}; font-size: 13px; font-weight: 600; }}
QLabel#ResultCI   {{ color: #8f83b8; font-size: 11px; }}

QFrame#MetricCard {{
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 {PANEL}, stop:1 {PANEL_2});
    border: 1px solid {BORDER};
    border-radius: 14px;
}}
QLabel#MetricK {{ color: {MUTED}; font-size: 10px; font-weight: 800; letter-spacing: 1.5px; }}
QLabel#MetricV {{ color: {PURPLE_4}; font-size: 24px; font-weight: 800; font-family: 'Consolas', monospace; }}

QFrame#ModelCard {{
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 {PANEL}, stop:1 {PANEL_2});
    border: 1.5px solid {BORDER};
    border-radius: 14px;
}}
QFrame#ModelCardWinner {{
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 {PANEL}, stop:1 {PANEL_2});
    border: 1.5px solid {ACCENT};
}}
QLabel#ModelName {{ color: {MUTED}; font-size: 11px; font-weight: 800; letter-spacing: 1.4px; }}
QLabel#ModelVal  {{ color: {PURPLE_4}; font-size: 32px; font-weight: 900; font-family: 'Consolas', monospace; }}
QLabel#ModelUnit {{ color: #8f83b8; font-size: 11px; }}
QLabel#BestBadge {{
    color: white; background: {ACCENT}; border-radius: 8px;
    font-size: 9px; font-weight: 800; letter-spacing: 1.5px;
    padding: 2px 8px;
}}

QScrollBar:vertical {{
    background: {PANEL_2}; width: 8px; border: 0; border-radius: 4px;
}}
QScrollBar::handle:vertical {{
    background: {BORDER_LT}; min-height: 20px; border-radius: 4px;
}}
QScrollBar::handle:vertical:hover {{ background: {PURPLE}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; background: none; }}

QTextEdit {{
    background: {PANEL}; border: 1px solid {BORDER}; border-radius: 12px;
    color: {TEXT}; padding: 14px; font-size: 13px;
}}
"""

# ============================================================
# SPLASH SCREEN
# ============================================================
class Splash(QWidget):
    done = Signal()
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.SplashScreen)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.resize(480, 300)
        self._center()
        self._angle = 0
        self._progress = 0.0
        self._status = "Loading..."
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(16)
        QTimer.singleShot(2500, self._close)

    def _center(self):
        s = QApplication.primaryScreen().geometry()
        self.move((s.width() - self.width()) // 2,
                  (s.height() - self.height()) // 2)

    def _tick(self):
        self._angle = (self._angle + 4) % 360
        self._progress = min(1.0, self._progress + 0.006)
        self.update()

    def set_status(self, s): self._status = s; self.update()

    def _close(self):
        self._timer.stop()
        self.done.emit()
        self.close()

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        # Background gradient
        g = QLinearGradient(0, 0, self.width(), self.height())
        g.setColorAt(0, QColor("#2a1150"))
        g.setColorAt(1, QColor("#08040f"))
        p.fillRect(self.rect(), QBrush(g))

        # Logo
        cx = self.width() // 2
        p.save()
        p.translate(cx, 100)
        p.rotate(self._angle)
        p.setPen(QPen(QColor(PURPLE_2), 4, Qt.SolidLine, Qt.RoundCap))
        p.drawArc(-30, -30, 60, 60, 0, 120 * 16)
        p.setPen(QPen(QColor(ACCENT), 4, Qt.SolidLine, Qt.RoundCap))
        p.drawArc(-30, -30, 60, 60, 180 * 16, 120 * 16)
        p.restore()

        # Title
        p.setPen(QColor("#ffffff"))
        p.setFont(QFont("Segoe UI", 20, QFont.Black))
        p.drawText(0, 160, self.width(), 32, Qt.AlignCenter, "Product Demand Forecasting")

        p.setPen(QColor(MUTED))
        p.setFont(QFont("Segoe UI", 11))
        p.drawText(0, 190, self.width(), 22, Qt.AlignCenter,
                   "Residual LSTM · LightGBM · Ensemble")

        # Progress bar
        bar_w, bar_h = 300, 5
        bx = (self.width() - bar_w) // 2
        by = 240
        p.setBrush(QColor(167, 139, 250, 40))
        p.setPen(Qt.NoPen)
        p.drawRoundedRect(bx, by, bar_w, bar_h, 3, 3)
        fill = int(bar_w * self._progress)
        if fill > 0:
            g2 = QLinearGradient(bx, 0, bx + bar_w, 0)
            g2.setColorAt(0, QColor(PURPLE))
            g2.setColorAt(0.5, QColor(PURPLE_4))
            g2.setColorAt(1, QColor(ACCENT))
            p.setBrush(QBrush(g2))
            p.drawRoundedRect(bx, by, fill, bar_h, 3, 3)

        # Status
        p.setPen(QColor(MUTED))
        p.setFont(QFont("Segoe UI", 9))
        p.drawText(0, by + 20, self.width(), 20, Qt.AlignCenter, self._status)

# ============================================================
# DATA LOADER THREAD
# ============================================================
class Loader(QObject):
    done = Signal(dict)
    status = Signal(str)

    def __init__(self):
        super().__init__()

    def run(self):
        try:
            self.status.emit("Loading LSTM model...")
            import tensorflow as tf
            model = tf.keras.models.load_model(MODEL_LSTM, compile=False)

            with open(MODEL_SCALER, "rb") as f:
                sp = pickle.load(f)
            scaler = sp["scaler"]; scaler_features = sp["features"]

            with open(MODEL_VOCAB, "rb") as f:
                voc = pickle.load(f)

            with open(MODEL_META) as f:
                meta = json.load(f)

            self.status.emit("Loading LightGBM...")
            with open(MODEL_LGBM, "rb") as f:
                lgb_model = pickle.load(f)["model"]

            with open(MODEL_ENSEMBLE, "rb") as f:
                ens = pickle.load(f)

            with open(MODEL_LABELS, "rb") as f:
                labels = pickle.load(f)

            self.status.emit("Loading dataset...")
            df = pd.read_parquet(DATA_PATH)
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values(["item_id","store_id","date"]).reset_index(drop=True)
            df["log_sales"] = np.log1p(df["sales"])

            doy = df["date"].dt.dayofyear
            df["doy_sin"]  = np.sin(2*np.pi*doy/365.25)
            df["doy_cos"]  = np.cos(2*np.pi*doy/365.25)
            df["wday_sin"] = np.sin(2*np.pi*df["wday"]/7)
            df["wday_cos"] = np.cos(2*np.pi*df["wday"]/7)
            df["event_any"] = ((df["event_name_1"].fillna("NoEvent")!="NoEvent") |
                               (df["event_name_2"].fillna("NoEvent")!="NoEvent")).astype(int)
            df["snap_flag"] = df[["snap_CA","snap_TX","snap_WI"]].max(axis=1).astype(int)

            self.status.emit("Engineering features...")
            g = df.groupby(["item_id","store_id"], sort=False)
            df["sell_price"] = g["sell_price"].transform(lambda s: s.ffill().bfill())
            for l in [1,2,3,7,14,28]:
                df[f"sales_log_lag_{l}"] = g["log_sales"].shift(l)
            for w in [7,14,28]:
                df[f"sales_log_roll_mean_{w}"] = g["log_sales"].transform(
                    lambda s: s.shift(1).rolling(w,min_periods=1).mean())
                df[f"sales_log_roll_std_{w}"]  = g["log_sales"].transform(
                    lambda s: s.shift(1).rolling(w,min_periods=1).std())
            df["sales_trend_7"]  = df["sales_log_lag_1"] - df["sales_log_lag_7"]
            df["sales_trend_28"] = df["sales_log_lag_1"] - df["sales_log_lag_28"]
            df["price_change_7d"] = g["sell_price"].transform(lambda s: s.pct_change(7))
            df["price_vs_28d_mean"] = g["sell_price"].transform(
                lambda s: s/(s.rolling(28,min_periods=1).mean()+1e-6)-1)
            df["prcp_mm"] = df["prcp_mm"].fillna(0.0)
            df["cpi_prev"] = df["cpi_prev"].ffill().bfill()
            df["unemployment_prev"] = df["unemployment_prev"].ffill().bfill()
            df = df.dropna(subset=["sales_log_lag_28"]).reset_index(drop=True)

            self.status.emit("Ready")
            self.done.emit({
                "model": model, "scaler": scaler, "scaler_features": scaler_features,
                "voc": voc, "meta": meta, "lgb_model": lgb_model, "ens": ens,
                "labels": labels, "df": df,
            })
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.done.emit({"error": str(e)})

# ============================================================
# PREDICTOR
# ============================================================
class Predictor:
    def __init__(self, ctx):
        self.model = ctx["model"]
        self.scaler = ctx["scaler"]
        self.SCALER_FEATURES = ctx["scaler_features"]
        self.voc = ctx["voc"]
        self.meta = ctx["meta"]
        self.lgb = ctx["lgb_model"]
        self.ens = ctx["ens"]
        self.df = ctx["df"]

        self.item2idx = self.voc["item2idx"]
        self.store2idx = self.voc["store2idx"]
        self.ITEMS = self.voc["ITEMS"]
        self.STORES = self.voc["STORES"]
        self.LOOKBACK = self.meta["lookback"]
        self.HIST = self.meta["hist_features"]
        self.FUT = self.meta["future_features"]
        self.W_LSTM = self.ens["lstm_weight"]
        self.W_LGBM = self.ens["lgbm_weight"]
        m = self.ens.get("metrics", {})
        self.CI80 = 1.28 * float(m.get("MAE", 3.81))

        # Valid combos
        combos = self.df[["item_id","store_id"]].drop_duplicates()
        self.item_to_stores = combos.groupby("item_id")["store_id"].apply(list).to_dict()
        self.valid_items = sorted(self.item_to_stores.keys())

    def get_history(self, item_id, store_id):
        sub = self.df[(self.df.item_id == item_id) & (self.df.store_id == store_id)]
        return sub.sort_values("date").tail(self.LOOKBACK).reset_index(drop=True)

    def _build_full_future_row(self, hist, last_row, last_price, target_date):
        doy = target_date.timetuple().tm_yday
        wday = target_date.weekday()
        last_log = float(last_row["log_sales"])
        row = {
            "sales_log_lag_1":  last_log,
            "sales_log_lag_2":  float(hist.iloc[-2]["log_sales"]) if len(hist)>=2 else last_log,
            "sales_log_lag_3":  float(hist.iloc[-3]["log_sales"]) if len(hist)>=3 else last_log,
            "sales_log_lag_7":  float(hist.iloc[-7]["log_sales"]) if len(hist)>=7 else last_log,
            "sales_log_lag_14": float(hist.iloc[-14]["log_sales"]) if len(hist)>=14 else last_log,
            "sales_log_lag_28": float(hist.iloc[-28]["log_sales"]) if len(hist)>=28 else last_log,
            "sales_log_roll_mean_7":  float(last_row.get("sales_log_roll_mean_7", last_log)),
            "sales_log_roll_std_7":   float(last_row.get("sales_log_roll_std_7", 0.0)),
            "sales_log_roll_mean_14": float(last_row.get("sales_log_roll_mean_14", last_log)),
            "sales_log_roll_std_14":  float(last_row.get("sales_log_roll_std_14", 0.0)),
            "sales_log_roll_mean_28": float(last_row.get("sales_log_roll_mean_28", last_log)),
            "sales_log_roll_std_28":  float(last_row.get("sales_log_roll_std_28", 0.0)),
            "sales_trend_7":  float(last_row.get("sales_trend_7", 0.0)),
            "sales_trend_28": float(last_row.get("sales_trend_28", 0.0)),
            "price_change_7d": float(last_row.get("price_change_7d", 0.0)),
            "price_vs_28d_mean": float(last_row.get("price_vs_28d_mean", 0.0)),
            "wday_sin": np.sin(2*np.pi*wday/7), "wday_cos": np.cos(2*np.pi*wday/7),
            "doy_sin": np.sin(2*np.pi*doy/365.25), "doy_cos": np.cos(2*np.pi*doy/365.25),
            "event_any": 0.0, "snap_flag": 0.0,
            "sell_price": last_price,
            "prcp_mm": float(last_row.get("prcp_mm", 0.0)),
            "cpi_prev": float(last_row.get("cpi_prev", 240.0)),
            "unemployment_prev": float(last_row.get("unemployment_prev", 6.0)),
        }
        for f in self.SCALER_FEATURES:
            row.setdefault(f, 0.0)
        return row

    def predict(self, item_id, store_id, days_ahead=1, model="Ensemble"):
        hist = self.get_history(item_id, store_id)
        if len(hist) < self.LOOKBACK:
            raise ValueError(f"Not enough history for {item_id}/{store_id}")

        last_row = hist.iloc[-1]
        last_price = float(last_row["sell_price"])
        target_date = last_row["date"] + timedelta(days=int(days_ahead))

        # LSTM hist
        hist_scaled = hist.copy()
        hist_scaled[self.SCALER_FEATURES] = self.scaler.transform(
            hist_scaled[self.SCALER_FEATURES])
        hist_arr = hist_scaled[self.HIST].to_numpy(np.float32)

        # Future
        full_row = self._build_full_future_row(hist, last_row, last_price, target_date)
        full_df = pd.DataFrame([full_row])[self.SCALER_FEATURES]
        full_scaled = self.scaler.transform(full_df)[0]
        scaled_dict = dict(zip(self.SCALER_FEATURES, full_scaled))
        fut_arr = np.array([scaled_dict[f] for f in self.FUT], dtype=np.float32)

        # LGBM
        H = hist_arr
        agg = np.concatenate([H.mean(axis=0), H.std(axis=0),
                              H.min(axis=0), H.max(axis=0), H[-1, :]])
        lgbm_X = np.concatenate([agg, fut_arr]).reshape(1, -1).astype(np.float32)
        base_log = float(hist_arr[-1, 0])

        # predictions
        def p_lstm():
            r = self.model.predict([hist_arr[None], fut_arr[None],
                                     np.array([self.item2idx[item_id]], dtype=np.int32),
                                     np.array([self.store2idx[store_id]], dtype=np.int32)],
                                    verbose=0)[0,0]
            return float(np.expm1(r + base_log))
        def p_lgbm():
            r = self.lgb.predict(lgbm_X)[0]
            return float(np.expm1(r + base_log))

        if model == "All":
            l_ = p_lstm(); g_ = p_lgbm()
            e_ = self.W_LSTM*l_ + self.W_LGBM*g_
            by = {"LSTM": l_, "LGBM": g_, "Ensemble": e_}
            chosen = e_
        elif model == "LSTM":
            chosen = p_lstm(); by = {"LSTM": chosen}
        elif model == "LGBM":
            chosen = p_lgbm(); by = {"LGBM": chosen}
        else:
            chosen = self.W_LSTM*p_lstm() + self.W_LGBM*p_lgbm(); by = {"Ensemble": chosen}

        return {
            "pred_sales": chosen, "by_model": by,
            "sales_history": hist["sales"].tolist(),
            "target_date": target_date.strftime("%Y-%m-%d"),
            "base_sales": float(hist["sales"].iloc[-1]),
            "naive_7d": float(hist["sales"].tail(7).mean()),
            "naive_28d": float(hist["sales"].mean()),
            "ci80": self.CI80,
        }

# ============================================================
# HELPERS
# ============================================================
def extract_id(raw, all_ids):
    if raw is None: return None
    s = str(raw)
    m = re.search(r"\[([A-Z0-9_]+)\]", s)
    if m and m.group(1) in all_ids: return m.group(1)
    if s.strip() in all_ids: return s.strip()
    for aid in all_ids:
        if aid in s: return aid
    return None

def item_disp(labels, iid):
    return f"{labels['en'][iid].split(' (')[0].strip()}  [{iid}]"

def store_disp(labels, sid):
    return f"{labels['en'][sid].split(' (')[0].strip()}  [{sid}]"

# ============================================================
# MAIN WINDOW
# ============================================================
class MainWindow(QMainWindow):
    def __init__(self, ctx):
        super().__init__()
        self.pred = Predictor(ctx)
        self.labels = ctx["labels"]

        self.setWindowTitle("Product Demand Forecasting")
        self.resize(1180, 800)
        self._center()

        # Root
        root = QWidget(); root.setObjectName("Root")
        self.setCentralWidget(root)
        rlayout = QVBoxLayout(root)
        rlayout.setContentsMargins(24, 24, 24, 24)
        rlayout.setSpacing(18)

        # Hero
        rlayout.addWidget(self._make_hero())

        # Body
        body = QHBoxLayout(); body.setSpacing(18)

        # Left column
        left = QVBoxLayout(); left.setSpacing(12)
        card = QFrame(); card.setObjectName("Card")
        cl = QVBoxLayout(card); cl.setContentsMargins(20, 20, 20, 20); cl.setSpacing(14)

        # Item
        cl.addWidget(self._section_label("ITEM"))
        self.cb_item = QComboBox()
        self.cb_item.setMinimumHeight(42)
        item_choices = [item_disp(self.labels, i) for i in self.pred.valid_items]
        self.cb_item.addItems(item_choices)
        self.cb_item.currentIndexChanged.connect(self._on_item_change)
        cl.addWidget(self.cb_item)

        # Store
        cl.addWidget(self._section_label("STORE"))
        self.cb_store = QComboBox()
        self.cb_store.setMinimumHeight(42)
        cl.addWidget(self.cb_store)

        # Model radio
        cl.addWidget(self._section_label("MODEL"))
        mr = QHBoxLayout(); mr.setSpacing(8)
        self.radio_group = QButtonGroup(self)
        for i, name in enumerate(["LSTM", "LGBM", "Ensemble", "All"]):
            rb = QRadioButton(name)
            rb.setMinimumHeight(36)
            if name == "Ensemble": rb.setChecked(True)
            self.radio_group.addButton(rb, i)
            mr.addWidget(rb)
        cl.addLayout(mr)

        # Days
        cl.addWidget(self._section_label("DAYS AHEAD"))
        dr = QHBoxLayout()
        self.lbl_days = QLabel("1")
        self.lbl_days.setStyleSheet(f"color:{PURPLE_4};font-size:15px;font-weight:800;")
        self.sl_days = QSlider(Qt.Horizontal)
        self.sl_days.setRange(1, 14); self.sl_days.setValue(1)
        self.sl_days.valueChanged.connect(lambda v: self.lbl_days.setText(str(v)))
        dr.addWidget(self.sl_days, 1); dr.addWidget(self.lbl_days)
        cl.addLayout(dr)

        # Run button
        self.btn_run = QPushButton("RUN FORECAST")
        self.btn_run.setObjectName("Primary")
        self.btn_run.setMinimumHeight(52)
        self.btn_run.setCursor(Qt.PointingHandCursor)
        self.btn_run.clicked.connect(self._run)
        cl.addWidget(self.btn_run)

        left.addWidget(card)
        left.addStretch()

        body.addLayout(left, 3)

        # Right column (result + chart + models)
        right = QVBoxLayout(); right.setSpacing(12)

        # Result card
        rc = QFrame(); rc.setObjectName("ResultCard")
        rcl = QVBoxLayout(rc); rcl.setContentsMargins(24, 22, 24, 22)
        rcl.setSpacing(4); rcl.setAlignment(Qt.AlignCenter)
        self.lbl_date = QLabel("AWAITING INPUT")
        self.lbl_date.setObjectName("ResultDate"); self.lbl_date.setAlignment(Qt.AlignCenter)
        self.lbl_num = QLabel("—")
        self.lbl_num.setObjectName("ResultNum"); self.lbl_num.setAlignment(Qt.AlignCenter)
        self.lbl_unit = QLabel("Run a forecast to begin")
        self.lbl_unit.setObjectName("ResultUnit"); self.lbl_unit.setAlignment(Qt.AlignCenter)
        self.lbl_ci = QLabel("")
        self.lbl_ci.setObjectName("ResultCI"); self.lbl_ci.setAlignment(Qt.AlignCenter)
        rcl.addWidget(self.lbl_date)
        rcl.addWidget(self.lbl_num)
        rcl.addWidget(self.lbl_unit)
        rcl.addWidget(self.lbl_ci)
        right.addWidget(rc)

        # Chart
        cc = QFrame(); cc.setObjectName("Card")
        ccl = QVBoxLayout(cc); ccl.setContentsMargins(14, 12, 14, 12)
        ccl.addWidget(self._section_label("SALES HISTORY & FORECAST"))
        pg.setConfigOptions(antialias=True)
        self.plot = pg.PlotWidget()
        self.plot.setBackground(PANEL_2)
        self.plot.showGrid(x=True, y=True, alpha=0.12)
        for ax_name in ("left","bottom"):
            ax = self.plot.getAxis(ax_name)
            ax.setPen(QColor(BORDER_LT)); ax.setTextPen(QColor(MUTED))
        self.plot.setMinimumHeight(220)
        self.plot.setMouseEnabled(x=False, y=False)
        ccl.addWidget(self.plot)
        right.addWidget(cc)

        # Model cards container
        self.models_row = QHBoxLayout(); self.models_row.setSpacing(12)
        right.addLayout(self.models_row)

        body.addLayout(right, 4)

        rlayout.addLayout(body, 1)

        # Init store dropdown
        self._on_item_change(0)

    def _center(self):
        s = QApplication.primaryScreen().geometry()
        self.move((s.width() - self.width()) // 2,
                  (s.height() - self.height()) // 2)

    def _section_label(self, txt):
        lbl = QLabel(txt); lbl.setObjectName("SectionLabel")
        return lbl

    def _make_hero(self):
        f = QFrame(); f.setObjectName("Hero")
        f.setMinimumHeight(150)
        l = QVBoxLayout(f); l.setContentsMargins(30, 24, 30, 24); l.setSpacing(6)

        title = QLabel("Product Demand Forecasting"); title.setObjectName("HeroTitle")
        sub = QLabel("Deep learning ensemble combining residual LSTM and gradient-boosted trees")
        sub.setObjectName("HeroSub")

        l.addWidget(title)
        l.addWidget(sub)
        l.addSpacing(14)

        stats = QHBoxLayout(); stats.setSpacing(36)
        for k, v in [("R-SQUARED","0.835"), ("MAE","3.81"),
                     ("RMSE","5.74"), ("MAPE","43.9%")]:
            col = QVBoxLayout(); col.setSpacing(2)
            kk = QLabel(k); kk.setObjectName("HeroK")
            vv = QLabel(v); vv.setObjectName("HeroV")
            col.addWidget(kk); col.addWidget(vv)
            stats.addLayout(col)
        stats.addStretch()
        l.addLayout(stats)
        return f

    def _on_item_change(self, idx):
        item_raw = self.cb_item.currentText()
        iid = extract_id(item_raw, set(self.pred.ITEMS))
        if not iid: return
        stores = self.pred.item_to_stores.get(iid, [])
        self.cb_store.blockSignals(True)
        self.cb_store.clear()
        for sid in stores:
            self.cb_store.addItem(store_disp(self.labels, sid))
        self.cb_store.blockSignals(False)

    def _run(self):
        self.btn_run.setEnabled(False)
        self.btn_run.setText("RUNNING...")
        QApplication.processEvents()

        try:
            item_raw = self.cb_item.currentText()
            store_raw = self.cb_store.currentText()
            iid = extract_id(item_raw, set(self.pred.ITEMS))
            sid = extract_id(store_raw, set(self.pred.STORES))
            if not iid or not sid:
                raise ValueError("Invalid selection")
            model_name = self.radio_group.checkedButton().text()
            days = self.sl_days.value()

            r = self.pred.predict(iid, sid, days_ahead=days, model=model_name)
            self._show_result(r, model_name)
        except Exception as e:
            self.lbl_date.setText("ERROR")
            self.lbl_num.setText("—")
            self.lbl_unit.setText(str(e))
            self.lbl_ci.setText("")
        finally:
            self.btn_run.setEnabled(True)
            self.btn_run.setText("RUN FORECAST")

    def _show_result(self, r, model_name):
        self.lbl_date.setText(f"FORECAST DATE · {r['target_date']}")
        self.lbl_num.setText(f"{r['pred_sales']:.1f}")
        self.lbl_unit.setText(f"Expected Units · {model_name}")

        lo = max(0, r["pred_sales"] - r["ci80"])
        hi = r["pred_sales"] + r["ci80"]
        self.lbl_ci.setText(
            f"80% Confidence: [{lo:.1f}, {hi:.1f}]   ·   "
            f"Last: {r['base_sales']:.0f}  ·   "
            f"7-day: {r['naive_7d']:.1f}  ·   "
            f"28-day: {r['naive_28d']:.1f}")

        self._draw_chart(r)
        self._render_model_cards(r["by_model"])

    def _draw_chart(self, r):
        self.plot.clear()
        hist = np.array(r["sales_history"], dtype=float)
        xs = np.arange(len(hist))
        self.plot.plot(xs, hist, pen=pg.mkPen(color=PURPLE_3, width=2.2),
                       symbol="o", symbolSize=5,
                       symbolBrush=PURPLE_3, symbolPen=None)
        fx = len(hist)
        fy = r["pred_sales"]
        self.plot.plot([xs[-1], fx], [hist[-1], fy],
                       pen=pg.mkPen(color=ACCENT, width=1.6, style=Qt.DashLine))
        self.plot.plot([fx], [fy], pen=None,
                       symbol="o", symbolSize=16,
                       symbolBrush=pg.mkBrush(ACCENT_2),
                       symbolPen=pg.mkPen("#ffffff", width=2))
        vline = pg.InfiniteLine(pos=fx-0.5, angle=90,
                                 pen=pg.mkPen(color=BORDER_LT, style=Qt.DashLine))
        self.plot.addItem(vline)

    def _render_model_cards(self, by_model):
        # clear
        while self.models_row.count():
            child = self.models_row.takeAt(0)
            if child.widget(): child.widget().deleteLater()

        if not by_model: return
        best = max(by_model, key=by_model.get)
        for name in ["LSTM","LGBM","Ensemble"]:
            if name not in by_model: continue
            f = QFrame()
            f.setObjectName("ModelCardWinner" if name == best else "ModelCard")
            f.setMinimumHeight(100)
            l = QVBoxLayout(f); l.setContentsMargins(14, 12, 14, 12); l.setSpacing(2)
            l.setAlignment(Qt.AlignCenter)
            if name == best:
                badge = QLabel("BEST"); badge.setObjectName("BestBadge")
                badge.setAlignment(Qt.AlignCenter)
                badge.setMaximumWidth(60)
                l.addWidget(badge, 0, Qt.AlignCenter)
            n = QLabel(name.upper()); n.setObjectName("ModelName"); n.setAlignment(Qt.AlignCenter)
            v = QLabel(f"{by_model[name]:.1f}"); v.setObjectName("ModelVal"); v.setAlignment(Qt.AlignCenter)
            u = QLabel("units"); u.setObjectName("ModelUnit"); u.setAlignment(Qt.AlignCenter)
            l.addWidget(n); l.addWidget(v); l.addWidget(u)
            self.models_row.addWidget(f)

# ============================================================
# MAIN
# ============================================================
def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Product Demand Forecasting")
    app.setStyleSheet(QSS)
    app.setFont(QFont("Segoe UI", 10))

    splash = Splash()
    splash.show()

    result = {}
    def on_done(ctx):
        result.update(ctx)

    loader = Loader()
    loader.done.connect(on_done)

    def run_loader():
        loader.run()
    QTimer.singleShot(200, run_loader)

    def after_splash():
        if "error" in result:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.critical(None, "Load error", result["error"])
            app.quit()
            return
        win = MainWindow(result)
        win.show()
        global _win
        _win = win

    splash.done.connect(after_splash)

    sys.exit(app.exec())

if __name__ == "__main__":
    main()
