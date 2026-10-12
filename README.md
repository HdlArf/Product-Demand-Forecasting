# P R O D U C T |  D E M A N D

<p align="center">
  <img src="assets/hero_banner.gif" width="100%" alt="Product Demand Forecasting">
</p>

<p align="center">
  <b>LSTM × LIGHTGBM × ENSEMBLE</b><br>
  <sub>train · evaluate · ensemble · deploy</sub>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/R%C2%B2-0.835-8b5cf6?style=for-the-badge">
  <img src="https://img.shields.io/badge/MAE-3.81-a855f7?style=for-the-badge">
  <img src="https://img.shields.io/badge/MAPE-43.9%25-d946ef?style=for-the-badge">
  <img src="https://img.shields.io/badge/license-MIT-8b5cf6?style=for-the-badge">
</p>

---

## SIGNAL

A production-ready **demand forecasting system** that predicts daily retail sales across **52 products × 10 stores** using a **deep learning ensemble**.

Trained on **5 years** of data (2011–2016) — **400,000+ daily records** — with a **Residual LSTM** network and a **gradient-boosted** companion model, combined into an **optimized ensemble**.

<p align="center">
  <img src="assets/neural_signal.gif" width="100%" alt="Neural Signal">
</p>

---

## PIPELINE

<p align="center">
  <img src="assets/workflow.gif" width="100%" alt="Pipeline">
</p>

---

## RESULTS

<p align="center">
  <img src="assets/results_chart.gif" width="100%" alt="Results Comparison">
</p>

<table align="center">
  <thead>
    <tr>
      <th align="left">Model</th>
      <th align="center">MAE</th>
      <th align="center">RMSE</th>
      <th align="center">R²</th>
      <th align="center">MAPE</th>
      <th align="center">Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Naive (previous day)</td>
      <td align="center">5.162</td>
      <td align="center">7.764</td>
      <td align="center">0.699</td>
      <td align="center">61.0%</td>
      <td align="center">⬜⬜⬜⬜⬜⬜⬜⬛⬛⬛</td>
    </tr>
    <tr>
      <td>7-day rolling mean</td>
      <td align="center">4.389</td>
      <td align="center">6.643</td>
      <td align="center">0.779</td>
      <td align="center">53.7%</td>
      <td align="center">⬜⬜⬜⬜⬜⬜⬜⬜⬛⬛</td>
    </tr>
    <tr>
      <td>28-day rolling mean</td>
      <td align="center">4.278</td>
      <td align="center">6.457</td>
      <td align="center">0.791</td>
      <td align="center">53.1%</td>
      <td align="center">⬜⬜⬜⬜⬜⬜⬜⬜⬛⬛</td>
    </tr>
    <tr>
      <td>LSTM v3</td>
      <td align="center">3.819</td>
      <td align="center">5.751</td>
      <td align="center"><b>0.835</b></td>
      <td align="center">44.3%</td>
      <td align="center">🟪🟪🟪🟪🟪🟪🟪🟪🟪⬛</td>
    </tr>
    <tr>
      <td>LightGBM v4</td>
      <td align="center">3.905</td>
      <td align="center">5.955</td>
      <td align="center">0.823</td>
      <td align="center"><b>43.6%</b></td>
      <td align="center">🟪🟪🟪🟪🟪🟪🟪🟪⬛⬛</td>
    </tr>
    <tr>
      <td><b>Ensemble v4</b> 🏆</td>
      <td align="center"><b>3.808</b></td>
      <td align="center"><b>5.744</b></td>
      <td align="center"><b>0.835</b></td>
      <td align="center"><b>43.9%</b></td>
      <td align="center">🟪🟪🟪🟪🟪🟪🟪🟪🟪🟪</td>
    </tr>
  </tbody>
</table>

<br>

<p align="center">
  <img src="https://img.shields.io/badge/%E2%86%91_MAPE_improvement-10--17%25_over_baselines-d946ef?style=for-the-badge&labelColor=1c0e3a">
  <img src="https://img.shields.io/badge/%E2%86%93_MAE_reduction-27%25_vs_LSTM_v2-8b5cf6?style=for-the-badge&labelColor=1c0e3a">
  <img src="https://img.shields.io/badge/best_model-Ensemble_v4-a855f7?style=for-the-badge&labelColor=1c0e3a">
</p>

---

## MODELS

<p align="center">
  <img src="assets/models_cards.gif" width="100%" alt="Three Models">
</p>

<table align="center">
  <tr>
    <td align="center" width="33%">
      <h3>01 — LSTM v3</h3>
      <sub>Deep residual network</sub><br><br>
      LSTM (128 → 64) + embeddings<br>
      Residual learning · Huber loss<br>
      <b>26 hist + 12 future features</b>
    </td>
    <td align="center" width="33%">
      <h3>02 — LightGBM v4</h3>
      <sub>Gradient-boosted trees</sub><br><br>
      1500 estimators · depth 8<br>
      L1 objective · window aggregates<br>
      <b>Fast & interpretable</b>
    </td>
    <td align="center" width="33%">
      <h3>03 — Ensemble v4 🏆</h3>
      <sub>Weighted combination</sub><br><br>
      0.75 × LSTM + 0.25 × LightGBM<br>
      Optimized on validation MAE<br>
      <b>Best overall</b>
    </td>
  </tr>
</table>

---

## GETTING STARTED

```bash
git clone https://github.com/HdlArf/Product-Demand-Forecasting.git
cd Product-Demand-Forecasting
pip install -r requirements.txt
python app/app.py
```

**Requirements:** Python 3.10–3.12 · ~2 GB disk · 5 minutes setup

---

## STACK

<p align="center">
  <img src="assets/stack_visual.gif" width="100%" alt="Tech Stack">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/TensorFlow-2.16-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white">
  <img src="https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white">
  <img src="https://img.shields.io/badge/LightGBM-02569B?style=for-the-badge">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white">
  <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white">
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white">
  <img src="https://img.shields.io/badge/PySide6-41CD52?style=for-the-badge&logo=qt&logoColor=white">
  <img src="https://img.shields.io/badge/Gradio-FF7C00?style=for-the-badge&logo=gradio&logoColor=white">
</p>

---

## PROJECT STRUCTURE

```
Product-Demand-Forecasting/
  ├── app/
  │   └── app.py                    Desktop application (PySide6)
  ├── data/
  │   └── master_data.parquet       Historical sales (400K rows)
  ├── models/
  │   ├── Product_Demand_LSTM_v3.keras
  │   ├── Product_Demand_LGBM_v4.pkl
  │   ├── Product_Demand_Ensemble_v4.pkl
  │   └── ...
  ├── assets/                       Animated GIFs
  ├── requirements.txt
  └── README.md
```

---

## EXPLORING NEXT

`Probabilistic Forecasting` · `Temporal Fusion Transformer` · `Explainability (SHAP)`

---

## MIAO APPROVES

<p align="center">
  <img src="assets/miao_observer.gif" width="100%" alt="Miao observing the forecast">
</p>

---

## CONNECT

<p align="center">
  <a href="https://github.com/HdlArf"><img src="https://img.shields.io/badge/GitHub-HdlArf-181717?style=for-the-badge&logo=github&logoColor=white"></a>
  <a href="https://www.kaggle.com/hadeelalnomay"><img src="https://img.shields.io/badge/Kaggle-hadeelalnomay-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white"></a>
  <a href="https://www.instagram.com/hdl_arf/"><img src="https://img.shields.io/badge/Instagram-hdl__arf-E4405F?style=for-the-badge&logo=instagram&logoColor=white"></a>
  <a href="https://www.linkedin.com/in/hadeel-hadeel-10a449331/"><img src="https://img.shields.io/badge/LinkedIn-Hadeel-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white"></a>
  <a href="mailto:hadeelalnomai@gmail.com"><img src="https://img.shields.io/badge/Gmail-Contact-EA4335?style=for-the-badge&logo=gmail&logoColor=white"></a>
</p>

---

<p align="center">
  <sub>still forecasting.</sub><br><br>
  <b>HΛDΞL</b><br>
  <sub>BUILD · EXPLORE · UNDERSTAND</sub>
</p>
