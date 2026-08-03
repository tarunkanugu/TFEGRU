# TFEGRU – Cloud Workload Prediction

Time-Frequency Enhanced GRU for Alibaba Cloud workload forecasting.

## Project Structure

```
TFEGRU_Project/
├── app.py                  # Flask web application
├── requirements.txt        # Python dependencies
├── models/
│   ├── __init__.py
│   ├── tfegru_model.py     # All 4 model architectures
│   ├── data_utils.py       # Dataset generation & DataLoaders
│   └── trainer.py          # Training loop & evaluation
└── templates/
    └── index.html          # Web dashboard (Bootstrap5 + Chart.js)
```

## Models

| Model | Description |
|-------|-------------|
| **TFEGRU** | Time-Freq Enhanced Block + GRU + Multi-Head Attention |
| **Hybrid-BiGRU** | TFEB + Bidirectional GRU + Attention |
| **Hybrid-LSTM** | TFEB + GRU + LSTM + Attention |
| **Baseline-GRU** | Plain GRU (baseline for comparison) |

## Setup & Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the Flask app
python app.py

# 3. Open browser
http://localhost:5000
```

## Usage

1. Select a model from the dropdown
2. Set number of training epochs (slider)
3. Click **Train Model** — wait for ✅ in the log
4. Click **Predict** to see forecast vs actual chart
5. Train multiple models then click **Compare All** for MSE comparison

## Dataset

Synthetic Alibaba-style cloud workload with:
- 3000 samples (5-minute intervals)
- 5 features: `cpu_util`, `mem_util`, `net_in`, `net_out`, `disk_io`
- Diurnal patterns + weekly seasonality + random traffic spikes

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/dataset_info` | GET | Dataset metadata |
| `/api/train` | POST | Train a model `{"model":"TFEGRU","epochs":15}` |
| `/api/predict` | POST | Run prediction `{"model":"TFEGRU","steps":50}` |
| `/api/compare` | GET | Compare all trained models |
