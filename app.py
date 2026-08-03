import os, sys, numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, render_template, request, jsonify
from models.tfegru_model import build_model, MODEL_REGISTRY
from models.data_utils import prepare_loaders, generate_alibaba_workload
from models.trainer import train_model, evaluate_model, DEVICE

app = Flask(__name__)

# ── Shared state (module-level dict, lives for the lifetime of the process) ──
STATE = {
    "models":   {},   # name -> nn.Module
    "histories":{},   # name -> history dict
    "metrics":  {},   # name -> metrics dict
    "loaders":  None, # (train, val, test)
    "meta":     None,
}

SEQ_LEN    = 32
BATCH_SIZE = 64
N_SAMPLES  = 3000
MODEL_NAMES = list(MODEL_REGISTRY.keys())


def _ensure_loaders():
    if STATE["loaders"] is None:
        tr, vl, te, scaler, meta = prepare_loaders(
            n_samples=N_SAMPLES, seq_len=SEQ_LEN, batch_size=BATCH_SIZE)
        STATE["loaders"] = (tr, vl, te)
        STATE["meta"] = meta
    return STATE["loaders"], STATE["meta"]


def _make_model(name):
    meta = STATE["meta"]
    if name == "Baseline-GRU":
        return build_model(name, input_dim=meta["input_dim"],
                           seq_len=SEQ_LEN, output_dim=1)
    return build_model(name, input_dim=meta["input_dim"],
                       seq_len=SEQ_LEN, d_model=64, output_dim=1)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html", models=MODEL_NAMES)


@app.route("/api/dataset_info")
def dataset_info():
    _, meta = _ensure_loaders()
    return jsonify({
        "n_samples":  N_SAMPLES,
        "features":   meta["feature_cols"],
        "n_train":    meta["n_train"],
        "n_val":      meta["n_val"],
        "n_test":     meta["n_test"],
        "seq_len":    SEQ_LEN,
    })


@app.route("/api/train", methods=["POST"])
def train():
    body   = request.get_json(force=True)
    name   = body.get("model", "TFEGRU")
    epochs = int(body.get("epochs", 15))

    if name not in MODEL_NAMES:
        return jsonify({"error": f"Unknown model: {name}"}), 400

    try:
        (tr, vl, te), meta = _ensure_loaders()

        model   = _make_model(name)
        history = train_model(model, tr, vl, epochs=epochs, patience=5)
        metrics = evaluate_model(model, te)

        # Store in STATE — this is the fix for "not trained yet"
        STATE["models"][name]    = model
        STATE["histories"][name] = history
        STATE["metrics"][name]   = metrics

        return jsonify({
            "model":         name,
            "metrics":       metrics,
            "epochs_run":    history["epochs_run"],
            "training_time": history["training_time_s"],
            "train_loss":    history["train_loss"],
            "val_loss":      history["val_loss"],
            "train_mae":     history["train_mae"],
            "val_mae":       history["val_mae"],
        })
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


@app.route("/api/predict", methods=["POST"])
def predict():
    body  = request.get_json(force=True)
    name  = body.get("model", "TFEGRU")
    steps = int(body.get("steps", 50))

    if name not in STATE["models"]:
        return jsonify({"error": f"Model '{name}' not trained yet. Please train it first."}), 400

    try:
        _, meta   = _ensure_loaders()
        model     = STATE["models"][name].to(DEVICE)
        model.eval()

        feature_cols = meta["feature_cols"]
        df           = generate_alibaba_workload(N_SAMPLES)
        values       = df[feature_cols].values.astype(np.float32)

        # Use last SEQ_LEN rows as the initial window
        buf    = values[-SEQ_LEN:].copy()
        preds  = []
        actual = values[-(SEQ_LEN + steps): -SEQ_LEN, 0].tolist()
        # Pad actual if dataset not long enough
        if len(actual) < steps:
            actual = (actual + [0.0] * steps)[:steps]

        with torch.no_grad():
            for _ in range(steps):
                x_t = torch.tensor(buf[-SEQ_LEN:]).unsqueeze(0).to(DEVICE)
                p   = float(model(x_t).item())
                preds.append(p)
                new_row    = buf[-1].copy()
                new_row[0] = p
                buf        = np.vstack([buf, new_row])

        return jsonify({"predictions": preds, "actuals": actual})

    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


@app.route("/api/compare")
def compare():
    if not STATE["metrics"]:
        return jsonify({"error": "No models trained yet."}), 400
    return jsonify(STATE["metrics"])


if __name__ == "__main__":
    app.run(debug=False, port=5000)
