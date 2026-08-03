import time
import numpy as np
import torch
import torch.nn as nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _run_epoch(model, loader, criterion, optimizer, train):
    model.train() if train else model.eval()
    total_loss, preds_all, targets_all = 0.0, [], []
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            p = model(xb)
            loss = criterion(p, yb)
            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total_loss += loss.item() * len(xb)
            preds_all.append(p.detach().cpu().numpy())
            targets_all.append(yb.detach().cpu().numpy())
    p = np.concatenate(preds_all)
    t = np.concatenate(targets_all)
    return total_loss / len(loader.dataset), p, t


def train_model(model, train_loader, val_loader, epochs=20, lr=1e-3, patience=5):
    model = model.to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    history = {"train_loss": [], "val_loss": [], "train_mae": [], "val_mae": []}
    best_val = float("inf")
    best_state = None
    patience_cnt = 0
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        tl, tp, tt = _run_epoch(model, train_loader, criterion, optimizer, True)
        vl, vp, vt = _run_epoch(model, val_loader, criterion, None, False)
        scheduler.step(vl)
        history["train_loss"].append(float(tl))
        history["val_loss"].append(float(vl))
        history["train_mae"].append(float(np.mean(np.abs(tt - tp))))
        history["val_mae"].append(float(np.mean(np.abs(vt - vp))))

        if vl < best_val - 1e-6:
            best_val = vl
            patience_cnt = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)

    history["training_time_s"] = round(time.time() - t0, 2)
    history["epochs_run"] = epoch
    return history


def evaluate_model(model, test_loader):
    model = model.to(DEVICE)
    model.eval()
    preds_all, targets_all = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(DEVICE)
            p = model(xb).cpu().numpy()
            preds_all.append(p)
            targets_all.append(yb.numpy())
    p = np.concatenate(preds_all)
    t = np.concatenate(targets_all)
    mse = float(np.mean((t - p) ** 2))
    mae = float(np.mean(np.abs(t - p)))
    return {
        "MSE": round(mse, 6),
        "MAE": round(mae, 6),
        "RMSE": round(float(np.sqrt(mse)), 6),
    }
