#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, random, glob, argparse
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
import torch.backends.cudnn as cudnn

# =========================
# Modelo
# =========================
class ECGImageCNN(nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = 5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout1 = nn.Dropout(p=0.4)
        self.fc1 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(p=0.3)
        self.classifier = nn.Linear(128,  num_classes)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout1(x)
        x = F.relu(self.fc1(x), inplace=True)
        x = self.dropout2(x)
        return self.classifier(x)

# =========================
# Utils
# =========================
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def build_amp():
    # Usa API nueva si está, si no cae a la clásica
    try:
        from torch.amp import autocast as ac_new, GradScaler as GS_new
        device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
        try:
            scaler = GS_new(device_type)  # API nueva
        except TypeError:
            scaler = GS_new(enabled=torch.cuda.is_available())
        def ctx(): return ac_new(device_type)
        return ctx, scaler
    except Exception:
        from torch.cuda.amp import autocast as ac_old, GradScaler as GS_old
        scaler = GS_old(enabled=torch.cuda.is_available())
        def ctx(): return ac_old(enabled=torch.cuda.is_available())
        return ctx, scaler

def stratified_split(df: pd.DataFrame, split):
    parts = []
    for label, grp in df.groupby("label", sort=False):
        grp = grp.sample(frac=1.0, random_state=split.get("seed", 42))
        n = len(grp)
        if n < 5:
            n_train, n_val = n, 0
        else:
            n_train = max(1, int(n * split["train"]))
            n_val   = int(n * split["val"])
            if n >= 10 and n_val == 0: n_val = 1
            n_train = min(n_train, n - n_val)
            if n_train < 1: n_train = 1
            if n_train + n_val > n: n_val = max(0, n - n_train)
        g_train = grp.iloc[:n_train]
        g_val   = grp.iloc[n_train:n_train+n_val]
        g_test  = grp.iloc[n_train+n_val:]
        parts += [("train", g_train), ("val", g_val), ("test", g_test)]
    df_train = pd.concat([g for p,g in parts if p=="train"]).reset_index(drop=True)
    df_val   = pd.concat([g for p,g in parts if p=="val"]).reset_index(drop=True)
    df_test  = pd.concat([g for p,g in parts if p=="test"]).reset_index(drop=True)
    return df_train, df_val, df_test

def compute_class_weights(counts: pd.Series) -> torch.Tensor:
    freq = counts.astype(float).copy()
    if (freq > 0).any():
        min_nz = freq[freq > 0].min()
        freq[freq == 0] = min_nz
    else:
        freq[:] = 1.0
    N, C = freq.sum(), len(freq)
    weights = N / (C * freq)
    return torch.tensor(weights.values, dtype=torch.float32)

def make_weighted_sampler(labels_idx: List[int], num_classes: int) -> WeightedRandomSampler:
    counts = np.bincount(np.array(labels_idx), minlength=num_classes).astype(float)
    inv_freq = 1.0 / (counts + 1e-9)
    sample_weights = inv_freq[np.array(labels_idx)]
    return WeightedRandomSampler(weights=torch.from_numpy(sample_weights),
                                 num_samples=len(sample_weights),
                                 replacement=True)

class CSVImageDataset(Dataset):
    def __init__(self, df: pd.DataFrame, label2idx: Dict[str, int], img_size: Tuple[int, int], train: bool=True):
        self.df = df.reset_index(drop=True)
        self.label2idx = label2idx
        self.tf = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        y = self.label2idx[r["label"]]
        img = Image.open(r["filepath"])
        img = self.tf(img)
        return img, y

# =========================
# Train / Eval
# =========================
def train_one_epoch(model, loader, optimizer, criterion, device, autocast_ctx, scaler):
    model.train()
    running_loss, running_acc = 0.0, 0.0
    for imgs, y in loader:
        imgs, y = imgs.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        with autocast_ctx():
            logits = model(imgs)
            loss = criterion(logits, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item() * imgs.size(0)
        running_acc  += (logits.argmax(1) == y).sum().item()
    n = len(loader.dataset)
    return running_loss / n, running_acc / n

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, running_acc = 0.0, 0.0
    for imgs, y in loader:
        imgs, y = imgs.to(device), y.to(device)
        logits = model(imgs)
        loss = criterion(logits, y)
        running_loss += loss.item() * imgs.size(0)
        running_acc  += (logits.argmax(1) == y).sum().item()
    n = len(loader.dataset)
    return running_loss / n, running_acc / n

# =========================
# Main (SageMaker entry)
# =========================
def main():
    # ----- Args / HPO friendly -----
    parser = argparse.ArgumentParser()
    # SageMaker env defaults
    parser.add_argument('--data-dir',   type=str, default=os.environ.get('SM_CHANNEL_TRAINING', '/opt/ml/input/data/training'))
    parser.add_argument('--model-dir',  type=str, default=os.environ.get('SM_MODEL_DIR', '/opt/ml/model'))
    parser.add_argument('--output-dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', '/opt/ml/output/data'))
    # Hiperparámetros
    parser.add_argument('--img-height', type=int, default=450)
    parser.add_argument('--img-width',  type=int, default=1500)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--epochs',     type=int, default=20)
    parser.add_argument('--lr',         type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--num-workers',  type=int, default=2)
    parser.add_argument('--seed',       type=int, default=42)
    parser.add_argument('--use-sampler', action='store_true', help='Si se activa: WeightedRandomSampler')
    args = parser.parse_args()

    cudnn.benchmark = True
    set_seed(args.seed)

    # ----- Localización de datos (dentro del contenedor) -----
    # Espera: data_dir/
    #   ├─ labels.csv
    #   └─ images_leadI/<CLASE>/*.png
    data_dir   = args.data_dir
    model_dir  = args.model_dir
    output_dir = args.output_dir
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    labels_csv = os.path.join(data_dir, "labels.csv")
    if not os.path.exists(labels_csv):
        raise FileNotFoundError(f"No se encontró labels.csv en {labels_csv}")

    # Resolver raíz de imágenes
    img_root_candidates = [
        os.path.join(data_dir, "images_leadI"),
        os.path.join(data_dir, "images_leadl"),
    ]
    img_root = None
    for c in img_root_candidates:
        if os.path.isdir(c):
            img_root = c; break
    if img_root is None:
        # busca por patrón
        cands = glob.glob(os.path.join(data_dir, "images_lead*"))
        for c in cands:
            if os.path.isdir(c):
                img_root = c; break
    if img_root is None:
        raise FileNotFoundError(f"No se encontró carpeta de imágenes bajo {data_dir} (images_leadI/*).")

    # Cargar labels y normalizar rutas al data_dir
    df = pd.read_csv(labels_csv)
    if "label" not in df.columns or "ecg_id" not in df.columns:
        raise ValueError("labels.csv debe tener columnas 'label' y 'ecg_id'.")

    if "filepath" not in df.columns:
        df["filepath"] = ""
    # reparar rutas si es necesario
    def file_exists(p):
        try: return os.path.exists(p)
        except: return False
    bad = ~df["filepath"].apply(file_exists)
    if bad.any():
        def rebuild_path(row):
            ecg = str(row["ecg_id"])
            if ecg.endswith(".0"): ecg = ecg[:-2]
            return os.path.join(img_root, str(row["label"]), f"{ecg}.png")
        df.loc[bad, "filepath"] = df[bad].apply(rebuild_path, axis=1)
        bad2 = ~df["filepath"].apply(file_exists)
        if bad2.any():
            missing = df[bad2][["ecg_id","label","filepath"]].head(10)
            raise FileNotFoundError(f"Rutas de imagen inexistentes. Ejemplos:\n{missing.to_string(index=False)}")
        # guarda copia reparada en outputs
        df.to_csv(os.path.join(output_dir, "labels_repaired.csv"), index=False)

    # ----- Clases y splits -----
    classes = sorted(df["label"].unique())
    label2idx = {c:i for i,c in enumerate(classes)}
    with open(os.path.join(model_dir, "label2idx.json"), "w", encoding="utf-8") as f:
        json.dump(label2idx, f, ensure_ascii=False, indent=2)

    split_cfg = {"train": 0.8, "val": 0.1, "test": 0.1, "seed": args.seed}
    df_train, df_val, df_test = stratified_split(df, split_cfg)

    # ----- Datasets / Loaders -----
    img_size = (args.img_height, args.img_width)
    ds_train = CSVImageDataset(df_train, label2idx, img_size=img_size, train=True)
    ds_val   = CSVImageDataset(df_val,   label2idx, img_size=img_size, train=False)
    ds_test  = CSVImageDataset(df_test,  label2idx, img_size=img_size, train=False)

    num_classes = len(classes)
    pin = torch.cuda.is_available()

    if args.use-sampler:
        train_labels_idx = [label2idx[l] for l in df_train["label"].tolist()]
        sampler = make_weighted_sampler(train_labels_idx, num_classes)
        dl_train = DataLoader(ds_train, batch_size=args.batch_size, sampler=sampler,
                              num_workers=args.num_workers, pin_memory=pin)
    else:
        dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=pin)

    dl_val   = DataLoader(ds_val,   batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers, pin_memory=pin)
    dl_test  = DataLoader(ds_test,  batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers, pin_memory=pin)

    # Pesos por clase (opcional)
    counts_train = df_train["label"].value_counts().reindex(classes, fill_value=0)
    class_weights = compute_class_weights(counts_train)

    # ----- Modelo / Opt / Loss -----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ECGImageCNN(in_channels=1, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    autocast_ctx, scaler = build_amp()

    cudnn.benchmark = True
    best_val_acc = 0.0
    ckpt_path = os.path.join(model_dir, "best_model.pt")

    # ----- Training loop -----
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, dl_train, optimizer, criterion, device, autocast_ctx, scaler)
        va_loss, va_acc = evaluate(model, dl_val, criterion, device)
        print(f"[{epoch:02d}/{args.epochs}] train_loss={tr_loss:.4f} acc={tr_acc:.4f} | val_loss={va_loss:.4f} acc={va_acc:.4f}", flush=True)

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "classes": classes,
                "label2idx": label2idx,
                "val_acc": best_val_acc,
                "img_size": img_size,
            }, ckpt_path)
            print(f"✔ Guardado mejor modelo en {ckpt_path} (val_acc={best_val_acc:.4f})", flush=True)

    # ----- Eval final -----
    te_loss, te_acc = evaluate(model, dl_test, criterion, device)
    print(f"TEST: loss={te_loss:.4f} acc={te_acc:.4f}", flush=True)

    # Escribe métrica a /opt/ml/output (útil para HPO)
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump({"val_acc": float(best_val_acc), "test_acc": float(te_acc)}, f)

    # (Opcional) export TorchScript para inferencia más cómoda
    # dummy = torch.randn(1,1,*img_size, device=device)
    # traced = torch.jit.trace(model.eval(), dummy)
    # traced.save(os.path.join(model_dir, "model_traced.pt"))

if __name__ == "__main__":
    main()
