#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, random, glob
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

# =========================================================
# Configuración (AJUSTA SOLO ESTAS RUTAS SI HACE FALTA)
# =========================================================
OUTPUT_DIR = "/content/drive/MyDrive/AWS/output"

IMG_ROOT_DEFAULTS = [
    os.path.join(OUTPUT_DIR, "images_leadI"),
    os.path.join(OUTPUT_DIR, "images_leadl"),  # por si quedó con 'l'
]
LABELS_CSV = os.path.join(OUTPUT_DIR, "labels.csv")
RUN_DIR    = os.path.join(OUTPUT_DIR, "runs_ecg")
os.makedirs(RUN_DIR, exist_ok=True)

# Tamaño de entrada del modelo (H, W). Si OOM, prueba (256,1024) o (224,896)
IMG_SIZE     = (450, 1500)
BATCH_SIZE   = 16
EPOCHS       = 20
LR           = 1e-3
WEIGHT_DECAY = 1e-4
VAL_EVERY    = 1
SEED         = 42
NUM_WORKERS  = 2
SPLIT        = {"train": 0.8, "val": 0.1, "test": 0.1}

cudnn.benchmark = True  # acelera convoluciones con tamaño fijo

def build_amp():
    try:
        from torch.amp import autocast as ac_new, GradScaler as GS_new
        def make(device_type):
            try:
                scaler = GS_new(device_type)  # API nueva
                def ctx():
                    return ac_new(device_type)
                return ctx, scaler
            except TypeError:
                # API semi-nueva (sin device_type en ctor)
                scaler = GS_new(enabled=torch.cuda.is_available())
                def ctx():
                    return ac_new(device_type)
                return ctx, scaler
        return make('cuda' if torch.cuda.is_available() else 'cpu')
    except Exception:
        from torch.cuda.amp import autocast as ac_old, GradScaler as GS_old
        scaler = GS_old(enabled=torch.cuda.is_available())
        def ctx():
            return ac_old(enabled=torch.cuda.is_available())
        return ctx, scaler

# =========================================================
# Utilidades
# =========================================================
def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def resolve_img_root() -> str:
    for c in IMG_ROOT_DEFAULTS:
        if os.path.isdir(c):
            return c
    cands = glob.glob(os.path.join(OUTPUT_DIR, "images_lead*"))
    for c in cands:
        if os.path.isdir(c):
            return c
    raise FileNotFoundError("No se encontró carpeta de imágenes dentro de OUTPUT_DIR (images_lead*).")

def load_labels_and_fix_paths(labels_csv: str, img_root: str) -> pd.DataFrame:
    if not os.path.exists(labels_csv):
        raise FileNotFoundError(f"No existe {labels_csv}")
    df = pd.read_csv(labels_csv)

    for col in ("label", "ecg_id"):
        if col not in df.columns:
            raise ValueError(f"labels.csv debe contener la columna '{col}'.")

    if "filepath" not in df.columns:
        df["filepath"] = ""

    def file_exists(p):
        try: return os.path.exists(p)
        except: return False

    bad = ~df["filepath"].apply(file_exists)
    if bad.any():
        def rebuild_path(row):
            ecg = str(row["ecg_id"])
            if ecg.endswith(".0"):  # por si viene con .0 del CSV
                ecg = ecg[:-2]
            return os.path.join(img_root, str(row["label"]), f"{ecg}.png")
        df.loc[bad, "filepath"] = df[bad].apply(rebuild_path, axis=1)

        bad2 = ~df["filepath"].apply(file_exists)
        if bad2.any():
            missing = df[bad2][["ecg_id", "label", "filepath"]].head(10)
            raise FileNotFoundError(
                f"Aún hay rutas que no existen ({bad2.sum()} archivos). "
                f"Ejemplos:\n{missing.to_string(index=False)}\n"
                f"Verifica {img_root}/<label>/<ecg_id>.png"
            )
        df.to_csv(labels_csv, index=False)
        print("✔ 'filepath' reparado y guardado en labels.csv")
    else:
        print("✔ Todas las rutas de 'filepath' existen")
    return df

def stratified_split(df: pd.DataFrame, split=SPLIT):
    """
    Estratificado garantizando al menos 1 muestra en train por clase si existe.
    Si n<5, mando todo a train para que el modelo tenga algo que aprender.
    """
    parts = []
    for label, grp in df.groupby("label", sort=False):
        grp = grp.sample(frac=1.0, random_state=SEED)
        n = len(grp)

        if n < 5:
            n_train, n_val = n, 0
        else:
            n_train = max(1, int(n * split["train"]))
            n_val   = int(n * split["val"])
            if n >= 10 and n_val == 0:
                n_val = 1
            n_train = min(n_train, n - n_val)
            if n_train < 1: n_train = 1
            if n_train + n_val > n:
                n_val = max(0, n - n_train)

        g_train = grp.iloc[:n_train]
        g_val   = grp.iloc[n_train:n_train+n_val]
        g_test  = grp.iloc[n_train+n_val:]

        parts.append(("train", g_train))
        parts.append(("val",   g_val))
        parts.append(("test",  g_test))

    df_train = pd.concat([g for p,g in parts if p=="train"]).reset_index(drop=True)
    df_val   = pd.concat([g for p,g in parts if p=="val"]).reset_index(drop=True)
    df_test  = pd.concat([g for p,g in parts if p=="test"]).reset_index(drop=True)

    print("\nSplit counts:")
    print("train:\n", df_train["label"].value_counts().sort_index().to_string())
    print("\nval:\n",   df_val["label"].value_counts().sort_index().to_string())
    print("\ntest:\n",  df_test["label"].value_counts().sort_index().to_string())

    return df_train, df_val, df_test

def compute_class_weights(counts: pd.Series) -> torch.Tensor:
    """
    Pesos balanceados: w_c = N / (C * n_c)
    Si n_c=0, sustituimos por el mínimo no-cero para evitar infinitos.
    """
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
    sampler = WeightedRandomSampler(weights=torch.from_numpy(sample_weights),
                                    num_samples=len(sample_weights),
                                    replacement=True)
    return sampler

# =========================================================
# Modelo
# =========================================================
class ECGImageCNN(nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = 5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout1 = nn.Dropout(p=0.4)
        self.fc1 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(p=0.3)
        self.classifier = nn.Linear(128, num_classes)
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

# =========================================================
# Dataset
# =========================================================
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
        img = Image.open(r["filepath"])   # sin .convert("L")
        img = self.tf(img)
        return img, y

# =========================================================
# Train / Eval (con AMP auto-seleccionada)
# =========================================================
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

# =========================================================
# Main
# =========================================================
def main():
    set_seed(SEED)

    if not os.path.exists(LABELS_CSV):
        raise FileNotFoundError(f"No existe labels.csv en {LABELS_CSV}")
    img_root = resolve_img_root()
    print("IMG_ROOT =", img_root)

    df = load_labels_and_fix_paths(LABELS_CSV, img_root)
    if df.empty:
        raise RuntimeError("labels.csv está vacío tras cargar.")

    classes = sorted(df["label"].unique())
    label2idx = {c:i for i,c in enumerate(classes)}
    num_classes = len(classes)
    with open(os.path.join(RUN_DIR, "label2idx.json"), "w", encoding="utf-8") as f:
        json.dump(label2idx, f, ensure_ascii=False, indent=2)
    print("Clases:", classes)

    df_train, df_val, df_test = stratified_split(df, SPLIT)

    ds_train = CSVImageDataset(df_train, label2idx, img_size=IMG_SIZE, train=True)
    ds_val   = CSVImageDataset(df_val,   label2idx, img_size=IMG_SIZE, train=False)
    ds_test  = CSVImageDataset(df_test,  label2idx, img_size=IMG_SIZE, train=False)

    train_labels_idx = [label2idx[l] for l in df_train["label"].tolist()]
    sampler = make_weighted_sampler(train_labels_idx, num_classes)

    counts = df_train["label"].value_counts().reindex(classes, fill_value=0)
    class_weights = compute_class_weights(counts)
    print("\nRecuento train por clase:")
    print(counts.to_string())
    print("\nClass weights usados (orden de classes):")
    print({c: float(w) for c, w in zip(classes, class_weights)})

    pin = torch.cuda.is_available()
    dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, sampler=sampler,
                          num_workers=NUM_WORKERS, pin_memory=pin)
    dl_val   = DataLoader(ds_val,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=pin)
    dl_test  = DataLoader(ds_test,  batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=pin)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ECGImageCNN(in_channels=1, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    autocast_ctx, scaler = build_amp()  # ← API nueva si está disponible

    best_val_acc = 0.0
    ckpt_path = os.path.join(RUN_DIR, "best_model.pt")

    for epoch in range(1, EPOCHS+1):
        tr_loss, tr_acc = train_one_epoch(model, dl_train, optimizer, criterion, device, autocast_ctx, scaler)
        va_loss, va_acc = evaluate(model, dl_val, criterion, device)

        print(f"[{epoch:02d}/{EPOCHS}] "
              f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f}  "
              f"val_loss={va_loss:.4f} val_acc={va_acc:.4f}")

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "label2idx": label2idx,
                "classes": classes,
                "val_acc": best_val_acc,
                "img_size": IMG_SIZE,
            }, ckpt_path)
            print(f"✔ Nuevo mejor modelo guardado en {ckpt_path} (val_acc={best_val_acc:.4f})")

    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        print(f"\nCargado mejor checkpoint (epoch {ckpt['epoch']}, val_acc={ckpt['val_acc']:.4f})")

    te_loss, te_acc = evaluate(model, dl_test, criterion, device)
    print(f"\nTEST: loss={te_loss:.4f} acc={te_acc:.4f}")

if __name__ == "__main__":
    main()
