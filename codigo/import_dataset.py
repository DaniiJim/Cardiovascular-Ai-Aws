#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import ast
import shutil
import wfdb
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# =========================
# Configuración
# =========================
PTBXL_ROOT = r"C:/Users/diego/OneDrive/Documentos/GitHub/Cardiovascular-Ai-Aws/ptb-xl-1.0.3"
CSV_META   = os.path.join(PTBXL_ROOT, "ptbxl_database.csv")

OUTPUT_DIR = "output"
IMG_ROOT   = os.path.join(OUTPUT_DIR, "images_leadI")
LABELS_CSV = os.path.join(OUTPUT_DIR, "labels.csv")
COUNTS_CSV = os.path.join(OUTPUT_DIR, "class_counts.csv")
DIST_CSV   = os.path.join(OUTPUT_DIR, "class_distribution.csv")
QUOTA_CSV  = os.path.join(OUTPUT_DIR, "class_quota.csv")

# --- Rango de subcarpetas en records500 (inclusive): 00000, 01000, ..., 21000
PREFIX_START = 0
PREFIX_END   = 21000
PREFIX_STEP  = 1000
PREFIXES = [f"records500/{i:05d}/" for i in range(PREFIX_START, PREFIX_END + 1, PREFIX_STEP)]

# --- Máximo de imágenes por etiqueta (tras fusión)
MAX_PER_LABEL = 1000

# --- Clases permitidas (solo generaremos estas)
ALLOWED_LABELS = {"NORM", "NDT", "ISC_", "PVC", "AVB", "AFLT"}

# --- Fusión de etiquetas para no tocar el modelo
MERGE_MAP = {
    "1AVB": "AVB",
    "2AVB": "AVB",
    "3AVB": "AVB",
    "SR":   "NORM",
}

# --- Purga del labels.csv: eliminar filas de clases no permitidas
PURGE_NOT_ALLOWED = True
DELETE_FILES = False  # si True, borra también las imágenes de clases no permitidas

# Códigos detectables con derivación I
CODES_I = [
    # Ritmos
    "NORM","SR","STACH","SBRAD","SARRH","AFIB","AFLT","SVTAC","PSVT","SVARR",
    # Ectopia y patrones
    "PAC","PVC","PRC(S)","BIGU","TRIGU",
    # Conducción AV / PR
    "1AVB","2AVB","3AVB","LPR",
    # Morfología QRS/T global
    "ABQRS","QWAVE","INVT","LOWT","TAB_","NT_","NDT",
    # ST-T global / fisiopatológico
    "STE_","STD_","ISC_","ANEUR","DIG","EL",
    # Voltajes
    "HVOLT","LVOLT","VCLVH"
]

# =========================
# Utilidades
# =========================
def ensure_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(IMG_ROOT, exist_ok=True)

def parse_scp_codes(s):
    if not isinstance(s, str):
        return {}
    try:
        d = ast.literal_eval(s)
        return {k: float(v) for k, v in d.items() if v is not None}
    except Exception:
        return {}

def get_principal_label(scp_codes_raw):
    """
    Devuelve (label_principal, valor) si hay un único código con el valor máximo
    y está en CODES_I. Si no, (None, None).
    """
    scp = parse_scp_codes(scp_codes_raw)
    if not scp:
        return None, None
    max_val = max(scp.values())
    top = [k for k, v in scp.items() if v == max_val]
    if len(top) != 1:
        return None, None
    principal = top[0]
    return (principal, max_val) if principal in CODES_I else (None, None)

def pick_lead_index(record, preferred="I"):
    try:
        names = [n.upper() for n in record.sig_name]
        return names.index(preferred.upper())
    except Exception:
        return 0

def print_label_percentages(df_labels: pd.DataFrame):
    counts = df_labels["label"].value_counts().sort_index()
    total = int(counts.sum())
    perc = (counts / total * 100)
    print("\nDistribución de etiquetas (n y %):\n")
    width = max(6, max(len(str(x)) for x in counts.index)) if len(counts) else 6
    for label in counts.index:
        print(f"{label:<{width}}  {counts[label]:>6d}  {perc[label]:6.2f}%")
    print(f"\nTotal muestras: {total}")
    dist_df = pd.DataFrame({"count": counts, "percent": perc.round(4)})
    dist_df.to_csv(DIST_CSV, index=True, encoding="utf-8")
    print(f"✔ class_distribution.csv: {DIST_CSV}")
    return counts, perc

def print_quota_status(counts: pd.Series, max_per_label: int = MAX_PER_LABEL):
    labels = sorted(counts.index.tolist())
    rows = []
    print("\nEstado de cuotas por clase (tienes / objetivo / faltan):")
    for lbl in labels:
        have = int(counts.get(lbl, 0))
        left = max(0, max_per_label - have)
        print(f"- {lbl}: {have:>5d} / {max_per_label:<5d} (faltan {left})")
        rows.append({"label": lbl, "count": have, "target": max_per_label, "remaining": left})
    pd.DataFrame(rows).to_csv(QUOTA_CSV, index=False, encoding="utf-8")
    print(f"✔ class_quota.csv: {QUOTA_CSV}")

def safe_move(src, dst):
    """Mover archivo creando carpeta destino; ignora si src==dst o no existe."""
    if os.path.abspath(src) == os.path.abspath(dst):
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(src):
        shutil.move(src, dst)

# =========================
# Generación (records500 + fusión + filtro de clases)
# =========================
def main():
    ensure_dirs()

    # Cargar metadata y filtrar por records500/<rango>
    df = pd.read_csv(
        CSV_META,
        index_col="ecg_id",
        usecols=["ecg_id", "filename_lr", "filename_hr", "scp_codes"]
    )
    mask = df["filename_hr"].astype(str).str.startswith(tuple(PREFIXES))
    df = df[mask]
    if df.empty:
        print("No hay registros en records500 dentro del rango solicitado.")
        return

    # Reanudar: labels previos, aplicar fusión y (opcional) purgar no permitidas
    prev = None
    seen_paths = set()
    current_counts = {}

    if os.path.exists(LABELS_CSV):
        prev = pd.read_csv(LABELS_CSV)
        if not prev.empty:
            prev["filepath"] = prev["filepath"].astype(str).str.replace("\\", "/", regex=False)

            # 1) Fusionar etiquetas existentes
            prev["label"] = prev["label"].replace(MERGE_MAP)

            # 2) Migrar físicamente imágenes a carpetas fusionadas si cambió la etiqueta
            moved = 0
            for i, r in prev.iterrows():
                old_path = str(r["filepath"])
                ecg_id = str(r["ecg_id"])
                new_label = str(r["label"])
                new_path = os.path.join(IMG_ROOT, new_label, f"{ecg_id}.png").replace("\\", "/")
                if old_path != new_path:
                    if os.path.exists(old_path) and not os.path.exists(new_path):
                        safe_move(old_path, new_path)
                        prev.at[i, "filepath"] = new_path
                        moved += 1
                    else:
                        if os.path.exists(new_path):
                            prev.at[i, "filepath"] = new_path
            if moved:
                print(f"✔ Migradas {moved} imágenes a carpetas fusionadas.")

            # 3) (Opcional) Purga de clases no permitidas en el CSV
            if PURGE_NOT_ALLOWED:
                mask_allowed = prev["label"].isin(ALLOWED_LABELS)
                to_drop = prev[~mask_allowed]
                if not to_drop.empty:
                    print(f"⚠ Eliminando del labels.csv {len(to_drop)} filas de clases no permitidas...")
                    if DELETE_FILES:
                        for p in to_drop["filepath"]:
                            try:
                                if isinstance(p, str) and os.path.exists(p):
                                    os.remove(p)
                            except Exception:
                                pass
                    prev = prev[mask_allowed].reset_index(drop=True)

            # Guardar CSV normalizado
            prev.to_csv(LABELS_CSV, index=False, encoding="utf-8")

            # Estado para reanudar
            seen_paths = set(prev["filepath"].tolist())
            current_counts = prev["label"].value_counts().to_dict()

    def can_add(label_fused: str) -> bool:
        return current_counts.get(label_fused, 0) < MAX_PER_LABEL

    rows_new = []
    total = len(df)
    ok = 0
    skipped_load = skipped_label = skipped_quota = skipped_exists = skipped_not_allowed = 0

    # stats extra
    not_allowed_seen = defaultdict(int)   # cuántas veces vemos clases no permitidas
    quota_skipped_by_label = defaultdict(int)

    # Barajar para no sesgar por orden
    df = df.sample(frac=1.0, random_state=42)

    for idx, (ecg_id, row) in enumerate(df.iterrows(), 1):
        if idx % 1000 == 0 or idx == total:
            print(f"[{idx}/{total}] procesando...")

        # Etiqueta principal válida para D1
        label, pval = get_principal_label(row["scp_codes"])
        if label is None:
            skipped_label += 1
            continue

        # Fusionar (AVB y SR)
        fused = MERGE_MAP.get(label, label)

        # Filtrar por clases permitidas
        if fused not in ALLOWED_LABELS:
            not_allowed_seen[fused] += 1
            skipped_not_allowed += 1
            continue

        # Respetar cuota por etiqueta
        if not can_add(fused):
            quota_skipped_by_label[fused] += 1
            skipped_quota += 1
            continue

        # Cargar registro (records500)
        record_base = os.path.join(PTBXL_ROOT, row["filename_hr"])
        try:
            record = wfdb.rdrecord(record_base)
        except Exception:
            skipped_load += 1
            continue

        if record.p_signal is None or record.p_signal.size == 0:
            skipped_load += 1
            continue

        # Derivación I
        li = pick_lead_index(record, preferred="I")
        sig = record.p_signal[:, li]
        fs = float(record.fs) if hasattr(record, "fs") else 500.0
        t = np.arange(sig.shape[0]) / fs

        # Salida (carpeta ya fusionada)
        out_dir = os.path.join(IMG_ROOT, fused)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{ecg_id}.png")
        out_path_norm = out_path.replace("\\", "/")

        # Evitar duplicados
        if out_path_norm in seen_paths or os.path.exists(out_path):
            skipped_exists += 1
            continue

        # Graficar sin texto
        try:
            plt.figure(figsize=(10, 3))
            plt.plot(t, sig, color="black", linewidth=1.0)
            plt.axis("off")
            plt.grid(False)
            plt.tight_layout(pad=0)
            plt.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0, facecolor="white")
        except Exception:
            pass
        finally:
            plt.close()

        rows_new.append({
            "ecg_id": ecg_id,
            "filepath": out_path_norm,
            "label": fused,                 # etiqueta ya fusionada
            "principal_value": pval,
            "scp_codes": row["scp_codes"],
            "filename_lr": row["filename_lr"],
            "filename_hr": row["filename_hr"]
        })

        current_counts[fused] = current_counts.get(fused, 0) + 1
        seen_paths.add(out_path_norm)
        ok += 1

    # Guardar/actualizar labels y reportes
    if rows_new:
        new_df = pd.DataFrame(rows_new)
        if prev is not None and not prev.empty:
            all_df = pd.concat([prev, new_df], ignore_index=True)
            all_df.drop_duplicates(subset=["filepath"], inplace=True)
        else:
            all_df = new_df

        # Asegura orden alfabético de labels en conteos
        all_df.to_csv(LABELS_CSV, index=False, encoding="utf-8")

        # Recuentos y distribución (solo clases permitidas porque prev fue purgado)
        class_counts = all_df["label"].value_counts().sort_index()
        class_counts.to_csv(COUNTS_CSV, header=["count"])
        print(f"\n✔ Imágenes guardadas en: {IMG_ROOT}")
        print(f"✔ labels.csv: {LABELS_CSV}  (total filas: {len(all_df)})")
        print(f"✔ class_counts.csv: {COUNTS_CSV}")

        # Estadísticas principales
        counts, perc = print_label_percentages(all_df)
        print_quota_status(counts, MAX_PER_LABEL)

        # Resumen de descartes
        print("\nResumen de descartes:")
        print(f"- Etiqueta inválida/empate (no principal o no CODES_I): {skipped_label}")
        print(f"- No permitidas (encontradas pero filtradas):         {skipped_not_allowed}")
        if not_allowed_seen:
            print("  Detalle no permitidas vistas:")
            for k in sorted(not_allowed_seen):
                print(f"    · {k}: {not_allowed_seen[k]}")
        print(f"- Ya existían en disco/CSV:                            {skipped_exists}")
        print(f"- Superaban cuota:                                     {skipped_quota}")
        if quota_skipped_by_label:
            print("  Detalle por clase (superaban cuota):")
            for k in sorted(quota_skipped_by_label):
                print(f"    · {k}: {quota_skipped_by_label[k]}")
        print(f"- Errores de carga/señal vacía:                        {skipped_load}")

    else:
        print("No se generaron imágenes nuevas (quizá alcanzaste cuotas o no hubo casos válidos).")

if __name__ == "__main__":
    main()
