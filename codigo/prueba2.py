#!/usr/bin/env python3
import os
import ast

import wfdb
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Configuración fija
PTBXL_ROOT = "/home/jaime/Desktop/github/Cardiovascular-Ai-Aws/codigo/physionet.org/files/ptb-xl/1.0.3"
CSV_META = os.path.join(PTBXL_ROOT, "ptbxl_database.csv")
OUTPUT_DIR = "output"

# Crear carpeta de salida si no existe
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1) Cargar el CSV y quedarse solo con filename_lr, filename_hr, scp_codes
df = pd.read_csv(
    CSV_META,
    index_col="ecg_id",
    usecols=["ecg_id", "filename_lr", "filename_hr", "scp_codes"],
)

# 2) Filtrar filas con filename_lr que empiecen por records100/00000/
mask = df["filename_lr"].astype(str).str.startswith("records100/00000/")
df_sub = df[mask].copy()

if df_sub.empty:
    print("No se encontraron filas con filename_lr comenzando en 'records100/00000/'.")
    exit(1)

# 3) Recorrer y generar gráficos usando scp_codes
for ecg_id, row in df_sub.iterrows():
    ecg_str = str(ecg_id)
    filename_lr = row["filename_lr"]
    scp_codes_raw = row["scp_codes"]
    record_base = os.path.join(PTBXL_ROOT, filename_lr)  # sin extensión

    # Parsear scp_codes
    codes_list = []
    try:
        scp_codes = ast.literal_eval(scp_codes_raw) if isinstance(scp_codes_raw, str) else {}
        # Tomar solo códigos con valor > 0 (puede ajustar el umbral si se desea)
        for code, val in scp_codes.items():
            try:
                if float(val) > 0:
                    codes_list.append(f"{code}:{float(val):.2f}")
            except Exception:
                # en caso de que val no sea convertible, lo muestra bruto
                codes_list.append(f"{code}:{val}")
    except Exception as e:
        codes_list = [f"(error parseando scp_codes: {e})"]

    if not codes_list:
        codes_list = ["(sin scp_codes)"]

    # Acortar si es muy largo
    codes_str = ", ".join(codes_list)
    if len(codes_str) > 120:
        codes_str = codes_str[:117] + "..."

    title = f"{ecg_str} | SCP: {codes_str}"

    # Cargar señal
    try:
        record = wfdb.rdrecord(record_base)
    except Exception as e:
        print(f"[{ecg_str}] Error cargando {record_base}: {e}")
        continue

    if record.p_signal.size == 0:
        print(f"[{ecg_str}] Señal vacía.")
        continue

    # Primera derivación
    first_lead = record.p_signal[:, 0]
    fs = record.fs
    t = np.arange(first_lead.shape[0]) / fs
    lead_name = record.sig_name[0] if record.sig_name else "Lead0"

    # Graficar
    plt.figure(figsize=(10, 3))
    plt.plot(t, first_lead)
    plt.xlabel("Tiempo (s)")
    plt.ylabel(lead_name)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()

    # Guardar
    safe_name = ecg_str.replace("/", "_")
    out_path = os.path.join(OUTPUT_DIR, f"{safe_name}.png")
    try:
        plt.savefig(out_path, dpi=150)
        print(f"[{ecg_str}] Guardado en {out_path}")
    except Exception as e:
        print(f"[{ecg_str}] Error guardando imagen: {e}")
    finally:
        plt.close()
