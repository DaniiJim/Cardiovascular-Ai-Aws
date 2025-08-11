#!/usr/bin/env python3
import os
import ast

import wfdb
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Configuración fija
PTBXL_ROOT = "C:/Users/diego/OneDrive/Documentos/GitHub/Cardiovascular-Ai-Aws/ptb-xl-1.0.3"
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


# 1.1)  Lista de códigos detectables con derivación I (hay que balancear los nomales y SR, hay muchos)
CODES_I = [
        # Ritmos
    "SR","STACH","SBRAD","SARRH","AFIB","AFLT","SVTAC","PSVT","SVARR",
    # Ectopia y patrones
    "PAC","PVC","PRC(S)","BIGU","TRIGU",
    # Conducción AV / PR
    "1AVB","2AVB","3AVB","LPR",
    # Morfología QRS/T global
    "ABQRS","QWAVE","INVT","LOWT","TAB_","NT_","NDT",
    # ST-T global inespecífico / fisiopatológico
    "STE_","STD_","ISC_","ANEUR","DIG","EL",
    # Voltajes globales
    "HVOLT","LVOLT","VCLVH",
    # (opcional en dataset) normal
    "NORM"
]


def has_code_I_principal(scp_codes_raw):
    try:
        scp = ast.literal_eval(scp_codes_raw) if isinstance(scp_codes_raw, str) else {}
        # Convertir a float y descartar valores no numéricos
        scp = {k: float(v) for k, v in scp.items() if v is not None}
        if not scp:
            return False

        max_val = max(scp.values())
        top = [k for k, v in scp.items() if v == max_val]

        # Solo aceptar si hay un único principal y está en CODES_I
        return all(k in CODES_I for k in top)
    except Exception:
        return False


df = df[df["scp_codes"].apply(has_code_I_principal)]


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
    plt.plot(t, first_lead, color='black')
    plt.axis('off')      # Quita ejes
    plt.grid(False)      # Quita grid
    plt.tight_layout(pad=0)
    # Añadir título con la etiqueta
    plt.title(codes_str, fontsize=8, color='red', loc='left')


    # Guardar
    safe_name = ecg_str.replace("/", "_")
    out_path = os.path.join(OUTPUT_DIR, f"{safe_name}.png")
    try:
        plt.savefig(out_path, dpi=150, bbox_inches='tight', pad_inches=0)
        plt.close()
        # Convertir a binario
        from PIL import Image
        img = Image.open(out_path).convert('L')
        img_bin = img.point(lambda p: 255 if p > 128 else 0, mode='1')
        img_bin.save(out_path)  # Sobrescribe o cambia el nombre si prefieres
        print(f"[{ecg_str}] Guardado en {out_path}")
    except Exception as e:
        print(f"[{ecg_str}] Error guardando imagen: {e}")
    finally:
        plt.close()
