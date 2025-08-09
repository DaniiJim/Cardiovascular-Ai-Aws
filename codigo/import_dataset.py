#!/usr/bin/env python3
import wfdb
import matplotlib.pyplot as plt
import numpy as np

record_base = "/home/jaime/Desktop/github/Cardiovascular-Ai-Aws/codigo/physionet.org/files/ptb-xl/1.0.3/records100/00000/00001_lr"
record = wfdb.rdrecord(record_base)

fs = record.fs
first_lead = record.p_signal[:, 0]
t = np.arange(first_lead.shape[0]) / fs

plt.plot(t, first_lead)
plt.xlabel("Tiempo (s)")
plt.ylabel(record.sig_name[0])
plt.title(f"Primera derivación: {record.sig_name[0]}")
plt.grid(True)
plt.tight_layout()
plt.show()

