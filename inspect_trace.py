#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
trace_type = np.dtype([
    ("prompt", np.complex64),
    ("cn0", np.float64),
    ("lock_ind", np.float64),
    ("code_disc", np.float64),
    ("carr_disc", np.float64),
    ("code_phase", np.float64),
    ("carr_phase", np.float64),
    ("carr_freq", np.float32),
    ("code_freq", np.float64),
])

trace = np.fromfile("out_trace", dtype=trace_type)
#out_i16 = np.astype(np.fromfile("out", dtype=np.int16), np.float32) / 32767.0
#out = out_i16[::2] + 1.0j*out_i16[1::2]
#t = np.linspace(0, len(out), len(out)) / 4e6
