#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import chi2
from gen_ca import gen_ca, cpsels
from argparse import ArgumentParser

arg_parser = ArgumentParser()
arg_parser.add_argument("--format", type=str, choices=("i16", "f32", "f64"), default="i16")
arg_parser.add_argument("--sample_rate", type=int, default=4e6)
arg_parser.add_argument("--prn", type=int)
arg_parser.add_argument("--n_coherent", type=int, default=1)
arg_parser.add_argument("--n_noncoherent", type=int, default=1)
arg_parser.add_argument("--doppler_min", type=int, default=-7500)
arg_parser.add_argument("--doppler_max", type=int, default=7500)
arg_parser.add_argument("--n_cands", type=int, default=10)
arg_parser.add_argument("in_file", type=str)
args = arg_parser.parse_args()

CODE_FREQ = 1.023e6

chips = 2 * np.array(list(gen_ca(*cpsels[args.prn - 1]))) - 1
duration_samples = int(args.sample_rate * (len(chips) / CODE_FREQ) * args.n_coherent)
code_phase_step = CODE_FREQ / args.sample_rate
code_phase = np.astype(code_phase_step * np.arange(2 * duration_samples), np.int32) % len(chips)
replica = np.take(chips, code_phase)

if args.format == "i16":
    print("reading i16")
    samples = np.fromfile(args.in_file, dtype=np.int16)
    samples = (samples[::2] + 1j*samples[1::2]) / 32767.0
elif args.format == "f32":
    print("reading f32")
    samples = np.fromfile(args.in_file, dtype=np.complex64)
elif args.format == "f64":
    print("reading f64")
    samples = np.fromfile(args.in_file, dtype=np.complex128)

doppler_step = int(1 / (len(replica) / args.sample_rate))
doppler_shift_min = args.doppler_min // doppler_step
doppler_shift_max = args.doppler_max // doppler_step

replica_fft = np.fft.fft(replica)
corr_res = np.zeros((doppler_shift_max - doppler_shift_min + 1, 2 * duration_samples))

for noncoh_i in range(args.n_noncoherent):
    samples_start = duration_samples * noncoh_i
    samples_end = samples_start + duration_samples
    coh_samples = samples[samples_start:samples_end]
    sample_var = np.var(coh_samples)
    real_var = np.var(coh_samples.real)
    imag_var = np.var(coh_samples.imag)
    print(f"mean={np.mean(coh_samples)}, var={np.var(coh_samples)}, real_var={real_var}, imag_var={imag_var}")
    coh_samples.real /= np.sqrt(duration_samples * real_var)
    coh_samples.imag /= np.sqrt(duration_samples * imag_var)
    coh_samples = np.pad(coh_samples, (0, len(replica) - len(coh_samples)))
    samples_fft = np.conj(np.fft.fft(coh_samples))
    #samples_fft /= np.sqrt(duration_samples * sample_var)

    for doppler_shift in range(doppler_shift_min, doppler_shift_max + 1):
        corr = np.fft.ifft(np.roll(samples_fft, -doppler_shift) * replica_fft)
        corr_res[doppler_shift - doppler_shift_min, :] += np.square(np.abs(corr))

corr_res = corr_res[:, :(duration_samples // args.n_coherent)]

max_indices = np.argpartition(corr_res.flatten(), -args.n_cands)[-args.n_cands:]
for cand_idx in max_indices:
    doppler_idx = cand_idx // corr_res.shape[1]
    code_idx = cand_idx % corr_res.shape[1]

    cand_freq = (doppler_idx + doppler_shift_min) * doppler_step
    cand_code_offset = code_idx / corr_res.shape[1]
    cand_val = corr_res.flatten()[cand_idx]
    chi2_dof = 2 * args.n_coherent * args.n_noncoherent
    max_count = corr_res.flatten().shape[0]
    print(f"freq={cand_freq}, code_offset={cand_code_offset}, val={cand_val}, chi2_dof={chi2_dof}, max_count={max_count}, p={1 - (chi2.cdf(cand_val, chi2_dof))**max_count}")

plt.imshow(corr_res, extent=(0, corr_res.shape[1]/((len(chips) / CODE_FREQ) * args.sample_rate), doppler_shift_max * doppler_step, doppler_shift_min * doppler_step), aspect="auto", interpolation="none")
plt.show()
