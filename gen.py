#!/usr/bin/env python3
import numpy as np

from gen_ca import gen_ca, cpsels
from argparse import ArgumentParser


arg_parser = ArgumentParser()
arg_parser.add_argument("--format", type=str, choices=("i16", "f32", "f64"), default="i16")
arg_parser.add_argument("--sample_rate", type=int, default=4e6)
arg_parser.add_argument("--duration", type=float, default=10)
arg_parser.add_argument("--prn", type=int)
arg_parser.add_argument("--doppler", type=float, default=0.0)
arg_parser.add_argument("--code_offset", type=float, default=0.0)
arg_parser.add_argument("--rx_pwr_dbm", type=float, default=-120)
arg_parser.add_argument("--noise_temp", type=float, default=300)
arg_parser.add_argument("--gain_scale", type=float, default=1.0)
arg_parser.add_argument("out_file", type=str)
args = arg_parser.parse_args()


CODE_FREQ = 1.023e6
BOLTZMANN_K = 1.3806503e-23

duration_samples = int(args.sample_rate * args.duration)
chips = 2 * np.array(list(gen_ca(*cpsels[args.prn - 1]))) - 1
code_phase_step = CODE_FREQ / args.sample_rate
code_phase = np.astype(len(chips) * args.code_offset + code_phase_step * np.arange(duration_samples), np.int32) % len(chips)

carr_phase = (2 * np.pi * args.doppler * np.arange(duration_samples) / args.sample_rate) % (2 * np.pi)

rx_pwr = 10**(args.rx_pwr_dbm/10)
noise_pwr = BOLTZMANN_K * args.noise_temp * args.sample_rate
print(f"noise_pwr={10*np.log10(noise_pwr)} dB, SNR={10*np.log10(rx_pwr/ noise_pwr)} dB")
samples = np.sqrt(rx_pwr) * np.take(chips, code_phase) * np.exp(1j*carr_phase)
samples += np.random.normal(scale=np.sqrt(noise_pwr / 2), size=samples.shape) + 1j*np.random.normal(scale=np.sqrt(noise_pwr / 2), size=samples.shape) 
samples *= args.gain_scale / np.max(np.abs(samples)) 

if args.format == "i16":
    samples *= 32767.0
    samples = samples.view(np.float64).astype(np.int16)
elif args.format == "f32":
    samples = samples.view(np.float64).astype(np.float32)
elif args.format == "f64":
    samples = samples.view(np.float64)
samples.tofile(args.out_file)

