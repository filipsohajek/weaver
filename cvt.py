#!/usr/bin/env python3
import numpy as np

from argparse import ArgumentParser


arg_parser = ArgumentParser()
arg_parser.add_argument("--in_format", type=str, choices=("i16", "f32", "f64"), default="i16")
arg_parser.add_argument("--out_format", type=str, choices=("i16", "f32", "f64"), default="i16")
arg_parser.add_argument("in_file", type=str)
arg_parser.add_argument("out_file", type=str)
args = arg_parser.parse_args()


if args.in_format == "i16":
    samples = np.fromfile(args.in_file, dtype=np.int16)
    samples = (samples[::2] + 1j*samples[1::2]) / 32767.0
elif args.in_format == "f32":
    samples = np.fromfile(args.in_file, dtype=np.complex64)
elif args.in_format == "f64":
    samples = np.fromfile(args.in_file, dtype=np.complex128)

if args.out_format == "i16":
    samples *= 32767.0
    samples = samples.view(np.float64).astype(np.int16)
elif args.out_format == "f32":
    samples = samples.view(np.float64).astype(np.float32)
elif args.out_format == "f64":
    samples = samples.view(np.float64)

samples.tofile(args.out_file)

