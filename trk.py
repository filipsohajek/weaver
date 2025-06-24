#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

from gen_ca import gen_ca, cpsels
from argparse import ArgumentParser

arg_parser = ArgumentParser()
arg_parser.add_argument("--format", type=str, choices=("i16", "f32", "f64"), default="i16")
arg_parser.add_argument("--sample_rate", type=int, default=4e6)
arg_parser.add_argument("--duration", type=float, default=10)
arg_parser.add_argument("--prn", type=int)
arg_parser.add_argument("--code_offset", type=float)
arg_parser.add_argument("--loop_type", type=str, choices=("kalman", "pll",), default="kalman")
arg_parser.add_argument("--doppler", type=float)
arg_parser.add_argument("in_file", type=str)
args = arg_parser.parse_args()

CODE_FREQ = 1.023e6
CARRIER_FREQ = 1575.42e6
CHIPS = 2 * np.array(list(gen_ca(*cpsels[args.prn - 1]))) - 1
CODE_CHIP_COUNT = len(CHIPS)
CODE_PERIOD = CODE_CHIP_COUNT / CODE_FREQ

def disc_carrier(prompt):
    return np.atan(prompt.imag / prompt.real) / (2 * np.pi)
def disc_carrier_freq(prompt1, prompt2, delta_t):
    cross = prompt1.real * prompt2.imag - prompt2.real * prompt1.imag
    dot = prompt1.real * prompt2.real + prompt1.imag * prompt2.imag
    #disc = (cross * np.sign(dot)) / (delta_t * 2 * np.pi)
    disc = (cross) / (delta_t * 2 * np.pi)
    disc /= 0.5 * (np.square(np.abs(prompt1)) + np.square(np.abs(prompt2)))
    return disc
def disc_code(early, late):
    e = np.abs(early)
    l = np.abs(late)
    return 0.5 * (e - l)/(e + l)

def correlate(samples, sample_rate, code_phase, code_freq, carrier_phase, carrier_freq, code_offset):
    code_phase_step = code_freq / args.sample_rate
    code_phase = CODE_CHIP_COUNT * (code_phase + (code_offset / CODE_CHIP_COUNT) + code_phase_step * np.arange(len(samples)))
    code_phase = (code_phase + CODE_CHIP_COUNT).astype(np.int32) % CODE_CHIP_COUNT
        
    carrier_freq = -2.0 * np.pi * carrier_freq
    carrier_phase = -2.0 * np.pi * carrier_phase + (carrier_freq / sample_rate) * np.arange(len(samples))

    replica = np.take(CHIPS, code_phase) * np.exp(1j * carrier_phase)
    return np.sum(samples * replica)

class KalmanLoop:
    def __init__(self, int_time_s, code_phase, carrier_freq):
        self._aiding_factor = (CODE_FREQ/CARRIER_FREQ) / CODE_CHIP_COUNT
        self.A = np.array([[1, 0, self._aiding_factor * int_time_s, 0.5 * self._aiding_factor * int_time_s**2], [0, 1, int_time_s, 0.5 * int_time_s**2], [0, 0, 1, int_time_s], [0, 0, 0, 1]])
        self.H = np.array([[1, 0, -0.5 * self._aiding_factor * int_time_s, -(1/6) * self._aiding_factor * int_time_s**2], [0, 1, -int_time_s, -(1/6) * int_time_s**2]])
        self.Q = np.diag([1e-7, 1e-6, 1e-5, 0])
        self.P = np.diag([1e-3, 1, 500**2, 1])
        self.R = np.diag([1e-6, 1e-8])
        self.state = np.array([0, 0, 0, 0])

        self._int_time_s = int_time_s
        self.carrier_freq = carrier_freq

    def update(self, carrier_disc, code_disc, _carrier_freq_disc):
        z = np.array([code_disc, carrier_disc])
        pred_state = self.A @ self.state
        pred_cov = self.A @ self.P @ self.A.transpose() + self.Q
        K = pred_cov @ self.H.transpose() @ np.linalg.inv(self.H @ pred_cov @ self.H.transpose() + self.R)
        self.state = pred_state + K @ (z - self.H @ pred_state)
        self.P = (np.eye(4) - K @ self.H) @ pred_cov

        print(self.state)
        self.carrier_freq += self.state[2]
        code_phase_err = self.state[0]
        carr_phase_err = self.state[1]
        self.state[0] = 0
        self.state[1] = 0
        self.state[2] = 0

        return (CODE_FREQ/CODE_CHIP_COUNT) + self._aiding_factor * self.carrier_freq, self.carrier_freq, code_phase_err, carr_phase_err
    
    def update_cn0(self, cn0):
        code_disc_var = (2 * (0.5 + 1/(cn0 * CODE_PERIOD))**2) / (CODE_CHIP_COUNT * CODE_CHIP_COUNT)
        carr_disc_var = ((1/(2 * cn0 * CODE_PERIOD)) * (1 + (1/(2 * cn0 * CODE_PERIOD)))) / (4 * np.pi**2)
        self.R = np.diag([code_disc_var, carr_disc_var])
        print(self.R)

class PLLLoop:
    def __init__(self, int_time_s, code_phase, carrier_freq):
        self.dll_omega = 0.1
        self.pll_omega = 15
        self.fll_omega = 2
        self._int_time_s = int_time_s
        self.code_freq_acc = 1/CODE_PERIOD
        self.freq_acc = carrier_freq
        self.freq_rate_acc = 0
    
    def update(self, carrier_disc, code_disc, carrier_freq_disc):
        carrier_freq = 2.4 * self.pll_omega * carrier_disc + self.freq_acc
        self.freq_acc += self._int_time_s * (1.1 * self.pll_omega**2 * carrier_disc + self.freq_rate_acc + np.sqrt(2) * self.fll_omega * carrier_freq_disc)
        self.freq_rate_acc += self._int_time_s * (self.pll_omega**3 * carrier_disc + carrier_freq_disc * self.fll_omega**2)

        code_freq = self.dll_omega * code_disc + ((CODE_FREQ/CARRIER_FREQ) * carrier_freq + CODE_FREQ) / CODE_CHIP_COUNT
        
        return code_freq, carrier_freq, 0, 0

    def update_cn0(self, cn0):
        pass
        
n_read_samples = int(args.duration * args.sample_rate)
if args.format == "i16":
    print("reading i16")
    samples = np.fromfile(args.in_file, dtype=np.int16, count=2*n_read_samples)
    samples = (samples[::2] + 1j*samples[1::2]) / 32767.0
elif args.format == "f32":
    print("reading f32")
    samples = np.fromfile(args.in_file, dtype=np.complex64, count=n_read_samples)
elif args.format == "f64":
    print("reading f64")
    samples = np.fromfile(args.in_file, dtype=np.complex128, count=n_read_samples)

n_int_samples = int(CODE_PERIOD * args.sample_rate)

if args.loop_type == "kalman":
    loop = KalmanLoop(CODE_PERIOD, args.code_offset, args.doppler)
elif args.loop_type == "pll":
    loop = PLLLoop(CODE_PERIOD, args.code_offset, args.doppler)

code_phases = [args.code_offset]
code_freqs = [1/CODE_PERIOD]
carrier_freqs = [args.doppler]
carrier_phases = [0]
carrier_discs = []
carrier_freq_discs = []
code_discs = []
earlys = []
prompts = [0]
last_prompts = np.zeros(20, dtype=np.complex64)
cn0s = []
lates = []
t = [0]
while len(samples):
    cur_samples, samples = samples[:n_int_samples], samples[n_int_samples:]
    earlys.append(correlate(cur_samples, args.sample_rate, code_phases[-1], code_freqs[-1], carrier_phases[-1], carrier_freqs[-1], -0.5))
    prompts.append(correlate(cur_samples, args.sample_rate, code_phases[-1], code_freqs[-1], carrier_phases[-1], carrier_freqs[-1], 0))
    lates.append(correlate(cur_samples, args.sample_rate, code_phases[-1], code_freqs[-1], carrier_phases[-1], carrier_freqs[-1], 0.5))
    carrier_phases.append((carrier_phases[-1] + CODE_PERIOD * carrier_freqs[-1]) % 1)
    code_phases.append((code_phases[-1] + CODE_PERIOD * code_freqs[-1]) % 1)

    t.append(t[-1] + CODE_PERIOD)
    carrier_discs.append(disc_carrier(prompts[-1]))
    carrier_freq_discs.append(-disc_carrier_freq(prompts[-1], prompts[-2], CODE_PERIOD))
    code_discs.append(-disc_code(earlys[-1], lates[-1]) / CODE_CHIP_COUNT)

    last_prompts = np.roll(last_prompts, 1)
    last_prompts[0] = prompts[-1]
    m2 = (1/len(last_prompts)) * np.sum(np.square(np.abs(last_prompts)))
    m4 = (1/len(last_prompts)) * np.sum(np.pow(np.abs(last_prompts), 4))
    pd = np.sqrt(2 * m2**2 - m4)
    pn = m2 - pd
    cn0 = (1/CODE_PERIOD) * (pd/pn)
    if np.isnan(cn0):
        cn0 = 1e3
    cn0s.append(cn0)
    loop.update_cn0(cn0s[-1])

    code_freq, carrier_freq, code_phase_err, carr_phase_err = loop.update(carrier_discs[-1], code_discs[-1], carrier_freq_discs[-1])
    code_freqs.append(code_freq)
    carrier_freqs.append(carrier_freq)
    code_phases[-1] += code_phase_err
    carrier_phases[-1] += carr_phase_err

    print(f"t={t[-1]:.2f} s, code_phase={code_phases[-1]:.8f}, code_freq={code_freqs[-1]:.3f} carrier_freq={carrier_freqs[-1]:.1f}, carrier_phase={carrier_phases[-1]:.3f}, cn0={10*np.log10(cn0s[-1]):.1f}")
    print(f"abs(prompt)={np.abs(prompts[-1]):.2f}, carrier_disc={carrier_discs[-1]:.5f}, carrier_freq_disc={carrier_freq_discs[-1]:.5f}, code_disc={code_discs[-1]:.5f}\n")


