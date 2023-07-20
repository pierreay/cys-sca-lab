#!/usr/bin/env python3

import collections
import tqdm
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (10, 4)

def expe1():
    full_trace_power = np.load("data/02_full_trace_example.npy")
    plt.plot(full_trace_power)
    plt.xlabel("Sample")
    plt.ylabel("Scope ADC value")
    plt.title("Example Power Trace")
    plt.show()

    full_trace_em = np.load("data/02_full_trace_example_em.npy")
    plt.plot(full_trace_em)
    plt.xlabel("Sample")
    plt.ylabel("Scope ADC value")
    plt.title("Example EM Trace")
    plt.show()

if __name__ == "__main__":
    expe1()
