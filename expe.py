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

def expe2():
    data_em = np.load("data/02_data_em.npy")
    data_power = np.load("data/02_data_power.npy")
    print("data_em:", data_em.shape)
    print("data_power:", data_power.shape)

    # Visualize.

    plt.plot(data_power[0])
    plt.xlabel("Sample")
    plt.ylabel("Scope ADC value")
    plt.title("First Power Trace")
    plt.show()

    plt.plot(data_em[0])
    plt.xlabel("Sample")
    plt.ylabel("Scope ADC value")
    plt.title("First EM Trace")
    plt.show()

    # Variance.

    data_power_var = np.var(data_power, axis=0)
    assert data_power_var.shape == (data_power.shape[1], )

    plt.plot(data_power_var)
    plt.xlabel("Sample")
    plt.ylabel("Variance of ADC samples")
    plt.title("Variance of the power traces")
    plt.show()

    data_em_var = np.var(data_em, axis=0)
    assert data_em_var.shape == (data_em.shape[1], )

    plt.plot(data_em_var)
    plt.xlabel("Sample")
    plt.ylabel("Variance of ADC samples")
    plt.title("Variance of the EM traces")
    plt.show()

if __name__ == "__main__":
    # expe1()
    expe2()
