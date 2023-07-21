#!/usr/bin/env python3

import collections
import tqdm
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (10, 4)

# * Experience #1

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

# * Experience #2

def signal_to_noise(data, labels):
    # Array of ordered integers from 0 to 255.
    unique_classes = np.unique(labels)
    mean_per_class = []
    var_per_class = []
    for value_of_z in unique_classes:
        mean_per_class.append(np.mean(data[labels == value_of_z], axis=0))
        var_per_class.append(np.var(data[labels == value_of_z], axis=0))
        
    numerator = np.var(np.array(mean_per_class), axis=0)
    demunerator = np.mean(np.array(var_per_class), axis=0)
    return numerator / demunerator

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

    # SNR.

    labels_256 = np.load("data/02_labels_p.npy")
    print(labels_256.shape)
    print(labels_256[0])

    snr_em = signal_to_noise(data_em, labels_256)
    snr_power = signal_to_noise(data_power, labels_256)
    plt.plot(snr_em, label="EM")
    plt.plot(snr_power, label="Power")
    plt.legend()
    plt.xlabel("Sample")
    plt.ylabel("Signal to noise ratio")
    plt.title("Signal to noise ratio (256 classes)")
    plt.show()

# * Experience #3

def bit_decomp(x, with_intercept=True):
    bits = np.unpackbits(x).reshape((x.shape[0], 8))
    if with_intercept:
        return np.c_[bits, np.ones(x.shape[0])]
    else:
        return bits

def least_square_regression(X, y):
    """
    Implementation of linear regression using the least square method
    """
    if y.ndim == 2:
        y = y[np.newaxis, :]
    scores = []
    sols = []
    ss_tot = np.sum((X - np.mean(X, axis=0)) ** 2, axis=0)
    for target in y:
        r = np.linalg.lstsq(target, X, rcond=None)
        sols.append(r[0])
        ss_res = r[1]
        if ss_res.shape == (0,):
            raise ValueError(
                "unable to find a least-square solution. This is often caused "
                "by the target variable having colinear entries"
            )
        score = 1 - ss_res / ss_tot
        scores.append(score)
    return np.array(scores), np.array(sols)

def expe3():
    data_em = np.load("data/02_data_em.npy")
    label_z = np.load("data/02_labels_z.npy")
    print(label_z.shape)
    label_z_bits = bit_decomp(label_z)
    for i in range(4):
        print(label_z[i])
        print(f"0x{label_z[i]:02x} ->", label_z_bits[i])

    scores, models = least_square_regression(data_em, label_z_bits)
    print("scores:", scores.shape)
    print("models:", models.shape)

    plt.plot(scores[0])
    plt.ylabel("Coefficient of determination")
    plt.xlabel("Sample index")
    plt.show()

    best_loc = np.argsort(scores[0])[::-1]
    print("top 10:", best_loc[:10])
    m = models[0].T
    for i in range(1):
        coeffs = m[best_loc[i]]
        plt.bar(np.arange(8), coeffs[:-1], alpha=0.5)
    plt.ylabel("magnitude")
    plt.xlabel("Coefficient number")
    plt.show()

# * Experience 4

def expe4():
    pass

# * Main

if __name__ == "__main__":
    # expe1()
    # expe2()
    # expe3()
    expe4()
