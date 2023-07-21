#!/usr/bin/env python3

import collections
import tqdm
import numpy as np
import matplotlib.pyplot as plt
from lab_utils import aes_sbox, NIST_KEY
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

def sbox_output(key, plaintexts):
    return aes_sbox(key ^ plaintexts)

def expe4():
    data = np.load("data/03_aes_r0_data.npy")
    plaintexts = np.load("data/03_aes_r0_pts.npy")

    print("number of traces :", data.shape[0])
    print("number of samples:", data.shape[1])
    print("number of plaintexts:", plaintexts.shape[0])
    print("size of plaintexts (bytes):", plaintexts.shape[1])
    print("plaintexts subbytes #0:", plaintexts[:, 0].shape, plaintexts[:, 0])
    print("plaintexts subbytes #1:", plaintexts[:, 1].shape, plaintexts[:, 1])

    plt.plot(data[:10].T)
    plt.xlabel("Sample")
    plt.ylabel("Scope ADC value")
    plt.title("Example Power Trace")
    plt.show()

    z_byte_0 = sbox_output(0, plaintexts[:, 0])
    z_byte_1 = sbox_output(42, plaintexts[:, 1])
    print("sbox outputs for subbyte #0 with key guess of 0:", z_byte_0.shape, z_byte_0)    
    print("sbox outputs for subbyte #1: with key guess of 42", z_byte_1.shape, z_byte_1)

    target_byte = 0

    # Step 1: linear regression analysis
    z_bits = bit_decomp(sbox_output(NIST_KEY[target_byte], plaintexts[:, target_byte]))
    scores, models = least_square_regression(data, z_bits)

    # Step 2: visualize R2
    plt.plot(scores[0])
    plt.ylabel("Coefficient of determination")
    plt.xlabel("Sample index")
    plt.show()

    # Step 3: visualize model at the best location
    best_loc = np.argsort(scores[0])[::-1]
    print("top 10:", best_loc[:10])
    m = models[0].T
    for i in range(1):
        coeffs = m[best_loc[i]]
        plt.bar(np.arange(8), coeffs[:-1], alpha=0.5)
    plt.ylabel("magnitude")
    plt.xlabel("Coefficient number")
    plt.show()

    target_byte = 0

    r2_scores = np.zeros((256, data.shape[1]))
    for key_guess in tqdm.trange(256):
        z = bit_decomp(sbox_output(key_guess, plaintexts[:, target_byte]))
        scores, _ = least_square_regression(data, z)
        r2_scores[key_guess] = scores[0]
    plt.plot(r2_scores.T, color="blue", alpha=0.1)
    plt.plot(r2_scores[NIST_KEY[target_byte]], color="red", label="correct key", alpha=0.6)
    plt.legend()
    plt.xlabel("Sample Index")
    plt.ylabel("R2 value")
    plt.show()

    best_key = np.argmax(np.max(r2_scores, axis=1))
    print("best key:", best_key)
    print("correct key:", NIST_KEY[target_byte])

# * Main

if __name__ == "__main__":
    # expe1()
    # expe2()
    # expe3()
    expe4()
