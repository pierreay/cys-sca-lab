#!/usr/bin/env python3

import collections
import tqdm
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (10, 4)
plt.rcParams["backend"] = "qtagg" # Needs pyqt6 or pyqt5

# * Constants

AES_SBOX = np.array([
        0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5,
        0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
        0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0,
        0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
        0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc,
        0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
        0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a,
        0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
        0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0,
        0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
        0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b,
        0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
        0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85,
        0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
        0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5,
        0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
        0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17,
        0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
        0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88,
        0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
        0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c,
        0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
        0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9,
        0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
        0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6,
        0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
        0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e,
        0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
        0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94,
        0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
        0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68,
        0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16], dtype=np.uint8)

def aes_sbox(x):
    return AES_SBOX[x]

NIST_KEY = np.array([43, 126,  21,  22,  40, 174, 210,
                     166, 171, 247,  21, 136,   9, 207,
                     79,  60], dtype=np.uint8)

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
