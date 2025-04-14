import numpy as np
import matplotlib.pyplot as plt
from commpy.channelcoding.convcode import Trellis, conv_encode, viterbi_decode
import matplotlib.patches as patches
import textwrap

data = np.random.randint(0, 2, 100)
print("Generated Bit Stream", data)

trellis = Trellis(np.array([7]), np.array([[171, 133]]))

encoded_data = conv_encode(data, trellis)
print("Convolutionally Encoded Data", encoded_data)

bpsk_modulated = 2*encoded_data - 1
print("BPSK Modulated Signal:", bpsk_modulated)

t = np.arange(0, len(bpsk_modulated))

window1=plt.figure(figsize=(10,5))
plt.subplot(2,1,1)
plt.plot(t, bpsk_modulated, marker='o', linestyle='-', color='b')
window1.canvas.manager.set_window_title("BPSK Signals")
plt.title("BPSK Modulated Signal")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.grid()

def awgn_noise(signal, snr_db):
    snr_linear = 10**(snr_db/10)
    power_signal = np.mean(signal**2)
    power_noise = power_signal / snr_linear
    noise = np.sqrt(power_noise) * np.random.randn(len(signal))
    return signal + noise

bpsk_noisy = awgn_noise(bpsk_modulated, 10)
bpsk_received = (bpsk_noisy > 0).astype(int)
decoded_data = viterbi_decode(bpsk_received, trellis, tb_depth=5)
print("Decoded Data:", decoded_data)

plt.subplot(2,1,2)
plt.plot(bpsk_noisy)
plt.title("Noisy BPSK Signal")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.grid()
plt.tight_layout()

bpsk_demodulated = (bpsk_noisy > 0).astype(int)
print("Demodulated Data:", bpsk_demodulated)

bit_errors = np.sum(decoded_data[:len(data)] != data)
ber_corrected = bit_errors / len(data)
print(f"Hata Düzeltilmiş Bit Hata Orani (BER): {ber_corrected:.4f}")
snr_values = np.linspace(0, 10, 30)
ber_values = []
for snr in snr_values:
    bpsk_noisy_signal = awgn_noise(bpsk_modulated, snr)
    bpsk_received = (bpsk_noisy_signal > 0).astype(int)
    decoded_data = viterbi_decode(bpsk_received, trellis, tb_depth=5) 
    bit_errors = np.sum(decoded_data[:len(data)] != data)
    ber = bit_errors / len(data)
    ber_values.append(ber)

window2=plt.figure(figsize=(8, 5))
plt.semilogy(snr_values, ber_values, marker='s', linestyle='-', color='r', markersize=5, label="BPSK with FEC")
plt.xlabel("SNR (dB)")
plt.ylabel("BER (Bit Error Rate)")  
window2.canvas.manager.set_window_title("SNR Graph")
plt.title("BER - SNR Graph")
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend()

fig, ax = plt.subplots(figsize=(12, 7))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off") 

rect = patches.FancyBboxPatch((0.05, 0.05), 0.9, 0.9, boxstyle="round,pad=0.1",
                              edgecolor="black", facecolor="lightgray", alpha=0.6)
ax.add_patch(rect)

def format_array(arr, width=70):
    return "\n".join(textwrap.wrap(str(arr[:20]), width))
y_position = 0.90  
data_blocks = {
    "Generated Data": data ,
    "Convolutionally Encoded Data": encoded_data,
    "BPSK Modulated Signal": bpsk_modulated,
    "Demodulated Data": bpsk_demodulated,
    "Bit Error Rate (BER)": f"{ber_corrected:.4f}"
}

for title, content in data_blocks.items():
    ax.text(0.1, y_position, title + ":", fontsize=14, fontweight="bold", color="red", fontfamily="monospace")
    y_position -= 0.04  
    text_box = format_array(content)
    ax.text(0.1, y_position, text_box, fontsize=12, fontfamily="monospace",
            bbox={"facecolor": "white", "alpha": 0.8, "pad": 5})
    y_position -= 0.10

fig.canvas.manager.set_window_title("Communication System Results")
plt.title("Communication System Summary", fontsize=16, fontweight="bold")
plt.show()  