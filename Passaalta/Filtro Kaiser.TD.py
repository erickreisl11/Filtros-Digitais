import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# ============================================
# PARÂMETROS DO PROJETO
# ============================================
fs = 8000                    # frequência de amostragem (Hz)
fc_rad = 110                 # frequência de corte (rad/s)
fc_hz = fc_rad / (2 * np.pi) # ≈ 17,51 Hz
fs_rad = 490                 # frequência de rejeição (rad/s)
fs_hz = fs_rad / (2 * np.pi) # ≈ 77,99 Hz

# Especificações
Rs = 19             # atenuação na banda de rejeição (dB)
delta_s = 10**(-Rs/20)       # δs = 0,1259

# Cálculo da banda de transição normalizada
omega_c_norm = (fc_rad / (2 * np.pi * fs)) * 2 * np.pi  # = fc_rad / fs
omega_s_norm = (fs_rad / (2 * np.pi * fs)) * 2 * np.pi  # = fs_rad / fs
delta_omega = omega_s_norm - omega_c_norm

print("=" * 50)
print("ESPECIFICAÇÕES DO FILTRO")
print("=" * 50)
print(f"Frequência de amostragem: {fs} Hz")
print(f"Frequência de corte: {fc_rad:.1f} rad/s ({fc_hz:.2f} Hz)")
print(f"Frequência de rejeição: {fs_rad:.1f} rad/s ({fs_hz:.2f} Hz)")
print(f"Banda de transição (Δω): {delta_omega:.6f} rad/amostra")
print(f"Atenuação desejada (Rs): {Rs} dB")
print(f"δs = {delta_s:.4f}")

# ============================================
# CÁLCULO DA ORDEM DO FILTRO
# ============================================
if Rs >= 8:
    M_estimado = (Rs - 8) / (2.285 * delta_omega)
else:
    M_estimado = 0

print("\n" + "=" * 50)
print("CÁLCULO DA ORDEM")
print("=" * 50)
print(f"M estimado = ({Rs} - 8) / (2.285 × {delta_omega:.6f})")
print(f"M estimado = {1:.3f} / {2.285 * delta_omega:.6f}")
print(f"M estimado = {M_estimado:.2f}")

# Arredondar para o próximo inteiro par (filtro Tipo I)
M = int(np.ceil(M_estimado))
if M % 2 != 0:
    M += 1

print(f"\nOrdem adotada (par, Tipo I): M = {M}")
print(f"Número de coeficientes: {M + 1}")

# ============================================
# PARÂMETRO β DA JANELA DE KAISER
# ============================================
def calcular_beta_kaiser(Rs):
    if Rs < 21:
        return 0.0
    elif 21 <= Rs <= 50:
        return 0.5842 * (Rs - 21)**0.4 + 0.07886 * (Rs - 21)
    else:
        return 0.1102 * (Rs - 8.7)

beta = calcular_beta_kaiser(Rs)

print(f"\nParâmetro β (Kaiser): {beta:.4f}")
print(f"Rs = {Rs} dB → {'β = 0' if Rs < 21 else 'β calculado'}")

# ============================================
# PROJETO DO FILTRO FIR (Janela de Kaiser)
# ============================================
f_cutoff_norm = fc_hz / (fs / 2)

coeffs = signal.firwin(
    numtaps=M + 1,
    cutoff=f_cutoff_norm,
    window=('kaiser', beta),
    fs=fs,
    pass_zero='lowpass'
)

# ============================================
# EXIBIR COEFICIENTES (primeiros e últimos)
# ============================================
print("\n" + "=" * 60)
print("COEFICIENTES DO FILTRO FIR")
print("=" * 60)
print(f"Total de {len(coeffs)} coeficientes:\n")
print("coeffs = [", end="")
for i, c in enumerate(coeffs):
    if i == len(coeffs) - 1:
        print(f"{c:.6f}]")
    else:
        print(f"{c:.6f}, ", end="")

# ============================================
# RESPOSTA EM FREQUÊNCIA TEÓRICA (para comparação)
# ============================================
frequencias, resposta = signal.freqz(coeffs, worN=4096, fs=fs)
magnitude_db = 20 * np.log10(np.abs(resposta) + 1e-12)

# ============================================
# FREQUÊNCIAS DE INTERESSE (rad/s → Hz)
# ============================================
f_desejada_rad = 100.0      # rad/s
f_indesejada_rad = 500.0    # rad/s
f_desejada_hz = f_desejada_rad / (2 * np.pi)   # ≈ 15.9155 Hz
f_indesejada_hz = f_indesejada_rad / (2 * np.pi) # ≈ 79.5775 Hz

# --- Cálculo teórico da TD (via resposta em frequência) ---
idx_desejada = np.argmin(np.abs(frequencias - f_desejada_hz))
idx_indesejada = np.argmin(np.abs(frequencias - f_indesejada_hz))
ganho_desejado_teor = np.abs(resposta[idx_desejada])
ganho_indesejado_teor = np.abs(resposta[idx_indesejada])
TD_teor = (ganho_indesejado_teor / ganho_desejado_teor) * 100.0

# ============================================
# SIMULAÇÃO TEMPORAL (ESTILO MATLAB)
# ============================================
# Duração do sinal: 0.2 segundos (mesmo do MATLAB)
duracao = 0.2   # segundos
T = 1 / fs
N = int(duracao * fs) + 1   # +1 para incluir t=0.2 (como no MATLAB: 0:T:0.2)
t = np.linspace(0, duracao, N)

# Sinal de entrada: cos(100*t) + cos(500*t)  (t em segundos)
x = np.cos(f_desejada_rad * t) + np.cos(f_indesejada_rad * t)

# Filtrar
y = signal.lfilter(coeffs, 1.0, x)

# FFT do sinal filtrado
Y_fft = np.fft.fft(y)
Y_mag = np.abs(Y_fft) / N   # normalização igual ao MATLAB (divide por N)

# Vetor de frequências positivas (apenas metade)
f_pos = np.fft.fftfreq(N, T)[:N // 2]
Y_pos = Y_mag[:N // 2]

# Encontrar índices das frequências desejada e indesejada no vetor f_pos
idx_d_sim = np.argmin(np.abs(f_pos - f_desejada_hz))
idx_i_sim = np.argmin(np.abs(f_pos - f_indesejada_hz))

Yd_sim = Y_pos[idx_d_sim]
Yi_sim = Y_pos[idx_i_sim]
TD_sim = (Yi_sim / Yd_sim) * 100.0

# ============================================
# EXIBIÇÃO DOS RESULTADOS
# ============================================
print("\n" + "=" * 60)
print("TAXA DE DISTORÇÃO (TD) – COMPARAÇÃO")
print("=" * 60)
print(f"Frequência desejada: {f_desejada_rad} rad/s ({f_desejada_hz:.4f} Hz)")
print(f"Frequência indesejada: {f_indesejada_rad} rad/s ({f_indesejada_hz:.4f} Hz)\n")

print("--- Método teórico (freqz) ---")
print(f"Ganho na desejada: {ganho_desejado_teor:.6f} (linear) -> {20*np.log10(ganho_desejado_teor):.2f} dB")
print(f"Ganho na indesejada: {ganho_indesejado_teor:.6f} (linear) -> {20*np.log10(ganho_indesejado_teor):.2f} dB")
print(f"TD = {TD_teor:.4f}%")
print(f"Critério (TD < 2%): {'ATENDE' if TD_teor < 2 else 'NÃO ATENDE'}")

print("\n--- Método via simulação temporal (FFT do sinal filtrado) ---")
print(f"Magnitude |Y(fd)| (FFT): {Yd_sim:.6f}")
print(f"Magnitude |Y(fi)| (FFT): {Yi_sim:.6f}")
print(f"TD = {TD_sim:.4f}%")
print(f"Critério (TD < 2%): {'ATENDE' if TD_sim < 2 else 'NÃO ATENDE'}")

# ============================================
# GRÁFICOS (SINAIS NO TEMPO E ESPECTROS)
# ============================================
# 1) Sinais no tempo
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(t, x, 'b', linewidth=1)
plt.title('Sinal de Entrada x(t) = cos(100t) + cos(500t)')
plt.xlabel('Tempo (s)')
plt.ylabel('Amplitude')
plt.grid(True)

plt.subplot(2, 2, 2)
plt.plot(t, y, 'r', linewidth=1)
plt.title('Sinal Filtrado y(t)')
plt.xlabel('Tempo (s)')
plt.ylabel('Amplitude')
plt.grid(True)

# 2) Espectros (amplitude vs frequência)
# Espectro de entrada (FFT)
X_fft = np.fft.fft(x)
X_mag = np.abs(X_fft) / N
X_pos = X_mag[:N//2]

plt.subplot(2, 2, 3)
plt.plot(f_pos, X_pos, 'b', linewidth=1)
plt.title('Espectro do Sinal de Entrada')
plt.xlabel('Frequência (Hz)')
plt.ylabel('|X(f)|')
plt.xlim(0, 150)
plt.grid(True)

plt.subplot(2, 2, 4)
plt.plot(f_pos, Y_pos, 'r', linewidth=1)
plt.title('Espectro do Sinal Filtrado')
plt.xlabel('Frequência (Hz)')
plt.ylabel('|Y(f)|')
plt.xlim(0, 150)
plt.grid(True)

plt.tight_layout()
plt.show()

# ============================================
# GRÁFICO DA RESPOSTA EM FREQUÊNCIA (igual ao original)
# ============================================
mask = frequencias <= 1000
frequencias_filtradas = frequencias[mask]
magnitude_filtrada = magnitude_db[mask]

plt.figure(figsize=(10, 6))
plt.plot(frequencias_filtradas, magnitude_filtrada, 'b-', linewidth=1.5)
plt.axhline(y=-Rs, color='r', linestyle='--', label=f'Atenuação desejada = {Rs} dB')
plt.axvline(x=fc_hz, color='g', linestyle='--', label=f'Fc = {fc_hz:.1f} Hz')
plt.axvline(x=fs_hz, color='orange', linestyle='--', label=f'Fs = {fs_hz:.1f} Hz')
plt.axvline(x=f_desejada_hz, color='magenta', linestyle=':', label=f'Desejada = {f_desejada_hz:.2f} Hz')
plt.axvline(x=f_indesejada_hz, color='brown', linestyle=':', label=f'Indesejada = {f_indesejada_hz:.2f} Hz')
plt.title(f'Resposta em Frequência - Filtro FIR (Janela Kaiser, β={beta:.3f}, M={M})')
plt.ylabel('Magnitude (dB)')
plt.xlabel('Frequência (Hz)')
plt.grid(True, alpha=0.3)
plt.xlim(0, 1000)
plt.ylim(-50, 5)
plt.legend()
plt.tight_layout()
plt.show()

# ============================================
# TABELA DE GANHOS (para comparação)
# ============================================
freqs_teste = [15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 120, 150]

print("\n" + "=" * 60)
print("GANHO EM FREQUÊNCIAS ESPECÍFICAS (teórico via freqz)")
print("=" * 60)
print(f"{'Freq (Hz)':>10} | {'Ganho (linear)':>15} | {'Ganho (dB)':>12}")
print("-" * 60)

for f in freqs_teste:
    idx = np.argmin(np.abs(frequencias - f))
    ganho_linear = np.abs(resposta[idx])
    ganho_db = magnitude_db[idx]
    print(f"{f:>10.1f} | {ganho_linear:>15.6f} | {ganho_db:>12.2f}")