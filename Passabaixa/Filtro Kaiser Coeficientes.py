import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# ============================================
# PARÂMETROS DO PROJETO
# ============================================
fs = 8000                    # frequência de amostragem (Hz)
fc_rad = 110                 # frequência de corte (rad/s)
fc_hz = fc_rad / (2 * np.pi) # ≈ 23,87 Hz
fs_rad = 490                 # frequência de rejeição (rad/s)
fs_hz = fs_rad / (2 * np.pi) # ≈ 71,62 Hz

# Especificações
Rs = 19                    # atenuação na banda de rejeição (dB)
delta_s = 10**(-Rs/20)       # δs = 0,3548

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
print(f"M estimado = {Rs - 8:.3f} / {2.285 * delta_omega:.6f}")
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
    else:  # Rs > 50
        return 0.1102 * (Rs - 8.7)

beta = calcular_beta_kaiser(Rs)

print(f"\nParâmetro β (Kaiser): {beta:.4f}")
print(f"Rs = {Rs} dB → {'β = 0' if Rs < 21 else 'β calculado'}")

# ============================================
# PROJETO DO FILTRO
# ============================================
coeffs = signal.firwin(
    numtaps=M + 1,
    cutoff=fc_hz,
    window=('kaiser', beta),
    fs=fs,
    pass_zero='lowpass'
)

# ============================================
# EXIBIR COEFICIENTES
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
# RESPOSTA EM FREQUÊNCIA
# ============================================
frequencias, resposta = signal.freqz(coeffs, worN=4096, fs=fs)

# Converter magnitude para dB
magnitude_db = 20 * np.log10(np.abs(resposta) + 1e-12)

# Filtrar faixa de 0 a 1000 Hz
mask = frequencias <= 1000
frequencias_filtradas = frequencias[mask]
magnitude_filtrada = magnitude_db[mask]

# ============================================
# PLOT - MAGNITUDE
# ============================================
plt.figure(figsize=(10, 6))
plt.plot(frequencias_filtradas, magnitude_filtrada, 'b-', linewidth=1.5)
plt.axhline(y=-Rs, color='r', linestyle='--', label=f'Atenuação desejada = {Rs} dB')
plt.axvline(x=fc_hz, color='g', linestyle='--', label=f'Fc = {fc_hz:.1f} Hz')
plt.axvline(x=fs_hz, color='orange', linestyle='--', label=f'Fs = {fs_hz:.1f} Hz')
plt.title(f'Resposta em Frequência - Filtro FIR (Janela Kaiser, β={beta:.4f}, M={M})')
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
print("GANHO EM FREQUÊNCIAS ESPECÍFICAS")
print("=" * 60)
print(f"{'Freq (Hz)':>10} | {'Ganho (linear)':>15} | {'Ganho (dB)':>12}")
print("-" * 60)

for f in freqs_teste:
    idx = np.argmin(np.abs(frequencias - f))
    ganho_linear = np.abs(resposta[idx])
    ganho_db = magnitude_db[idx]
    print(f"{f:>10.1f} | {ganho_linear:>15.6f} | {ganho_db:>12.2f}")

# ============================================
# CÁLCULO DA TAXA DE DISTORÇÃO (TD)
# ============================================
# Frequências de interesse (convertidas de rad/s para Hz)
f_desejada = 100 / (2 * np.pi)      # ≈ 15.9155 Hz
f_indesejada = 500 / (2 * np.pi)    # ≈ 79.5775 Hz

# Encontrar os índices mais próximos no vetor de frequências
idx_d = np.argmin(np.abs(frequencias - f_desejada))
idx_i = np.argmin(np.abs(frequencias - f_indesejada))

# Ganhos lineares do filtro nas duas frequências
H_desejada = np.abs(resposta[idx_d])
H_indesejada = np.abs(resposta[idx_i])

# Taxa de distorção percentual
TD = (H_indesejada / H_desejada) * 100

print("\n" + "=" * 60)
print("TAXA DE DISTORÇÃO (TD) DO FILTRO")
print("=" * 60)
print(f"Frequência desejada   : {f_desejada:.4f} Hz  → |H| = {H_desejada:.8f}")
print(f"Frequência indesejada : {f_indesejada:.4f} Hz → |H| = {H_indesejada:.8f}")
print(f"\nTD = (|H_indesejada| / |H_desejada|) × 100% = {TD:.6f}%")
if TD < 2:
    print("Resultado: FILTRO APROVADO (TD < 2%)")
else:
    print("Resultado: FILTRO REPROVADO (TD ≥ 2%)")
print("=" * 60)