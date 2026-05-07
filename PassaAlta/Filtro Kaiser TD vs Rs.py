import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import pandas as pd
import os

# ============================================
# PARÂMETROS FIXOS DO FILTRO
# ============================================
fs = 8000                    # frequência de amostragem (Hz)
fc_rad = 310                # frequência de corte (rad/s)
fc_hz = fc_rad / (2 * np.pi) # ≈ 17.51 Hz
fs_rad = 470            # frequência de rejeição (rad/s)
fs_hz = fs_rad / (2 * np.pi) # ≈ 77.99 Hz

# Banda de transição normalizada (constante)
omega_c_norm = fc_rad / fs
omega_s_norm = fs_rad / fs
delta_omega = omega_s_norm - omega_c_norm   # ≈ 0.0475 rad/amostra

# Frequências de interesse (rad/s -> Hz)
f_desejada_rad = 500.0
f_indesejada_rad = 100.0
f_desejada_hz = f_desejada_rad / (2 * np.pi)   # ≈ 79.5775 Hz
f_indesejada_hz = f_indesejada_rad / (2 * np.pi) # ≈ 15.9155 Hz

# ============================================
# FUNÇÃO PARA CALCULAR β DA JANELA DE KAISER
# ============================================
def calcular_beta_kaiser(Rs):
    if Rs < 21:
        return 0.0
    elif 21 <= Rs <= 50:
        return 0.5842 * (Rs - 21)**0.4 + 0.07886 * (Rs - 21)
    else:
        return 0.1102 * (Rs - 8.7)

# ============================================
# FUNÇÃO PARA CALCULAR A ORDEM M (PAR, TIPO I)
# ============================================
def calcular_ordem(Rs, delta_omega):
    if Rs >= 8:
        M_est = (Rs - 8) / (2.285 * delta_omega)
    else:
        M_est = 0
    M = int(np.ceil(M_est))
    if M % 2 != 0:
        M += 1
    return M

# ============================================
# FUNÇÃO PARA CALCULAR TD TEÓRICA (E ORDEM)
# ============================================
def calcular_TD_e_ordem(Rs, delta_omega, fs, fc_hz, f_desejada_hz, f_indesejada_hz):
    M = calcular_ordem(Rs, delta_omega)
    beta = calcular_beta_kaiser(Rs)
    
    # CORREÇÃO: usar fc_hz (em Hz) diretamente, pois o argumento 'fs' é fornecido
    coeffs = signal.firwin(
        numtaps=M + 1,
        cutoff=fc_hz,               # <--- corrigido: agora em Hz (17.51 Hz)
        window=('kaiser', beta),
        fs=fs,
        pass_zero='highpass'
    )
    
    frequencias, resposta = signal.freqz(coeffs, worN=4096, fs=fs)
    
    idx_d = np.argmin(np.abs(frequencias - f_desejada_hz))
    idx_i = np.argmin(np.abs(frequencias - f_indesejada_hz))
    ganho_d = np.abs(resposta[idx_d])
    ganho_i = np.abs(resposta[idx_i])
    
    TD = (ganho_i / ganho_d) * 100.0
    return TD, M

# ============================================
# LOOP PARA VÁRIOS VALORES DE Rs
# ============================================
Rs_range = np.arange(10, 61, 1)   # de 10 dB a 60 dB, passo 1 dB
TD_values = []
M_values = []

print("Calculando TD teórica para Rs variando de 10 a 60 dB...")
for Rs in Rs_range:
    td, m = calcular_TD_e_ordem(Rs, delta_omega, fs, fc_hz, f_desejada_hz, f_indesejada_hz)
    TD_values.append(td)
    M_values.append(m)
    print(f"Rs = {Rs:2d} dB -> TD = {td:6.4f}%, Ordem M = {m:3d}")

# ============================================
# SALVAR DADOS EM ARQUIVO EXCEL
# ============================================
# Diretório de destino
output_dir = r"C:\Users\erick\Desktop\Projeto\Filtro passa alta"
output_file = os.path.join(output_dir, "resultados_filtro_kaiser.xlsx")

# Criar DataFrame com as três colunas
df = pd.DataFrame({
    'Rs (dB)': Rs_range,
    'TD (%)': TD_values,
    'Ordem M': M_values
})

# Garantir que o diretório existe; se não, criar
os.makedirs(output_dir, exist_ok=True)

# Salvar arquivo Excel
df.to_excel(output_file, index=False, sheet_name='Dados')
print(f"\nArquivo Excel salvo em: {output_file}")

# ============================================
# GRÁFICO APENAS DA TAXA DE DISTORÇÃO
# ============================================
plt.figure(figsize=(10, 6))
plt.plot(Rs_range, TD_values, 'b-o', linewidth=2, markersize=4, label='TD (%)')
plt.axhline(y=2.0, color='r', linestyle='--', label='Limite de 2%')
plt.xlabel('Atenuação na banda de rejeição - Rs (dB)')
plt.ylabel('Taxa de Distorção Teórica - TD (%)')
plt.title('Variação da TD com Rs (filtro FIR Kaiser, fc=110 rad/s, fs=490 rad/s)')
plt.grid(True, alpha=0.3)
plt.legend()
plt.ylim(bottom=0)
plt.tight_layout()
plt.show()

# ============================================
# INFORMAÇÃO ADICIONAL
# ============================================
atende = [td < 2.0 for td in TD_values]
if any(atende):
    idx_first = np.where(atende)[0][0]
    Rs_min = Rs_range[idx_first]
    M_min = M_values[idx_first]
    print(f"\nPara Rs >= {Rs_min} dB, a TD teórica é inferior a 2%. Neste ponto, M = {M_min}.")
else:
    print("\nNenhum Rs testado atingiu TD < 2% (intervalo 10-60 dB).")