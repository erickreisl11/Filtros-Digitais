import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import pandas as pd
import os
from itertools import product

# ============================================
# PARÂMETROS FIXOS DO FILTRO
# ============================================
fs = 8000                    # frequência de amostragem (Hz)

# Frequências de interesse (rad/s -> Hz)
f_desejada_rad = 500.0
f_indesejada_rad = 100.0
f_desejada_hz = f_desejada_rad / (2 * np.pi)   # ≈ 79.5775 Hz
f_indesejada_hz = f_indesejada_rad / (2 * np.pi) # ≈ 15.9155 Hz

# ============================================
# DEFINIÇÃO DOS RANGES PARA VARREDURA
# ============================================
# fc_rad deve ser > f_desejada_rad? NÃO! Para passa-altas, fc deve estar ENTRE as frequências
# f_indesejada (100) < fc < f_desejada (500)
fc_rad_range = np.arange(110, 491, 20)  # de 110 a 490 rad/s, passo 20
fs_rad_range = np.arange(50, 501, 20)    # de 50 a 500 rad/s, passo 20

# Rs range
Rs_range = np.arange(10, 61, 1)  # de 10 a 60 dB

print("=" * 80)
print("VARREDURA DE PARÂMETROS PARA FILTRO PASSA-ALTAS")
print("=" * 80)
print(f"Frequência desejada (preservar): {f_desejada_rad} rad/s ({f_desejada_hz:.2f} Hz)")
print(f"Frequência indesejada (rejeitar): {f_indesejada_rad} rad/s ({f_indesejada_hz:.2f} Hz)")
print(f"Range de fc_rad: {fc_rad_range[0]} a {fc_rad_range[-1]} rad/s")
print(f"Range de fs_rad: {fs_rad_range[0]} a {fs_rad_range[-1]} rad/s")
print(f"Range de Rs: {Rs_range[0]} a {Rs_range[-1]} dB")
print("=" * 80)

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
    
    # Se a ordem estimada já for maior que 150, nem tenta projetar
    if M >= 150:
        return 100.0, M  # Retorna TD alta para descartar
    
    beta = calcular_beta_kaiser(Rs)
    
    try:
        coeffs = signal.firwin(
            numtaps=M + 1,
            cutoff=fc_hz,
            window=('kaiser', beta),
            fs=fs,
            pass_zero='highpass'
        )
        
        frequencias, resposta = signal.freqz(coeffs, worN=4096, fs=fs)
        
        idx_d = np.argmin(np.abs(frequencias - f_desejada_hz))
        idx_i = np.argmin(np.abs(frequencias - f_indesejada_hz))
        ganho_d = np.abs(resposta[idx_d])
        ganho_i = np.abs(resposta[idx_i])
        
        if ganho_d < 1e-10:  # Evitar divisão por zero
            TD = 100.0
        else:
            TD = (ganho_i / ganho_d) * 100.0
            
        return TD, M
    except Exception as e:
        return 100.0, M

# ============================================
# VARREDURA COMPLETA
# ============================================
resultados = []

print("\nIniciando varredura...")
print("-" * 80)

total_combinacoes = len(fc_rad_range) * len(fs_rad_range) * len(Rs_range)
contador = 0

for fc_rad in fc_rad_range:
    fc_hz = fc_rad / (2 * np.pi)
    
    for fs_rad in fs_rad_range:
        # Verificar se fs_rad > fc_rad (banda de transição positiva)
        if fs_rad <= fc_rad:
            continue
            
        fs_hz = fs_rad / (2 * np.pi)
        
        # Calcular banda de transição normalizada
        omega_c_norm = fc_rad / fs
        omega_s_norm = fs_rad / fs
        delta_omega = omega_s_norm - omega_c_norm
        
        # Verificar se a banda de transição é positiva
        if delta_omega <= 0:
            continue
        
        # Para cada Rs
        for Rs in Rs_range:
            contador += 1
            if contador % 100 == 0:
                print(f"Progresso: {contador}/{total_combinacoes} combinações testadas...")
            
            td, m = calcular_TD_e_ordem(Rs, delta_omega, fs, fc_hz, f_desejada_hz, f_indesejada_hz)
            
            # Armazenar se atender aos critérios
            if td < 2.0 and m < 150:
                resultados.append({
                    'Rs (dB)': Rs,
                    'fc_rad (rad/s)': fc_rad,
                    'fc_hz (Hz)': fc_hz,
                    'fs_rad (rad/s)': fs_rad,
                    'fs_hz (Hz)': fs_hz,
                    'TD (%)': td,
                    'Ordem M': m,
                    'delta_omega': delta_omega,
                    'Beta': calcular_beta_kaiser(Rs)
                })

print("-" * 80)
print(f"Varredura concluída! Total de combinações: {contador}")
print(f"Combinações que atendem (TD < 2% e M < 150): {len(resultados)}")

# ============================================
# ORDENAR RESULTADOS POR TD (MENOR TD PRIMEIRO)
# ============================================
if resultados:
    resultados.sort(key=lambda x: x['TD (%)'])
    
    # ============================================
    # EXIBIR MELHORES RESULTADOS
    # ============================================
    print("\n" + "=" * 80)
    print("MELHORES RESULTADOS (menor TD):")
    print("=" * 80)
    
    # Mostrar top 20 resultados
    for i, res in enumerate(resultados[:20], 1):
        print(f"\n{i}. Rs = {res['Rs (dB)']:2d} dB | fc = {res['fc_rad (rad/s)']:3.0f} rad/s ({res['fc_hz (Hz)']:.2f} Hz) | fs = {res['fs_rad (rad/s)']:3.0f} rad/s ({res['fs_hz (Hz)']:.2f} Hz)")
        print(f"   TD = {res['TD (%)']:.6f}% | M = {res['Ordem M']:3d} | β = {res['Beta']:.3f} | Δω = {res['delta_omega']:.5f}")
    
    # ============================================
    # RESULTADO COM MENOR ORDEM
    # ============================================
    melhor_ordem = min(resultados, key=lambda x: x['Ordem M'])
    print("\n" + "=" * 80)
    print("FILTRO COM MENOR ORDEM (M mais baixo):")
    print("=" * 80)
    print(f"Rs = {melhor_ordem['Rs (dB)']} dB")
    print(f"fc_rad = {melhor_ordem['fc_rad (rad/s)']} rad/s ({melhor_ordem['fc_hz (Hz)']:.2f} Hz)")
    print(f"fs_rad = {melhor_ordem['fs_rad (rad/s)']} rad/s ({melhor_ordem['fs_hz (Hz)']:.2f} Hz)")
    print(f"TD = {melhor_ordem['TD (%)']:.6f}%")
    print(f"Ordem M = {melhor_ordem['Ordem M']}")
    print(f"Beta = {melhor_ordem['Beta']:.3f}")
    
    # ============================================
    # RESULTADO COM MENOR TD
    # ============================================
    melhor_td = resultados[0]  # Já ordenado por TD
    print("\n" + "=" * 80)
    print("FILTRO COM MENOR TD (melhor rejeição):")
    print("=" * 80)
    print(f"Rs = {melhor_td['Rs (dB)']} dB")
    print(f"fc_rad = {melhor_td['fc_rad (rad/s)']} rad/s ({melhor_td['fc_hz (Hz)']:.2f} Hz)")
    print(f"fs_rad = {melhor_td['fs_rad (rad/s)']} rad/s ({melhor_td['fs_hz (Hz)']:.2f} Hz)")
    print(f"TD = {melhor_td['TD (%)']:.6f}%")
    print(f"Ordem M = {melhor_td['Ordem M']}")
    print(f"Beta = {melhor_td['Beta']:.3f}")
    
    # ============================================
    # SALVAR RESULTADOS EM EXCEL
    # ============================================
    output_dir = r"C:\Users\erick\Desktop\Projeto\Filtro passa alta"
    output_file = os.path.join(output_dir, "otimizacao_filtro_passa_alta.xlsx")
    
    df = pd.DataFrame(resultados)
    
    # Ordenar por TD
    df = df.sort_values('TD (%)')
    
    os.makedirs(output_dir, exist_ok=True)
    df.to_excel(output_file, index=False, sheet_name='Melhores_Resultados')
    print(f"\n✓ Todos os resultados salvos em: {output_file}")
    
    # ============================================
    # GRÁFICO: TD vs Ordem para diferentes fc
    # ============================================
    plt.figure(figsize=(12, 8))
    
    # Separar por faixas de fc_rad
    fc_faixas = {
        'Baixo (110-150)': (110, 150),
        'Médio (151-300)': (151, 300),
        'Alto (301-490)': (301, 490)
    }
    
    cores = {'Baixo (110-150)': 'blue', 'Médio (151-300)': 'green', 'Alto (301-490)': 'red'}
    
    for faixa_nome, (f_min, f_max) in fc_faixas.items():
        faixa_resultados = [r for r in resultados if f_min <= r['fc_rad (rad/s)'] <= f_max]
        if faixa_resultados:
            tds = [r['TD (%)'] for r in faixa_resultados]
            ordens = [r['Ordem M'] for r in faixa_resultados]
            plt.scatter(ordens, tds, c=cores[faixa_nome], alpha=0.6, label=faixa_nome, s=30)
    
    plt.axhline(y=2.0, color='r', linestyle='--', linewidth=2, label='Limite TD = 2%')
    plt.axvline(x=150, color='gray', linestyle='--', linewidth=2, label='Limite M = 150')
    plt.xlabel('Ordem do Filtro (M)', fontsize=12)
    plt.ylabel('Taxa de Distorção TD (%)', fontsize=12)
    plt.title('Otimização do Filtro Passa-Altas: TD vs Ordem', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # ============================================
    # GRÁFICO DE CALOR: TD em função de fc e fs (para melhor Rs)
    # ============================================
    # Encontrar o melhor Rs para cada combinação fc/fs
    dados_calor = {}
    for res in resultados:
        chave = (res['fc_rad (rad/s)'], res['fs_rad (rad/s)'])
        if chave not in dados_calor or res['TD (%)'] < dados_calor[chave]['TD (%)']:
            dados_calor[chave] = res
    
    if dados_calor:
        fc_vals = sorted(set([k[0] for k in dados_calor.keys()]))
        fs_vals = sorted(set([k[1] for k in dados_calor.keys()]))
        
        td_matrix = np.zeros((len(fc_vals), len(fs_vals)))
        for i, fc in enumerate(fc_vals):
            for j, fs_val in enumerate(fs_vals):
                chave = (fc, fs_val)
                if chave in dados_calor:
                    td_matrix[i, j] = dados_calor[chave]['TD (%)']
                else:
                    td_matrix[i, j] = np.nan
        
        plt.figure(figsize=(10, 8))
        plt.imshow(td_matrix, aspect='auto', origin='lower', 
                   extent=[fs_vals[0], fs_vals[-1], fc_vals[0], fc_vals[-1]],
                   cmap='viridis_r')
        plt.colorbar(label='TD (%)')
        plt.xlabel('fs_rad (rad/s) - Frequência de rejeição', fontsize=12)
        plt.ylabel('fc_rad (rad/s) - Frequência de corte', fontsize=12)
        plt.title('Mapa de Calor: Melhor TD para cada combinação fc/fs', fontsize=14)
        plt.tight_layout()
        plt.show()

else:
    print("\n" + "=" * 80)
    print("NENHUMA COMBINAÇÃO ATENDEU AOS CRITÉRIOS!")
    print("=" * 80)
    print("Sugestões:")
    print("1. Aumentar o range de fc_rad e fs_rad")
    print("2. Aumentar o limite de ordem M (atualmente 150)")
    print("3. Verificar se as frequências desejada/indesejada estão corretas")
    print("4. Considerar um filtro de ordem mais alta")

# ============================================
# RESUMO FINAL
# ============================================
print("\n" + "=" * 80)
print("RESUMO DA OTIMIZAÇÃO")
print("=" * 80)
print(f"Total de combinações analisadas: {contador}")
print(f"Soluções encontradas (TD<2% e M<150): {len(resultados)}")
if resultados:
    print(f"Melhor TD: {resultados[0]['TD (%)']:.6f}% com M={resultados[0]['Ordem M']}")
    print(f"Menor ordem: {melhor_ordem['Ordem M']} com TD={melhor_ordem['TD (%)']:.6f}%")
print("=" * 80)