% ============================================
% PARÂMETROS FIXOS DO FILTRO
% ============================================
clc; clear; close all;

fs = 8000;                    % frequência de amostragem (Hz)
fc_rad = 110;                 % frequência de corte (rad/s)
fc_hz = fc_rad / (2 * pi);    % ? 17.51 Hz
fs_rad = 490;                 % frequência de rejeição (rad/s)
fs_hz = fs_rad / (2 * pi);    % ? 77.99 Hz

% Banda de transição normalizada (constante)
omega_c_norm = fc_rad / fs;
omega_s_norm = fs_rad / fs;
delta_omega = omega_s_norm - omega_c_norm;   % ? 0.0475 rad/amostra

% Frequências de interesse (rad/s -> Hz)
f_desejada_rad = 100.0;
f_indesejada_rad = 500.0;
f_desejada_hz = f_desejada_rad / (2 * pi);   % ? 15.9155 Hz
f_indesejada_hz = f_indesejada_rad / (2 * pi); % ? 79.5775 Hz

% ============================================
% LOOP PARA VÁRIOS VALORES DE Rs
% ============================================
Rs_range = 10:1:60;            % de 10 dB a 60 dB, passo 1 dB
nRs = length(Rs_range);
TD_values = zeros(1, nRs);
M_values = zeros(1, nRs);

fprintf('Calculando TD teórica para Rs variando de 10 a 60 dB...\n');
for i = 1:nRs
    Rs = Rs_range(i);
    [td, m] = calcular_TD_e_ordem(Rs, delta_omega, fs, fc_hz, ...
                                  f_desejada_hz, f_indesejada_hz);
    TD_values(i) = td;
    M_values(i) = m;
    fprintf('Rs = %2d dB -> TD = %6.4f%%, Ordem M = %3d\n', Rs, td, m);
end

% ============================================
% GRÁFICO APENAS DA TAXA DE DISTORÇÃO (SEM A ORDEM M)
% ============================================
figure('Position', [100 100 800 500]);
plot(Rs_range, TD_values, 'b-o', 'LineWidth', 2, 'MarkerSize', 4);
hold on;
yline(2.0, 'r--', 'LineWidth', 1.5);
xlabel('Atenuação na banda de rejeição - Rs (dB)');
ylabel('Taxa de Distorção Teórica - TD (%)');
title('Variação da TD com Rs (filtro FIR Kaiser, fc=110 rad/s, fs=490 rad/s)');
grid on;
legend('TD (%)', 'Limite de 2%', 'Location', 'best');
ylim([0 inf]);
hold off;

% ============================================
% INFORMAÇÃO ADICIONAL
% ============================================
atende = TD_values < 2.0;
if any(atende)
    idx_first = find(atende, 1, 'first');
    Rs_min = Rs_range(idx_first);
    M_min = M_values(idx_first);
    fprintf('\nPara Rs >= %d dB, a TD teórica é inferior a 2%%. Neste ponto, M = %d.\n', Rs_min, M_min);
else
    fprintf('\nNenhum Rs testado atingiu TD < 2%% (intervalo 10-60 dB).\n');
end

% ============================================
% FUNÇÕES LOCAIS
% ============================================

function beta = calcular_beta_kaiser(Rs)
    % Calcula o parâmetro beta da janela de Kaiser
    if Rs < 21
        beta = 0.0;
    elseif Rs >= 21 && Rs <= 50
        beta = 0.5842 * (Rs - 21)^0.4 + 0.07886 * (Rs - 21);
    else
        beta = 0.1102 * (Rs - 8.7);
    end
end

function M = calcular_ordem(Rs, delta_omega)
    % Calcula a ordem M (par, tipo I) com base na atenuação Rs
    if Rs >= 8
        M_est = (Rs - 8) / (2.285 * delta_omega);
    else
        M_est = 0;
    end
    M = ceil(M_est);
    if mod(M, 2) ~= 0
        M = M + 1;   % garante que M seja par
    end
end

function [TD, M] = calcular_TD_e_ordem(Rs, delta_omega, fs, fc_hz, f_desejada_hz, f_indesejada_hz)
    % Calcula a Taxa de Distorção (TD) e a ordem M para um dado Rs
    M = calcular_ordem(Rs, delta_omega);
    beta = calcular_beta_kaiser(Rs);
    
    % Frequência de corte normalizada (0 a 1, onde 1 corresponde a fs/2)
    f_cutoff_norm = fc_hz / (fs / 2);
    
    % Projeto do filtro FIR low-pass com janela de Kaiser
    % fir1(ordem, Wn, 'low', janela)
    ordem = M;                       % número de coeficientes = ordem+1
    janela = kaiser(ordem + 1, beta);
    b = fir1(ordem, f_cutoff_norm, 'low', janela, 'noscale');
    
    % Resposta em frequência (4096 pontos)
    [h, f] = freqz(b, 1, 4096, fs);
    H_mag = abs(h);
    
    % Encontra as frequências mais próximas das desejadas
    [~, idx_d] = min(abs(f - f_desejada_hz));
    [~, idx_i] = min(abs(f - f_indesejada_hz));
    ganho_d = H_mag(idx_d);
    ganho_i = H_mag(idx_i);
    
    % Taxa de Distorção (%)
    TD = (ganho_i / ganho_d) * 100.0;
end