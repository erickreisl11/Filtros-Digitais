% ============================================
% PARÂMETROS DO PROJETO
% ============================================
fs = 8000;                    % frequência de amostragem (Hz)
fc_rad = 110;                 % frequência de corte (rad/s)
fc_hz = fc_rad / (2 * pi);    % ? 23,87 Hz
fs_rad = 490;                 % frequência de rejeição (rad/s)
fs_hz = fs_rad / (2 * pi);    % ? 71,62 Hz

% Especificações
Rs = 19;                      % atenuação na banda de rejeição (dB)
delta_s = 10^(-Rs/20);        % ?s = 0,3548

% Cálculo da banda de transição normalizada
omega_c_norm = fc_rad / fs;
omega_s_norm = fs_rad / fs;
delta_omega = omega_s_norm - omega_c_norm;

fprintf('==================================================\n');
fprintf('ESPECIFICAÇÕES DO FILTRO\n');
fprintf('==================================================\n');
fprintf('Frequência de amostragem: %d Hz\n', fs);
fprintf('Frequência de corte: %.1f rad/s (%.2f Hz)\n', fc_rad, fc_hz);
fprintf('Frequência de rejeição: %.1f rad/s (%.2f Hz)\n', fs_rad, fs_hz);
fprintf('Banda de transição (??): %.6f rad/amostra\n', delta_omega);
fprintf('Atenuação desejada (Rs): %d dB\n', Rs);
fprintf('?s = %.4f\n', delta_s);

% ============================================
% CÁLCULO DA ORDEM DO FILTRO
% ============================================
if Rs >= 8
    M_estimado = (Rs - 8) / (2.285 * delta_omega);
else
    M_estimado = 0;
end

fprintf('\n==================================================\n');
fprintf('CÁLCULO DA ORDEM\n');
fprintf('==================================================\n');
fprintf('M estimado = (%d - 8) / (2.285 × %.6f)\n', Rs, delta_omega);
numerador = Rs - 8;
fprintf('M estimado = %.3f / %.6f\n', numerador, 2.285 * delta_omega);
fprintf('M estimado = %.2f\n', M_estimado);

% Arredondar para o próximo inteiro par (filtro Tipo I)
M = ceil(M_estimado);
if mod(M, 2) ~= 0
    M = M + 1;
end

fprintf('\nOrdem adotada (par, Tipo I): M = %d\n', M);
fprintf('Número de coeficientes: %d\n', M + 1);

% ============================================
% PARÂMETRO ? DA JANELA DE KAISER
% ============================================
if Rs < 21
    beta = 0;
elseif Rs <= 50
    beta = 0.5842 * (Rs - 21)^0.4 + 0.07886 * (Rs - 21);
else
    beta = 0.1102 * (Rs - 8.7);
end

fprintf('\nParâmetro ? (Kaiser): %.4f\n', beta);
if Rs < 21
    fprintf('Rs = %d dB ? ? = 0\n', Rs);
else
    fprintf('Rs = %d dB ? ? calculado\n', Rs);
end
fprintf('Parâmetro ? (Kaiser): %.4f (Rs ? 21 dB)\n', beta);

% ============================================
% PROJETO DO FILTRO
% ============================================
cutoff_norm = fc_hz / (fs/2);     % frequência de corte normalizada (Nyquist = 1)
janela = kaiser(M + 1, beta);     % janela de Kaiser com comprimento M+1
coeffs = fir1(M, cutoff_norm, 'low', janela);

% ============================================
% EXIBIR TODOS OS COEFICIENTES
% ============================================
fprintf('\n============================================================\n');
fprintf('COEFICIENTES DO FILTRO FIR\n');
fprintf('============================================================\n');
fprintf('Total de %d coeficientes:\n\n', length(coeffs));
fprintf('coeffs = [');
for i = 1:length(coeffs)
    if i == length(coeffs)
        fprintf('%.6f]\n', coeffs(i));
    else
        fprintf('%.6f, ', coeffs(i));
    end
end

% ============================================
% RESPOSTA EM FREQUÊNCIA
% ============================================
[h, w] = freqz(coeffs, 1, 4096, fs);
magnitude_db = 20 * log10(abs(h) + 1e-12);

% Limitar faixa de 0 a 1000 Hz
mascara = w <= 1000;
w_filt = w(mascara);
mag_filt = magnitude_db(mascara);

% ============================================
% PLOT - APENAS MAGNITUDE
% ============================================
figure;
plot(w_filt, mag_filt, 'b-', 'LineWidth', 1.5);
hold on;
yline(-Rs, 'r--', 'LineWidth', 1.5);
xline(fc_hz, 'g--', 'LineWidth', 1.5);
xline(fs_hz, 'Color', [1 0.5 0], 'LineStyle', '--', 'LineWidth', 1.5);
title(sprintf('Resposta em Frequência - Filtro FIR (Janela Kaiser, ?=%.4f, M=%d)', beta, M));
ylabel('Magnitude (dB)');
xlabel('Frequência (Hz)');
grid on;
xlim([0 1000]);
ylim([-50 5]);
legend('Magnitude', sprintf('Atenuação desejada = %d dB', Rs), ...
       sprintf('Fc = %.1f Hz', fc_hz), sprintf('Fs = %.1f Hz', fs_hz), ...
       'Location', 'best');
hold off;

% ============================================
% TABELA DE GANHOS (para comparação com dados experimentais)
% ============================================
freqs_teste = [15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 120, 150];

fprintf('\n============================================================\n');
fprintf('GANHO EM FREQUÊNCIAS ESPECÍFICAS\n');
fprintf('============================================================\n');
fprintf('%10s | %15s | %12s\n', 'Freq (Hz)', 'Ganho (linear)', 'Ganho (dB)');
fprintf('------------------------------------------------------------\n');

for f = freqs_teste
    [~, idx] = min(abs(w - f));
    ganho_linear = abs(h(idx));
    ganho_db = magnitude_db(idx);
    fprintf('%10.1f | %15.6f | %12.2f\n', f, ganho_linear, ganho_db);
end