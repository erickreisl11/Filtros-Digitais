% ============================================
% PARÂMETROS DO PROJETO
% ============================================
fs = 8000;                    % frequência de amostragem (Hz)
fc_rad = 310;                 % frequência de corte (rad/s)
fc_hz = fc_rad / (2 * pi);    % ? 23.87 Hz
fs_rad = 470;                 % frequência de rejeição (rad/s)
fs_hz = fs_rad / (2 * pi);    % ? 71.62 Hz

% Especificações
Rs = 13;                      % atenuação na banda de rejeição (dB)
delta_s = 10^(-Rs/20);        % ?s = 0.3548

% Cálculo da banda de transição normalizada
omega_c_norm = (fc_rad / (2 * pi * fs)) * 2 * pi;  % = fc_rad / fs
omega_s_norm = (fs_rad / (2 * pi * fs)) * 2 * pi;  % = fs_rad / fs
delta_omega = omega_s_norm - omega_c_norm;

fprintf('%s\n', repmat('=', 1, 50));
fprintf('ESPECIFICAÇÕES DO FILTRO\n');
fprintf('%s\n', repmat('=', 1, 50));
fprintf('Frequência de amostragem: %d Hz\n', fs);
fprintf('Frequência de corte: %.1f rad/s (%.2f Hz)\n', fc_rad, fc_hz);
fprintf('Frequência de rejeição: %.1f rad/s (%.2f Hz)\n', fs_rad, fs_hz);
fprintf('Banda de transição (??): %.6f rad/amostra\n', delta_omega);
fprintf('Atenuação desejada (Rs): %d dB\n', Rs);
fprintf('?s = %.4f\n', delta_s);

% ============================================
% CÁLCULO DA ORDEM DO FILTRO
% ============================================
% Fórmula empírica para ordem mínima:
% M ? (Rs - 8) / (2.285 * ??)

if Rs >= 8
    M_estimado = (Rs - 8) / (2.285 * delta_omega);
else
    M_estimado = 0;
end

fprintf('\n%s\n', repmat('=', 1, 50));
fprintf('CÁLCULO DA ORDEM\n');
fprintf('%s\n', repmat('=', 1, 50));
fprintf('M estimado = (%d - 8) / (2.285 × %.6f)\n', Rs, delta_omega);
fprintf('M estimado = %.3f / %.6f\n', Rs - 8, 2.285 * delta_omega);
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
% Para Rs ? 21 dB, a literatura recomenda ? = 0

% Função inline para calcular beta da janela de Kaiser
calcular_beta_kaiser = @(Rs) ...
    (Rs < 21) * 0.0 + ...
    (Rs >= 21 && Rs <= 50) * (0.5842 * (Rs - 21)^0.4 + 0.07886 * (Rs - 21)) + ...
    (Rs > 50) * (0.1102 * (Rs - 8.7));

beta = calcular_beta_kaiser(Rs);

fprintf('\nParâmetro ? (Kaiser): %.4f\n', beta);
fprintf('Rs = %d dB ? ? = %.4f\n', Rs, beta);

% ============================================
% PROJETO DO FILTRO
% ============================================
% Frequência de corte normalizada (Nyquist = 1)
f_cutoff_norm = fc_hz / (fs / 2);

% Projetar o filtro FIR com janela de Kaiser
coeffs = fir1(M, f_cutoff_norm, 'high', kaiser(M+1, beta));

% ============================================
% EXIBIR TODOS OS COEFICIENTES
% ============================================
fprintf('\n%s\n', repmat('=', 1, 60));
fprintf('COEFICIENTES DO FILTRO FIR\n');
fprintf('%s\n', repmat('=', 1, 60));
fprintf('Total de %d coeficientes:\n\n', length(coeffs));

% Exibir coeficientes em formato vetor MATLAB
fprintf('coeffs = [');
for i = 1:length(coeffs)
    if i == length(coeffs)
        fprintf('%.6f];\n', coeffs(i));
    else
        fprintf('%.6f, ', coeffs(i));
    end
end

% ============================================
% RESPOSTA EM FREQUÊNCIA
% ============================================
[resposta, frequencias] = freqz(coeffs, 1, 4096, fs);

% Converter magnitude para dB
magnitude_db = 20 * log10(abs(resposta) + 1e-12);

% Filtrar faixa de 0 a 1000 Hz
mask = frequencias <= 1000;
frequencias_filtradas = frequencias(mask);
magnitude_filtrada = magnitude_db(mask);

% ============================================
% PLOT - APENAS MAGNITUDE
% ============================================
figure('Position', [100, 100, 1000, 600]);

plot(frequencias_filtradas, magnitude_filtrada, 'b-', 'LineWidth', 1.5);
hold on;

% Linhas de referência sem rótulos verticais
h1 = yline(-Rs, 'r--', 'LineWidth', 1.5);
h2 = xline(fc_hz, 'g--', 'LineWidth', 1.5);
h3 = xline(fs_hz, '--', 'Color', [0.85, 0.5, 0], 'LineWidth', 1.5);

title(sprintf('Resposta em Frequência - Filtro FIR (Janela Kaiser, ?=%.4f, M=%d)', beta, M));
ylabel('Magnitude (dB)');
xlabel('Frequência (Hz)');
grid on;
xlim([0, 1000]);
ylim([-50, 5]);

% Legenda com todas as informações
legend({'Resposta do Filtro', ...
    sprintf('Atenuação desejada = %d dB', Rs), ...
    sprintf('Fc = %.1f Hz', fc_hz), ...
    sprintf('Fs = %.1f Hz', fs_hz)}, ...
    'Location', 'best');

% ============================================
% TABELA DE GANHOS (para comparação com dados experimentais)
% ============================================
freqs_teste = [15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 120, 150];

fprintf('\n%s\n', repmat('=', 1, 60));
fprintf('GANHO EM FREQUÊNCIAS ESPECÍFICAS\n');
fprintf('%s\n', repmat('=', 1, 60));
fprintf('%10s | %15s | %12s\n', 'Freq (Hz)', 'Ganho (linear)', 'Ganho (dB)');
fprintf('%s\n', repmat('-', 1, 60));

for i = 1:length(freqs_teste)
    f = freqs_teste(i);
    [~, idx] = min(abs(frequencias - f));
    ganho_linear = abs(resposta(idx));
    ganho_db = magnitude_db(idx);
    fprintf('%10.1f | %15.6f | %12.2f\n', f, ganho_linear, ganho_db);
end