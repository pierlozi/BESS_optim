data_time = readmatrix('test_rainflow_DoDs.csv');
SOC = data_time(:,3)';


%Parametri modello 
a_sei= 5.75e-2;
b_sei=121;
k_d1=1.4e5;
k_d2=-5.01e-1;
k_d3=-1.23e5;
k_s=1.04;
k_T=6.93e-2;
k_t=4.14e-10;
s_ref=0.5;
T_ref=25;
T=25; %°C

% Funzioni
S_d = @(d) (k_d1*d.^(k_d2)+k_d3).^(-1); S_s=@(s) exp(k_s.*(s-s_ref));
S_T = @(T) exp(k_T* (T-T_ref).*T_ref./T); 
S_t = @(t) k_t*t;
ff_cyc = @(d,s,T) S_d(d).*S_s(s).*S_T(T);
ff_cal = @(s, t,T) S_s(s).*S_t(t).*S_T(T);


% Elaborazione dati
L_sei = zeros(1,10);
SOC_mean = zeros(1,10);
DoD_mean = zeros(1,10);

% Define the group size
groupSize = 8760;

for ii = 1:10
    
    % Calculate the indices for the current group
    startIdx = (ii - 1) * groupSize + 1;
    endIdx = ii * groupSize;
    
    % Extract the elements for the current group
    SOCC = SOC(startIdx:endIdx);

    rf = rainflow(SOCC);
    
    dod=2*rf(1,:); %calcolo del depth of discharge (DoD) di ogni ciclo.
    soc= rf(2,:); %calcolo del SOC medio di ogni ciclo.
    f_cyc= rf(3,:).*ff_cyc(dod, soc, T); %Moltiplicazione del peso del ciclo per il degrado % di quel ciclo. Vettore che determina il degrado di ogni ciclo.
    f_cal = ff_cal(mean(SOCC), 3600*length(SOCC), T);
    f_d = sum(f_cyc) + f_cal;
    L = 1-exp(-f_d);
    L_sei(ii) = 1 - a_sei * exp(-b_sei*f_d) - (1-a_sei)*exp(-f_d);
    SOC_mean(ii) = mean(SOCC);


end

%writematrix(L_sei', 'L_sei_MATLAB.csv')



% Life=L(f_d);
% figure(3)
% plot([0 tempi/24/30], [100 (1-Life)*100],'-', 'linewidth',2); xlabel('Mesi')
% ylabel('Capacità massima (%)')
% grid on
