clear;

%simulation parameters I
y_AP = 20; H = 5; % The location of the AP
xR = 40; HR = 20; % The location of the RIS 
L0 = 0.1; d0 = 1;  % The channel power gain at a reference distance of d0=1 m -10db
alpha_d = 3.5; alpha_r = 2.5; alpha_AP = 2; % The channel attenuation coefficients
Ky = 8; Kz = 3; % IS s reflecting elements
K = Ky*Kz;
M = 8;
N = 8;

%
xs = 20; D = 40;
x_UEs = xs + D*rand(N,1);
y_UEs = xs + D*rand(N,1);

%HAP
H_AP = Generation_of_RIS_to_AP_channel(y_AP,H,HR,xR,L0,d0,alpha_AP,Ky,Kz,M);
%h_r
h_r = zeros(K,N);
for loop = 1 : N
    xn = x_UEs(loop);
    yn = y_UEs(loop);
    h_r(:,loop) = Generation_of_relay_channel(xn,yn,HR,xR,L0,d0,alpha_r,Ky,Kz);
end
%h_d
h_d = zeros(M,N);
for loop = 1 : N
    xn = x_UEs(loop);
    yn = y_UEs(loop);
    h_d(:,loop) = Generation_of_direct_channel_LOS(xn,yn,y_AP,H,L0,d0,alpha_d,M);
end

%simulation parameters II
%T , B  ε, ε1, ε3; En, Cn, κn
T = 5; % Length of time slot in s
EN = 10; % Energy budget in J
EN_bar = EN/T;
noise = 10^-9; % noise power −60dBm
epsilon_sigh = 10^-15; % εΨ is a positive threshold with very small value close to zero
Cn = 200; %Required CPU cycles per bit of UEs in cycles/bit
kn = 10^-28; %The effective switched capacitance of UEs
knn = kn^(1/3);
B = 4*10^7; %The total system bandwidth 40MHZ
%B = 40;
%
% Initializaion 
% X phi_0 a0 W0
X = 0;


theta = 2*pi*rand(1,K);
phi_0 = exp(1j*theta);

%a0 = rand(N,1);
a0 = ones(N,1);

W0 = zeros(M,N);
for column = 1 : N
    for row = 1 : M
    w_real = rand(1)+rand(1);
    w_img = rand(1)+rand(1);
    W0(row,column) = w_real + w_img*1i;
    end
end






