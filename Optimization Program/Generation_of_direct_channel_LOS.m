% generate the direct channel between UE n and AP under LOS condition.
function h_d = Generation_of_direct_channel_LOS(xn,yn,y_AP,H,L0,d0,alpha_d,M)

d = sqrt(H^2 + xn^2 + (yn - y_AP)^2);
% AOA
sin_beta = xn/d; 
% path loss
L_d = L0*(d/d0)^(-alpha_d);
%
m = 1:M;
e_d = exp(1i*pi*(m-1)*sin_beta).';
%
h_d = sqrt(L_d)*e_d;

end

