% generate the channel between RIS and AP, H_AP
function H_AP = Generation_of_RIS_to_AP_channel(y_AP,H,HR,xR,L0,d0,alpha_AP,Ky,Kz,M)

d = sqrt(xR^2 + y_AP^2 + (H-HR)^2);
%AOA
sin_beta_r = xR/d;
%AOD
sin_beta_t = y_AP/d;
cos_beta_t = sqrt(xR^2 + (H-HR)^2)/d;
sin_gamma_t = xR/sqrt(xR^2 + (H-HR)^2);

%path loss
L_AP = (10^0.3)*L0*(d/d0)^(-alpha_AP);
%
m = 1:M;
e_r = exp(1i*pi*(m-1)*sin_beta_r).';
%
ky = 1:Ky;
e_t_y = exp(1i*pi*(ky-1)*sin_beta_t*sin_gamma_t).';
kz = 1:Kz;
e_t_z = exp(1i*pi*(kz-1)*cos_beta_t*sin_gamma_t).';
e_t = kron(e_t_y,e_t_z)';
%
H_AP = sqrt(L_AP)*e_r*e_t;

end

