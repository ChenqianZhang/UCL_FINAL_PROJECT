
% generate the relay channel information h_r for one UE n
function h_r = Generation_of_relay_channel(xn,yn,HR,xR,L0,d0,alpha_r,Ky,Kz)

d = sqrt(HR^2 + yn^2 + (xR-xn)^2);
%Taking the positive direction of the Z-axis as the reference direction for determining the azimuth angle
sin_beta = yn/d;
cos_beta = sqrt(HR^2 + (xR-xn)^2) / d;
sin_gamma = abs(xR - xn) / sqrt(HR^2 + (xR-xn)^2);
%path loss 
L_r = (10^0.3) * L0 * (d/d0)^(-alpha_r);
%
ky = 1:Ky; 
e_y = exp(1i*pi*(ky-1)*sin_beta*sin_gamma).';
kz = 1:Kz;
e_z = exp(1i*pi*(kz-1)*cos_beta*sin_gamma).';
%
e_r = kron(e_y,e_z);
%
h_r = sqrt(L_r)*e_r;

end