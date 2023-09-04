CSI_INFO_LOC2 = zeros(0,0);
RIS_INFO_LOC2 = zeros(0,0);
x_UEs_LOC2 = zeros(0,0);
y_UEs_LOC2 = zeros(0,0);
d_UEs_LOC2 = zeros(0,0);

sample_num = 1;
turns = 0;
for tt = 1:sample_num
    turns = turns + 1;
    % Create a list of all variables in the workspace
    vars = who;
    % Specify the variables that you want to keep
    varsToKeep = {'CSI_INFO_LOC2', 'RIS_INFO_LOC2','sample_num','turns','x_UEs_LOC2','y_UEs_LOC2','d_UEs_LOC2',};
    % Get the variables that you want to delete
    varsToDelete = setdiff(vars, varsToKeep);
    % Delete the specified variables
    clear(varsToDelete{:});
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%% Initialization %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %simulation parameters I
    y_AP = 20; H = 5; % The location of the AP
    %xn = 12.2; yn = 15.3; %
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
    d_UEs = sqrt(HR^2 + y_UEs.^2 + (x_UEs - xR).^2);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    x_UEs_LOC2(:,turns) = x_UEs;
    y_UEs_LOC2(:,turns) = y_UEs;
    d_UEs_LOC2(:,turns) = d_UEs;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
    %
    
    theta = 2*pi*rand(1,K);%not used
    phi_0 = exp(1j*theta);%not used
    
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

    %%%
    %Generate dataset for DNN-loc-2
    %CSI-h,d
    outputVector1 = zeros(M * N * 2, 1);
    
    % Traverse matrix elements and store the real and imaginary parts in the vector
    index = 1;
    for col = 1:N
        for row = 1:M
            outputVector1(index) = real(h_d(row, col));
            outputVector1(index + 1) = imag(h_d(row, col));
            index = index + 2;
        end
    end

    %CSI-h,r
    outputVector2 = zeros(K * N * 2, 1);
    % Traverse matrix elements and store the real and imaginary parts in the vector
    index = 1;
    for col = 1:N
        for row = 1:K
            outputVector2(index) = real(h_r(row, col));
            outputVector2(index + 1) = imag(h_r(row, col));
            index = index + 2;
        end
    end
    
    %CSI-H_AP
    outputVector3 = zeros(K * N * 2, 1);
    
    % Traverse matrix elements and store the real and imaginary parts in the vector
    index = 1;
    for col = 1:K
        for row = 1:M
            outputVector3(index) = real(H_AP(row, col));
            outputVector3(index + 1) = imag(H_AP(row, col));
            index = index + 2;
        end
    end

    CSI_INFO_temp = [outputVector1; outputVector2; outputVector3];
    CSI_INFO_LOC2(:,turns) = CSI_INFO_temp;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%% Simulation %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    temp = 0;
    %load('myWorkspace_2.mat');
    scale_factor = 10^4;
    scale_factor_2 = 1;
    cov_condition_1 = 0.01;
    
    W_X = zeros(M,N,1000);
    phi_X = zeros(K,1,1000);
    X = 0;
    
    
    aux = 1; % auxiliary scalar ξ
    phi_tilde = [phi_0,aux].'; % RIS reflecting coefficients with auxiliary scalar
    sigh_l = phi_tilde*phi_tilde'; % Ψ
    
    
    
    
    TCTB_X = 0;
    convergence_condition_0 = 0.005;
    flag_0 = true;
    %%%%%%%%%%%%%%%%%%%%%%%  Start of Repaet 1 %%%%%%%%%%%%%%%%%%%%%%%%%%%
    while flag_0
        X = X + 1;
        TCTB_a = 0;
        TCTB_m = 0;
    
        l = 0;
        TCTB_1 = zeros(1,500);
        convergence_condition_1 = 0.01;
        flag_1 = true;
        while flag_1
            l = l + 1;
            cvx_quiet(false); 
            cvx_begin
                obj_1 = 0; %object function of problem 1.1
                variable sigh(K+1,K+1) complex semidefinite
            
            % calculate F_1_n
                for n = 1:N
                    left = 0;
                    for j = 1:N
                        pj = (a0(j,1)*EN)/T;
                        
                        h_RIS = W0(:,n)'*H_AP*diag(h_r(:,j));
                        h_dni = W0(:,n)'*h_d(:,j);
    
                        Q = zeros(K+1);     
                        Q(1:K, 1:K) = h_RIS'*h_RIS;
                        Q(1:K, K+1) = h_RIS'*h_dni;
                        Q(K+1, 1:K) = h_dni'*h_RIS;
                        Q(K+1, K+1) = 0;
                        %
                        left = left + scale_factor_2*pj*(trace(Q*sigh)+abs(h_dni)^2);
                    end
                    F_1_n = log(left + scale_factor_2*noise*norm(W0(:,n)')^2)/log(2)+log(10)/log(2);
    
            % calculate F_2_n_bar
                    top = 0;
                    left = 0;
                    for j = 1:N
                        if j==n
                            continue;
                        end
                        pj = (a0(j,1)*EN)/T;
                        h_RIS = W0(:,n)'*H_AP*diag(h_r(:,j));
                        h_dni = W0(:,n)'*h_d(:,j);
    
                        Q = zeros(K+1);     
                        Q(1:K, 1:K) = h_RIS'*h_RIS;
                        Q(1:K, K+1) = h_RIS'*h_dni;
                        Q(K+1, 1:K) = h_dni'*h_RIS;
                        Q(K+1, K+1) = 0;
                        %
                        left = left + scale_factor_2*pj*(trace(Q*sigh_l)+abs(h_dni)^2);
                        gap = sigh-sigh_l;
                        top = top + pj*dot(gap(:),Q(:));
                    end
                    F_2_n_l = log(left + scale_factor_2*(noise)*norm(W0(:,n)')^2)/log(2)+log(10)/log(2);
                    bottom = log(2)*F_2_n_l;
                    %F_2_n_bar = F_2_n_l + top/bottom;
                    F_2_n_bar = F_2_n_l + top/bottom;
                    obj_1 = obj_1 + F_1_n - real(F_2_n_bar);
                end
                sub = subgradient_spectral_norm(sigh_l);
                gap = sigh-sigh_l;
                ipsilon = norm(sigh_l) + dot(gap(:),sub(:));
                maximize(obj_1) 
                subject to
                    for kk = 1:K+1
                        sigh(kk,kk)==1;
                    end     
                    %trace(sigh) - ipsilon <= cov_condition_1;
                    trace(sigh) - ipsilon <= cov_condition_1;
            cvx_end
            sigh_l = sigh;
        
            %calculate TCTB (5) (12)
            for n = 1:N
                TCTB_1(1,l+1) = TCTB_1(1,l+1) + Get_F1(noise,EN,T,K,N,a0,W0,H_AP,h_r,h_d,sigh,n) - Get_F2(noise,EN,T,K,N,a0,W0,H_AP,h_r,h_d,sigh,n);
            end
            if (abs(TCTB_1(1,l+1) - TCTB_1(1,l))) <= convergence_condition_1
                flag_1 = false;
            end
            disp(abs((TCTB_1(1,l+1) - TCTB_1(1,l))));
            disp(abs(TCTB_1(1,l+1)));
        end
        % Get the result of phi_X of the X-th iteration
        [V,D] = eig(sigh);
        [~, index] = max(diag(D));
        phi_X_0 = sqrt(D(index,index)) * V(:,index);
        phi_X(:,:,X) = phi_X_0(1:K)/phi_X_0(K+1,1);
        
    
    %%%%%%%%%%%%%%%%%%%%%%%%  End of Repaet 1 %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%  Start of Lemma 2 %%%%%%%%%%%%%%%%%%%%%%%%%%%  
        phi_0 = phi_X(:,:,X);
    
        for n = 1:N
            % Equation (20)
            h_n_phi = H_AP*diag(phi_0)*h_r(:,n) + h_d(:,n);
            pn = (a0(j,1)*EN)/T; 
    
            Theta_n = pn*h_n_phi*(h_n_phi');
            Theta_n_neg = zeros(M,M);
            for j = 1:N
                if j==n
                    continue;
                end
                pj = (a0(j,1)*EN)/T;
                h_i_phi = H_AP*diag(phi_0)*h_r(:,j) + h_d(:,j);
                Theta_n_neg = Theta_n_neg + pj*h_i_phi*(h_i_phi');
            end
            Theta_n_neg = Theta_n_neg + noise*eye(M);
            intermediate_variable = inv(Theta_n_neg)*Theta_n;
            [V, D] = eig(intermediate_variable);  % Calculating eigenvalues and eigenvectors
            [~, max_idx] = max(diag(D));  % Find the index corresponding to the maximum eigenvalue 
            W0(:,n) = V(:, max_idx);  % Update the Beamforming matrix with the corresponding eigenvectors
            
        end
        W_X(:,:,X) = W0;
        %calculate TCTB (5) (12)
        TCTB_2 = 0;
        for n = 1:N
            TCTB_2 = TCTB_2 + Get_F1(noise,EN,T,K,N,a0,W0,H_AP,h_r,h_d,sigh,n) - Get_F2(noise,EN,T,K,N,a0,W0,H_AP,h_r,h_d,sigh,n);
        end
        disp(abs(TCTB_2));
    %%%%%%%%%%%%%%%%%%%%%%%%  End of Lemma 2 %%%%%%%%%%%%%%%%%%%%%%%%%%% 
        %Calculate the TCTB
        TCTB = 0;
        for n = 1:N
            TCTB = TCTB + B*T*(Get_F1(noise,EN,T,K,N,a0,W0,H_AP,h_r,h_d,sigh,n) - Get_F2(noise,EN,T,K,N,a0,W0,H_AP,h_r,h_d,sigh,n));
        end
        TCTB_X =[TCTB_X,TCTB];
    
        disp((TCTB_X(X+1)-TCTB_X(X))*10^-9);
        if (TCTB_X(X+1)-TCTB_X(X))*10^-9 <= convergence_condition_0
            flag_0 = false;
        end
        temp = temp + 1;
        if temp >= 1
            flag_0 = false;
        end
    
        disp('the value of TCTB is');
        disp(TCTB_X);
    end
    RIS_INFO_LOC2_angle = zeros(K,1);
    optimized_phi = phi_X(:,:,temp); % Kx1
    for i = 1:K
        RIS_INFO_LOC2_angle(i,1) =  pi + angle(optimized_phi(i,1));
    end
    RIS_INFO_LOC2(:,turns) = RIS_INFO_LOC2_angle;
    disp(turns);
end
%%%%%
%x_UEs_F_6 = x_UEs_LOC2;
%y_UEs_F_6 = y_UEs_LOC2;
%d_UEs_F_6 = d_UEs_LOC2;
%%%%%
%CSI_INFO_LOC2_F_6 = CSI_INFO_LOC2;
%RIS_INFO_LOC2_F_6 = RIS_INFO_LOC2;

%save('CSI_INFO_LOC2_F_6.mat', 'CSI_INFO_LOC2_F_6');
%save('RIS_INFO_LOC2_F_6.mat', 'RIS_INFO_LOC2_F_6');
%save('x_UEs_F_6.mat', 'x_UEs_F_6');
%save('y_UEs_F_6.mat', 'y_UEs_F_6');
%save('d_UEs_F_6.mat', 'd_UEs_F_6');
%clear;
%clc;


