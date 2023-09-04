temp = 0;

cov_condition_1 = 0.001;

W_X = zeros(M,N,1000);
phi_X = zeros(K,1,1000);
X = 0;


aux = 1; % auxiliary scalar ξ
phi_tilde = [phi_0,aux].'; % RIS reflecting coefficients with auxiliary scalar
sigh_l = phi_tilde*phi_tilde'; % Ψ




TCTB_X = 0;
convergence_condition_0 = 0.0005;
flag_0 = true;
%%%%%%%%%%%%%%%%%%%%%%%  Start of Repaet 1 %%%%%%%%%%%%%%%%%%%%%%%%%%%
while flag_0
    X = X + 1;
    TCTB_a = 0;
    TCTB_m = 0;

    l = 0;
    TCTB_1 = zeros(1,500);
    convergence_condition_1 = 1;
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
                    left = left + pj*(trace(Q*sigh)+abs(h_dni)^2);
                end
                F_1_n = log(left + noise*norm(W0(:,n)')^2)/log(2)+log(10)/log(2);

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
                    left = left + pj*(trace(Q*sigh_l)+abs(h_dni)^2);
                    gap = sigh-sigh_l;
                    top = top + pj*dot(gap(:),Q(:));
                end
                F_2_n_l = log(left + (noise)*norm(W0(:,n)')^2)/log(2)+log(10)/log(2);
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
                disp(trace(sigh) - ipsilon);
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
    if temp >= 10
        flag_0 = false;
    end

    disp('the value of TCTB is');
    disp(TCTB_X);
end