function F1_n = Get_F1(noise,EN,T,K,N,a0,W0,H_AP,h_r,h_d,sigh,n)
        
                left = 0;
                % calculate F1_n
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
                F1_n = log(left + (noise)*norm(W0(:,n)')^2)/log(2);
       
end

