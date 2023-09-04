function subgradient = subgradient_spectral_norm(A)
    % 计算奇异值分解
    [U,S,V] = svd(A);

    % 获取最大奇异值对应的奇异向量
    max_singular_value = S(1, 1);  % 最大奇异值
    max_singular_vector = U(:, 1);  % 最大奇异值对应的奇异向量

    % 计算谱范数的子梯度
    subgradient = max_singular_vector*max_singular_vector';
end

