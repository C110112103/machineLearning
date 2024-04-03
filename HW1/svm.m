%開始計時
tic;

% 假設X和Y的值
X = [1, 2; 2, 3; 3, 4];
Y = [-1; 1; -1];


w = rand(size(X, 2), 1); % 權重向量
b = 0; % 偏值

learning_rate = 0.01;
lambda = 0.1; 

% 迭代更新參數
max_iter = 1000;
for iter = 1:max_iter
    % 計算梯度
    grad_w = lambda * w; % 正規化的梯度
    for i = 1:size(X, 1)
        if Y(i) * (X(i, :) * w + b) < 1
            grad_w = grad_w - Y(i) * X(i, :)'; % hinge loss的梯度
        end
    end
    
    % 更新參數
    w = w - learning_rate * grad_w;
    
    % 損失函數
    loss = 1/2 * norm(w)^2 + lambda * sum(max(0, 1 - Y .* (X * w + b)));
    
end

% 输出權重和截距
fprintf('權重: '); disp(w);
fprintf('截距: %f\n', b);

%結束計時
elapsed_time = toc;
fprintf('運算時長為：%.4f 秒\n', elapsed_time);
