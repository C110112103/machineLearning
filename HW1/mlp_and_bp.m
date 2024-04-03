%開始計時
tic;

% 設定訓練數據
X = [0 0; 0 1; 1 0; 1 1]; % 特徵
Y = [0; 1; 1; 0]; % 標籤

% 參數設定
input_size = size(X, 2); 
hidden_size = 4; 
output_size = 1; 
learning_rate = 0.1; 
epochs = 1000; 

% 初始化權重和偏值
W1 = randn(input_size, hidden_size); % 輸入層到隱藏層的權重
b1 = zeros(1, hidden_size); %隱藏層的偏值
W2 = randn(hidden_size, output_size); % 隱藏層到輸出層的權重
b2 = zeros(1, output_size); % 輸出層的偏值
% 訓練
for epoch = 1:epochs
    % 前向傳播
    z1 = X * W1 + b1; % 隱藏層輸入
    a1 = sigmoid(z1); % 隱藏層激活函數
    z2 = a1 * W2 + b2; % 輸出層輸入
    y_pred = sigmoid(z2); % 輸出層預測值
    
    % 計算準確率
    accuracy = sum((y_pred > 0.5) == Y) / length(Y);
    
    % 反向傳播
    delta2 = (y_pred - Y) .* sigmoid_derivative(z2); % 輸出層的誤差
    delta1 = (delta2 * W2') .* sigmoid_derivative(z1); % 隱藏層的誤差
    
    % 更新權重和偏值
    W2 = W2 - learning_rate * (a1' * delta2);
    b2 = b2 - learning_rate * sum(delta2);
    W1 = W1 - learning_rate * (X' * delta1);
    b1 = b1 - learning_rate * sum(delta1);
    
end
disp(['準確率: ', num2str(accuracy)]);
%結束計時
elapsed_time = toc;
fprintf('運算時長：%.4f 秒\n', elapsed_time);

% 定義 sigmoid 函數及其導數
function y = sigmoid(x)
    y = 1 ./ (1 + exp(-x));
end

function y = sigmoid_derivative(x)
    y = sigmoid(x) .* (1 - sigmoid(x));
end
