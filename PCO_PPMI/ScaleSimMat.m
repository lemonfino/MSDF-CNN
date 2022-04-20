%% Scale Similar Matrix by Row %%通过行缩放相似矩阵

function W = ScaleSimMat(W)

%scale 
W = W - diag(diag(W));  %diagonal elements must be 0   对角线元素变为0
W =W + diag(sum(W)==0);
D = diag(sum(W), 0);    %degree matrix

W = D^(-1)*W;