function [U, S] = pca(X)
%PCA Run principal component analysis on the dataset X
%   [U, S, X] = pca(X) computes eigenvectors of the covariance matrix of X
%   Returns the eigenvectors U, the eigenvalues (on diagonal) in S
%

% Useful values
[m, n] = size(X);

% You need to return the following variables correctly.
U = zeros(n);
S = zeros(n);

% ====================== YOUR CODE HERE ======================
% Instructions: You should first compute the covariance matrix. Then, you
%               should use the "svd" function to compute the eigenvectors
%               and eigenvalues of the covariance matrix. 
%
% Note: When computing the covariance matrix, remember to divide by m (the
%       number of examples).
%

Sigma = (X'*X)/m;

[U, S, V] = svd(Sigma);

d = diag(S);

denom_sum = 0;

for i = 1:size(d)
    denom_sum = denom_sum + d(i);
end

num_sum = 0;
final_k = 0;

for k = 1:size(d)
    num_sum = num_sum + d(k);
    test_k = num_sum / denom_sum;  
    if (test_k >= 0.991)
        disp(k);
        disp(test_k);          
        final_k = k;
        break
    end
end





% =========================================================================

end
