function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%

% theta looks like:
% u1    0.2854   -1.6843    0.2629
% u2    0.5050   -0.4546    0.3175
% u3   -0.4319   -0.4788    0.8467
% u4    0.7286   -0.2719    0.3268

% R is movie x user
%        u1   u2    u3    u4
% m1     1     1     0     0
% m2     1     0     0     0
% m3     1     0     0     0
% m4     1     0     0     0
% m5     1     0     0     0

% Y is movies x user
%        u1    u2
% m1     5     4     0     0
% m2     3     0     0     0
% m3     4     0     0     0
% m4     3     0     0     0
% m5     3     0     0     0

% disp('size R:')
% size(R)
% disp('size theta:')
% disp(Theta)
% disp('size X:')
% size(X)



%works, but lame
% temp = (X * Theta' - Y) .* R;
% temp = temp .^2;
% J = sum(temp(:))/2;

h=(X * Theta' - Y) .* R;

reg_j=lambda/2 * (sum(sum(Theta.^2))) + lambda/2 * (sum(sum(X.^2)));

reg_g=lambda * X;

reg_t=lambda * Theta;

J = 1/2 .* sum(sum(h.^2))+ reg_j;

X_grad = h * Theta + reg_g;

Theta_grad = h' * X + reg_t;











% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
