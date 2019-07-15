function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

%sizeX = size(X)          % 12 x 2
%sizey = size(y)          % 12 x 1
%sizeTheta = size(theta)   %  2 x 1
%sizeLambda = size(lambda) %  1 x 1
%m                         % = 12
%sizeGrad = size(grad)     %  2 x 1

H = X * theta;

theta0 = [0; theta(2:end)]; %do NOT regularize theta0 (i.e., theta(1) = 0)

J = 1/(2*m) * sum((H-y).^2) + lambda/(2*m) * sum(theta0.^2); 

grad = X'*(H-y)/m + lambda/m * theta0; 








% =========================================================================

grad = grad(:);

end
