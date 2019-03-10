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
prediction= X*theta-y
Z=theta.^2;
Z(1)=0;
J=sum(prediction.^2) + lambda*sum(Z);
J=J/(2*m);

grad=((prediction)'*X)'./m + (lambda*theta)./m;
grad(1) = prediction'*X(:,1)./m;
grad= grad +1-1;










% =========================================================================

grad = grad(:);

end
