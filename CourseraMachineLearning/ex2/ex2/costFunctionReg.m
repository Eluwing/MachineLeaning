function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

z = X * theta

hypo_x = sigmoid(z);

costfunc = ((1/m) * (-y' * log(hypo_x) - ((1 - y)' * log(1-hypo_x)))) 
cost_lamda_formula = ((lambda/(2 * m)) * sum((theta).^2))
cost_regul_theta_1 = ((lambda/(2 * m)) * (theta(1).^2)) 



grad_lamda_formula = ((lambda / m) * theta)
grad = ((1/m) * (X)' *((hypo_x - y))) + grad_lamda_formula
% When regularizetion, need to subtract theta 1 for customary reasons
grad(1,1) = ((1/m) * (X(:,1))' * ((hypo_x - y))) 




% When regularizetion, need to subtract theta 1 for customary reasons
J =  costfunc + cost_lamda_formula - cost_regul_theta_1 



% =============================================================

end
