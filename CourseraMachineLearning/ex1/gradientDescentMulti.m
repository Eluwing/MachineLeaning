function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

theta_len = length(theta)
tmp_theta = zeros(size(theta),1)
for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %

    %gradientDescent.m
    %for theta_index = 1:theta_len
    %  theta(theta_index) = theta(theta_index) - ((alpha / m) * sum(((X*theta)-y) .* X(:, theta_index)));
    %end
  

    %Gradient descent by length of theta
    for theta_index = 1:theta_len
      tmp_theta(theta_index)= theta(theta_index) - ((alpha * (1/ m)) * sum(((X*theta)-y) .* X(:, theta_index)));
      
      %Save temporary theta to original theta when finished gradient descent
      if theta_index == theta_len
        for index = 1:theta_len
          theta(index) = tmp_theta(index)
        endfor
      endif
      
    end






    % ============================================================
  
    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
