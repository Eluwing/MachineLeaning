function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


% then K is 10, make y matrix of neural network
nn_y = zeros(m,num_labels);
  
for i = 1 : m
  
  nn_y(i,y(i)) = 1;

end 

% hypo x calculation of neural network
a1_nn_padding_matrix = ones(size(X,1),1);
a1 = [a1_nn_padding_matrix X];
z2 = a1 * (Theta1)';

ori_a2 = sigmoid(z2);
a2_nn_padding_matrix = ones(size(ori_a2,1),1);
a2 = [a2_nn_padding_matrix ori_a2];
z3 = a2 * (Theta2)';

nn_hypo_x = sigmoid(z3);

%% Part 1: Feedforward the neural network

costfunc = ((1 / m) * sum(sum( (-nn_y .* log(nn_hypo_x)) - ((1 - nn_y) .* log(1 - nn_hypo_x)))))

theta1_index_1_value = (lambda/(2 * m)) * sum(Theta1(:,1).^2)   
theta2_index_1_value = (lambda/(2 * m)) * sum(Theta2(:,1).^2) 
cost_regular = ((lambda/(2 * m)) * (sum(sum((Theta1).^2)) + sum(sum((Theta2).^2))))

% Why this process that "- (theta1_index_1_value + theta2_index_1_value)"
% because When regularizetion, need to subtract value of first index theta , for customary reasons
% Therefore, it subtracts the first parameters of Theta1 and Theta2.

J = costfunc + (cost_regular - (theta1_index_1_value + theta2_index_1_value))

%% Part 2: Implement the backpropagation algorithm

% Random initialization of neural network, imploement 
% because need to value initialized

%% step 2 Å®Å@lower Delta calculation of Neural NetWorks last layer

% lower_delta3 Å® l_delta3  
l_delta3 = nn_hypo_x - nn_y;

%% step 3 Å®Å@lower delta2 calculation
l_delta_z2_nn_padding_matrix = ones(size(z2,1),1);
l_delta_z2 = [l_delta_z2_nn_padding_matrix z2];

l_delta2 = (l_delta3 * (Theta2) .* sigmoidGradient(l_delta_z2))
l_delta2 = l_delta2(:,2:end)

%% step 4

% upper_delta3 Å® u_delta3
u_delta2 = l_delta3' * a2
u_delta1 = l_delta2' * a1

% value of Unregularized Neural NetWorks but,
% then Regularized Neural Networks, value of j equal 0
% D(1) = (1 / m) * ?(1)
Delta2 = (1 / m) * u_delta2
Delta1 = (1 / m) * u_delta1

%% Part 3: Implement regularization with the cost function and gradients.

% because, the Regularized Neural Networks aissumes that j is greater than 1
% Therefore, padding 0 at the first index of Theta1 and Theta2
Theta1_zero_padding = zeros(size(Theta1),1)
Theta2_zero_padding = zeros(size(Theta2),1)

r_Theta2 = [Theta1_zero_padding Theta2(:, 2:end)]
r_Theta1 = [Theta2_zero_padding Theta1(:, 2:end)]

% value of Unregularized Neural NetWorks
r_Delta2 = Delta2 + (lambda / m) * r_Theta2
r_Delta1 = Delta1 + (lambda / m) * r_Theta1

Theta2_grad = r_Delta2 
Theta1_grad = r_Delta1  

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
