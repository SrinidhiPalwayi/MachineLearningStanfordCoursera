function [J, grad] = nnCostFunction(nn_params, ...
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
yv = [1:num_labels] == y;
%disp(size(yv));
%disp(size (ones(m+1)));
A1 = [ones(m, 1) X];

Z2 = A1 * Theta1';%multiply first set of parameters
A2 = [ones(size(Z2),1),sigmoid(Z2)];
Z3 = A2*Theta2';
A3 = sigmoid(Z3);
%forward propagation

%disp(size(A3));
d3 = A3 - yv;
d2 =( d3 * (Theta2(:, 2:end))) .* (A2(:, 2:end).*(1-A2(:, 2:end)));

Delta_1 = d2' * A1;
Delta_2 = d3' * A2;
disp(size(Delta_2));

Theta1_gradMost = 1/m * (Delta_1(:, 2:end) + lambda * Theta1( :, 2:end));
Theta1_first = 1/m * Delta_1( :, 1:1);

Theta1_grad = Theta1_grad + [Theta1_first Theta1_gradMost];


Theta2_gradMost = 1/m * (Delta_2( :, 2:end) + lambda * Theta2( :, 2:end));
Theta2_first = 1/m * Delta_2( :, 1:1);
Theta2_grad = Theta2_grad + [Theta2_first Theta2_gradMost];
disp(size(Theta2_grad));

%stepA1 = [ones(size(stepM1),1), sigmoid(stepM1)] * Theta2' % multiply second theta and use sigmoid function
%disp(size(stepA1));
stepAdd = -yv .* log(A3) - (ones(size(yv,1), 1) - yv) .* log(1 - A3);

%unrollTheta1 =   Theta1( :, 2:size(Theta1));
%unrollTheta2 =   Theta2( :, 2:size(Theta2));

p = (sum(sum(Theta1(:, 2:end).^2))+sum(sum(Theta2(:, 2:end).^2))) * lambda / (2*m);

%Jreg = (sum(sum(Theta1( :, 2:size(Theta1)) .^ 2)) + sum(sum(Theta2( :, 2:size(Theta2)) .^2))) / (2 *m) ;

J = 1/m * sum(sum(stepAdd)) + p ;

disp(d3);

%disp(size(X * Theta1'));
%step1 = (sigmoid((X * Theta1'))') * yv ;
%step2 = (ones(size(Theta1,1),1) - sigmoid((X * Theta1'))') * (ones(m,1) - yv);

%step3 = step1 + step2;

%step4 = step3
%disp(size(step3));

%step3 = [ones(size(step3', 1),1) step3'];
%disp(size(step3));
%step4 = log(sigmoid((step3 * Theta2'))') * yv' ;
%step5 = log(ones(size(Theta2,1),1) - sigmoid((step3 * Theta2'))') * (ones(m,1) - yv);

%step6= step4 + step5;

%disp(step1);
%J = - 1/m * sum(step6);

%disp(size(stepAdd));

%grad = (X' * (sigmoid((X * theta)) - y))/m; 
















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
