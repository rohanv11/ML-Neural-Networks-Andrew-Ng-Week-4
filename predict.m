function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% Adding ones

X = [ones(m,1) X]

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%


% totol 3 layers
% 1 input 400 nurons
% 2 hidden 25 neurons
% 3 output 10 neurons

% X = m * (n + 1) = m * 401 ; n = 400 ..added ones above
% theta1 = 25 by 401
% theta2 = 10 by 26


  
layer2 = X * Theta1'   % m * 25
layer2 = sigmoid(layer2)    % m * 25
layer2 = [ones(m, 1) layer2]    % m * 26

layer3_output = layer2 * Theta2'   % m * 10
layer3_output = sigmoid(layer3_output) % m * 10

[values, indices] = max(layer3_output, [], 2)

p = indices % m by 1





% =========================================================================


end
