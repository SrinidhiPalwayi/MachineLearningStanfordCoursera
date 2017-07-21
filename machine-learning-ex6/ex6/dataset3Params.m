function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~=yval))
%

counter = 1;
in1 = 0.01;
in3 = 0.03;
holder = 10000000; 
cholder1 = 0.01;
cholder3 = 0.03
x1 = [1 2 1]; x2 = [0 4 -1]; 
for c = 1:5
    for r = 1:5
         model = svmTrain(X, y, cholder1, @(x1, x2) gaussianKernel(x1, x2, in1));
         pred = svmPredict(X, model);
         error = mean(double(pred ~=yval));
         if error < holder 
             
         
    end
end
 
    
    
   
    







% =========================================================================

end
