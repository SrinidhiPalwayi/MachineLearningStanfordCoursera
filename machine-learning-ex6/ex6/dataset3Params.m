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
CVals = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
sigmaVals = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];

holder = 10000000; 
cfinal = 0;
sigmafinal =0;
disp(CVals(1));
model = svmTrain(X, y, 0.01, @(x1, x2) gaussianKernel(x1, x2, 0.01));
pred = svmPredict(model,X);
%error = mean(double(pred ~=yval));
%disp(error);

for c = 1: size(CVals)
    for r = 1: size(sigmaVals)

         model = svmTrain(X, y, 0.01, @(x1, x2) gaussianKernel(x1, x2, 0.01));
         
         visualizeBoundary(X, y, model);
         pred = svmPredict(model,Xval);
         disp(size(pred));
         disp(size(yval));
         error = mean(double(pred ~=yval));
         if error < holder 
           holder = error;
           sigmafinal = sigmaVals(r);
           cfinal = CVals(c);
         end
         
    end
      
end

C = cfinal;
sigma = sigmafinal;
 
    
    
   
    







% =========================================================================

end
