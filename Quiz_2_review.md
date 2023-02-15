Given y = w_0 + w_1 x:
x (independent), y (dependent), w_0 (intercept), w_1 (slope) 

Linear models - when function consists of no powers on x or independent 
variable.
y = w_0 + w_1 x_1 + w_2 x_2 ... w_n x_n

Non-Linear models - when function consists of powers on x or independent 
variable.
y = w_0 + w_1 x_1 + w_2 (x_2)^2 ... w_n (x_n)^n

Residual Sum of Squares (RSS) = SUMMATION_i (y_i - f_w (x_i))^2
Loss or Mean Squared Error (MSE) = 1/N RSS
Root MSE (RMSE) = Square_root of MSE

Total Sum Square (TSS) = SUMMATION_i (y_i - AVG y)^2
Coefficient of determination (R^2) = 1 - (RSS / TSS) 

Least squares - analysitcal solution for minimum error
Simple formula, simple implementation, for big data matrix requires large 
memory

gradient descent methods - iterative solution or approximation to find 
minimum or maximum
Fast even with big data, Stocastic gradient descent is very memory 
efficient
𝛼 is a tuning parameter (or learning rate) to
control the speed and accuracy of convergence.

Underfitting when model cannot adequately capture the stucture of data 
(Linear)

Overfitting when model fits too closely to the training data (Too curvy)

Scaling - converting feature values on a similar scale

Normalization - Changing shape or distribution of data

If λ(penalty term or shrinkage factor) = 0, linear regression without 
regularzation. 

λ should be chosen wisely, it controls tradeoff between penalizing not 
fitting the data and penalizing overlayb complex models. 

Ridge and LASSO is effected by Tuning parameter λ.

Elastic Net is effected by 𝛼

Elastic Net becomes LASSO when 𝛼 = 1, Ridge i f𝛼 approaches 0

----------------------------------------------------------------------
Population -> Sample -> Data set

Bootstrap Sampling is creating set by drawing n examples at random with 
replacement.

CV is resampling method, K-fold cross-validation, shuffle data set 
randomly and split into k groups.

