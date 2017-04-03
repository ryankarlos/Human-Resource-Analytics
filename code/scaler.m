function [Xscaled, mu, stddev] = scaler(X)
% This function standardizes the features to mu=0 and stddev=1.

Xscaled = X;
mu = zeros(1, size(X, 2));
stddev = zeros(1, size(X, 2));

% Perform feature scaling for every feature
for i=1:size(mu,2)
    mu(1,i) = mean(X(:,i)); % calculate the mean
    stddev(1,i) = std(X(:,i)); % calculate the stddev
    Xscaled(:,i) = (X(:,i)-mu(1,i))/stddev(1,i); % subtract the mean and devide by stddev
end