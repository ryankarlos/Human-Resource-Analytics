clear; clc;

% load data and shuffle
df = importdata('train.csv');
data = df.data;
n = randperm(length(data));
data = data(n, :);  % permuatation
label = data(:, end);  % target variable

%% reduce data set version 1)
% run either this version or version 2) 
Xtrain = data(1:10000, 1:8);
Xtest = data(10001:20000, 1:8);
ytrain = label(1:10000);
ytest = label(10001:20000);

%% reduce data set version 2)
% reduce such that every class is evenly distributed
train_zero = data(label==0, :);
train_0 = train_zero(1:3500, :);
test_0 = train_zero(3501:5000, :);

train_one = data(label==1, :);
train_1 = train_one(1:3500, :);
test_1 = train_one(3501:5000, :);

train_two = data(label==2, :);
train_2 = train_two(1:2500, :);
test_2 = train_two(2501:3800, :);

train = vertcat(train_0, train_1, train_2);
test = vertcat(test_0, test_1, test_2);
Xtrain = train(:, 1:8);
Xtest = test(:, 1:8);
ytrain = train(:, end);
ytest = test(:, end);

% class distribution 
tabulate(ytrain)  
tabulate(ytest)


%% Feature Scaling 

% Apply feature scaling with mu=0 and stddev=1 to the features. 
% This is crucial for the subsequent SVM and Neural Net.

% Save the output as the scaled Xtrain matrix as well as the corresponding 
% mu and stddev of every feature column.

[Xtrain, mu, stddev] = scaler(Xtrain); 

% Standardize the test set with mu and std of the training set
% It is crucial to use mu and stddev from the training set as
% our test set represents an unseen set of samples.

for i=1:size(Xtest, 2)
    Xtest(:,i) = (Xtest(:,i)-mu(1,i))/stddev(1,i);
end

fprintf('\nStandardizing training and test data with mu=0 and std=1:')
fprintf('\n')
fprintf('\nMean of standardized Xtrain: %.3f\nStd of standardized Xtrain: %.3f',...
    mean(mean(Xtrain(:,:))), mean(std(Xtrain(:,:))))
fprintf('\n')
fprintf('\nMean of standardized Xtest: %.3f\nStd of standardized Xtest: %.3f\n',...
    mean(mean(Xtest(:,:))), mean(std(Xtest(:,:))))


%% SVM training

%% 5 fold CV (< 40min)
%rng(10);

%cv = fitcecoc(Xtrain, ytrain, 'Kfold', 5);  
%fprintf('\nSVM 5-fold CV accuracy: %0.2f\n', (1 - kfoldLoss(cv)) * 100)


%% HP grid search (55min)
for i = {'linear', 'gaussian', 'polynomial'}
    t = templateSVM('KernelFunction', i{1});
    svm = fitcecoc(Xtrain, ytrain, 'learners', t, 'Kfold', 5);
    accuracy = 1- kfoldLoss(svm);
    fprintf('\nAccuracy score of SVM with %s Kernel: %0.2f %', i{1}, accuracy)
end


%% train on best HP and evaluate generalisation performance on test

% train on best HP values
rng(10);
t = templateSVM('KernelFunction', 'gaussian');
svm = fitcecoc(Xtrain, ytrain, 'learners', t);
train_error_svm = loss(svm, Xtrain, ytrain);
fprintf('\nSVM train accuracy: %0.2f\n', (1 - train_error_svm) * 100)
% test on test set
[ypred_svm, score_svm] = predict(svm, Xtest);  
test_error_svm = loss(svm, Xtest, ytest);
fprintf('\nSVM test accuracy: %0.2f\n', (1 - test_error_svm) * 100)


%% Vanilla Feedforward Neural Network

data_x = Xtrain';
label = ytrain';
label(label == 2) = 3;  % need to be labelled from 1 to feed in nn in matlab
label(label == 1) = 2;
label(label == 0) = 1;
vec = ind2vec(label) ; %convert each label to a spearate column with binary values 
target_var = full(vec) ; % as nn toolbox requires it in this format
%% Hyperparameter tuning and k fold cross validation
clear train 
x = data_x;
t = target_var;
i = 1;
for hiddenLayerSize = [100];  % number of hidden neurons
for epochs = [500]; % maximum number of epochs 
for lr = [0.01 0.1 0.9]; % learning rate 
for numLayers = [3 5 10];    
net = feedforwardnet(hiddenLayerSize, 'trainscg'); % Stochastic conjugate gradient
net.trainParam.epochs = epochs;	% Maximum number of epochs to train		
net.trainParam.lr = lr; % learning rate	
net.trainParam.goal = 0.01;	% stop training if error gold reached
net.numLayers = numLayers;
indices = crossvalind('Kfold',x(1,:),5);
percentErrors = zeros(1,5);
for j = 1:5  % for each fold
      testIdx = (indices == i);   
      trainIdx = ~testIdx  ;    
      trInd = find(trainIdx); % get indices of test instances
      tstInd = find(testIdx); % get indices training instances
      
net.divideFcn = 'divideind'; 
net.divideParam.trainInd=trInd;
net.divideParam.testInd=tstInd;
            
% Train the Network
[net,tr] = train(net, x, t);

% Test the Network

y = net(x);
%e = gsubtract(t, y);
tind = vec2ind(t);
yind = vec2ind(y);
percentErrors(:,j) = sum(tind ~= yind)/numel(tind);

% View the Network
%view(net)

% Plots
% Uncomment these lines to enable various plots.
%figure(i), plotperform(tr)
%figure(i), plottrainstate(tr)
%figure(i), ploterrhist(e)
%figure(i), plotconfusion(tind, yind)
%figure(i), plotroc(tind, yind) 

end 

fprintf('Average CV accuracy for following parameter settings: hidden layer:%d, epochs:%d,lr rate:%.2f,num Layers:%d, = %.4f \n', hiddenLayerSize,epochs, lr,numLayers, 100*(1-mean(percentErrors)));
i = i + 1;

end 
end
end
end

%% Train the best classifer again and test it on unseen data 
x = data_x;
t = target_var;
trainFcn = 'trainbr' % Here we apply a more robust method: Bayesian Regularisation Backpropagation 
hiddenLayerSize =100;  % number of hidden neurons
epochs = 300; % maximum number of epochs
lr = 0.01; % learning rate 
numLayers = 5
net = feedforwardnet(hiddenLayerSize, trainFcn);
net.trainParam.epochs = epochs;	% Maximum number of epochs to train
net.trainParam.lr = lr; % learning rate	
net.trainParam.goal = 0.01;	% stop training if error gold reached
net.numLayers  = numLayers % number of layers 
net.trainParam.max_fail = 6 % early stopping criterion
% Setup Division of Data for Training, Validation, Testing
net.divideParam.trainRatio = 70/100; % keep 70% of data for training the model 
net.divideParam.valRatio = 15/100 % 15% for validation 
net.divideParam.testRatio = 15/100; % keep 30% of the training data for model evaluation


% Evaluate the Network on the split test dataset
[net,tr] = train(net, x, t);

y= zeros(size(t));
y = net(x);
%e = gsubtract(t, y);
tind = vec2ind(t);
yind = vec2ind(y);
percentErrors = sum(tind ~= yind)/numel(tind);

fprintf('Training accuracy with hidden layer:%d, epochs:%d,lr rate:%.2f = %.4f \n', hiddenLayerSize,epochs, lr, 100*(1-percentErrors));

% View the Network
%view(net)

% Plots
% Uncomment these lines to enable various plots.
figure(), plotperform(tr)
%figure(i), plottrainstate(tr)
%figure(i), ploterrhist(e)
%figure(i), plotconfusion(t, y)
%figure(i), plotroc(t, y) 

%%%%%%% Test the network on unseen test data %%%%%%%%%%%%%%%

% Here we will test the model on the completely unseen test dataset
data_x_test = Xtest';
label_test = ytest';

label_test(label_test == 2) = 3;  % need to be labelled from 1 to feed in nn in matlab
label_test(label_test == 1) = 2;
label_test(label_test == 0) = 1;
vec = ind2vec(label_test) ; %convert each label to a spearate column with binary values 
target_var_test = full(vec) ; % as nn toolbox requires it in this format

output_test = net(data_x_test);
tind = vec2ind(output_test);
yind = vec2ind(target_var_test);
percentErrors_test = sum(tind ~= yind)/numel(tind);

fprintf('Test accuracy is %.4f \n', 100*(1-percentErrors_test));


%% Confusion Matrix, Precision, Recall, F1 Score for SVM 

% svm
[Csvm, order] = confusionmat(ytest, ypred_svm);
precision_svm_class1 = Csvm(1,1)./(Csvm(1,1)+Csvm(1,2)+Csvm(1,3));
precision_svm_class2 = Csvm(2,2)./(Csvm(1,2)+Csvm(2,2)+Csvm(3,2));
precision_svm_class3 = Csvm(3,3)./(Csvm(1,3)+Csvm(2,3)+Csvm(3,3));
recall_svm_class1 =  Csvm(1,1)./(Csvm(1,1)+Csvm(1,2)+Csvm(1,3));
recall_svm_class2 =  Csvm(2,1)./(Csvm(2,1)+Csvm(2,2)+Csvm(2,3));
recall_svm_class3 =  Csvm(3,1)./(Csvm(3,1)+Csvm(3,2)+Csvm(3,3));
f1Score_svm_class1 =  2*(precision_svm_class1.*recall_svm_class1)./(precision_svm_class1+recall_svm_class1);
f1Score_svm_class2 =  2*(precision_svm_class2.*recall_svm_class2)./(precision_svm_class2+recall_svm_class2);
f1Score_svm_class3 =  2*(precision_svm_class3.*recall_svm_class3)./(precision_svm_class3+recall_svm_class3);


fprintf('F1_class1: %0.3f\n', f1Score_svm_class1) 
fprintf('F1_class2: %0.3f\n', f1Score_svm_class2) 
fprintf('F1_class3: %0.3f\n', f1Score_svm_class3) 

%% Confusion Matrix, Precision, Recall, F1 Score for Neural Net 

[Cnn,c] = confusionmat(tind,yind);
precision_nn_class1 = Cnn(1,1)./(Cnn(1,1)+Cnn(1,2)+Cnn(1,3));
precision_nn_class2 = Cnn(2,2)./(Cnn(1,2)+Cnn(2,2)+Cnn(3,2));
precision_nn_class3 = Cnn(3,3)./(Cnn(1,3)+Cnn(2,3)+Cnn(3,3));
recall_nn_class1 =  Cnn(1,1)./(Cnn(1,1)+Cnn(1,2)+Cnn(1,3));
recall_nn_class2 =  Cnn(2,1)./(Cnn(2,1)+Cnn(2,2)+Cnn(2,3));
recall_nn_class3 =  Cnn(3,1)./(Cnn(3,1)+Cnn(3,2)+Cnn(3,3));
f1Score_nn_class1 =  2*(precision_nn_class1.*recall_nn_class1)./(precision_nn_class1+recall_nn_class1);
f1Score_nn_class2 =  2*(precision_nn_class2.*recall_nn_class2)./(precision_nn_class2+recall_nn_class2);
f1Score_nn_class3 =  2*(precision_nn_class3.*recall_nn_class3)./(precision_nn_class3+recall_nn_class3);

fprintf('F1_class1: %0.3f\n', f1Score_nn_class1) 
fprintf('F1_class2: %0.3f\n', f1Score_nn_class2) 
fprintf('F1_class3: %0.3f\n', f1Score_nn_class3) 

plotconfusion(target_var_test,output_test)