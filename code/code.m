clear; clc;

% load data and shuffle
df = importdata('data_clean.csv');
rng(10);
n = randperm(length(df));
data = df(n, :);  % permuatation

% Split data set into 70 % training and 30 % testing
Xtrain = data(1:10500, 1:20);
Xtest = data(10501: end, 1:20);
ytrain = data(1:10500, end);
ytest = data(10501:end, end);

% renaming the labels as nntoolbox does not negative or zero values 
ytrain(ytrain == 1) = 2; 
ytrain(ytrain == 0) = 1;
ytest(ytest == 1) = 2;
ytest(ytest == 0) = 1;

% plot class distribution 
fprintf('\nClass distribution training data:')
fprintf('\n')
tabulate(ytrain)

fprintf('\nClass distributtion test data:')
fprintf('\n')
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



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Support Vector Machines (SVM)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%1) HP grid search 1) Search for best kernel function between linear,
%gaussian and polynomial kernels

% Create random partition for stratified 5-fold cross validation. Each fold
% roughly has the same class proportions.

cv = cvpartition(ytrain,'Kfold',5);

% loop over different kernel functions with 5 fold stratified cross
% validation
for i = {'linear', 'gaussian', 'polynomial'}
    % fitcecoc requires an SVM template
    t = templateSVM('KernelFunction', i{1});
    svm = fitcecoc(Xtrain, ytrain, 'learners', t, 'CVPartition', cv);
    accuracy = 1- kfoldLoss(svm);
    fprintf('\nAccuracy score of SVM with %s Kernel: %0.2f %', i{1}, accuracy)
end

% results:
%Accuracy score of SVM with linear Kernel: 0.78 
%Accuracy score of SVM with gaussian Kernel: 0.97 
%Accuracy score of SVM with polynomial Kernel: 0.95 

%% Continue with gaussian kernel and tune C and sigma

% create HP object
params = hyperparameters('fitcecoc', Xtrain, ytrain, 'svm');
% change range of C
params(2).Range = [0.1, 1];
% change range of sigma
params(3).Range = [0.1, 1];

% fit random search 
fitcecoc(Xtrain, ytrain, 'OptimizeHyperparameters', params,...
    'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName',...
    'expected-improvement-plus', 'Optimizer', 'randomsearch', 'MaxObjectiveEvaluations',...
    10, 'CVPartition', cv));


%% train on best HP and evaluate generalisation performance on test

% train on best HP values
t = templateSVM('KernelFunction', 'gaussian', 'KernelScale', 1, 'BoxConstraint', 1);
rng(10);
svm = fitcecoc(Xtrain, ytrain, 'learners', t);
% compute loss
train_error_svm = loss(svm, Xtrain, ytrain);
fprintf('\nSVM train accuracy: %0.2f\n', (1 - train_error_svm) * 100)
% test on test set
[ypred_svm, score_svm] = predict(svm, Xtest);  
test_error_svm = loss(svm, Xtest, ytest);
fprintf('\nSVM test accuracy: %0.2f\n', (1 - test_error_svm) * 100)

%% Confusion Matrix, Precision, Recall, F1 Score for SVM 

% svm
[Csvm, order] = confusionmat(ytest, ypred_svm);
precision_svm = Csvm(2,2)./(Csvm(2,2)+Csvm(1,2));
recall_svm =  Csvm(2,2)./(Csvm(2,2)+Csvm(2,1));
f1Score_svm =  2*(precision_svm.*recall_svm)./(precision_svm+recall_svm);

fprintf('Precision: %0.3f\n', precision_svm) 
fprintf('Recall: %0.3f\n', recall_svm) 
fprintf('F1: %0.3f\n', f1Score_svm) 

% Plot the confusion matrix
% Convert the integer label vector to a class-identifier matrix.
isLabels = unique(ytest);
nLabels = numel(isLabels);
[n,p] = size(Xtest);
[~,grpOOF] = ismember(ypred_svm,isLabels); 
oofLabelMat = zeros(nLabels,n); 
idxLinear = sub2ind([nLabels n],grpOOF,(1:n)'); 
oofLabelMat(idxLinear) = 1; % Flags the row corresponding to the class 
[~,grpY] = ismember(ytest,isLabels); 
YMat = zeros(nLabels,n); 
idxLinearY = sub2ind([nLabels n],grpY,(1:n)'); 
YMat(idxLinearY) = 1; 

figure;
plotconfusion(YMat,oofLabelMat);
h = gca;
h.XTickLabel = [num2cell(isLabels); {''}];
h.YTickLabel = [num2cell(isLabels); {''}];



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Mutli Layer Perceptron (MLP)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

x_train = Xtrain'; % training data 
label_train = ytrain'; % labels 

%convert each label to a spearate column as per the format required by the toolbox
vec = ind2vec(label_train) ;    
t_train = full(vec);

%% Hyperparameter tuning and k fold cross validation

%%%%%% Total time for hyperparameter tuning: 2.5 - 3hrs 

clear train 
for hiddenLayerSize = [10 20 40 60 80];  % number of hidden neurons
epochs = 500; % maximum number of epochs 
for lr = [0.05 0.1 0.3 0.6 0.9]; % learning rate 
for numLayers = [3 5 7];  % number of layers in the MLP 
net = feedforwardnet(hiddenLayerSize, 'trainscg'); % Stochastic conjugate gradient
net.trainParam.epochs = epochs;	% Maximum number of epochs to train	
net.trainParam.lr = lr; % learning rate	
net.trainParam.goal = 0.01;	% stop training if error gold reached
net.numLayers = numLayers; % number of layers in the MLP
% generate cross validation indices for partition of data into 5 folds
indices = crossvalind('Kfold',x_train(1,:),5);  
performance_cv = zeros(1,5);
for j = 1:5  % for each fold
    % samples which are present in fold j are true 
      testIdx = (indices == j); % boolean vector of test indices      
      trainIdx = ~testIdx  ; % boolean vector of train indices (which are not test)    
      trInd = find(trainIdx); % get training sample indices 
      tstInd = find(testIdx); % get test sample indices 
      
net.divideFcn = 'divideind'; % dividing the samples into sets using indices
net.divideParam.trainInd=trInd; % separate samples into train set using train indices 
net.divideParam.testInd=tstInd; % separate samples into test set using test indices
            
% Train the Network
[net,tr] = train(net, x_train, t_train);

% Fit the model on the training data 
pred_cv = net(x_train);
% calculate the difference between predicted and target values
e = gsubtract(t_train, pred_cv);
% compute performance of the network for a single fold 
performance_cv(:,j) = perform(net,t_train,pred_cv);

% View the Network
%view(net)

% Plots
% Uncomment these lines to enable various plots.
%figure(i), plotperform(tr)
%figure(i), plottrainstate(tr)
%figure(i), ploterrhist(e)

end 

% average cross validation accuracy after tuning network on 5 folds
fprintf('Average CV performance for following parameter settings: hidden layer:%d, epochs:%d,lr rate:%.2f,num Layers:%d, = %.4f \n', hiddenLayerSize,epochs, lr,numLayers, 100*(mean(performance_cv)));

end 
end
end


% Results from hyperparameter tuning (Average CV scores)
% best HP: hiddenlayersize : 60, lr 0.1, numLayers 3 


%% Re-Train the best classifer after hyperparameter tuning again

% total time for training model with best HP: ~30 mins

trainFcn = 'trainbr' % Here we apply a more robust method: Bayesian Regularisation Backpropagation 
hiddenLayerSize =60;  % number of hidden neurons
epochs = 500; % maximum number of epochs
lr = 0.1; % learning rate 
numLayers = 3;
% creating a MLP object setting hidden layer size and backprop algorithm 
net = feedforwardnet(hiddenLayerSize, trainFcn);  

% setting other parameters for the MLP

net.trainParam.epochs = epochs;	% Maximum number of epochs to train
net.trainParam.lr = lr; % learning rate	
net.trainParam.goal = 0.01;	% stop training if error gold reached
net.numLayers  = numLayers; % number of layers in MLP
% Setup Division of Data for Training, Validation, Testing
net.divideParam.trainRatio = 70/100; % keep 70% of data for training the model 
net.divideParam.valRatio = 30/100; % keep 30% of the training data for model evaluation

% Evaluate the Network on the split test dataset
[net,tr] = train(net, x_train, t_train);

% fit the model to the training data
pred_train = net(x_train);
% compute thedifference between the train and predicted
e = gsubtract(t_train, pred_train);

% evaluate performance (mean square error and cross entropy) on the train data
perf_train_mse = mse(net,t_train,pred_train) %MSE 
perf_train_crossentropy =  crossentropy(net,t_train,pred_train)% crossentropy 

% mse and cross entropy give the same values in this case 


%% Calculating the training accuracy 

% For each observation, we set the higher of the two predicted value to 1
% and the lower to 0.

% Now we compute accuracy by comparing predicted labels with the target labels 

for i =1:size(pred_train,2) % loop through all the columns 
    if pred_train(1,i)> pred_train(2,i) % if predicted value for one class is greater than the other 
        pred_train(1,i) = 1; % set the class with higher value to 1
        pred_train(2,i) = 0; % class with lower value set to 0
    else
        pred_train(1,i) = 0; % if statement above is not true, then do the opposite 
        pred_train(2,i) = 1;
    end 
end 

count = 0; % initialise the count to 0
for i =1:size(pred_train,2) % loop through all the columns 
if pred_train(1,i) == t_train(1,i); % if predicted is equal to target 
    count = count + 1;  % increment the count 
else 
    count = count; % otherwise leave it unchanged
end 
end 

train_accuracy = count/size(pred_train,2); % calculate proportion of correct classifications

fprintf('Training accuracy = %.4f \n', train_accuracy);

%% Test the model on unseen test data 

% View the Network
%view(net)

% Plots
% Uncomment these lines to enable various plots.
%figure(), plotperform(tr)
%figure(i), plottrainstate(tr)
%figure(i), ploterrhist(e)

%%%%%%% Test the network on unseen test data %%%%%%%%%%%%%%%

% Here we will test the model on the completely unseen test dataset
x_test = Xtest';
label_test = ytest';
vec = ind2vec(label_test) ; %convert each label to a spearate column with binary values 
t_test = full(vec) ; % as nn toolbox requires it in this format

% fit the model on the test data
pred_test = net(x_test);
% evaluate performance (mean square error and cross entropy) on the test data
perf_test_mse = mse(net,t_test,pred_test) %MSE 
perf_test_crossentropy =  crossentropy(net,t_test,pred_test)% crossentropy 

% mse and cross entropy give the same values in this case 

% Calculate the test accuracy 

for i =1:size(pred_test,2)
    if pred_test(1,i)> pred_test(2,i) % if predicted value for one class is greater than the other 
        pred_test(1,i) = 1; % set the class with higher value to 1
        pred_test(2,i) = 0; % class with lower value set to 0
    else
        pred_test(1,i) = 0; % if statement above is not true, then do the opposite 
        pred_test(2,i) = 1;
    end 
end 

count = 0; % initialise the count to 0
for i =1:size(pred_test,2) % loop through all the columns 
if pred_test(1,i) == t_test(1,i);% if predicted is equal to target 
    count = count + 1; % increment the count 
else 
    count = count; % otherwise leave it unchanged
end 
end 

test_accuracy = count/size(pred_test,2); % compute accuracy as proportion of correct classifications 


fprintf('Test accuracy is %.4f \n', test_accuracy);

%% Confusion Matrix, Precision, Recall, F1 Score for Neural Net 

% plot the confusion matrix 
plotconfusion(t_test,pred_test)

% c is the fraction of samples misclassified 
%Cnn is the 2 x 2 confusion matrix 
[c,Cnn] = confusion(t_test,pred_test) 

% computing the precision, recall and F1 score 
precision_nn = Cnn(2,2)./(Cnn(2,2)+Cnn(2,1));
recall_nn =  Cnn(2,2)./(Cnn(2,2)+Cnn(1,2));
f1Score_nn =  2*(precision_nn.*recall_nn)./(precision_nn+recall_nn);
fprintf('Precision: %0.3f\n', precision_nn) 
fprintf('Recall: %0.3f\n', recall_nn) 
fprintf('F1: %0.3f\n', f1Score_nn) 

