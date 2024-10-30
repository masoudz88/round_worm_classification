% Load or Read Data
wormdata = readtable("WormData.csv");
imds = imageDatastore('WormImages', 'IncludeSubfolders', true, 'Labels', categorical(wormdata.Status));

% Preprocess Data
[trainData, testData] = splitEachLabel(imds, 0.6, "randomized");
trainds = augmentedImageDatastore([224 224], trainData, "ColorPreprocessing", "gray2rgb");
testds = augmentedImageDatastore([224 224], testData, "ColorPreprocessing", "gray2rgb");

% Define or Load Network
modelPath = 'trainedWormsNet.mat';
if exist(modelPath, 'file')
    load(modelPath, 'wormsNet');
else
    % Modify googlenet layers
    net = googlenet;
    lgraph = layerGraph(net);
    newFc = fullyConnectedLayer(2, "Name", "new_fc");
    lgraph = replaceLayer(lgraph, "loss3-classifier", newFc);
    newOut = classificationLayer("Name", "new_out");
    lgraph = replaceLayer(lgraph, "output", newOut);

    % Set Training Options
    options = trainingOptions("sgdm", "InitialLearnRate", 0.001);

    % Train the Network
    wormsNet = trainNetwork(trainds, lgraph, options);

    save(modelPath, 'wormsNet');
end

[preds, scores] = classify(wormsNet, testds);

truetest = testData.Labels;
nnz(preds == truetest)/numel(preds)

confusionchart(truetest, preds)
