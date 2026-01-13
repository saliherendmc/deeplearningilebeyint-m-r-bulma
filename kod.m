clear; clc; close all;

fprintf('Program baslatildi...\n');

%% 1) Veri Seti Yolunu Ayarla
fprintf('Veri seti yolları ayarlaniyor...\n');

datasetPath = 'SalihDamacı_230757015_MuhammetÖzeler_230757025_BuğraEmirErbakan_230757009/VeriSeti';

trainFolder = fullfile(datasetPath,'train');
valFolder   = fullfile(datasetPath,'valid');
testFolder  = fullfile(datasetPath,'test');

assert(isfolder(trainFolder), 'Training klasörü bulunamadı');
assert(isfolder(valFolder),   'Validation klasörü bulunamadı');
assert(isfolder(testFolder),  'Testing klasörü bulunamadı');

fprintf('Klasörler basariyla bulundu.\n');

imdsTrain = imageDatastore(trainFolder, ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');

imdsVal = imageDatastore(valFolder, ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');

imdsTest = imageDatastore(testFolder, ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');

fprintf('Train / Validation / Test datastore olusturuldu.\n');

%% 2) ResNet-50 Hazırlığı
fprintf('ResNet-50 yukleniyor...\n');

net = resnet50;
inputSize = net.Layers(1).InputSize;

lgraph = layerGraph(net);
numClasses = numel(categories(imdsTrain.Labels));

fprintf('Sinif sayisi: %d\n', numClasses);

newLayers = [
    fullyConnectedLayer(numClasses, ...
        'Name','fc_new', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10)
    softmaxLayer('Name','softmax')
    classificationLayer('Name','classoutput')
];

lgraph = replaceLayer(lgraph,'fc1000',newLayers(1));
lgraph = replaceLayer(lgraph,'fc1000_softmax',newLayers(2));
lgraph = replaceLayer(lgraph,'ClassificationLayer_fc1000',newLayers(3));

fprintf('Son katmanlar guncellendi.\n');

%% 3) Veri Ön İşleme (Augmentation + Gray → RGB)
fprintf('Veri on isleme ve augmentation ayarlaniyor...\n');

pixelRange = [-10 10];

augTrain = augmentedImageDatastore( ...
    inputSize(1:2), imdsTrain, ...
    'DataAugmentation', imageDataAugmenter( ...
        'RandXReflection',true, ...
        'RandRotation',[-10 10], ...
        'RandXTranslation',pixelRange, ...
        'RandYTranslation',pixelRange), ...
    'ColorPreprocessing','gray2rgb');

augVal = augmentedImageDatastore( ...
    inputSize(1:2), imdsVal, ...
    'ColorPreprocessing','gray2rgb');

augTest = augmentedImageDatastore( ...
    inputSize(1:2), imdsTest, ...
    'ColorPreprocessing','gray2rgb');

fprintf('Augmentation ve Gray -> RGB donusumu tamamlandi.\n');

%% 4) Eğitim Ayarları
fprintf('Egitim ayarlari yapiliyor...\n');

options = trainingOptions('adam', ...
    'MiniBatchSize',64, ...
    'MaxEpochs',1, ...
    'InitialLearnRate',5e-5, ...
    'ValidationData',augVal, ...
    'ValidationFrequency',65, ...
    'Shuffle','every-epoch', ...
    'Plots','training-progress', ...  
    'Verbose',false);

fprintf('Egitim basliyor...\n');

%% 5) Modeli Eğit
netTransfer = trainNetwork(augTrain, lgraph, options);

fprintf('Egitim tamamlandi.\n');

%% 6) Test Başarısı
fprintf('Test verisi uzerinde tahmin yapiliyor...\n');

predLabels = classify(netTransfer, augTest);
trueLabels = imdsTest.Labels;

testAccuracy = mean(predLabels == trueLabels);
fprintf('Test Accuracy: %.2f%%\n', testAccuracy*100);

%% 7) Confusion Matrix 
fprintf('Confusion Matrix gorsellestiriliyor...\n');

figure;
cm = confusionchart(trueLabels, predLabels);
cm.Title = 'Confusion Matrix';
cm.RowSummary = 'row-normalized';
cm.ColumnSummary = 'column-normalized';

fprintf('Program basariyla tamamlandi.\n');