clear;
clc;

inputFolder = '../../CardClips720out';
modelFile  = 'football_social_lstm_multiple_csv.mat';
outputRoot = 'simulink_results';

opts = struct;
opts.FrameSkip = 1;
opts.UseHydraulicCorrection = true;
opts.PredictionStepSeconds = 1/25;
opts.SimulinkModelName = 'sldemo_hydcyl4';
opts.SimulinkOutputName = 'sldemo_hydcyl4_output';
opts.SimulinkBlockPaths = {};

files = dir(fullfile(inputFolder,'*_tracks.csv'));

fprintf('Found %d ByteTrack files\n',numel(files));

allResults = cell(numel(files),1);
load_system(opts.SimulinkModelName);
for k = 62:numel(files)

    byteFile = fullfile(files(k).folder,files(k).name);

    [~,base,~] = fileparts(files(k).name);

    fprintf('\n=================================================\n');
    fprintf('Processing %d of %d\n',k,numel(files));
    fprintf('%s\n',files(k).name);
    fprintf('=================================================\n');

    try

        outputFolder = fullfile(outputRoot,base);

        if ~exist(outputFolder,'dir')
            mkdir(outputFolder);
        end

        allResults{k} = ...
            runFootballByteSocialSimulinkParameters( ...
                byteFile,...
                modelFile,...
                outputFolder,...
                opts);

        fprintf('SUCCESS: %s\n',files(k).name);

    catch ME

        fprintf('FAILED: %s\n',files(k).name);
        fprintf('%s\n',ME.message);

    end

end
close_system(opts.SimulinkModelName,0);
save(fullfile(outputRoot,'all_results.mat'),'allResults','-v7.3');

fprintf('\nFinished processing all files.\n');