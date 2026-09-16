function results = runFootballByteSocialSimulinkParameters(byteFile, modelFile, outputFolder, opts)
%RUNFOOTBALLBYTESOCIALHYDRAULIC Integrated football analysis pipeline.
% 1) detects players, 2) tracks them with FootballByteTracker,
% 4) predicts future positions with the
% trained occupancy-pooled Social-LSTM model, 5) applies an optional
% hydraulic-inspired second-order smoothing correction, and 6) computes
% ADE/FDE where future tracked positions are available.
%
% Required existing files:
%   FootballByteTracker.m, FootballTrack.m,
%   makeOccupancyGrid.m, preparePlayerPositions.m
%
% Example:
% detector = yolov4ObjectDetector('csp-darknet53-coco');
% opts = struct;
% opts.FrameSkip = 1;
% opts.UseHydraulicCorrection = true;
% results = runFootballByteSocialSimulinkParameters('match.mp4', detector, ...
%     'football_social_lstm_model.mat', 'results', opts);

if nargin < 3 || isempty(outputFolder), outputFolder = 'results'; end
if nargin < 4, opts = struct; end
opts = setDefaults(opts);
if ~exist(outputFolder,'dir'), mkdir(outputFolder); end

S = load(modelFile);
requiredModelFields = {'net','obsLength','predLength','gridSize','neighbourhoodRadius'};
for k = 1:numel(requiredModelFields)
    if ~isfield(S,requiredModelFields{k})
        error('Model file is missing variable: %s',requiredModelFields{k});
    end
end

T = readtable(byteFile);

Tp = preparePlayerPositions(T);

[X,YCell,meta] = buildSocialSequences(...
     Tp,...
     S.obsLength,...
     S.predLength,...
     S.gridSize,...
     S.neighbourhoodRadius);

[~,base,~] = fileparts(byteFile);

% Build test windows from ByteTrack trajectories. The trained model is not
% retrained here; this stage tests/predicts on the current video.
Tp = preparePlayerPositions(T);
[X,YCell,meta] = buildSocialSequences(Tp,S.obsLength,S.predLength, ...
    S.gridSize,S.neighbourhoodRadius);
if isempty(X)
    warning('Tracking completed, but no complete prediction windows exist.');
    results = struct('Tracks',T,'Predictions',table,'Summary',table);
    return;
end
YTrue = double(vertcat(YCell{:}));
valid = all(isfinite(YTrue),2);
for k=1:numel(X)
    valid(k)=valid(k) && all(isfinite(X{k}(:)));
end
X=X(valid); YTrue=YTrue(valid,:); meta=meta(valid);

YPred = predict(S.net,X,'MiniBatchSize',64);
if iscell(YPred), YPred=vertcat(YPred{:}); end
if size(YPred,1)==2*S.predLength && size(YPred,2)==numel(X), YPred=YPred'; end

% Generate player parameters from Social-LSTM predictions, save a
% Weka-compatible CSV/ARFF, update Simulink blocks, and run the model.
YCorrected = YPred;
parameterTable = table;
if opts.UseHydraulicCorrection
    parameterPrefix = fullfile(outputFolder,[base '_predicted_player_parameters']);
    
    [~,basev,~] = fileparts(byteFile);
    basev = erase(basev,'_tracks');

    videoFile = fullfile( '../../CardClips720',[basev '.mp4']);

    Tp = assignStableTeamColoursFromVideo(Tp,videoFile);
    
    [YCorrected,parameterTable,hydraulicInfo] = ...
        applyPlayerParameterSimulinkResponse(YPred,meta,Tp,S.predLength, ...
        opts.PredictionStepSeconds,opts.SimulinkModelName, ...
        opts.SimulinkBlockPaths,opts.SimulinkOutputName,parameterPrefix);
else
    hydraulicInfo = struct;
end


% Build full frame-level metrics before and after Simulink correction.
% "injury" is a relative model-generated risk label, not a diagnosis.

[metricsBefore, metricsAfter, injuredPlayerID, playerRiskTable] = ...
    buildBeforeAfterPlayerMetrics( ...
        YPred, ...
        YCorrected, ...
        meta, ...
        T, ...
        S.predLength, ...
        opts.PredictionStepSeconds, ...
        'football_injury_weka_sub.csv', ...
        5, ...
        0.5, ...
        0.5);

riskFile = fullfile( ...
    outputFolder, ...
    [base '_combined_player_risk.csv']);

writetable(playerRiskTable, riskFile);

if ~isempty(parameterTable)

    parameterTable.Injury_Next_Season(:)=0;

    parameterTable.Injury_Next_Season( ...
        parameterTable.Player_ID==injuredPlayerID)=1;

    parameterPrefix = fullfile( ...
        outputFolder,...
        [base '_predicted_player_parameters']);

    writetable(parameterTable,[parameterPrefix '.csv']);
    writeWekaARFF([parameterPrefix '.arff'],parameterTable);

end

metricsBeforeFile = fullfile( ...
    outputFolder, [base '_player_metrics_before_correction.csv']);

metricsAfterFile = fullfile( ...
    outputFolder, [base '_player_metrics_after_correction.csv']);

writetable(metricsBefore, metricsBeforeFile);
writetable(metricsAfter, metricsAfterFile);

fprintf('Highest injury-risk player ID: %g\n', injuredPlayerID);
fprintf('Before-correction metrics: %s\n', metricsBeforeFile);
fprintf('After-correction metrics:  %s\n', metricsAfterFile);

raw = evaluateTrajectoryAccuracy(YPred,YTrue,meta,S.predLength);
corrected = evaluateTrajectoryAccuracy(YCorrected,YTrue,meta,S.predLength);

n=size(YTrue,1); playerID=zeros(n,1); startFrame=zeros(n,1); 

parfor k=1:n
    playerID(k)=meta(k).TrackID; startFrame(k)=meta(k).StartFrame;
    q=T(T.TrackID==playerID(k),:);
end
P=table(playerID,startFrame,raw.PerSequence.ADE,raw.PerSequence.FDE, ...
    corrected.PerSequence.ADE,corrected.PerSequence.FDE, ...
    'VariableNames',{'TrackID','StartFrame','RawADE','RawFDE', ...
    'CorrectedADE','CorrectedFDE'});
writetable(P,fullfile(outputFolder,[base '_prediction_accuracy.csv']));

summary=table(mean(P.RawADE),mean(P.RawFDE),mean(P.CorrectedADE), ...
    mean(P.CorrectedFDE),height(P),'VariableNames', ...
    {'MeanRawADE','MeanRawFDE','MeanCorrectedADE','MeanCorrectedFDE','NumWindows'});
writetable(summary,fullfile(outputFolder,[base '_accuracy_summary.csv']));

results = struct( ...
    'Tracks', T, ...
    'Predictions', P, ...
    'Summary', summary, ...
    'PlayerParameters', parameterTable, ...
    'MetricsBeforeCorrection', metricsBefore, ...
    'MetricsAfterCorrection', metricsAfter, ...
    'HighestInjuryRiskPlayerID', injuredPlayerID, ...
    'HydraulicInfo', hydraulicInfo, ...
    'InputSequences', X, ...
    'RawPrediction', YPred, ...
    'CorrectedPrediction', YCorrected, ...
    'GroundTruth', YTrue);

disp(summary);
end

function opts=setDefaults(opts)
def=struct('FrameSkip',1,'HighThreshold',0.60,'LowThreshold',0.10, ...
 'FirstMatchIoU',0.20,'SecondMatchIoU',0.20,'NewTrackThreshold',0.70, ...
 'MaxLostFrames',30,'MinConfirmedHits',2, ...
 'UseHydraulicCorrection',true,'PredictionStepSeconds',0.04, ...
 'SimulinkModelName','sldemo_hydcyl4','SimulinkBlockPaths',{{}}, ...
 'SimulinkOutputName','sldemo_hydcyl4_output');
fn=fieldnames(def);
for k=1:numel(fn), if ~isfield(opts,fn{k}), opts.(fn{k})=def.(fn{k}); end, end
end

function [beforeTable, afterTable, injuredPlayerID, playerRiskTable] = ...
    buildBeforeAfterPlayerMetrics( ...
        YBefore, YAfter, meta, trackTable, predLength, dt, ...
        sensorCSVFile, numberOfNeighbours, ...
        trajectoryWeight, sensorWeight)
%BUILDBEFOREAFTERPLAYERMETRICS
% Combine trajectory-derived player risk with labelled sensor data using
% position-aware, distance-weighted k-nearest neighbours.
%
% Expected sensor CSV columns:
%   Position
%   Reaction_Time_ms
%   Sprint_Speed_10m_s
%   Agility_Score
%   Stress_Level_Score
%   Injury_Next_Season
%
% Outputs:
%   beforeTable:
%       frame-level metrics before hydraulic correction
%
%   afterTable:
%       frame-level metrics after hydraulic correction
%
%   injuredPlayerID:
%       TrackID of the single player with the highest combined risk
%
%   playerRiskTable:
%       one row per player containing trajectory risk, sensor KNN risk,
%       combined risk, nearest-neighbour information and final injury flag
%
% Exactly one player is assigned:
%   injury = 1
%
% This is a relative model-generated risk indicator and is not a medical
% diagnosis.

    %% ---------------------------------------------------------------
    %  Default arguments
    %  ---------------------------------------------------------------

    if nargin < 7 || isempty(sensorCSVFile)
        sensorCSVFile = 'football_injury_weka_sub.csv';
    end

    if nargin < 8 || isempty(numberOfNeighbours)
        numberOfNeighbours = 5;
    end

    if nargin < 9 || isempty(trajectoryWeight)
        trajectoryWeight = 0.5;
    end

    if nargin < 10 || isempty(sensorWeight)
        sensorWeight = 0.5;
    end

    numberOfNeighbours = max(1, round(numberOfNeighbours));

    %% ---------------------------------------------------------------
    %  Build frame-level trajectory metric tables
    %  ---------------------------------------------------------------

    beforeTable = trajectoryMetricsTable( ...
        YBefore, ...
        meta, ...
        trackTable, ...
        predLength, ...
        dt);

    afterTable = trajectoryMetricsTable( ...
        YAfter, ...
        meta, ...
        trackTable, ...
        predLength, ...
        dt);

    % Ensure both tables contain an injury column.
    if ~ismember('injury', beforeTable.Properties.VariableNames)
        beforeTable.injury = zeros(height(beforeTable), 1);
    else
        beforeTable.injury(:) = 0;
    end

    if ~ismember('injury', afterTable.Properties.VariableNames)
        afterTable.injury = zeros(height(afterTable), 1);
    else
        afterTable.injury(:) = 0;
    end

    %% ---------------------------------------------------------------
    %  Find all players
    %  ---------------------------------------------------------------

    playerIDs = unique(beforeTable.player_id);
    playerIDs = playerIDs(isfinite(playerIDs));

    numberOfPlayers = numel(playerIDs);

    if numberOfPlayers == 0
        injuredPlayerID = NaN;
        playerRiskTable = table;

        warning( ...
            'No valid players were available for injury-risk analysis.');

        return;
    end

    %% ---------------------------------------------------------------
    %  Allocate player-level variables
    %  ---------------------------------------------------------------

    playerPosition = strings(numberOfPlayers, 1);

    rawReaction = zeros(numberOfPlayers, 1);
    rawSpeed = zeros(numberOfPlayers, 1);
    rawAgility = zeros(numberOfPlayers, 1);
    rawStress = zeros(numberOfPlayers, 1);

    correctionDistance = zeros(numberOfPlayers, 1);

    %% ---------------------------------------------------------------
    %  Aggregate trajectory metrics by player
    %  ---------------------------------------------------------------

    parfor i = 1:numberOfPlayers

        currentPlayerID = playerIDs(i);

        beforeRows = ...
            beforeTable.player_id == currentPlayerID;

        afterRows = ...
            afterTable.player_id == currentPlayerID;

        rawReaction(i) = safeMedian( ...
            beforeTable.reaction_time(beforeRows));

        rawSpeed(i) = safeMedian( ...
            beforeTable.speed(beforeRows));

        rawAgility(i) = safeMedian( ...
            beforeTable.agility(beforeRows));

        rawStress(i) = safeMedian( ...
            beforeTable.stress(beforeRows));

        % Determine the player's most frequent position.
        positionValues = string( ...
            beforeTable.position(beforeRows));

        positionValues = positionValues( ...
            positionValues ~= "" & ...
            ~strcmpi(positionValues, "Unknown"));

        if isempty(positionValues)
            playerPosition(i) = "Unknown";
        else
            playerPosition(i) = ...
                string(mode(categorical(positionValues)));
        end

        % Calculate the median distance between raw and corrected paths.
        beforeXY = [ ...
            beforeTable.x(beforeRows), ...
            beforeTable.y(beforeRows)];

        afterXY = [ ...
            afterTable.x(afterRows), ...
            afterTable.y(afterRows)];

        commonCount = min( ...
            size(beforeXY, 1), ...
            size(afterXY, 1));

        if commonCount > 0

            displacement = sqrt(sum( ...
                (beforeXY(1:commonCount, :) - ...
                 afterXY(1:commonCount, :)).^2, ...
                2));

            correctionDistance(i) = ...
                safeMedian(displacement);
        else
            correctionDistance(i) = 0;
        end
    end

    %% ---------------------------------------------------------------
    %  Calculate trajectory-only risk
    %  ---------------------------------------------------------------

    reactionTrajectoryN = normaliseRiskColumn(rawReaction);
    speedTrajectoryN = normaliseRiskColumn(rawSpeed);
    agilityTrajectoryN = normaliseRiskColumn(rawAgility);
    stressTrajectoryN = normaliseRiskColumn(rawStress);
    correctionTrajectoryN = ...
        normaliseRiskColumn(correctionDistance);

    trajectoryRisk = ...
        0.10 .* reactionTrajectoryN + ...
        0.20 .* speedTrajectoryN + ...
        0.15 .* agilityTrajectoryN + ...
        0.40 .* stressTrajectoryN + ...
        0.15 .* correctionTrajectoryN;

    trajectoryRisk = max(0, min(1, trajectoryRisk));

    %% ---------------------------------------------------------------
    %  Initialise KNN outputs
    %  ---------------------------------------------------------------

    mappedReaction = nan(numberOfPlayers, 1);
    mappedSpeed = nan(numberOfPlayers, 1);
    mappedAgility = nan(numberOfPlayers, 1);
    mappedStress = nan(numberOfPlayers, 1);

    sensorKNNRisk = zeros(numberOfPlayers, 1);
    sensorKNNClass = zeros(numberOfPlayers, 1);

    nearestSensorDistance = nan(numberOfPlayers, 1);
    meanNeighbourDistance = nan(numberOfPlayers, 1);

    neighboursUsed = zeros(numberOfPlayers, 1);
    samePositionNeighbours = false(numberOfPlayers, 1);

    nearestSensorPosition = strings(numberOfPlayers, 1);
    neighbourInjuryVotes = strings(numberOfPlayers, 1);

    %% ---------------------------------------------------------------
    %  Load and validate sensor data
    %  ---------------------------------------------------------------

    sensorAvailable = ...
        ~isempty(sensorCSVFile) && isfile(sensorCSVFile);

    if ~sensorAvailable

        warning( ...
            ['Sensor CSV was not found: %s. ', ...
             'Only trajectory risk will be used.'], ...
            sensorCSVFile);

        trajectoryWeight = 1;
        sensorWeight = 0;

    else

        sensorTable = readtable( ...
            sensorCSVFile, ...
            'VariableNamingRule', ...
            'preserve');

        requiredColumns = { ...
            'Position', ...
            'Reaction_Time_ms', ...
            'Sprint_Speed_10m_s', ...
            'Agility_Score', ...
            'Stress_Level_Score', ...
            'Injury_Next_Season'};

        for columnIndex = 1:numel(requiredColumns)

            if ~ismember( ...
                    requiredColumns{columnIndex}, ...
                    sensorTable.Properties.VariableNames)

                error( ...
                    ['Missing sensor CSV column: %s\n', ...
                     'Available columns are: %s'], ...
                    requiredColumns{columnIndex}, ...
                    strjoin( ...
                        sensorTable.Properties.VariableNames, ...
                        ', '));
            end
        end

        sensorPosition = string( ...
            sensorTable.("Position"));

        sensorReaction = convertToNumeric( ...
            sensorTable.("Reaction_Time_ms"));

        sensorSpeed = convertToNumeric( ...
            sensorTable.("Sprint_Speed_10m_s"));

        sensorAgility = convertToNumeric( ...
            sensorTable.("Agility_Score"));

        sensorStress = convertToNumeric( ...
            sensorTable.("Stress_Level_Score"));

        sensorLabels = convertToNumeric( ...
            sensorTable.("Injury_Next_Season"));

        sensorLabels = double(sensorLabels >= 0.5);

        validSensorRows = ...
            sensorPosition ~= "" & ...
            all(isfinite([ ...
                sensorReaction, ...
                sensorSpeed, ...
                sensorAgility, ...
                sensorStress]), 2) & ...
            isfinite(sensorLabels);

        sensorPosition = sensorPosition(validSensorRows);
        sensorReaction = sensorReaction(validSensorRows);
        sensorSpeed = sensorSpeed(validSensorRows);
        sensorAgility = sensorAgility(validSensorRows);
        sensorStress = sensorStress(validSensorRows);
        sensorLabels = sensorLabels(validSensorRows);

        sensorFeatures = [ ...
            sensorReaction, ...
            sensorSpeed, ...
            sensorAgility, ...
            sensorStress];

        if isempty(sensorFeatures)

            warning( ...
                ['The sensor CSV contains no complete valid rows. ', ...
                 'Only trajectory risk will be used.']);

            trajectoryWeight = 1;
            sensorWeight = 0;

        else

            %% -------------------------------------------------------
            %  Map video-derived features to sensor feature ranges
            %
            %  Video speed is measured in image-coordinate units, while
            %  sensor speed is measured on the sensor dataset's scale.
            %  Percentile mapping preserves relative player ordering
            %  without modifying values in the hydraulic model.
            %  -------------------------------------------------------

            mappedReaction = percentileMapToReference( ...
                rawReaction, sensorReaction);

            mappedSpeed = percentileMapToReference( ...
                rawSpeed, sensorSpeed);

            mappedAgility = percentileMapToReference( ...
                rawAgility, sensorAgility);

            mappedStress = percentileMapToReference( ...
                rawStress, sensorStress);

            mappedPlayerFeatures = [ ...
                mappedReaction, ...
                mappedSpeed, ...
                mappedAgility, ...
                mappedStress];

            %% -------------------------------------------------------
            %  Standardise features using sensor-data statistics
            %  -------------------------------------------------------

            sensorMean = mean( ...
                sensorFeatures, ...
                1, ...
                'omitnan');

            sensorStandardDeviation = std( ...
                sensorFeatures, ...
                0, ...
                1, ...
                'omitnan');

            invalidScale = ...
                ~isfinite(sensorStandardDeviation) | ...
                sensorStandardDeviation < eps;

            sensorStandardDeviation(invalidScale) = 1;

            standardSensorFeatures = ...
                (sensorFeatures - sensorMean) ./ ...
                sensorStandardDeviation;

            standardPlayerFeatures = ...
                (mappedPlayerFeatures - sensorMean) ./ ...
                sensorStandardDeviation;

            %% -------------------------------------------------------
            %  Position-aware distance-weighted KNN
            %  -------------------------------------------------------

            for i = 1:numberOfPlayers

                currentPosition = playerPosition(i);

                % Prefer sensor examples with the same position.
                samePositionRows = strcmpi( ...
                    sensorPosition, ...
                    currentPosition);

                if any(samePositionRows)

                    candidateIndexes = find(samePositionRows);
                    samePositionNeighbours(i) = true;

                else

                    % Fallback to all sensor rows if the player's
                    % position is unknown or absent from the sensor data.
                    candidateIndexes = ...
                        (1:size(sensorFeatures, 1))';

                    samePositionNeighbours(i) = false;
                end

                candidateFeatures = ...
                    standardSensorFeatures(candidateIndexes, :);

                currentFeature = ...
                    standardPlayerFeatures(i, :);

                distances = sqrt(sum( ...
                    (candidateFeatures - currentFeature).^2, ...
                    2));

                [sortedDistances, sortedOrder] = ...
                    sort(distances, 'ascend');

                kUsed = min( ...
                    numberOfNeighbours, ...
                    numel(sortedOrder));

                localNeighbourIndexes = ...
                    sortedOrder(1:kUsed);

                sensorNeighbourIndexes = ...
                    candidateIndexes(localNeighbourIndexes);

                selectedDistances = ...
                    sortedDistances(1:kUsed);

                selectedLabels = ...
                    sensorLabels(sensorNeighbourIndexes);

                % Prevent division by zero for an exact match.
                distanceWeights = ...
                    1 ./ (selectedDistances + 1e-9);

                sensorKNNRisk(i) = ...
                    sum(distanceWeights .* selectedLabels) ./ ...
                    sum(distanceWeights);

                sensorKNNClass(i) = ...
                    double(sensorKNNRisk(i) >= 0.5);

                nearestSensorDistance(i) = ...
                    selectedDistances(1);

                meanNeighbourDistance(i) = ...
                    mean(selectedDistances);

                neighboursUsed(i) = kUsed;

                nearestSensorPosition(i) = ...
                    sensorPosition(sensorNeighbourIndexes(1));

                neighbourInjuryVotes(i) = ...
                    strjoin( ...
                        string(selectedLabels'), ...
                        ',');
            end
        end
    end

    %% ---------------------------------------------------------------
    %  Combine trajectory and sensor risk
    %  ---------------------------------------------------------------

    trajectoryWeight = max(0, double(trajectoryWeight));
    sensorWeight = max(0, double(sensorWeight));

    totalWeight = trajectoryWeight + sensorWeight;

    if totalWeight <= eps
        trajectoryWeight = 1;
        sensorWeight = 0;
    else
        trajectoryWeight = ...
            trajectoryWeight / totalWeight;

        sensorWeight = ...
            sensorWeight / totalWeight;
    end

    combinedRisk = ...
        trajectoryWeight .* trajectoryRisk + ...
        sensorWeight .* sensorKNNRisk;

    combinedRisk(~isfinite(combinedRisk)) = 0;
    combinedRisk = max(0, min(1, combinedRisk));

    %% ---------------------------------------------------------------
    %  Select exactly one highest-risk player
    %
    %  Ties are resolved in this order:
    %    1. highest combined risk
    %    2. highest sensor KNN risk
    %    3. highest trajectory risk
    %    4. lowest TrackID
    %  ---------------------------------------------------------------

    rankingTable = table( ...
        (1:numberOfPlayers)', ...
        combinedRisk, ...
        sensorKNNRisk, ...
        trajectoryRisk, ...
        playerIDs, ...
        'VariableNames', { ...
            'OriginalIndex', ...
            'CombinedRisk', ...
            'SensorRisk', ...
            'TrajectoryRisk', ...
            'PlayerID'});

    rankingTable = sortrows( ...
        rankingTable, ...
        { ...
            'CombinedRisk', ...
            'SensorRisk', ...
            'TrajectoryRisk', ...
            'PlayerID'}, ...
        { ...
            'descend', ...
            'descend', ...
            'descend', ...
            'ascend'});

    selectedIndex = rankingTable.OriginalIndex(1);
    injuredPlayerID = playerIDs(selectedIndex);

    %% ---------------------------------------------------------------
    %  Set the final selected player's injury label to 1
    %  ---------------------------------------------------------------

    beforeTable.injury(:) = 0;
    afterTable.injury(:) = 0;

    beforeTable.injury( ...
        beforeTable.player_id == injuredPlayerID) = 1;

    afterTable.injury( ...
        afterTable.player_id == injuredPlayerID) = 1;

    finalInjuryLabel = ...
        double(playerIDs == injuredPlayerID);

    %% ---------------------------------------------------------------
    %  Build one-row-per-player risk table
    %  ---------------------------------------------------------------

    playerRiskTable = table( ...
        playerIDs, ...
        playerPosition, ...
        rawReaction, ...
        rawSpeed, ...
        rawAgility, ...
        rawStress, ...
        correctionDistance, ...
        mappedReaction, ...
        mappedSpeed, ...
        mappedAgility, ...
        mappedStress, ...
        trajectoryRisk, ...
        sensorKNNRisk, ...
        sensorKNNClass, ...
        combinedRisk, ...
        neighboursUsed, ...
        samePositionNeighbours, ...
        nearestSensorPosition, ...
        nearestSensorDistance, ...
        meanNeighbourDistance, ...
        neighbourInjuryVotes, ...
        finalInjuryLabel, ...
        'VariableNames', { ...
            'Player_ID', ...
            'Position', ...
            'Raw_Reaction_Time', ...
            'Raw_Speed', ...
            'Raw_Agility', ...
            'Raw_Stress', ...
            'Correction_Distance', ...
            'Mapped_Reaction_Time_ms', ...
            'Mapped_Sprint_Speed_10m_s', ...
            'Mapped_Agility_Score', ...
            'Mapped_Stress_Level_Score', ...
            'Trajectory_Risk', ...
            'Sensor_KNN_Risk', ...
            'Sensor_KNN_Class', ...
            'Combined_Risk', ...
            'K_Used', ...
            'Same_Position_Neighbours', ...
            'Nearest_Sensor_Position', ...
            'Nearest_Sensor_Distance', ...
            'Mean_Neighbour_Distance', ...
            'Neighbour_Injury_Votes', ...
            'Injury_Next_Season'});

    playerRiskTable = sortrows( ...
        playerRiskTable, ...
        'Combined_Risk', ...
        'descend');

    fprintf( ...
        'Highest combined injury-risk player ID: %g\n', ...
        injuredPlayerID);

    fprintf( ...
        'Trajectory weight: %.3f\n', ...
        trajectoryWeight);

    fprintf( ...
        'Sensor KNN weight: %.3f\n', ...
        sensorWeight);

    fprintf( ...
        'Selected player combined risk: %.6f\n', ...
        combinedRisk(selectedIndex));

    fprintf( ...
        'Selected player sensor KNN risk: %.6f\n', ...
        sensorKNNRisk(selectedIndex));

    fprintf( ...
        'Selected player trajectory risk: %.6f\n', ...
        trajectoryRisk(selectedIndex));
end


function mappedValues = percentileMapToReference( ...
    sourceValues, referenceValues)
%PERCENTILEMAPTOREFERENCE
% Map source values to the empirical range of a reference dataset while
% preserving the relative ranking of source values.

    sourceValues = double(sourceValues(:));
    referenceValues = double(referenceValues(:));

    validReference = ...
        referenceValues(isfinite(referenceValues));

    mappedValues = nan(size(sourceValues));

    if isempty(validReference)
        mappedValues(:) = 0;
        return;
    end

    validReference = sort(validReference);

    validSourceIndexes = find(isfinite(sourceValues));

    if isempty(validSourceIndexes)
        mappedValues(:) = median(validReference);
        return;
    end

    sourceSubset = sourceValues(validSourceIndexes);

    if max(sourceSubset) - min(sourceSubset) < eps

        mappedValues(validSourceIndexes) = ...
            median(validReference);

    else

        % tiedrank ensures equal source values receive equal percentiles.
        sourceRanks = tiedrank(sourceSubset);

        sourcePercentiles = ...
            (sourceRanks - 0.5) ./ numel(sourceSubset);

        referencePercentiles = ...
            ((1:numel(validReference))' - 0.5) ./ ...
            numel(validReference);

        mappedValues(validSourceIndexes) = interp1( ...
            referencePercentiles, ...
            validReference, ...
            sourcePercentiles, ...
            'linear', ...
            'extrap');
    end

    mappedValues(~isfinite(mappedValues)) = ...
        median(validReference);

    mappedValues = max( ...
        min(mappedValues, max(validReference)), ...
        min(validReference));
end


function values = convertToNumeric(inputValues)
%CONVERTTONUMERIC Convert numeric, string, categorical or cell values to
% a numeric column vector.

    if isnumeric(inputValues) || islogical(inputValues)

        values = double(inputValues);

    elseif iscategorical(inputValues)

        values = str2double(string(inputValues));

    elseif isstring(inputValues)

        values = str2double(inputValues);

    elseif iscell(inputValues)

        values = str2double(string(inputValues));

    else

        values = str2double(string(inputValues));
    end

    values = values(:);
end


function normalisedValues = normaliseRiskColumn(values)
%NORMALISERISKCOLUMN Min-max normalisation with safe handling of missing
% or constant values.

    values = double(values(:));

    finiteValues = values(isfinite(values));

    if isempty(finiteValues)
        normalisedValues = zeros(size(values));
        return;
    end

    replacementValue = median(finiteValues);
    values(~isfinite(values)) = replacementValue;

    minimumValue = min(values);
    maximumValue = max(values);

    if maximumValue - minimumValue < eps
        normalisedValues = 0.5 .* ones(size(values));
    else
        normalisedValues = ...
            (values - minimumValue) ./ ...
            (maximumValue - minimumValue);
    end
end

function metricsTable = trajectoryMetricsTable( ...
    Y, meta, trackTable, predLength, dt)

%TRAJECTORYMETRICSTABLE Convert trajectory windows to frame-level metrics.

    if iscell(Y)
        Y = vertcat(Y{:});
    end

    rowCount = size(Y,1) * predLength;

    frame = zeros(rowCount,1);
    player_id = zeros(rowCount,1);
    x = zeros(rowCount,1);
    y = zeros(rowCount,1);
    position = strings(rowCount,1);
    reaction_time = zeros(rowCount,1);
    speed = zeros(rowCount,1);
    agility = zeros(rowCount,1);
    stress = zeros(rowCount,1);
    injury = zeros(rowCount,1);

    outputRow = 0;

    for k = 1:size(Y,1)
        pid = meta(k).TrackID;

        relativeXY = reshape(Y(k,:), 2, predLength)';

        % Predictions are displacements relative to the last observation.
        absoluteXY = relativeXY + ...
            [meta(k).LastObservedX, meta(k).LastObservedY];

        velocity = [zeros(1,2); diff(absoluteXY,1,1)] / dt;
        speedSeries = sqrt(sum(velocity.^2,2));

        acceleration = [0; abs(diff(speedSeries)) / dt];

        agilitySeries = zeros(predLength,1);

        for q = 2:predLength
            previousVelocity = velocity(q-1,:);
            currentVelocity = velocity(q,:);

            denominator = ...
                norm(previousVelocity) * norm(currentVelocity);

            if denominator > eps
                cosineAngle = dot( ...
                    previousVelocity, currentVelocity) / denominator;

                cosineAngle = max(-1, min(1, cosineAngle));
                agilitySeries(q) = acos(cosineAngle);
            end
        end

        reactionSeries = 1000 * agilitySeries;
        stressSeries = 0.7 * speedSeries + 0.3 * acceleration;

        playerRows = trackTable.TrackID == pid;

        frameHeight = estimateFrameHeight(trackTable);

        for q = 1:predLength
            outputRow = outputRow + 1;

            currentX = absoluteXY(q,1);
            currentY = absoluteXY(q,2);

            frame(outputRow) = ...
                meta(k).StartFrame + q - 1;

            player_id(outputRow) = pid;
            x(outputRow) = currentX;
            y(outputRow) = currentY;

            position(outputRow) = ...
                classifyPlayerPosition(currentY, frameHeight);

            reaction_time(outputRow) = reactionSeries(q);
            speed(outputRow) = speedSeries(q);
            agility(outputRow) = agilitySeries(q);
            stress(outputRow) = stressSeries(q);
        end
    end

    metricsTable = table( ...
        frame, player_id, x, y, position, ...
        reaction_time, speed, agility, stress, injury);
end


function positionName = classifyPlayerPosition(yCoordinate, frameHeight)

%CLASSIFYPLAYERPOSITION Assign a descriptive football position.
%
% This is an image-location heuristic rather than a tactical classifier.

    if ~isfinite(yCoordinate) || ...
            ~isfinite(frameHeight) || frameHeight <= 0

        positionName = "Unknown";
        return;
    end

    normalisedY = yCoordinate / frameHeight;

    if normalisedY < 0.15
        positionName = "Goalkeeper";
    elseif normalisedY < 0.40
        positionName = "Defender";
    elseif normalisedY < 0.70
        positionName = "Midfielder";
    else
        positionName = "Forward";
    end
end


function frameHeight = estimateFrameHeight(trackTable)

%ESTIMATEFRAMEHEIGHT Estimate image height from the tracking table.

    if ismember('Y', trackTable.Properties.VariableNames) && ...
            ismember('Height', trackTable.Properties.VariableNames)

        bottomEdge = double(trackTable.Y) + ...
            double(trackTable.Height);

        bottomEdge = bottomEdge(isfinite(bottomEdge));

        if isempty(bottomEdge)
            frameHeight = 1;
        else
            frameHeight = max(bottomEdge);
        end

    elseif ismember('PlayerY', ...
            trackTable.Properties.VariableNames)

        values = double(trackTable.PlayerY);
        values = values(isfinite(values));

        if isempty(values)
            frameHeight = 1;
        else
            frameHeight = max(values);
        end

    else
        frameHeight = 1;
    end
end


function value = safeMedian(values)

%SAFEMEDIAN Return zero for an empty set.

    values = values(isfinite(values));

    if isempty(values)
        value = 0;
    else
        value = median(values);
    end
end

