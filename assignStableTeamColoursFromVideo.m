function T = assignStableTeamColoursFromVideo(T, videoFile)

% -------------------------------------------------------------------------
% Input:
%
% T must contain:
%   Frame
%   TrackID
%   X
%   Y
%   Width
%   Height
%
% Output:
%
% Adds:
%   JerseyR
%   JerseyG
%   JerseyB
%   Team
% -------------------------------------------------------------------------

required = { ...
    'Frame', ...
    'TrackID', ...
    'X', ...
    'Y', ...
    'Width', ...
    'Height'};

for k = 1:numel(required)

    if ~ismember(required{k},T.Properties.VariableNames)

        error('Missing required column: %s',required{k});

    end

end

fprintf('\nExtracting jersey colours from video...\n');

v = VideoReader(videoFile);

ids = unique(T.TrackID);

trackColour = nan(numel(ids),3);

% Store frame colours back into tracking table

T.JerseyR = nan(height(T),1);
T.JerseyG = nan(height(T),1);
T.JerseyB = nan(height(T),1);

for playerIndex = 1:numel(ids)

    playerID = ids(playerIndex);

    rows = find(T.TrackID == playerID);

    if isempty(rows)
        continue;
    end

    sampleCount = min(20,numel(rows));

    sampleRows = ...
        rows(round(linspace( ...
        1, ...
        numel(rows), ...
        sampleCount)));

    rgbSamples = [];

    fprintf('TrackID %d (%d samples)\n', ...
        playerID, ...
        sampleCount);

    for s = 1:numel(sampleRows)

        rowIdx = sampleRows(s);

        frameNumber = T.Frame(rowIdx);

        try

            v.CurrentTime = ...
                max(0,(frameNumber-1)/v.FrameRate);

            frame = readFrame(v);

        catch
            continue;
        end

        bbox = [ ...
            T.X(rowIdx), ...
            T.Y(rowIdx), ...
            T.Width(rowIdx), ...
            T.Height(rowIdx)];

        crop = getJerseyCropSafe(frame,bbox);

        if isempty(crop)
            continue;
        end

        pixels = reshape(double(crop),[],3);

        validPixels = ...
            all(isfinite(pixels),2);

        pixels = pixels(validPixels,:);

        if isempty(pixels)
            continue;
        end

        rgb = median(pixels,1);

        rgbSamples(end+1,:) = rgb; %#ok<AGROW>

        T.JerseyR(rowIdx) = rgb(1);
        T.JerseyG(rowIdx) = rgb(2);
        T.JerseyB(rowIdx) = rgb(3);

    end

    if ~isempty(rgbSamples)

        trackColour(playerIndex,:) = ...
            median(rgbSamples,1);

    end

end

%% ------------------------------------------------------------------------
% Fill missing colour rows for each player
%% ------------------------------------------------------------------------

for playerIndex = 1:numel(ids)

    rows = T.TrackID == ids(playerIndex);

    rgb = trackColour(playerIndex,:);

    if all(isfinite(rgb))

        T.JerseyR(rows) = rgb(1);
        T.JerseyG(rows) = rgb(2);
        T.JerseyB(rows) = rgb(3);

    end

end

%% ------------------------------------------------------------------------
% Team clustering
%% ------------------------------------------------------------------------

validTracks = all(isfinite(trackColour),2);

teamPerTrack = zeros(numel(ids),1);

if sum(validTracks) >= 2

    rng(1,'twister');

    teamPerTrack(validTracks) = ...
        kmeans( ...
            trackColour(validTracks,:), ...
            2, ...
            'Replicates',20, ...
            'Distance','sqeuclidean');

elseif sum(validTracks) == 1

    teamPerTrack(validTracks) = 1;

end

%% ------------------------------------------------------------------------
% Add Team column
%% ------------------------------------------------------------------------

T.Team = zeros(height(T),1);

for playerIndex = 1:numel(ids)

    T.Team(T.TrackID == ids(playerIndex)) = ...
        teamPerTrack(playerIndex);

end

%% ------------------------------------------------------------------------
% Print summary
%% ------------------------------------------------------------------------

fprintf('\n');
fprintf('=====================================\n');
fprintf('TEAM ASSIGNMENT SUMMARY\n');
fprintf('=====================================\n');

fprintf('Team 1 players: %d\n', ...
    sum(teamPerTrack == 1));

fprintf('Team 2 players: %d\n', ...
    sum(teamPerTrack == 2));

fprintf('Players assigned: %d\n', ...
    sum(teamPerTrack > 0));

end


% =========================================================================
% JERSEY CROP
% =========================================================================

function crop = getJerseyCropSafe(frame,bbox)

x = round(bbox(1));
y = round(bbox(2));

w = round(bbox(3));
h = round(bbox(4));

y2 = y + round(h*0.30);
h2 = round(h*0.40);

x = max(1,x);
y2 = max(1,y2);

if x+w > size(frame,2)
    w = size(frame,2)-x;
end

if y2+h2 > size(frame,1)
    h2 = size(frame,1)-y2;
end

if w<=0 || h2<=0

    crop = [];

    return;

end

crop = imcrop(frame,[x y2 w h2]);

if isempty(crop)
    return;
end

% central torso only

x1 = round(size(crop,2)*0.25);
x2 = round(size(crop,2)*0.75);

crop = crop(:,x1:x2,:);

% remove green grass

hsvImg = rgb2hsv(crop);

H = hsvImg(:,:,1);
S = hsvImg(:,:,2);

grassMask = ...
    H > 0.20 & ...
    H < 0.45 & ...
    S > 0.30;

for c = 1:3

    tmp = double(crop(:,:,c));

    tmp(grassMask) = NaN;

    crop(:,:,c) = uint8(tmp);

end

end