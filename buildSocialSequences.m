function [X, Y, meta] = buildSocialSequences(T, obsLength, predLength, gridSize, radius)
%BUILDSOCIALSEQUENCES Build O-LSTM samples from ByteTrack positions.
% Each observed step contains [dx;dy;occupancy-grid]. Targets are future
% displacements relative to the final observed player position.

X = {}; Y = {}; meta = struct('TrackID',{},'StartFrame',{}, ...
    'LastObservedX',{},'LastObservedY',{});
ids = unique(T.TrackID);

for a = 1:numel(ids)
    id = ids(a);
    P = T(T.TrackID==id,:);
    P = sortrows(P,'Frame');
    n = height(P);
    window = obsLength + predLength;

    for s = 1:(n-window+1)
        W = P(s:s+window-1,:);
        % Require consecutive frames for meaningful temporal prediction.
        if any(diff(W.Frame) ~= 1), continue; end

        features = zeros(2 + gridSize*gridSize, obsLength);
        for q = 1:obsLength
            if q == 1
                dx = 0; dy = 0;
            else
                dx = W.PlayerX(q)-W.PlayerX(q-1);
                dy = W.PlayerY(q)-W.PlayerY(q-1);
            end
            occ = makeOccupancyGrid(T, W.Frame(q), id, W.PlayerX(q), ...
                W.PlayerY(q), gridSize, radius);
            features(:,q) = [dx; dy; occ(:)];
        end

        x0 = W.PlayerX(obsLength); y0 = W.PlayerY(obsLength);
        future = [W.PlayerX(obsLength+1:end)-x0, ...
                  W.PlayerY(obsLength+1:end)-y0];
        target = reshape(future', 1, []);

        X{end+1,1} = features; %#ok<AGROW>
        Y{end+1,1} = target; %#ok<AGROW>
        meta(end+1,1).TrackID = id; %#ok<AGROW>
        meta(end).StartFrame = W.Frame(1);
        meta(end).LastObservedX = x0;
        meta(end).LastObservedY = y0;
    end
end
end
