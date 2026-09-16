function T = preparePlayerPositions(T)
%PREPAREPLAYERPOSITIONS Convert ByteTrack boxes to bottom-centre positions.
required = {'Frame','TrackID','X','Y','Width','Height'};
for k = 1:numel(required)
    if ~ismember(required{k}, T.Properties.VariableNames)
        error('Missing CSV column: %s', required{k});
    end
end

% Keep people if a Class column exists.
if ismember('Class', T.Properties.VariableNames)
    c = cellstr(string(T.Class));
    T = T(strcmpi(c,'person'),:);
end

T.PlayerX = double(T.X) + double(T.Width)/2;
T.PlayerY = double(T.Y) + double(T.Height);
T = sortrows(T, {'Frame','TrackID'});
end

