function [YCorrected,paramTable,info] = applyPlayerParameterSimulinkResponse( ...
    YPred,meta,trackTable,predLength,dt,modelName,blockPaths,outputName,filePrefix)
% Team-aware hydraulic correction.
% 1. Assigns one stable jersey-colour team to each TrackID.
% 2. Runs each team separately.
% 3. Runs players in batches of at most four because the model has x1-x4.
% 4. Maps x1, x2, x3 and x4 to separate players. Outputs are never merged.

n = size(YPred,1);
YPred = double(YPred);
YCorrected = YPred;

if numel(meta) ~= n
    error('meta must contain one entry per prediction row.');
end

% Add stable Team information from existing Team or jersey RGB columns.
trackTable = assignStableTeamColours(trackTable);

trackID = reshape([meta.TrackID],[],1);
startFrame = reshape([meta.StartFrame],[],1);
team = zeros(n,1);
position = zeros(n,1);
reaction = zeros(n,1);
speed = zeros(n,1);
agility = zeros(n,1);
stress = zeros(n,1);
injury = zeros(n,1);

maxY = max(double(trackTable.Y) + double(trackTable.Height),[],'omitnan');
if isempty(maxY) || ~isfinite(maxY) || maxY <= 0
    maxY = 1;
end

for k = 1:n
    xy = reshape(YPred(k,:),2,predLength)';
    velocity = [zeros(1,2); diff(xy,1,1)] / max(dt,eps);
    speedSeries = sqrt(sum(velocity.^2,2));
    acceleration = [0; abs(diff(speedSeries))/max(dt,eps)];
    angles = zeros(predLength,1);

    for q = 2:predLength
        den = norm(velocity(q-1,:))*norm(velocity(q,:));
        if den > eps
            c = dot(velocity(q-1,:),velocity(q,:))/den;
            angles(q) = acos(max(-1,min(1,c)));
        end
    end

    speed(k) = mean(speedSeries,'omitnan');
    agility(k) = mean(angles,'omitnan');
    reaction(k) = 1000*agility(k);
    stress(k) = 0.7*speed(k) + 0.3*mean(acceleration,'omitnan');

    rows = trackTable.TrackID == trackID(k);
    validTeams = trackTable.Team(rows);
    validTeams = validTeams(isfinite(validTeams) & validTeams > 0);
    if ~isempty(validTeams)
        team(k) = mode(validTeams);
    end

    yn = meta(k).LastObservedY / maxY;
    if yn < 0.15
        position(k) = 1;
    elseif yn < 0.40
        position(k) = 2;
    elseif yn < 0.70
        position(k) = 3;
    else
        position(k) = 4;
    end
end

paramTable = table(trackID,startFrame,team,position,reaction,speed,agility,stress,injury, ...
    'VariableNames',{'Player_ID','Start_Frame','Team','Position', ...
    'Reaction_Time_ms','Sprint_Speed_10m_s','Agility_Score', ...
    'Stress_Level_Score','Injury_Next_Season'});

writetable(paramTable,[filePrefix '.csv']);
writeWekaARFF([filePrefix '.arff'],paramTable);

if ~bdIsLoaded(modelName)
    load_system(modelName);
end
cleanup = onCleanup(@() close_system(modelName,0)); %#ok<NASGU>

if isempty(blockPaths)
    blockPaths = findCompatibleBlocks(modelName);
end

% The active hydraulic model exposes four independent player outputs x1-x4.
maxPlayersPerRun = 4;
if numel(blockPaths) < 1
    error(['No compatible Simulink blocks were found. Provide full block paths ' ...
           'for blocks containing K, Ac, Beta, C1, rho or V30.']);
end

teamValues = unique(team);
teamValues = teamValues(teamValues > 0 & isfinite(teamValues));
runLog = table;

for t = reshape(teamValues,1,[])
    teamPlayerIDs = unique(trackID(team == t));

    fprintf('\nTEAM %d: %d players\n',t,numel(teamPlayerIDs));

    % Process every player in the team, four at a time. This prevents
    % hydraulic output channels from being shared or merged across players.
    for batchStart = 1:maxPlayersPerRun:numel(teamPlayerIDs)
        batchIDs = teamPlayerIDs(batchStart:min(batchStart+maxPlayersPerRun-1,numel(teamPlayerIDs)));
        batchCount = numel(batchIDs);
        P = zeros(batchCount,4);

        for j = 1:batchCount
            r = team == t & trackID == batchIDs(j);
            P(j,:) = [median(stress(r),'omitnan'), ...
                      median(speed(r),'omitnan'), ...
                      median(agility(r),'omitnan'), ...
                      mode(position(r))];
        end

        stressN = localNormalise(P(:,1));
        speedN = localNormalise(P(:,2));
        agilityN = localNormalise(P(:,3));
        K = 5e4.*(0.5+1.5*stressN);
        Ac = 1e-3.*(0.5+1.5*speedN);
        Beta = 7e8.*(0.5+1.5*agilityN);
        C1 = 2e-8.*(0.95+0.10*speedN);
        rho = 800.*(0.95+0.10*stressN);
        V30 = 2.5e-5.*arrayfun(@positionFactor,P(:,4));

        if numel(blockPaths) < batchCount
            error('Need at least %d compatible block paths for this batch.',batchCount);
        end

        for j = 1:batchCount
            b = blockPaths{j};
            setIfPresent(b,'K',K(j));
            setIfPresent(b,'Ac',Ac(j));
            setIfPresent(b,'Beta',Beta(j));
            setIfPresent(b,'C1',C1(j));
            setIfPresent(b,'rho',rho(j));
            setIfPresent(b,'V30',V30(j));
        end

        simIn = Simulink.SimulationInput(modelName);
        simIn = simIn.setModelParameter('StopTime', ...
            num2str(max(dt*(predLength-1),dt)));
        
        
        %simOut = sim(simIn);
try

    simOut = sim(simIn);

    responses = extractIndependentResponses( ...
        simOut, ...
        outputName, ...
        batchCount);

catch ME

    warning( ...
        'Team %d Batch %d failed: %s', ...
        t, ...
        batchStart, ...
        ME.message);

    continue;

end

        % responses(:,1)=x1, responses(:,2)=x2, etc. No averaging or merging.
        responses = extractIndependentResponses(simOut,outputName,batchCount);

        for j = 1:batchCount
            pid = batchIDs(j);
            playerWindows = find(team == t & trackID == pid);
            playerResponse = resampleResponse(responses(:,j),predLength);

            for w = reshape(playerWindows,1,[])
                xy = reshape(YPred(w,:),2,predLength)';
                delta = [xy(1,:); diff(xy,1,1)];
                correctedXY = cumsum(delta.*[playerResponse playerResponse],1);
                YCorrected(w,:) = reshape(correctedXY',1,[]);
            end

            newLog = table(t,batchStart,pid,j,string("x"+j),K(j),Ac(j),Beta(j),C1(j),rho(j),V30(j), ...
                'VariableNames',{'Team','BatchStart','PlayerID','ResponseColumn', ...
                'HydraulicOutput','K','Ac','Beta','C1','rho','V30'});
            runLog = [runLog; newLog]; %#ok<AGROW>
        end
    end
end

info = struct('RunLog',runLog,'PlayerIDs',trackID,'Teams',team);
writetable(runLog,[filePrefix '_hydraulic_player_output_map.csv']);
end

function T = assignStableTeamColours(T)
% Use an existing valid Team column; otherwise cluster median jersey RGB per TrackID.
if ~ismember('TrackID',T.Properties.VariableNames)
    error('trackTable must contain TrackID.');
end

if ismember('Team',T.Properties.VariableNames)
    existing = double(T.Team);
    if all(isfinite(existing(existing>0))) && numel(unique(existing(existing>0))) >= 2
        return;
    end
end

rgbColumns = {'JerseyR','JerseyG','JerseyB'};
if ~all(ismember(rgbColumns,T.Properties.VariableNames))
    %error(['Team cannot be inferred from X/Y alone. The track CSV must contain ' ...
     %      'JerseyR, JerseyG and JerseyB, or an existing Team column.']);

    warning('No team information found. Assigning all players to Team 1.');

    T.Team = ones(height(T),1);
    return;

end

ids = unique(T.TrackID);
C = nan(numel(ids),3);
for k = 1:numel(ids)
    rows = T.TrackID == ids(k);
    rgb = double([T.JerseyR(rows),T.JerseyG(rows),T.JerseyB(rows)]);
    rgb = rgb(all(isfinite(rgb),2),:);
    if ~isempty(rgb)
        C(k,:) = median(rgb,1);
    end
end

valid = all(isfinite(C),2);
teamByID = zeros(numel(ids),1);
if sum(valid) >= 2
    rng(1,'twister');
    teamByID(valid) = kmeans(C(valid,:),2,'Replicates',10,'Distance','sqeuclidean');
elseif sum(valid) == 1
    teamByID(valid) = 1;
end

T.Team = zeros(height(T),1);
for k = 1:numel(ids)
    T.Team(T.TrackID == ids(k)) = teamByID(k);
end
end

function responses = extractIndependentResponses(simOut,outputName,m)
% Return one separate column for each hydraulic output. Never merge channels.
vals = [];
try
    obj = simOut.get(outputName);
    if iscell(obj), obj = obj{1}; end
    if isprop(obj,'Values'), vals = obj.Values; end
catch
end
if isempty(vals)
    try
        vals = simOut.sldemo_hydcyl4_output{1}.Values;
    catch
    end
end
if isempty(vals)
    error('Could not obtain hydraulic output values.');
end

names = {'x1','x2','x3','x4'};
channel = cell(m,1);
for j = 1:m
    if isstruct(vals) && isfield(vals,names{j})
        channel{j} = squeeze(double(vals.(names{j}).Data));
    elseif isobject(vals) && isprop(vals,names{j})
        channel{j} = squeeze(double(vals.(names{j}).Data));
    else
        error('Hydraulic output %s was not found.',names{j});
    end
    if ~isvector(channel{j})
        channel{j} = channel{j}(:,1);
    end
    channel{j} = channel{j}(:);
end

commonLength = min(cellfun(@numel,channel));
responses = zeros(commonLength,m);
for j = 1:m
    responses(:,j) = channel{j}(1:commonLength);
end
end

function z = localNormalise(x)
x = double(x);
lo = min(x,[],'omitnan');
hi = max(x,[],'omitnan');
if isempty(lo) || ~isfinite(lo) || ~isfinite(hi) || hi-lo < eps
    z = 0.5*ones(size(x));
else
    z = (x-lo)/(hi-lo);
end
end

function f = positionFactor(pos)
switch round(pos)
    case 1, f = 1.10;
    case 2, f = 1.05;
    case 3, f = 1.00;
    case 4, f = 0.95;
    otherwise, f = 1.00;
end
end

function blocks = findCompatibleBlocks(modelName)
allBlocks = find_system(modelName,'Type','Block');
blocks = {};
for i = 1:numel(allBlocks)
    try
        d = get_param(allBlocks{i},'DialogParameters');
        if isstruct(d) && any(isfield(d,{'K','Ac','Beta','C1','rho','V30'}))
            blocks{end+1} = allBlocks{i}; %#ok<AGROW>
        end
    catch
    end
end
end

function setIfPresent(block,param,value)
d = get_param(block,'DialogParameters');
if isstruct(d) && isfield(d,param)
    set_param(block,param,num2str(value,16));
end
end

function r = resampleResponse(r,n)
r = double(r(:));
r = r(isfinite(r));
if isempty(r)
    error('The assigned player hydraulic response is empty.');
end
r = interp1(linspace(0,1,numel(r)),r,linspace(0,1,n),'linear','extrap')';
r = r-r(1);
s = max(abs(r));
if s < eps
    r = ones(n,1);
else
    r = max(0,min(1,r/s));
    r(1) = 1;
end
end
