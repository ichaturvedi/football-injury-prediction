function metrics = evaluateTrajectoryAccuracy(YPred, YTrue, meta, predLength)
%EVALUATETRAJECTORYACCURACY Compute ADE and FDE.
% ADE: mean Euclidean error over all predicted time steps.
% FDE: Euclidean error at the final predicted time step.

if iscell(YPred), YPred = vertcat(YPred{:}); end
if iscell(YTrue), YTrue = vertcat(YTrue{:}); end
n = size(YPred,1);
ADE = zeros(n,1); FDE = zeros(n,1);

for k = 1:n
    p = reshape(YPred(k,:),2,predLength)';
    g = reshape(YTrue(k,:),2,predLength)';
    d = sqrt(sum((p-g).^2,2));
    ADE(k) = mean(d);
    FDE(k) = d(end);
end

trackID = reshape([meta.TrackID],[],1);
startFrame = reshape([meta.StartFrame],[],1);
perSequence = table(trackID,startFrame,ADE,FDE, ...
    'VariableNames',{'TrackID','StartFrame','ADE','FDE'});
summary = table(mean(ADE),median(ADE),mean(FDE),median(FDE),n, ...
    'VariableNames',{'MeanADE','MedianADE','MeanFDE','MedianFDE','NumTestSequences'});
metrics = struct('PerSequence',perSequence,'Summary',summary);
end
