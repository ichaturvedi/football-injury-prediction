function grid = makeOccupancyGrid(T, frameNo, ownID, x0, y0, gridSize, radius)
%MAKEOCCUPANCYGRID Paper's occupancy-map pooling variant.
grid = zeros(gridSize, gridSize);
N = T(T.Frame==frameNo & T.TrackID~=ownID,:);
cellWidth = 2*radius/gridSize;

for k = 1:height(N)
    rx = N.PlayerX(k)-x0;
    ry = N.PlayerY(k)-y0;
    if abs(rx) <= radius && abs(ry) <= radius
        col = floor((rx+radius)/cellWidth)+1;
        row = floor((ry+radius)/cellWidth)+1;
        col = min(max(col,1),gridSize);
        row = min(max(row,1),gridSize);
        grid(row,col) = grid(row,col)+1;
    end
end
end
