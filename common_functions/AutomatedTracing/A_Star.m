% this function runs A star algorithm on a 3D image to find the optimal trace
% x_s, y_s, z_s: coordinates of the start point
% x_g, y_g, z_g: coordinates of the goal point
% c: c = 1 -> A Star, c = 0 -> Dijkstra
% trace: trace index
% I_trace: indensity on the trace

function [trace, I_trace] = A_Star(IM,x_s,y_s,z_s,x_g,y_g,z_g,c)

cost_f=@(x) 1./(x+0.01); 

start = sub2ind(size(IM),x_s,y_s,z_s);
goal = sub2ind(size(IM),x_g,y_g,z_g);
cost_so_far=-ones(size(IM));
came_from = -ones(size(IM));
priority = inf(size(IM));
sizeIm = size(IM);
front = start; % indexes of x,y,z
cost_so_far(start)= 0;
N6_ind=[-1;+1;-sizeIm(1);+sizeIm(1);-sizeIm(1)*sizeIm(2);+sizeIm(1)*sizeIm(2)];

while ~isempty(front) && ~any(front == goal)
    [~,I] = min(priority(front)); % to prioretize the front (difference between Dijkstra and BFS
    next=[front(I)+N6_ind(1),front(I)+N6_ind(2),front(I)+N6_ind(3), ...
        front(I)+N6_ind(4),front(I)+N6_ind(5),front(I)+N6_ind(6)]; % find front neighbors
    next = next(IM(next)~=-1); %if the next is on the boundry do not proceed
    new_cost = cost_so_far(front(I))*ones(size(next)) + 1/2*(cost_f(IM(front(I)))*ones(size(next))+cost_f(IM(next)));%calculating cost
    ind = (came_from(next) == -1) | (new_cost < cost_so_far(next));%Find new path to go, I has some neighbors, to find points we havent found | find the new 
    next = next(ind);
    new_cost = new_cost(ind);
    cost_so_far(next) = new_cost;
    [x_b,y_b,z_b] = ind2sub_AS(size(IM),next);
    %     heu_dist = abs(x_g-x_b)+abs(y_g-y_b)+abs(z_g-z_b); % manhattan_dist
    heu_dist = sqrt((x_g-x_b).^2+(y_g-y_b).^2+(z_g-z_b).^2); % euclidean_dist
    priority(next) = new_cost + c*heu_dist;
    front = [front,next];
    came_from(next) = front(I);
    front(I) = [];
end

here = goal;
trace = here;
I_trace = [];
while here ~= start
    here = came_from(here);
    trace = [trace;here];
    I_trace = [I_trace;IM(here)];
end

