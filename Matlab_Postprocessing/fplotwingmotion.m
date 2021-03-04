function [] = fplotwingmotion(pos)
% wing length
R = 4.5;

str = pos(:,1);
dev = pos(:,2);
rot = pos(:,3);

L     = 2;
rot_r = 0.15;
time = (0:length(pos)-1)/length(pos);


for k =1:length(str)
    % find wing tip location at axis of rotation
    if (k ~= 0 && k ~= length(str))
        w_Axis = [str(k)*R,-dev(k)*R];
    end
    
    % find wing leading edge
    w_Head = w_Axis + [ (rot_r)*L*sin(rot(k)) (rot_r)*L*cos(rot(k))];
    w_Headx(k) = w_Head(1);
    w_Heady(k) = w_Head(2);
    
    % find wing trailing edge
    w_Tail = w_Axis - [ (1-rot_r)*L*sin(rot(k)) (1-rot_r)*L*cos(rot(k))];
    w_Tailx(k) = w_Tail(1);
    w_Taily(k) = w_Tail(2);
end
drawArrow = @(x,y,varargin) quiver(x(1),y(1),x(2)-x(1),y(2)-y(1),0);
    
for k= 1:7:floor(length(time)-1)
    %% Plot
    plot(w_Headx(k),w_Heady(k),'k.','Markersize',30,'linewidth',1.5);
    hold on
    line([w_Headx(k) w_Tailx(k)],[w_Heady(k) w_Taily(k)],'linewidth',2,'color','k');
    xlim([-8,8])
    axis equal
end
hold off

 end



