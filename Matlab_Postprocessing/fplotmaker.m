function fplotmaker(pos, groundt, gaussian_p, linear_r, savename)
    %% PLOT POSITION INITIALIZATION
    x = 700;
    y = 100;
    w = 400;
    h = 1080;
    time = (0:length(pos)-1)/length(pos);
    map = fbrewermap(5,'Set1');
    
    fig = figure(1);
    set(fig, 'Position', [x y w h],'defaultLineLineWidth',1.5)
    s0 = subplot(4,1,1);
    fplotwingmotion(pos)
%     s0.Position(1) = 0.035;
%     s0.Position(3) = 0.18;
    axis equal
    
    s1 = subplot(4,1,2);
    plot(time,pos*180/pi)
    ylabel('Position [deg]')
    xlim([time(1), time(end)])
    l1=legend('str','dev','rot');
    ylim([-100 100])
%     s1.Position(1) = 0.25;
%     s1.Position(3) = 0.18;
%     l1.Position(1) = s1.Position(1) + s1.Position(3) + 0.006;
    
    s2 = subplot(4,1,3);
    plot(time, groundt(:,1), 'color', [255 0 0]/255)
    hold on
    plot(time, gaussian_p(:,1), 'color',[141 28 28]/255)
    plot(time, linear_r(:,1), 'color', [245 107 15]/255)
    plot(time, groundt(:,2), 'color', [0 0 255]/255)
    plot(time, gaussian_p(:,2), 'color', [179 5 176]/255)
    plot(time, linear_r(:,2), 'color', [2 179 198]/255)
    hold off
    ylabel('CF')
    xlim([time(1), time(end)])
    ylim([-4 7])
    l2 = legend('Exp1', 'GP1', 'LR1', 'Exp2', 'GP2', 'LR2');
%     s2.Position(1) = 0.50;
%     s2.Position(3) = 0.18;
%     l2.Position(1) = s2.Position(1) + s2.Position(3) + 0.006;
    
    
    s3 = subplot(4,1,4);
    plot(time, groundt(:,3), 'color', [255 0 0]/255)
    hold on
    plot(time, gaussian_p(:,3), 'color',[141 28 28]/255)
    plot(time, linear_r(:,3), 'color', [245 107 15]/255)
    plot(time, groundt(:,4), 'color', [0 0 255]/255)
    plot(time, gaussian_p(:,4), 'color', [179 5 176]/255)
    plot(time, linear_r(:,4), 'color', [2 179 198]/255)
    plot(time, groundt(:,5), 'color', [0 255 0]/255)
    plot(time, gaussian_p(:,5), 'color', [35 131 0]/255)
    plot(time, linear_r(:,5), 'color', [0 209 118]/255)
    hold off
    ylabel('CM')
    xlim([time(1), time(end)])
    ylim([-11 13])
    l3 = legend('Exp1', 'GP1', 'LR1', 'Exp3', 'GP3', 'LR3','Exp3', 'GP3', 'LR3');
%     s3.Position(1) = 0.75;
%     s3.Position(3) = 0.18;
%     l3.Position(1) = s3.Position(1) + s3.Position(3) + 0.006;  
    if savename
        saveas(fig,savename)
    end
end