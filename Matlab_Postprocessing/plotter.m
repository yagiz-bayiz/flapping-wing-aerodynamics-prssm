clear
clc
close all


%% Figure settings
alw = 0.75;    % AxesLineWidth
fsz = 14;      % Fontsize
lw = 2;        % LineWidth
msz = 100;     % MarkerSize

map = fbrewermap(5,'Set1');
set(0, 'DefaultAxesFontSize', fsz, 'DefaultBarLineWidth', lw,'DefaultAxesLineWidth', alw)


%% Analysis Settings 
histogram_plots= false;
cycle_averaged_plots = false;
trajectory_plots = false;
latent_state_analysis = true;
traj_overview_plots = false;
save_switch = false;

N       = 10;
Ts      = 0.004*N;
nx      = 12;
inputLR = [];
outputLR = [];

num_train = 512;
num_test = 548-num_train;
traj_start_ind = 94;

%% Load/Organize Data
in_folder_name = '../Data/Input/';
out_folder_name = '../Data/Output/20_08_15_IN_LK7Normalized_OUT_FMwoFa5_Dimx12_Epochs1000_512Train/';
load([in_folder_name, 'flapping_wing_aerodynamics_lasso_fit']);             % Load GPSSM inputs
load([in_folder_name, 'flapping_wing_aerodynamics.mat']);                   % Load Lasso results and model
load([out_folder_name, 'matfiles/predict_train_n_test.mat']);               % Load GPSSM outputs

gp_mean_train = permute(gp_mean_train(:,traj_start_ind:end,:),[2,3,1]);
gp_mean_test = permute(gp_mean_test(:,traj_start_ind:end,:),[2,3,1]);
gp_var_train = permute(gp_var_train(:,traj_start_ind:end,:),[2,3,1]);
gp_var_test = permute(gp_var_test(:,traj_start_ind:end,:),[2,3,1]);
gt_train = permute(gt_train(:,traj_start_ind:end,:),[2,3,1]);
gt_test = permute(gt_test(:,traj_start_ind:end,:),[2,3,1]);
pos_train = permute(pos_train(:,traj_start_ind:end,:),[2,3,1]);
pos_test = permute(pos_test(:,traj_start_ind:end,:),[2,3,1]);
lr_train = permute(lr_y(1:512,traj_start_ind:end,:),[2,3,1]);
lr_test = permute(lr_y(513:end,traj_start_ind:end,:),[2,3,1]);

std_out = std_out(:)';
mean_out = mean_out(:)';

gp_train_flat = reshape(permute(gp_mean_train, [2,1,3]) , size(gp_mean_train,2)', [])';
gp_test_flat = reshape(permute(gp_mean_test, [2,1,3]) , size(gp_mean_test,2)', [])';
gt_train_flat = reshape(permute(gt_train, [2,1,3]) , size(gt_train,2)', [])';
gt_test_flat = reshape(permute(gt_test, [2,1,3]) , size(gt_test,2)', [])';
lr_train_flat = reshape(permute(lr_train, [2,1,3]) , size(lr_train,2)', [])';
lr_test_flat = reshape(permute(lr_test, [2,1,3]) , size(lr_test,2)', [])';
errorGP_train_flat = ferrorgpssm(gp_train_flat, gt_train_flat);
errorGP_test_flat = ferrorgpssm(gp_test_flat, gt_test_flat);
errorLR_train_flat = ferrorgpssm(lr_train_flat, gt_train_flat);
errorLR_test_flat = ferrorgpssm(lr_test_flat, gt_test_flat);

ave_rmse_GP_train = (mean((errorGP_train_flat.Er).^2,'all')).^0.5;
ave_rmse_LR_train = (mean((errorLR_train_flat.Er).^2,'all')).^0.5;
ave_rmse_GP_test = (mean((errorGP_test_flat.Er).^2,'all')).^0.5;
ave_rmse_LR_test = (mean((errorLR_test_flat.Er).^2,'all')).^0.5;

%% Histograms
CIFcn = @(x,p)prctile(x,abs([0,100]-(100-p)/2));
if  histogram_plots
    width = 1.1;     % Width in inches
    height = 1.1;    % Height in inches
    for i = 1: size(gt_test,2)
        fignum1 = 1000+i;
        fig = figure(fignum1);
        ax = axes;
        set(ax , 'Units', 'pixels');
        pos = get(ax , 'Position');
        set(ax,  'Position', [pos(1)+10 pos(2)+10 width*200, height*200]);
        hold on
        histogram(errorLR_train_flat.Er(:,i),-1.1:.01:1.1,'Normalization', 'probability','facecolor',map(2,:),'facealpha',.5,'edgecolor','none')
        histogram(errorGP_train_flat.Er(:,i),-1.1:.01:1.1,'Normalization', 'probability','facecolor',map(1,:),'facealpha',.5,'edgecolor','none')
        hold off
        box off
        axis([-1.1,1.1,0,0.04])
        LRCI = CIFcn(errorLR_train_flat.Er(:,i),95); 
        arrayfun(@(x)xline(x,'-m',['LRprctile',num2str(x,'%.2f')]),LRCI);
        GPCI = CIFcn(errorGP_train_flat.Er(:,i),95); 
        arrayfun(@(x)xline(x,'-c',['GPprctile',num2str(x,'%.2f')]),GPCI);
        if  i==1
            legend('LR','GP','location','northwest')
            legend boxoff
            ylabel('frequency')
        end
        saveas(fig, [out_folder_name,'figures/hist_figures/hist_figures_', num2str(fignum1)],'pdf')
        
        fignum2 = 1100+i;
        fig = figure(fignum2);
        ax = axes;
        set(ax , 'Units', 'pixels');
        pos = get(ax , 'Position');
        set(ax,  'Position', [pos(1)+10 pos(2)+10 width*200, height*200], 'XAxisLocation','top');
        
        
        hold on
        histogram(errorLR_test_flat.Er(:,i),-1.1:.01:1.1,'Normalization', 'probability','facecolor',map(2,:),'facealpha',.5,'edgecolor','none')
        histogram(errorGP_test_flat.Er(:,i),-1.1:.01:1.1,'Normalization', 'probability','facecolor',map(1,:),'facealpha',.5,'edgecolor','none')
        box off
        hold off
        ylabel('frequency')
        axis([-1.1,1.1,0,0.04])
        set(gca, 'ydir', 'reverse')
        xticklabels([])
        LRCI = CIFcn(errorLR_test_flat.Er(:,i),95); 
        arrayfun(@(x)xline(x,'-m',['LRprctile',num2str(x,'%.2f')]),LRCI);
        GPCI = CIFcn(errorGP_test_flat.Er(:,i),95); 
        arrayfun(@(x)xline(x,'-c',['GPprctile',num2str(x,'%.2f')]),GPCI);
        legend('LR','GP','location','northwest')
        legend boxoff
        saveas(fig, [out_folder_name,'figures/hist_figures/hist_figures_', num2str(fignum2)],'pdf')
    end
end

%% Trajectories
close all
size_mean = [2*size(gt_train,3), size(gt_train,2)];
gt_train_bar2 = nan(size_mean);
gp_train_bar2 = nan(size_mean);
lr_train_bar2 = nan(size_mean);
duration_train = nan(1, size(gt_train,3));

colorgrey999 = [0.7, 0.7, 0.7];

for i = 1:num_train
    position = pos_train(:,:,i);
    ground_t = gt_train(:,:,i);
    gaussian_p = gp_mean_train(:,:,i);
    linear_r = lr_train(:,:,i);
    
    [gt_bar_up, gp_bar_up, lr_bar_up, gt_bar_down, gp_bar_down, ...
        lr_bar_down,  pos, gt_ndim, gp_ndim, lr_ndim, duration] = ...
        fplotter(position, ground_t, gaussian_p, linear_r,...
        mean_out, std_out, Ts);
    
    gt_train_bar2(2*i-1, :) = gt_bar_up;
    gp_train_bar2(2*i-1, :) = gp_bar_up;
    lr_train_bar2(2*i-1, :) = lr_bar_up;
    
    gt_train_bar2(2*i, :) = gt_bar_down;
    gp_train_bar2(2*i, :) = gp_bar_down;
    lr_train_bar2(2*i, :) = lr_bar_down;
    
    duration_train(i) = duration;
    
    if trajectory_plots
        if save_switch
            savename = [out_folder_name,'figures/traj_figures/train_', num2str(i,'%03d'),'.pdf'];
        else
            savename = false;
        end
        fplotmaker(pos, gt_ndim, gp_ndim, lr_ndim, savename)
    end
    
    if traj_overview_plots
        color999 = colorgrey999;
        Nfig = 5;
        
        t = ((1:length(pos(1:Nfig:end,1)))-1)/(length(pos(1:Nfig:end,1))-1);
        hold on
        fig1788 = figure(1788);
        plot(t, pos(1:Nfig:end,1)*180/pi, 'color', color999)
        hold off
        hold on
        fig1789 = figure(1789);
        plot(t, pos(1:Nfig:end,2)*180/pi, 'color', color999)
        hold off
        hold on
        fig1790 = figure(1790);
        plot(t, pos(1:Nfig:end,3)*180/pi, 'color', color999)
        hold off
    end
end

size_mean = [2*size(gt_test,3), size(gt_test,2)];
gt_test_bar2 = nan(size_mean);
gp_test_bar2 = nan(size_mean);
lr_test_bar2 = nan(size_mean);
duration_test = nan(1, size(gt_test,3));

dummymap = fbrewermap(1,'Dark2');
map999 = [0        0.4470    0.7410;
    0.8500    0.3250    0.0980;
    0.9290    0.6940    0.1250;
    0.4940    0.1840    0.5560;
    0.4660    0.6740    0.1880;
    dummymap];


GP_RMSE = [];
LR_RMSE = [];
for i = 1:num_test
    position = pos_test(:,:,i);
    ground_t = gt_test(:,:,i);
    gaussian_p = gp_mean_test(:,:,i);
    linear_r = lr_test(:,:,i);
    
    [gt_bar_up, gp_bar_up, lr_bar_up, gt_bar_down, gp_bar_down, ...
        lr_bar_down,  pos, gt_ndim, gp_ndim, lr_ndim, duration] = ...
        fplotter(position, ground_t, gaussian_p, linear_r,...
        mean_out, std_out, Ts);
    
    GP_RMSE = [GP_RMSE; goodnessOfFit(gp_ndim(:,1),gt_ndim(:,1),'MSE')^0.5, goodnessOfFit(gp_ndim(:,2),gt_ndim(:,2),'MSE')^0.5];
    LR_RMSE = [LR_RMSE; goodnessOfFit(lr_ndim(:,1),gt_ndim(:,1),'MSE')^0.5, goodnessOfFit(lr_ndim(:,2),gt_ndim(:,2),'MSE')^0.5];

    
    gt_test_bar2(2*i-1, :) = gt_bar_up;
    gp_test_bar2(2*i-1, :) = gp_bar_up;
    lr_test_bar2(2*i-1, :) = lr_bar_up;
    
    gt_test_bar2(2*i, :) = gt_bar_down;
    gp_test_bar2(2*i, :) = gp_bar_down;
    lr_test_bar2(2*i, :) = lr_bar_down;
    
    duration_test(i) = duration;
    
    if trajectory_plots
        if save_switch
            savename = [out_folder_name,'figures/traj_figures/test_', num2str(i,'%03d'),'.pdf'];
        else
            savename = false;
        end
        fplotmaker(pos, gt_ndim, gp_ndim, lr_ndim, savename)
    end
    
    Nfig =1;
    switch i
        case 26
        color999 = map999(1,:);
        case 14
        color999 = map999(2,:);
        case 13
        color999 = map999(3,:);
        case 33
        color999 = map999(4,:);
        case 5
        color999 = map999(5,:);
        case 32
        color999 = map999(6,:);
        otherwise
        color999 = colorgrey999;
        Nfig = 5;
    end
    
    if traj_overview_plots
        t = ((1:length(pos(1:Nfig:end,1)))-1)/(length(pos(1:Nfig:end,1))-1);
        hold on
        fig1788 = figure(1788);
        plot(t, pos(1:Nfig:end,1)*180/pi, 'color', color999)
        ylim([-110,110])
        hold off
        hold on
        fig1789 = figure(1789);
        plot(t, pos(1:Nfig:end,2)*180/pi, 'color', color999)
        ylim([-110,110])
        hold off
        hold on
        fig1790 = figure(1790);
        plot(t, pos(1:Nfig:end,3)*180/pi, 'color', color999)
        ylim([-110,110])
        hold off
    end
end
if traj_overview_plots
    saveas(fig1788, [out_folder_name,'figures/trajoverview_stroke'],'pdf')
    saveas(fig1789, [out_folder_name,'figures/trajoverview_deviation'],'pdf')
    saveas(fig1790, [out_folder_name,'figures/trajoverview_rotation'],'pdf')
end 
%% Cycle averaged residiuals
close all
if cycle_averaged_plots
    width = 1.4;     % Width in inches
    height = 1.4;    % Height in inches
    fsz = 12;       % Fontsize
    lw = 2;      % LineWidth
    
    map = fbrewermap(5,'Set1');
    set(0, 'DefaultAxesFontSize', fsz, 'DefaultBarLineWidth', lw,'DefaultAxesLineWidth', alw)
    
    axesmat= [-2, 3, -0.75, 0.75;
        -2, 3, -0.75, 0.75;
        -6, 7, -2, 2;
        -6, 7, -2, 2;
        -6, 7, -2, 2];
    
    for i = 1: size(gt_test,2)
        fignum = 2000+i;
        fig = figure(fignum);
        ax = axes;
        set(ax , 'Units', 'pixels');
        pos = get(ax , 'Position');
        set(ax,  'Position', [pos(1)+10 pos(2)+10 width*200, height*200]);
        scatter(gt_train_bar2(:,i), gt_train_bar2(:,i)-lr_train_bar2(:,i),'o', 'MarkerFaceColor',map(2,:),'MarkerFaceAlpha', 0.5,'MarkerEdgeColor', map(2,:))
        hold on
        scatter(gt_train_bar2(:,i), gt_train_bar2(:,i)-gp_train_bar2(:,i),'o', 'MarkerFaceColor',map(1,:),'MarkerFaceAlpha', 0.5,'MarkerEdgeColor', map(1,:))
        scatter(gt_test_bar2(:,i), gt_test_bar2(:,i)-lr_test_bar2(:,i),'d', 'MarkerFaceColor',map(5,:),'MarkerFaceAlpha', 0.5,'MarkerEdgeColor', map(5,:))
        scatter(gt_test_bar2(:,i), gt_test_bar2(:,i)-gp_test_bar2(:,i),'d', 'MarkerFaceColor',map(3,:),'MarkerFaceAlpha', 0.5,'MarkerEdgeColor', map(3,:))
        LRCI = CIFcn([gt_train_bar2(:,i)-lr_train_bar2(:,i);gt_test_bar2(:,i)-lr_test_bar2(:,i)],95); 
        arrayfun(@(x)yline(x,'-m',['LRprctile',num2str(x,'%.2f')]),LRCI);
        GPCI = CIFcn([gt_train_bar2(:,i)-gp_train_bar2(:,i);gt_test_bar2(:,i)-gp_test_bar2(:,i)],95); 
        arrayfun(@(x)yline(x,'-c',['GPprctile',num2str(x,'%.2f')]),GPCI);
        box off
        ylabel('Error')
        xlabel('Observation')
        axis(axesmat(i,:))
        axis square
        rline = refline([0,0]);
        rline.Color = 'k';
        hold off
        legend('LR_{train}','GP_{train}','LR_{test}','GP_{test}','location','northeast')
        legend boxoff
        saveas(fig,[out_folder_name,'figures/errorvsobs_figures/fig_', num2str(fignum)],'pdf')
    end
end

%% Latent State Analysis
close all
limit_delay = 1;
limit_xcorr = 1;

if latent_state_analysis
    latent_mean_train = permute(latent_mean_train(:,traj_start_ind:end,:),[2,3,1]);
    latent_mean_test = permute(latent_mean_test(:,traj_start_ind:end,:),[2,3,1]);
    latent_mean_all = cat(3, latent_mean_train, latent_mean_test);
    pos_all = cat(3, pos_train, pos_test);
    inmat = permute(ds_u(:,traj_start_ind:end,:),[2,3,1]);
    duration_all = [duration_train, duration_test];
    maxlag_all = [];
    
    for i = 1:size(latent_mean_all,3)
        lags  = [];
        lags2 = [];
        lags3 = [];
        xcor1 = [];
        xcor2 = [];
        xcor3 = [];
        maxlagmat = nan(12,7,2);
        for j = 1:12
            for k = 1:7
                x = inmat(:,k,i);
                y = latent_mean_all(:,j,i);
                [rtemp,ltemp1] = crosscorr(x,y,'NumLags',300);
                ltemp1 = ltemp1/duration_all(i);
                ltemp = ltemp1(ltemp1>0 & ltemp1<limit_delay);
                rtemp = rtemp(ltemp1>0 & ltemp1<limit_delay);
                
                [maxcros,indxcros] = max(abs(rtemp));
                maxlag = ltemp(indxcros);
                maxlagmat(j,k,1) = maxlag;
                maxlagmat(j,k,2) = maxcros;
            end
        end
        maxlag_all = cat(4, maxlag_all, maxlagmat);
    end
    
    state_labels = {'x1','x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12'};
    input_labels = {'AoA', '|v|', 'a_N', 'a_T', 'a_A', 'rot_{vel}', 'rot_{acc}'};
    close all
    dummymap = fbrewermap(1,'Dark2');
    map2 = [0        0.4470    0.7410;
        0.8500    0.3250    0.0980;
        0.9290    0.6940    0.1250;
        0.4940    0.1840    0.5560;
        0.4660    0.6740    0.1880;
        dummymap;
        0.6350    0.0780    0.1840;];
    
    
    fillmap = colormap('lines');
    
    for j = 1:12
        fig1 = figure(3000+10*j);
        ax1 = axes;
        ax1.ActivePositionProperty = 'position';
        set(ax1 , 'Units', 'pixels');
        pos = get(ax1 , 'Position');
        set(ax1,  'Position', [pos(1), pos(2), 160, 160]);
        Xslag = [];
        Xsmag = [];
        grouping = {};
        
        for k = 1:7
            Xscat1 = maxlag_all(j,k,1,:);
            Xscat1 = Xscat1(:);
            Xscat2 = maxlag_all(j,k,2,:);
            Xscat2 = Xscat2(:);
            Xslag = [Xslag; Xscat1];
            Xsmag = [Xsmag; Xscat2];
            groupnames = repelem({['input',num2str(k)]},length(Xscat1));
            groupnames = groupnames(:);
            grouping = [grouping;groupnames];
            c = ksdensity([Xscat1,Xscat2],[Xscat1,Xscat2],'Support','positive','BoundaryCorrection','reflection','Bandwidth',0.09);            
                    
            %% Distribution Plots
            fig = figure(10*j+k);
            ax = axes;
            ax.ActivePositionProperty = 'position';
            ax.PositionConstraint = 'innerposition';
            set(ax , 'Units', 'pixels');
            pos = get(ax , 'Position');
            set(ax,  'Position', [pos(1), pos(2), 360, 360]);
            scatter(Xscat1, Xscat2, 50, c, 'o', 'filled', 'MarkerEdgeColor', 'none', 'MarkerFaceAlpha', 0.7);
            hslc = frgb2hsl(map2(k,:));
            xlim([0,limit_delay])
            ylim([0,limit_xcorr])
            caxis([0 20])
            H = hslc(1);
            S = hslc(2);
            L = [1, hslc(3)];
            colormap(fhslcolormap(10,H,S,L))
            box on
            saveas(fig,[out_folder_name,'figures/xcorr_figures/scatter',num2str(10*j+k)],'pdf')
            
            
                    
            %% Overall Plots
            % kernel smooth for distribution
            ngrid = 30; % used 300 for the plots in the paper
            xgrid = linspace(0, limit_delay,ngrid);
            ygrid = linspace(0,limit_xcorr,ngrid);
            [Xgrid, Ygrid] = meshgrid(xgrid, ygrid);
            xdummy = Xgrid(:);
            ydummy = Ygrid(:);
            [fdens, xydens] = ksdensity([Xscat1,Xscat2], [xdummy, ydummy], 'Support','positive','BoundaryCorrection','reflection','Bandwidth',0.085);
            Fdens = reshape(fdens, ngrid, ngrid);
            Freduceddens = Fdens(:);
            
            % find 70% conf interval
            total_int = sum(Freduceddens);
            reduced_int = total_int;
            Freduceddens(Freduceddens<1e-5) = nan;
            
            while reduced_int> 0.70*total_int
                [~, minloc] = min(Freduceddens,[], 'omitnan', 'linear');
                Freduceddens(minloc) = nan;
                reduced_int = sum(Freduceddens,'omitnan');
            end
            Freduceddens = reshape(Freduceddens, size(Fdens));
            [B,~] = bwboundaries(~isnan(Freduceddens),'noholes');
            boundary = B{1};
            figure(1)
            fill(xgrid(boundary(:,2)), ygrid(boundary(:,1)), 'b');
            xlim([0,limit_delay])
            ylim([0,limit_xcorr])
            
            
            fig1 = figure(3000+10*j);
            mean_plot = mean([Xscat1, Xscat2], 1);
            [~, maxdens_loc] = max(Fdens, [], 'all', 'omitnan', 'linear');
            max_loc = [Xgrid(maxdens_loc), Ygrid(maxdens_loc)];
            
            hold on
            plot(max_loc(1), max_loc(2), '+', 'Color', map2(k,:), 'MarkerSize', 12, 'LineWidth', 3);
            h = fill(xgrid(boundary(:,2)), ygrid(boundary(:,1)), map2(k,:), 'LineWidth', 1.5, 'EdgeColor', map2(k,:));
            set(h, 'facealpha', 0.2);

            xlim([0,limit_delay])
            ylim([0,limit_xcorr])
            xlabel('Lag')
            ylabel('Maximum Xcorr')
        end
        hold off
        saveas(fig1,[out_folder_name,'figures/xcorr_figures/overall',num2str(9900+j)],'pdf')
        
        
        %% Box Plots
        fig11 = figure(1000*j+1);
        ax = axes;
        ax.ActivePositionProperty = 'position';
        set(ax , 'Units', 'pixels');
        pos = get(ax , 'Position');
        set(ax,  'Position', [pos(1), pos(2), 360, 120]);
        h2 = boxplot(Xslag,grouping,'PlotStyle','compact','orientation','horizontal', 'Whisker', 2, 'label', {'','','','','','',''}, 'color',map2);
        xlim([0,limit_delay])
        set(ax,  'Position', [pos(1), pos(2), 360, 120]);
        saveas(fig11,[out_folder_name,'figures/xcorr_figures/box',num2str(1000*j+1)],'pdf')
        
        fig12 = figure(1000*j+2);
        ax = axes;
        ax.ActivePositionProperty = 'position';
        set(ax , 'Units', 'pixels');
        pos = get(ax , 'Position');
        set(ax,  'Position', [pos(1), pos(2), 120, 360]);
        h2 = boxplot(Xsmag,grouping,'PlotStyle','compact','orientation','vertical', 'Whisker', 2, 'label', {'','','','','','',''}, 'color',map2);
        ylim([0,limit_xcorr])
        set(ax,  'Position', [pos(1), pos(2), 120, 360]);
        saveas(fig12,[out_folder_name,'figures/xcorr_figures/box',num2str(1000*j+2)],'pdf')
        
        
        
        [Xslag_med, Xslag_iqr] = grpstats(Xslag,grouping,{@median,@iqr});
        Xsmag_med = grpstats(Xsmag,grouping,{@median});
        circ_size = Xsmag_med.^2*2000;
        hslc = frgb2hsl(map2(5,:));
        H = hslc(1);
        S = hslc(2);
        L = [hslc(3), 0.9];

        fig9876 = figure(9876);
        set(gcf,'position',[0 0  960 400 ])
        hold on
        scatter(j*ones(1,7), -1:-1:-7, circ_size, Xslag_iqr,'o', 'filled', 'MarkerEdgeColor', map2(5,:), 'MarkerFaceAlpha', 1)
        if j==12
            str_plot = [0.25, 0.5, 0.8, 0, 0.15, 0.3];
            scatter(13*ones(1,6), -1:-1:-6, [0.2; 0.5; 0.8; 0.5; 0.5; 0.5].^2*2000, [0.15; 0.15; 0.15; 0; 0.15; 0.3],'o', 'filled', 'MarkerEdgeColor', map2(5,:), 'MarkerFaceAlpha', 1)
            for k= 1:6
                text(14-0.25,-k,num2str(str_plot(k)))
            end
        end
        caxis([0 0.3])
        colormap(fhslcolormap(10,H,S,L))
        xlim([0,15])
        ylim([-8,0])
        for k= 1:7
            if Xsmag_med(k)>0.5
                text(j-0.2,-k,num2str(Xslag_med(k),'%3.3f'))
            end
        end
        
    end
    orient landscape
    saveas(fig9876,[out_folder_name,'figures/xcorr_figures/table'],'pdf')
end













