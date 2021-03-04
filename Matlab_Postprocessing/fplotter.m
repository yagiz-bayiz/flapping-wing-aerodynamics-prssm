function [gt_bar_up, gp_bar_up, lr_bar_up, gt_bar_down, gp_bar_down, lr_bar_down, pos, gt_ndim, gp_ndim, lr_ndim, duration] = fplotter(position, ground_t, gaussian_p, linear_r,  mean_out, std_out, Ts)
R = 0.235; %[m]
r_length = 0.180; %[m]
rho = 800; %kg*m^-3
cbar = 0.06; %[m]
Sarea = r_length*cbar;


endpoint_st = position(end,1);
delta = endpoint_st - position(end-1,1);
ind_array = find(abs(position(1:end-10,1)-endpoint_st)<delta, 10, 'last');
diff_ind = [diff(ind_array')~=1,true];
ind_array = ind_array(diff_ind)';
ind = ind_array(end-1);
total_length = length(position(:,1));
duration = total_length-ind;
first_str_rev = floor(total_length-(5*duration/4));
third_str_rev = first_str_rev + duration;

gt_dim = (ground_t(first_str_rev:third_str_rev,:).*std_out) + mean_out;
gp_dim = (gaussian_p(first_str_rev:third_str_rev,:).*std_out) + mean_out;
lr_dim = (linear_r(first_str_rev:third_str_rev,:).*std_out) + mean_out;
pos = position(first_str_rev:third_str_rev,:);

v = diff(pos,1)/Ts;
distance = trapz((1:length(v))*Ts,R*(v(:,1).^2+v(:,2).^2).^0.5);
Uref = distance/(Ts*duration);

gt_ndim = gt_dim;
gt_ndim(:,1:2) = gt_ndim(:,1:2)/(0.5*rho*Sarea*Uref^2);
gt_ndim(:,3:5) = gt_ndim(:,3:5)/(0.5*1000*rho*cbar*Sarea*Uref^2);

gp_ndim = gp_dim;
gp_ndim(:,1:2) = gp_ndim(:,1:2)/(0.5*rho*Sarea*Uref^2);
gp_ndim(:,3:5) = gp_ndim(:,3:5)/(0.5*1000*rho*cbar*Sarea*Uref^2);

lr_ndim = lr_dim;
lr_ndim(:,1:2) = lr_ndim(:,1:2)/(0.5*rho*Sarea*Uref^2);
lr_ndim(:,3:5) = lr_ndim(:,3:5)/(0.5*1000*rho*cbar*Sarea*Uref^2);

gt_ndim_up = gt_ndim(1:floor(duration/2),:);
gp_ndim_up = gp_ndim(1:floor(duration/2),:);
lr_ndim_up = lr_ndim(1:floor(duration/2),:);

gt_ndim_down = gt_ndim(floor(duration/2):end,:);
gp_ndim_down = gp_ndim(floor(duration/2):end,:);
lr_ndim_down = lr_ndim(floor(duration/2):end,:);

gt_bar_up = mean(gt_ndim_up,1);
gp_bar_up = mean(gp_ndim_up,1);
lr_bar_up = mean(lr_ndim_up,1);

gt_bar_down = mean(gt_ndim_down,1);
gp_bar_down = mean(gp_ndim_down,1);
lr_bar_down = mean(lr_ndim_down,1);

end

