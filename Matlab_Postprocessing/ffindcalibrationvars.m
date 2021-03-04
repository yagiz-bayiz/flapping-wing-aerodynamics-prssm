function x = ffindcalibrationvars(calibration)
    calibration(1:10,:) = [];
    calibration(:,[1,11]) = [];
    
    temppos = calibration(1:2:end,7:end);
    pos = [smooth(interp(temppos(:,1),2),50),smooth(interp(temppos(:,2),2),50),smooth(interp(temppos(:,3),2),50)];
    ind = find(vecnorm(pos-[0,0,-90],2,2)<1,1,'first');
    calibration(1:ind-1,:) = [];
    
    vel = (vecnorm(diff(pos),2,2)/0.004);
    zero_vel_bool = vel < 0.02;
    zero_vel_ind = 1:length(pos);
    zero_vel_ind = zero_vel_ind(zero_vel_bool);
    dzero_vel_ind = diff(zero_vel_ind);
    trajend_ind = find(dzero_vel_ind>100);
    trajend_ind = [zero_vel_ind(trajend_ind), zero_vel_ind(end)] ;
    trajend_ind(trajend_ind<800) = NaN;
    
    for i = 1:length(trajend_ind)
        trajend_ind(abs(trajend_ind-trajend_ind(i))<50 & abs(trajend_ind-trajend_ind(i))>1.1) = NaN;
    end
    trajend_ind = trajend_ind(~isnan(trajend_ind)); 
    mean_cal = [];
    for i = 1:length(trajend_ind)
        ind_cal = (trajend_ind-600):(trajend_ind-100); 
        cal = calibration(ind_cal,:);
        mean_cal = [mean_cal; mean(cal,1)];
    end
    
    F_m = mean_cal(:,1:3);
    M_m = mean_cal(:,4:6);
    position = mean_cal(:,7:end);
    output = [F_m/12, M_m/120];
    input = position;
    x0 = [-0.3, -3, 0.3, -25, 18, 21, 1, 800];
    options = optimoptions('lsqcurvefit', 'FunctionTolerance', 1e-8, 'StepTolerance', 1e-8, 'OptimalityTolerance', 1e-8, 'MaxFunctionEvaluations', 4000, 'Algorithm', 'levenberg-marquardt');
    [x,resnorm, residual ,exitflag,output] = lsqcurvefit(@fcaltransform, x0, input, output,[] ,[] , options);
end

