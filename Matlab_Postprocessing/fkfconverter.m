function KF = fkfconverter(input_LR_train)

AoA = input_LR_train(:,1);
vw_abs = input_LR_train(:,2);
a_w = input_LR_train(:,3:5);
vel_alpha = input_LR_train(:,6);
acc_alpha = input_LR_train(:,7);
[vx, vy]  = pol2cart(AoA+pi, vw_abs);


KF = [vw_abs.^2, AoA.*vw_abs.^2, AoA.^2.*vw_abs.^2, AoA.^3.*vw_abs.^2,...
      vel_alpha.*vw_abs, a_w(:,1:2), vel_alpha.*vx, vel_alpha.*vy,...
      acc_alpha, vel_alpha.*abs(vel_alpha)];
end

