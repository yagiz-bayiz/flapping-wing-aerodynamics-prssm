function Error=ferrorgpssm(P,T)
%
%                                ANNN1
%                       
%           NEURAL NETWORK SCRIPT LINK FOR HYDROLOGICAL PURPOSES
%
%                             Version 2.0
%                         
%                       Department of Hydroinformatics
%                                 Delft
%                        Gerald A. Corzo Perez
%                               UNESCO-IHE
%                               www.hi.ihe.nl
%                                --OO--
%
%
%DEFAULT PARAMETERS FOR TRAINING NETWORKS:if isempty(Par)
%     P= Predicted values (vector)
%     T= Target values  (vector)
%
%OTHER FILES IN DIRECTORY
%Error1, Error2, Error7, Error6 in adition works only with Matlab version
%above 7 and should have the NN toolbox
%
% Error.MARE=MARE -> Mean Absolute relative error
% Author: Gerald Corzo
% Made Jan 2004
% Updated May /2009
%
S=size(P);
if S(1)<S(2)
    P=P';
end

S=size(T);
if S(1)<S(2)
    T=T';
end

n=size(P,1);
%Traditional Measures of error
SSE = sum((P-T).^2);
RMSE = sqrt(SSE/size(P,1));
StdT = std(T,1);
StdP = std(P,1);
range = (max(T)-min(T));
NRMSE = RMSE./range;
%Cor=sum((P-mean(P)).*(T-mean(T)))/(sqrt(sum((P-mean(P)).^2))*sqrt(sum((T-mean(T)).^2)));

MAE=sum(abs(P-T))/size(P,1);
MARE=sum(abs((T-P)./T))/n;

MuT=mean(T);
MuP=mean(P);


%Passing the output structure
Error.RMSE=RMSE;
Error.NRMSE=NRMSE;
Error.MAE=MAE;
Error.StdT=StdT;
Error.StdP=StdP;
Error.MuT=MuT;
Error.MuP=MuP;
Error.SSE=SSE;
Error.MARE=MARE;
Error.Er=T-P;
