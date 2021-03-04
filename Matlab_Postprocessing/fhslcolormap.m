function rgb=fhslcolormap(varargin)
%% Generate hue-saturation-lightness colormaps
%
% USAGE: hslcolormap(N,H,S,L)
% 
% INPUTS:
%    N: number of colors in colormap.
%    H: hue stops (optional).    %can also be specified using a string: rygcbmRYGCBM
%    S: saturation stops (optional)
%    L: Lightness stops (optional)
%
% The stops are placed evenly over the colorscale. Parameter examples:
% H=[0 1/6]; %colorscale runs from red to yellow
% S=[0 1]; %gray to fully saturated
% L=[.3 1 .3]; % dark-bright-dark
%
% Using L with two-stops (e.g. L=[.2 .8]) makes it easy to distinguish
% high and low values. 
%
% Using a symmetric L with 3 stops (e.g. L=[.2 1 .2]) is useful for divergent
% colorscales. 
%
% Discontinuities in the colormap can be indicated with nans.
% 
%
% EXAMPLE:
%   surf(peaks(200),'edgecolor','none'); 
%   caxis([-6 6])
%   hslcolormap(300,'g.m',1,[.1  1 .1]);
%   colorbar
% 
%
%
%
% (c) Aslak Grinsted 2014
if nargin==0
    hslcolormap('mbbccc.YYYRRm',1,[.1 1 .1])
    return
end

N=varargin{1};
if (numel(N)>1)||(N(1)<3)
    N=300;
else
    varargin(1)=[];
end

if ischar(varargin{1})
    [~,ix]=ismember(varargin{1},'rygcbmRYGCBM');
    ix=((ix-1)/6);ix(ix<0)=nan;
    varargin{1}=ix;
end

z=linspace(0,1,N)';

hsl=zeros(N,3);
if length(varargin)<2
    hsl(:,2)=1;
end
if length(varargin)<3
    hsl(:,3)=z;
end
for ii=1:length(varargin)
    d=varargin{ii};
    if isempty(d)||all(isnan(d))
        continue
    end
    
    
    jj=1;Nc=0;d=[d(:);nan];
    while jj<=length(d)
        if isnan(d(jj))
            switch Nc
                case 0, d(jj)=[]; jj=jj-1;
                case 1, d=d([1:jj-1 jj-1 jj:end]);jj=jj+1;
            end
            Nc=0;
        else
            Nc=Nc+1;
        end
        jj=jj+1;
    end
    d(end)=[];
    
    nans=find(isnan(d));
    
    %zin=linspace(0,1,length(d)-length(nans)*2)';
    zin=[0;cumsum(~isnan(d(2:end)+d(1:end-1)))];
    zin(nans+1)=zin(nans+1)+1e-8;
    zin(nans)=[]; d(nans)=[];
    hsl(:,ii)=interp1q(zin/zin(end),d,z);
end

zin=linspace(0,1,size(hsl,1))';

hsl(:,1)=mod(hsl(:,1),1);


rgb=hsl2rgb(hsl);

if nargout==0
    colormap(rgb)
    clearvars rgb
end


function rgb=hsl2rgb(hsl) %based on http://www.mathworks.com/matlabcentral/fileexchange/28790-colorspace-transformations/content//colorspace/colorspace.m (by Pascal Getreuer)

H=hsl(:,1);
L = hsl(:,3);
Delta = hsl(:,2).*min(L,1-L);

H = min(max(H*6,0),6);
m0 = L-Delta;
m2 = L+Delta;
F = H - round(H/2)*2;
M = [m0, m0 + (m2-m0).*abs(F), m2];
Num = length(m0);
j = [2 1 0;1 2 0;0 2 1;0 1 2;1 0 2;2 0 1;2 1 0]*Num;
k = floor(H) + 1;
rgb = [M(j(k,1)+(1:Num).'),M(j(k,2)+(1:Num).'),M(j(k,3)+(1:Num).')];

function rgb=hcl2rgb(hcl) % from: http://en.wikipedia.org/wiki/HSL_and_HSV
%hue/chroma/luma
H=hcl(:,1);
C=hcl(:,2);
L=hcl(:,3);

H = mod(H,1)*6;
X = C.*(1-abs(mod(H,2)-1));
rgb=zeros(size(hcl));
N=size(rgb,1);
Ccol=mod(floor((H+1)/2),3); %zero-based
Xcol=mod(1-floor(H),3); %zero-based
rgb((1:N)'+Ccol*N)=C;
rgb((1:N)'+Xcol*N)=X;

m=L - (0.3*rgb(:,1)+0.59*rgb(:,2)+0.11*rgb(:,3));
rgb=max(min(bsxfun(@plus,rgb,m),1),0);



