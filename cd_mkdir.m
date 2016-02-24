function [a] = cd_mkdir(pathname)

if ~exist(pathname,'dir')
    a= pathname;
    system(['mkdir -p ' pathname])
else
    a=-10
end
cd(pathname)

% if ~exist(pathname,'dir')
%     system(['mkdir -p ' pathname])
% end
% cd(pathname)
    