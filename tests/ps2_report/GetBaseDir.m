function [basedir] = GetBaseDir()
% function [basedir] = GetBaseDir()
%
% function to set the base directory for subsequent analyses. checks to see
% whether analysis is being run on bronx, via mounted rhino, or directly on
% rhino.

[~,name] = system('hostname');
if strfind(name,'bronx.psych.upenn.edu')    % working locally on bronx
    basedir = '/Users/yezzyat/Lab';
elseif strfind(name,'rhino.psych.upenn.edu')    % working on rhino
    basedir = '';
elseif strfind(name,'rhino2.psych.upenn.edu')    % working on rhino
    basedir = '';
else
    error('GetBaseDir.m: cannot set base directory');
end


% if isdir('/Users/yezzyat/Lab/data/events/')      % working locally on bronx
%     
% elseif isdir('/Volumes/rhino/data/events/') % working with rhino mounted
%     basedir = '/Volumes/rhino';
% elseif isdir ('/data/events/')              % working directly on rhino
%     basedir = '';
% else
%     error('cannot find connection to rhino');
% end

%