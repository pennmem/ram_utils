%CLASS_INTERFACE Example MATLAB class wrapper to an underlying C++ class
classdef morlet_interface < handle
    properties (SetAccess = private, Hidden = true)
        objectHandle; % Handle to the underlying C++ class instance
    end
    methods
        %% Constructor - Create a new C++ class instance 
        function this = morlet_interface(varargin)
            this.objectHandle = morlet_interface_mex('new', varargin{:});
        end
        
        %% Destructor - Destroy the C++ class instance
        function delete(this)
            morlet_interface_mex('delete', this.objectHandle);
        end

        %% init
        function varargout = init(this, varargin)
            [varargout{1:nargout}] = morlet_interface_mex('init', this.objectHandle, varargin{:});
        end

        %% multiphasevec
        function varargout = multiphasevec(this, varargin)
            [varargout{1:nargout}] = morlet_interface_mex('multiphasevec', this.objectHandle, varargin{:});
        end
    end
end