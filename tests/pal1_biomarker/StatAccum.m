classdef StatAccum < handle
    properties(Access = public)
        n;
        sum;
        sum_sq;
        mean;
        stdev;
    end
        
    methods(Access = public)
        function initialize(this)
            this.n=0;
            this.sum = 0.0;
            this.sum_sq = 0.0;
            this.mean = 0.0;
            this.stdev = 0.0;
        end
        
        function receive(this, x)
            this.n = this.n + 1;
            this.sum = this.sum + x;
            this.sum_sq = this.sum_sq + x.*x;
            this.mean = this.sum / this.n;
            this.stdev = sqrt((this.sum_sq-this.sum.*this.sum/this.n)/(this.n-1));
        end
    end
end
