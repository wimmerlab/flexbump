function [bin, edges] = equalbins(x,n_bins)

edges = [-inf, quantile(x, n_bins - 1), inf];
bin = discretize(x,edges);

