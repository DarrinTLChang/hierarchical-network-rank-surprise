function [level, diagnostics] = meartools_otsu_level(values)
% Upstream: igm-team/meaRtools, commit 0b04c9c8421a8bc23fbe15da59f8b9a092e6a139,
% package 1.0.4 (GPL version 3 or later). See ../../THIRD_PARTY.md.
% meartools_otsu_level  Exact fixed-bin Otsu threshold used by meaRtools.
%
% This ports .graythresh from meaRtools 1.0.4 after the authors' 2019
% fixed-break correction. Values are assigned to the 256 R hist() bins
% [0,1], (1,2], ..., (255,256] after multiplication by 256.

values = double(values(:));
if isempty(values) || ~isreal(values) || any(~isfinite(values)) || ...
        min(values) < 0 || max(values) > 1
    error('meartools_otsu_level:invalidInput', ...
        'Input must be a nonempty finite real vector in [0,1].');
end

scaled = 256*values;
binIndex = max(1,ceil(scaled));
binIndex = min(256,binIndex);
counts = accumarray(binIndex,1,[256,1]).';

p = counts/sum(counts);
omega = cumsum(p);
mu = cumsum(p.*(1:256));
muTotal = mu(end);
betweenClassVariance = (muTotal*omega-mu).^2 ./ ...
    (omega.*(1-omega));
betweenClassVariance(isnan(betweenClassVariance)) = 0;
maxValue = max(betweenClassVariance);
if isfinite(maxValue)
    index = mean(find(betweenClassVariance == maxValue));
    level = (index-1)/255;
else
    index = NaN;
    level = 0;
end

if nargout > 1
    diagnostics = struct('counts',counts,'probabilities',p, ...
        'omega',omega,'mu',mu,'between_class_variance', ...
        betweenClassVariance,'maximizing_index_mean',index);
end
end
