function filtered = meartools_circular_filter(values, kernel)
% Upstream: igm-team/meaRtools, commit 0b04c9c8421a8bc23fbe15da59f8b9a092e6a139,
% package 1.0.4 (GPL version 3 or later). See ../../THIRD_PARTY.md.
% meartools_circular_filter  Centered circular convolution as in stats::filter.

values = double(values);
kernel = double(kernel(:));
if isempty(values)
    filtered = values;
    return
end
if isempty(kernel) || ~isreal(kernel) || any(~isfinite(kernel)) || ...
        mod(numel(kernel),2) ~= 1
    error('meartools_circular_filter:invalidKernel', ...
        'kernel must be a nonempty finite real vector of odd length.');
end

wasRowVector = isrow(values);
if wasRowVector
    values = values(:);
end
nRows = size(values,1);
halfSize = (numel(kernel)-1)/2;
extendedPositions = (1-halfSize):(nRows+halfSize);
wrappedRows = mod(extendedPositions-1,nRows)+1;
filtered = conv2(values(wrappedRows,:),kernel,'valid');
if wasRowVector
    filtered = filtered.';
end
end
