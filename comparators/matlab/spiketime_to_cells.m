function [SpikeTimes, IDs, Regions, Electrodes, Clusters, durMs] = ...
         spiketime_to_cells(matPath, SNR_LO, SNR_HI, FR_MIN, MIN_SPK)
% spiketime_to_cells  Convert a spikeTime.mat into Valkki's input format.
%
% Valkki's CalculateCMANoRound wants:
%   SpikeTimes : 1xM cell, each cell one spike train (ms, sorted)
%   IDs        : 1xM cell of IDs, one per train
%   ST_Duration_ms : recording length in ms (here returned as durMs)
%
% spikeTime.mat schema (one struct-array element per electrode channel):
%   .electrode           char, e.g. 'microGPi_L_1_CommonFiltered'
%   .snr                 numeric, one value per cluster
%   .time                cell, one spike train (ms) per cluster
%   .dataSegmentLength   scalar, seconds
%
% Each CLUSTER becomes one independent input spike train. Unit inclusion is
% common study preprocessing applied before the unchanged author detector,
% so all comparison methods receive the same eligible unit set.

if nargin < 2, SNR_LO  = 1.2;  end
if nargin < 3, SNR_HI  = 25; end
if nargin < 4, FR_MIN  = 0.8+eps(0.8); end
if nargin < 5, MIN_SPK = 2;    end

S  = load(matPath);
st = S.spikeTime;
st = st(:).';

SpikeTimes = {}; IDs = {}; Regions = {}; Electrodes = {}; Clusters = [];

for i = 1:numel(st)
    elec = st(i).electrode;  if ~ischar(elec), elec = char(elec); end
    reg  = strtok(regexprep(elec,'^micro',''), '_');   % token after 'micro'
    dur_s  = double(st(i).dataSegmentLength);
    tcells = st(i).time;     if ~iscell(tcells), tcells = {tcells}; end
    snr    = st(i).snr;      if iscell(snr), snr = cell2mat(snr); end

    for c = 1:numel(tcells)
        tr   = sort(double(tcells{c}(:)).');   % ms, row vector, sorted
        nspk = numel(tr);
        fr   = nspk / dur_s;
        snr_c = NaN; if numel(snr) >= c, snr_c = double(snr(c)); end

        keep = nspk >= MIN_SPK && fr >= FR_MIN && ...
               ~isnan(snr_c) && snr_c >= SNR_LO && snr_c <= SNR_HI;
        if keep
            SpikeTimes{end+1} = tr;                          %#ok<AGROW>
            IDs{end+1}        = sprintf('%s#%d', elec, c);   %#ok<AGROW>
            Regions{end+1}    = reg;                         %#ok<AGROW>
            Electrodes{end+1} = elec;                        %#ok<AGROW>
            Clusters(end+1)   = c;                           %#ok<AGROW>
        end
    end
end

if isempty(SpikeTimes)
    error('spiketime_to_cells:noUnits', ...
          'No units passed the filter (SNR %.2f-%.2f, FR>=%.2f, >=%d spikes).', ...
          SNR_LO, SNR_HI, FR_MIN, MIN_SPK);
end

durMs = double(st(1).dataSegmentLength) * 1000;
fprintf('Converted %d units across %d regions.\n', ...
        numel(SpikeTimes), numel(unique(Regions)));
end
