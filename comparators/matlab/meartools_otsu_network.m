function result = meartools_otsu_network(channelSpikesS, durationS, settings)
% Upstream: igm-team/meaRtools, commit 0b04c9c8421a8bc23fbe15da59f8b9a092e6a139,
% package 1.0.4 (GPL version 3 or later). See ../../THIRD_PARTY.md.
% meartools_otsu_network  Source-faithful MATLAB port of meaRtools network bursts.
%
% result = meartools_otsu_network(channelSpikesS,durationS,settings)
% accepts one cell per physical electrode, with spike timestamps in seconds.
% It follows meaRtools 1.0.4 source behavior, including the inner 10-second
% crop, 12.5-kHz quantization, right-closed 25-sample bins, circular
% filtering, source sigma mutation, fixed-bin Otsu threshold, inclusive
% threshold comparison, and source event-time offset convention.

if nargin < 3 || isempty(settings)
    settings = struct();
end
if ~iscell(channelSpikesS) || ~isvector(channelSpikesS)
    error('meartools_otsu_network:invalidSpikes', ...
        'channelSpikesS must be a cell vector of per-electrode spike times.');
end
if ~isscalar(durationS) || ~isreal(durationS) || ...
        ~isfinite(durationS) || durationS < 0
    error('meartools_otsu_network:invalidDuration', ...
        'durationS must be a finite nonnegative scalar.');
end
if ~isstruct(settings) || ~isscalar(settings)
    error('meartools_otsu_network:invalidSettings', ...
        'settings must be a scalar structure.');
end

settings = resolve_settings(settings);
channelSpikesS = channelSpikesS(:);
nChannels = numel(channelSpikesS);
for channel = 1:nChannels
    spikes = double(channelSpikesS{channel}(:));
    if ~isreal(spikes) || any(~isfinite(spikes))
        error('meartools_otsu_network:invalidSpikes', ...
            'Every spike timestamp must be finite and real.');
    end
    channelSpikesS{channel} = spikes;
end

allPositive = vertcat(channelSpikesS{:});
allPositive = allPositive(allPositive > 0);
if isempty(allPositive)
    error('meartools_otsu_network:noPositiveSpikes', ...
        'meaRtools source cropping requires at least one positive spike.');
end

cropStartS = floor(min(allPositive)/settings.skip_s)*settings.skip_s + ...
    settings.skip_s;
cropEndS = floor(max(allPositive)/settings.skip_s)*settings.skip_s;
timespanS = cropEndS-cropStartS;

selected = cell(nChannels,1);
for channel = 1:nChannels
    spikes = channelSpikesS{channel};
    selected{channel} = spikes(spikes >= cropStartS & spikes <= cropEndS);
end
selectedAll = vertcat(selected{:});
if isempty(selectedAll)
    result = empty_result(settings,durationS,nChannels,cropStartS, ...
        cropEndS,timespanS);
    return
end

offsetS = min(selectedAll);
sampleTimes = cell(nChannels,1);
for channel = 1:nChannels
    sampleTimes{channel} = r_round(selected{channel}*12500);
end
allSamples = vertcat(sampleTimes{:});
tMin = min(allSamples);
tMax = max(allSamples);
sampleRange = tMax-tMin+1;
nSourceBins = ceil(sampleRange/settings.bin_samples);
nRows = nSourceBins+1;

allRows = zeros(0,1);
allColumns = zeros(0,1);
for channel = 1:nChannels
    shifted = sampleTimes{channel}-tMin+1;
    shifted = shifted(shifted > 1); % source hist() trims the first spike
    rows = ceil((shifted-1)/settings.bin_samples);
    rows = rows(rows >= 1 & rows <= nRows);
    allRows = [allRows; rows]; %#ok<AGROW>
    allColumns = [allColumns; repmat(channel,numel(rows),1)]; %#ok<AGROW>
end
binnedSpikes = sparse(allRows,allColumns,1,nRows,nChannels);
binnedSpikes = spones(binnedSpikes); % at most one spike/electrode/bin
activeMask = full(max(binnedSpikes,[],1) > 0);
nActive = nnz(activeMask);

nSigma = numel(settings.sigmas);
sigmaInput = double(settings.sigmas(:)).';
sigmaEffective = sigmaInput;
sigmaEffective(mod(sigmaEffective,2) == 0) = ...
    sigmaEffective(mod(sigmaEffective,2) == 0)+1;
kernels = cell(1,nSigma);
signals = cell(1,nSigma);
masks = cell(1,nSigma);
eventsBySigma = cell(1,nSigma);
otsuLevels = NaN(1,nSigma);

for sigmaIndex = 1:nSigma
    effective = sigmaEffective(sigmaIndex);
    halfSize = (effective-1)/2;
    x = linspace(-halfSize,halfSize,effective).';
    kernel = exp(-(x.^2)/(2*effective^2));
    kernel = kernel/sum(kernel);
    kernels{sigmaIndex} = kernel;

    if nActive < settings.min_electrodes || nActive == 0
        signals{sigmaIndex} = zeros(0,1);
        masks{sigmaIndex} = false(0,1);
        eventsBySigma{sigmaIndex} = empty_event_table();
        continue
    end

    population = zeros(nRows,1);
    activeChannels = find(activeMask);
    for channel = activeChannels
        trace = full(binnedSpikes(:,channel));
        trace = meartools_circular_filter(trace,kernel);
        trace = trace/max(trace);
        population = population+trace;
    end
    population = population/max(population);
    population = meartools_circular_filter(population,kernel);
    population = population/max(population);
    population(population < 0) = 0;

    level = meartools_otsu_level(population);
    mask = population >= level;
    if settings.local_region_min_nae > 0
        [preliminaryStart,preliminaryEnd] = true_runs(mask);
        for eventIndex = 1:numel(preliminaryStart)
            rows = preliminaryStart(eventIndex):preliminaryEnd(eventIndex)-1;
            participating = nnz(any(binnedSpikes(rows,:) > 0,1));
            if participating < settings.local_region_min_nae
                mask(rows) = false;
            end
        end
    end

    [burstStart,burstEnd] = true_runs(mask);
    events = build_event_table(burstStart,burstEnd,binnedSpikes, ...
        sigmaInput(sigmaIndex),effective,offsetS);
    signals{sigmaIndex} = population;
    masks{sigmaIndex} = mask;
    eventsBySigma{sigmaIndex} = events;
    otsuLevels(sigmaIndex) = level;
end

result = struct();
result.settings = settings;
result.recording_duration_s = double(durationS);
result.crop_start_s = cropStartS;
result.crop_end_s = cropEndS;
result.timespan_s = timespanS;
result.bin_time_s = settings.bin_samples/settings.sample_rate_hz;
result.offset_s = offsetS;
result.n_channels = nChannels;
result.n_active_electrodes = nActive;
result.binned_spikes = binnedSpikes;
result.sigma_input = sigmaInput;
result.sigma_effective = sigmaEffective;
result.gaussian_kernels = kernels;
result.population_signals = signals;
result.otsu_levels = otsuLevels;
result.burst_masks = masks;
result.events_by_sigma = eventsBySigma;
result.events = vertcat(eventsBySigma{:});
end

function settings = resolve_settings(settings)
defaults = struct('sigmas',45, ...
    'min_electrodes',4,'local_region_min_nae',0, ...
    'sample_rate_hz',12500,'bin_samples',25,'skip_s',10);
names = fieldnames(defaults);
for index = 1:numel(names)
    name = names{index};
    if ~isfield(settings,name) || isempty(settings.(name))
        settings.(name) = defaults.(name);
    end
end
if ~isnumeric(settings.sigmas) || isempty(settings.sigmas) || ...
        ~isvector(settings.sigmas) || ~isreal(settings.sigmas) || ...
        any(~isfinite(settings.sigmas)) || any(settings.sigmas <= 0) || ...
        any(mod(settings.sigmas,1) ~= 0)
    error('meartools_otsu_network:invalidSigmas', ...
        'settings.sigmas must contain positive finite integers.');
end
validate_nonnegative_integer(settings.min_electrodes,'min_electrodes');
validate_nonnegative_integer(settings.local_region_min_nae, ...
    'local_region_min_nae');
if settings.sample_rate_hz ~= 12500 || settings.bin_samples ~= 25 || ...
        settings.skip_s ~= 10
    error('meartools_otsu_network:fixedSourceTiming', ...
        ['Exact meaRtools fidelity fixes sample_rate_hz=12500, ' ...
        'bin_samples=25, and skip_s=10.']);
end
end

function validate_nonnegative_integer(value,name)
if ~isnumeric(value) || ~isscalar(value) || ~isreal(value) || ...
        ~isfinite(value) || value < 0 || mod(value,1) ~= 0
    error('meartools_otsu_network:invalidSetting', ...
        'settings.%s must be a nonnegative integer.',name);
end
end

function rounded = r_round(values)
% R round() uses IEC 60559 ties-to-even; MATLAB round() does not.
lower = floor(values);
fraction = values-lower;
rounded = lower+(fraction > 0.5);
ties = fraction == 0.5;
rounded(ties) = lower(ties)+mod(lower(ties),2);
end

function [starts,endsExclusive] = true_runs(mask)
changes = diff([false; logical(mask(:)); false]);
starts = find(changes == 1);
endsExclusive = find(changes == -1);
end

function events = build_event_table(starts,endsExclusive,binnedSpikes, ...
        sigmaInput,sigmaEffective,offsetS)
nEvents = numel(starts);
inputColumn = repmat(sigmaInput,nEvents,1);
effectiveColumn = repmat(sigmaEffective,nEvents,1);
startS = starts*0.002+offsetS;
endS = endsExclusive*0.002+offsetS;
durationS = endS-startS;
participating = zeros(nEvents,1);
nSpikes = zeros(nEvents,1);
for eventIndex = 1:nEvents
    rows = starts(eventIndex):endsExclusive(eventIndex)-1;
    participating(eventIndex) = nnz(any(binnedSpikes(rows,:) > 0,1));
    nSpikes(eventIndex) = full(sum(sum(binnedSpikes(rows,:))));
end
events = table(inputColumn,effectiveColumn,starts,endsExclusive,startS, ...
    endS,durationS,participating,nSpikes, ...
    'VariableNames',{'sigma_input','sigma_effective','burst_start', ...
    'burst_end','start_s','end_s','duration_s', ...
    'participating_electrodes','n_spikes'});
end

function events = empty_event_table()
events = table(zeros(0,1),zeros(0,1),zeros(0,1),zeros(0,1), ...
    zeros(0,1),zeros(0,1),zeros(0,1),zeros(0,1),zeros(0,1), ...
    'VariableNames',{'sigma_input','sigma_effective','burst_start', ...
    'burst_end','start_s','end_s','duration_s', ...
    'participating_electrodes','n_spikes'});
end

function result = empty_result(settings,durationS,nChannels,cropStartS, ...
        cropEndS,timespanS)
nSigma = numel(settings.sigmas);
eventsBySigma = repmat({empty_event_table()},1,nSigma);
sigmaInput = double(settings.sigmas(:)).';
sigmaEffective = sigmaInput;
sigmaEffective(mod(sigmaEffective,2) == 0) = ...
    sigmaEffective(mod(sigmaEffective,2) == 0)+1;
result = struct('settings',settings, ...
    'recording_duration_s',double(durationS), ...
    'crop_start_s',cropStartS,'crop_end_s',cropEndS, ...
    'timespan_s',timespanS,'bin_time_s',0.002,'offset_s',NaN, ...
    'n_channels',nChannels,'n_active_electrodes',0, ...
    'binned_spikes',sparse(0,nChannels),'sigma_input',sigmaInput, ...
    'sigma_effective',sigmaEffective, ...
    'gaussian_kernels',{cell(1,nSigma)}, ...
    'population_signals',{cell(1,nSigma)}, ...
    'otsu_levels',NaN(1,nSigma),'burst_masks',{cell(1,nSigma)}, ...
    'events_by_sigma',{eventsBySigma},'events',vertcat(eventsBySigma{:}));
end
