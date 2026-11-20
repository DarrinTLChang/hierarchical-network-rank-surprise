function run_comparators(inputRoot, outputRoot, trialIds, dependencyRoot)
% RUN_COMPARATORS Reproduce the fixed comparator settings on archived inputs.
%   run_comparators('datasets/network_burst','runs/comparators')
%   accepts either a cohort directory containing trials/ or the trials/
%   directory itself. An optional third argument selects recording IDs.
%   An optional fourth argument locates the pinned external checkouts.
%
% This driver reads only simulated spikeTime.mat inputs, never planted labels.
% It does not generate datasets or alter detector participation/duration rules.
% Native population intervals remain separate from post-hoc anatomical labels.

if nargin < 2
    error('run_comparators:missingPaths','Provide input and new output paths.');
end
inputRoot = char(string(inputRoot)); outputRoot = char(string(outputRoot));
here = fileparts(mfilename('fullpath'));
if nargin < 4 || isempty(dependencyRoot)
    dependencyRoot = fullfile(fileparts(fileparts(here)),'external_dependencies');
end
dependencyRoot = char(string(dependencyRoot));
if isfolder(fullfile(inputRoot,'trials')), inputRoot = fullfile(inputRoot,'trials'); end
assert(isfolder(inputRoot),'Input cohort directory does not exist: %s',inputRoot);
if nargin < 3 || isempty(trialIds)
    listing = dir(fullfile(inputRoot,'trial*')); trialIds = [];
    for index = 1:numel(listing)
        if listing(index).isdir
            token = regexp(listing(index).name,'^trial(\d+)$','tokens','once');
            if ~isempty(token), trialIds(end+1) = str2double(token{1}); end %#ok<AGROW>
        end
    end
    trialIds = sort(trialIds);
end
assert(~isempty(trialIds) && all(isfinite(trialIds)) && ...
    all(trialIds>=1 & mod(trialIds,1)==0) && numel(unique(trialIds))==numel(trialIds));
assert(~isfolder(outputRoot) && ~isfile(outputRoot), ...
    'Output path must be new: %s',outputRoot);
assert(isfolder(fullfile(dependencyRoot,'SpyCode')) && ...
    isfolder(fullfile(dependencyRoot,'NetworkWideAdaptiveBurstDetection')), ...
    'Run python comparators/setup_external.py or provide pinned dependencyRoot.');
assert(exist('smooth','file')==2, ...
    'SPYCODE requires smooth() from MATLAB Curve Fitting Toolbox.');
assert(exist('uddpvparse','file')==2, ...
    'SPYCODE requires the legacy Signal Processing Toolbox peak helpers.');
assert(exist('skewness','file')==2, ...
    'Network-CMA requires skewness() from Statistics and Machine Learning Toolbox.');
previousPath = path; cleanup = onCleanup(@()path(previousPath)); %#ok<NASGU>
addpath(here);
maxNumCompThreads(1);
mkdir(outputRoot);
quality = struct('snr_lo',1.2,'snr_hi',25, ...
    'fr_min',0.8+eps(0.8),'min_spikes',2);
otsuSettings = struct('sigmas',45,'min_electrodes',4, ...
    'local_region_min_nae',0,'sample_rate_hz',12500,'bin_samples',25,'skip_s',10);

for trial = reshape(trialIds,1,[])
    tag = sprintf('trial%02d',trial);
    source = fullfile(inputRoot,tag);
    matPath = fullfile(source,'spikeTime.mat');
    assert(isfile(matPath) && isfile(fullfile(source,'channel_map.csv')) && ...
        isfile(fullfile(source,'event_manifest.csv')), ...
        'Expected an archived multiregion synthetic recording: %s',source);
    % Metadata existence identifies the cohort. No ground-truth values are read.
    [trains,ids,regions,electrodes,clusters,durationMs] = spiketime_to_cells( ...
        matPath,quality.snr_lo,quality.snr_hi,quality.fr_min,quality.min_spikes);
    trains = trains(:); ids = string(ids(:)); regions = string(regions(:));
    electrodes = string(electrodes(:)); clusters = clusters(:);
    assert(durationMs==250000 && numel(trains)<=75 && numel(unique(regions))==5, ...
        'This fixed-profile driver supports the released 250-s, five-group cohort.');
    folder = fullfile(outputRoot,tag); mkdir(folder);
    nSpikes = cellfun(@numel,trains);
    retained = table(electrodes,clusters-1,regions,ids,nSpikes, ...
        nSpikes*1000/durationMs,'VariableNames', ...
        {'Electrode','Cluster','Region','unit_id','spike_count','FR_Hz'});
    writetable(retained,fullfile(folder,'retained_units.csv'));

    logFolder = fullfile(folder,'logibei');
    logStatus = run_logisi_pasq(matPath,logFolder,[],quality,dependencyRoot);
    logEvents = readtable(fullfile(logFolder,'logisi_network_bursts.csv'),'TextType','string');
    logMembers = readtable(fullfile(logFolder,'logisi_native_member_bursts.csv'),'TextType','string');
    assert(logStatus.n_input_units==numel(trains));

    cmaFolder = fullfile(folder,'network_cma_perunit');
    cmaStatus = run_valkki_net(matPath,cmaFolder,quality,dependencyRoot);
    cmaUnits = readtable(fullfile(cmaFolder,'valkki_network_cma_perunit_bursts.csv'),'TextType','string');
    assert(cmaStatus.n_input_units==numel(trains));
    [segments,signalMean,signalSd] = make_signal(cmaUnits,durationMs);
    active = segments.n_simultaneously_bursting_inputs > signalMean+0.75*signalSd;
    edges = diff([false;active;false]);
    first = find(edges==1); last = find(edges==-1)-1;
    cmaStart = segments.segment_start_ms(first);
    cmaEnd = segments.segment_end_ms(last);
    writetable(segments,fullfile(folder,'cma_burst_signal_segments.csv'));

    channelNames = unique(electrodes,'stable');
    channelTrains = cell(numel(channelNames),1);
    channelRegions = strings(numel(channelNames),1);
    for channel = 1:numel(channelNames)
        mask = electrodes==channelNames(channel);
        labels = unique(regions(mask)); assert(isscalar(labels));
        channelRegions(channel) = labels;
        parts = cellfun(@(x)double(x(:))/1000,trains(mask),'UniformOutput',false);
        channelTrains{channel} = sort(vertcat(parts{:}));
    end
    otsuCore = meartools_otsu_network(channelTrains,durationMs/1000,otsuSettings);
    otsuEvents = otsuCore.events_by_sigma{1};
    otsuStart = otsuEvents.start_s*1000; otsuEnd = otsuEvents.end_s*1000;
    regionalParts = cell(numel(unique(channelRegions)),1);
    groupNames = unique(channelRegions,'stable');
    for group = 1:numel(groupNames)
        take = channelRegions==groupNames(group);
        localCore = meartools_otsu_network(channelTrains(take),durationMs/1000,otsuSettings);
        localEvents = localCore.events_by_sigma{1};
        regionalParts{group} = table( ...
            repmat(groupNames(group),height(localEvents),1), ...
            localEvents.start_s*1000,localEvents.end_s*1000, ...
            'VariableNames',{'region','start_ms','end_ms'});
    end
    otsuRegional = vertcat(regionalParts{:});
    writetable(otsuRegional,fullfile(folder,'otsu_regional_intervals.csv'));

    native = [population_rows("logIBEI",trial,logEvents.start_ms,logEvents.end_ms); ...
        population_rows("Network-CMA",trial,cmaStart,cmaEnd); ...
        population_rows("meaRtools/Otsu",trial,otsuStart,otsuEnd)];
    writetable(native,fullfile(folder,'native_population_events.csv'));
    members = cell(0,8);
    for row = 1:height(logMembers)
        members(end+1,:) = {"logIBEI",trial,0,logMembers.event_id(row), ... %#ok<AGROW>
            string(logMembers.Region(row)),logMembers.start_ms(row), ...
            logMembers.end_ms(row),string(logMembers.unit_id(row))};
    end
    for event = 1:numel(cmaStart)
        take = cmaUnits.burst_start_ms<cmaEnd(event) & cmaUnits.burst_end_ms>cmaStart(event);
        for row = reshape(find(take),1,[])
            members(end+1,:) = {"Network-CMA",trial,0,event, ... %#ok<AGROW>
                string(cmaUnits.Region(row)),cmaUnits.burst_start_ms(row), ...
                cmaUnits.burst_end_ms(row),string(cmaUnits.unit_id(row))};
        end
    end
    for event = 1:numel(otsuStart)
        take = otsuRegional.start_ms<otsuEnd(event) & otsuRegional.end_ms>otsuStart(event);
        for row = reshape(find(take),1,[])
            members(end+1,:) = {"meaRtools/Otsu",trial,0,event, ... %#ok<AGROW>
                string(otsuRegional.region(row)),otsuRegional.start_ms(row), ...
                otsuRegional.end_ms(row),""};
        end
    end
    support = cell2table(members,'VariableNames', ...
        {'method','trial','master_seed','event_id','region','start_ms','end_ms','unit_id'});
    writetable(support,fullfile(folder,'support_events.csv'));
    counts = struct('logIBEI',height(logEvents),'NetworkCMA',numel(cmaStart), ...
        'meaRtools',numel(otsuStart));
    proof = struct('trial',trial,'profile','general-selected-1-3-15-comparators', ...
        'quality',quality,'firing_rate_rule','strictly greater than 0.8 Hz', ...
        'logIBEI_peak_search_ms',100,'NetworkCMA_k',0.75,'meaRtools_Sigma',45, ...
        'population_counts',counts,'n_retained_units',numel(trains), ...
        'recording_duration_s',durationMs/1000, ...
        'meaRtools_crop_start_s',otsuCore.crop_start_s, ...
        'meaRtools_crop_end_s',otsuCore.crop_end_s, ...
        'meaRtools_exposure_s',otsuCore.timespan_s, ...
        'cma_time_weighted_mean',signalMean,'cma_population_sd',signalSd, ...
        'native_population_boundaries_unchanged',true,'output_merge_added',false, ...
        'output_duration_filter_added',false,'ground_truth_values_read',false, ...
        'MATLAB_version',string(version),'platform',string(computer));
    handle = fopen(fullfile(folder,'recording_settings.json'),'w');
    assert(handle>=0); fprintf(handle,'%s\n',jsonencode(proof,PrettyPrint=true)); fclose(handle);
    fprintf('COMPARATORS_COMPLETE %s logIBEI=%d CMA=%d meaRtools=%d\n', ...
        tag,counts.logIBEI,counts.NetworkCMA,counts.meaRtools);
end
end

function rows = population_rows(method,trial,starts,ends)
starts = double(starts(:)); ends = double(ends(:)); n = numel(starts);
assert(numel(ends)==n && all(starts<ends));
event_id = (1:n)';
prediction_id = strings(n,1);
if n>0, prediction_id = compose('%s:t%02d:%05d',method,trial,event_id); end
rows = table(repmat(string(method),n,1),repmat(trial,n,1),zeros(n,1), ...
    event_id,prediction_id,starts,ends,repmat("population",n,1),strings(n,1), ...
    'VariableNames',{'method','trial','master_seed','event_id','prediction_id', ...
    'start_ms','end_ms','kind','regions'});
end

function [segments,signalMean,signalSd] = make_signal(events,durationMs)
names = {'segment_start_ms','segment_end_ms','duration_ms','n_simultaneously_bursting_inputs'};
if isempty(events)
    segments = table(0,durationMs,durationMs,0,'VariableNames',names);
    signalMean = 0; signalSd = 0; return
end
starts = max(0,min(durationMs,double(events.burst_start_ms)));
ends = max(0,min(durationMs,double(events.burst_end_ms)));
valid = isfinite(starts)&isfinite(ends)&ends>starts; starts = starts(valid); ends = ends(valid);
[uniqueTimes,~,groups] = unique([starts;ends]);
deltas = accumarray(groups,[ones(numel(starts),1);-ones(numel(ends),1)]);
boundaries = unique([0;uniqueTimes;durationMs]);
startsOut = boundaries(1:end-1); endsOut = boundaries(2:end);
durations = endsOut-startsOut; counts = zeros(numel(startsOut),1); current = 0; changeIndex = 1;
for index = 1:numel(startsOut)
    while changeIndex<=numel(uniqueTimes) && uniqueTimes(changeIndex)<=startsOut(index)
        current = current+deltas(changeIndex); changeIndex = changeIndex+1;
    end
    counts(index) = current;
end
keep = durations>0;
segments = table(startsOut(keep),endsOut(keep),durations(keep),counts(keep),'VariableNames',names);
weights = segments.duration_ms/durationMs;
signalMean = sum(weights.*segments.n_simultaneously_bursting_inputs);
signalSd = sqrt(sum(weights.*(segments.n_simultaneously_bursting_inputs-signalMean).^2));
end
