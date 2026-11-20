function status = run_logisi_pasq(matPath, outDir, legacyN, quality, dependencyRoot)
% run_logisi_pasq  Adapter for the unmodified Pasquale/SPYCODE implementation.
%
% The active algorithm is called directly from external/SpyCode at commit
% fb9bc345d562939a4c6f4c90002e5f689de45db1. This file only converts the
% project's spikeTime.mat representation into SPYCODE's sparse peak trains
% and serializes the returned single-channel and network-burst results.
%
% SPYCODE defaults are retained:
%   Single-channel logISI: 10 bins/decade, lowess span 5, peak distance 2,
%   void threshold 0.70, intra-burst peak limit 100 ms.
%   Burst detection: MFR > 0.1 Hz, default/core ISI 100 ms, maximum adaptive
%   ISI 1000 ms, >=5 spikes, MBR > 0.4 bursts/min, sampling rate 10 kHz.
%   Network logIBEI: 10 bins/decade, lowess span 5, peak distance 2, void
%   threshold 0.90, intra-network peak limit 100 ms, default IBEI 100 ms,
%   and >=20 percent of burst-active input units participating.
%
% legacyN is accepted only so older drivers still call this function. It is
% deliberately ignored because the authors' network method has no N_REPORT
% parameter.

if nargin < 2 || isempty(outDir), outDir = pwd; end
if nargin < 3, legacyN = []; end
if nargin >= 3 && isstruct(legacyN) && (nargin < 4 || isempty(quality))
    quality = legacyN;
end
if nargin < 4 || isempty(quality)
    quality = struct('snr_lo',1.2,'snr_hi',25,'fr_min',0.8+eps(0.8),'min_spikes',2);
end
if exist(outDir,'dir') ~= 7, mkdir(outDir); end

root = fileparts(mfilename('fullpath'));
if nargin < 5 || isempty(dependencyRoot)
    dependencyRoot = fullfile(fileparts(fileparts(root)),'external_dependencies');
end
spyRoot = fullfile(dependencyRoot,'SpyCode');
assert(exist(spyRoot,'dir')==7,'Missing authors'' SPYCODE checkout: %s',spyRoot);

authorDirs = { ...
    fullfile(spyRoot,'05_SpikeAnalysis','ISI','ISI_subScripts'), ...
    fullfile(spyRoot,'12_Utilities','UtilitiesVP'), ...
    fullfile(spyRoot,'07_BurstDetection','burstDetection','burstDetection_subScripts'), ...
    fullfile(spyRoot,'08_BurstAnalysis','IBEI','IBEI_subScripts'), ...
    fullfile(spyRoot,'07_BurstDetection','netBurstDetection','netBurstDetection_subScripts')};
requiredAuthorFiles = { ...
    fullfile(authorDirs{1},'calcISILogHist.m'), ...
    fullfile(authorDirs{1},'calcISImax.m'), ...
    fullfile(authorDirs{2},'findpeaksISI.m'), ...
    fullfile(authorDirs{2},'smoothISI.m'), ...
    fullfile(authorDirs{2},'calcMFR.m'), ...
    fullfile(authorDirs{3},'burstDetection_oneChannelComput.m'), ...
    fullfile(authorDirs{4},'calcIBEImax.m'), ...
    fullfile(authorDirs{4},'findpeaksIBEI.m'), ...
    fullfile(authorDirs{5},'netBurstDetection_alg.m')};
for i = 1:numel(requiredAuthorFiles)
    assert(exist(requiredAuthorFiles{i},'file')==2, ...
        'Missing original SPYCODE source: %s',requiredAuthorFiles{i});
end

originalPath = path;
pathCleanup = onCleanup(@() path(originalPath));
for i = 1:numel(authorDirs), addpath(authorDirs{i}); end

[SpikeTimes, ~, Regions, Electrodes, Clusters, durMs] = ...
    spiketime_to_cells(matPath,quality.snr_lo,quality.snr_hi, ...
    quality.fr_min,quality.min_spikes);
nU = numel(SpikeTimes);

% Defaults read from the authors' SPYCODE GUI files at the pinned commit.
sf = 10000;
nSamples = ceil(durMs/1000*sf)+1;
isiParam = struct('nBinsPerDec',10,'smoothMethod','lowess','smoothSpan',5, ...
    'minPeakDistance',2,'voidParamTh',0.70,'timeTh',100);
burstParam = struct('MFRmin',0.1,'ISImaxTh',100,'maxISImaxTh',1000, ...
    'minNumSpikes',5,'MBRmin',0.4,'sf',sf,'cancWin',4);
ibeiParam = struct('nBinsPerDec',10,'smoothMethod','lowess','smoothSpan',5, ...
    'minPeakDistance',2,'voidParamTh',0.90,'timeTh',100);
networkParam = struct('th',0.20,'IBeIThDef',100,'sf',sf);

% Stage 1: original per-channel adaptive logISI thresholds.
bins = cell(nU,1);
histNorm = cell(nU,1);
for u = 1:nU
    peakTrain = make_peak_train(SpikeTimes{u},sf,nSamples);
    [bins{u},histNorm{u}] = calcISILogHist(peakTrain,isiParam.nBinsPerDec,sf);
end
histSmooth = smoothISI(histNorm,isiParam.smoothMethod,isiParam.smoothSpan);
[isiMax,pks,flags] = calcISImax(bins,histSmooth, ...
    isiParam.minPeakDistance,isiParam.voidParamTh,isiParam.timeTh,nU); %#ok<ASGLU>

analyzeFlag = logical(flags(:,1));
adaptiveFlag = logical(flags(:,2));
coreThreshold = nan(nU,1);
boundaryThreshold = nan(nU,1);
thresholdSource = strings(nU,1);
for u = 1:nU
    if ~analyzeFlag(u)
        thresholdSource(u) = "not_analyzed";
    elseif adaptiveFlag(u) && isiMax(u) < burstParam.maxISImaxTh
        coreThreshold(u) = min(isiMax(u),burstParam.ISImaxTh);
        boundaryThreshold(u) = isiMax(u);
        thresholdSource(u) = "adaptive";
    else
        coreThreshold(u) = burstParam.ISImaxTh;
        boundaryThreshold(u) = burstParam.ISImaxTh;
        thresholdSource(u) = "default_100ms";
    end
end

thresholdTable = table((1:nU)',string(Electrodes(:)),Clusters(:), ...
    string(Regions(:)),isiMax(:),analyzeFlag,adaptiveFlag, ...
    coreThreshold,boundaryThreshold,thresholdSource, ...
    'VariableNames',{'unit_id','Electrode','Cluster','Region', ...
    'adaptive_isi_threshold_ms','analyze_flag','adaptive_threshold_flag', ...
    'core_threshold_used_ms','boundary_threshold_used_ms','threshold_source'});
writetable(thresholdTable,fullfile(outDir,'logisi_unit_thresholds.csv'));

% Stage 2: original SPYCODE single-channel burst detector.
rows = cell(0,9);
BDTrains = zeros(0,3); % unit id, start sample, end sample
for u = 1:nU
    if ~analyzeFlag(u), continue; end
    peakTrain = make_peak_train(SpikeTimes{u},sf,nSamples);
    addISImaxTh = [u,isiMax(u),double(analyzeFlag(u)),double(adaptiveFlag(u))];
    [burstTrain,~,~] = burstDetection_oneChannelComput(peakTrain,addISImaxTh,burstParam);
    if isempty(burstTrain), continue; end
    % SPYCODE appends one recording-summary row after the event rows.
    eventRows = burstTrain(1:end-1,1:4);
    for b = 1:size(eventRows,1)
        sSample = eventRows(b,1);
        eSample = eventRows(b,2);
        sMs = (sSample-1)/sf*1000;
        eMs = (eSample-1)/sf*1000;
        rows(end+1,:) = {Electrodes{u},Clusters(u),Regions{u}, ... %#ok<AGROW>
            sMs,eMs,eMs-sMs,eventRows(b,3),u,'Pasquale_SPYCODE'};
        BDTrains(end+1,:) = [u,sSample,eSample]; %#ok<AGROW>
    end
end

perUnitNames = {'Electrode','Cluster','Region','burst_start_ms', ...
    'burst_end_ms','burst_duration_ms','num_spikes','unit_id','Method'};
if isempty(rows)
    perUnit = cell2table(cell(0,numel(perUnitNames)),'VariableNames',perUnitNames);
else
    perUnit = cell2table(rows,'VariableNames',perUnitNames);
end
writetable(perUnit,fullfile(outDir,'logisi_perunit_bursts.csv'));

% Stage 3: original SPYCODE logIBEI threshold and network-burst detector.
ibeiMax = 0;
ibeiFlag = false;
ibeiUsed = networkParam.IBeIThDef;
numBurstUnits = numel(unique(BDTrains(:,1)));
numUnitThreshold = numBurstUnits*networkParam.th;
requiredUnits = ceil(numUnitThreshold);
if isempty(BDTrains)
    NB = zeros(0,5);
    NBpattern = cell(0,1);
else
    burstOnsets = unique(BDTrains(:,2));
    cumulativeBurstEventTrain = sparse(burstOnsets,1,1,nSamples,1);
    [ibeiBins,ibeiHist] = calcISILogHist(cumulativeBurstEventTrain, ...
        ibeiParam.nBinsPerDec,sf);
    ibeiSmooth = smoothISI({ibeiHist},ibeiParam.smoothMethod,ibeiParam.smoothSpan);
    [ibeiMaxArray,~,ibeiFlags] = calcIBEImax({ibeiBins},ibeiSmooth, ...
        ibeiParam.minPeakDistance,ibeiParam.voidParamTh,ibeiParam.timeTh,1);
    ibeiMax = ibeiMaxArray(1);
    ibeiFlag = logical(ibeiFlags(1));
    if ibeiFlag, ibeiUsed = ibeiMax; end
    [NB,NBpattern] = netBurstDetection_alg(BDTrains, ...
        [ibeiMax,double(ibeiFlag)],numUnitThreshold,networkParam);
end

if isempty(NB)
    network = array2table(zeros(0,7),'VariableNames', ...
        {'start_ms','end_ms','duration_ms','n_single_unit_bursts', ...
        'n_participating_units','minimum_participating_units','participating_fraction'});
else
    startMs = (NB(:,1)-1)/sf*1000;
    endMs = (NB(:,2)-1)/sf*1000;
    network = table(startMs,endMs,endMs-startMs,NB(:,3),NB(:,5), ...
        repmat(requiredUnits,size(NB,1),1),NB(:,5)/max(numBurstUnits,1), ...
        'VariableNames',{'start_ms','end_ms','duration_ms', ...
        'n_single_unit_bursts','n_participating_units', ...
        'minimum_participating_units','participating_fraction'});
end
writetable(network,fullfile(outDir,'logisi_network_bursts.csv'));

participationRows = cell(0,4);
for b = 1:numel(NBpattern)
    unitIds = unique(NBpattern{b}(:,1));
    for j = 1:numel(unitIds)
        u = unitIds(j);
        participationRows(end+1,:) = {b,u,Regions{u},Electrodes{u}}; %#ok<AGROW>
    end
end
participationNames = {'network_event_id','unit_id','Region','Electrode'};
if isempty(participationRows)
    participation = cell2table(cell(0,4),'VariableNames',participationNames);
else
    participation = cell2table(participationRows,'VariableNames',participationNames);
end
writetable(participation,fullfile(outDir,'logisi_network_participation.csv'));

% Preserve exact accepted onset-group members for anatomical classification.
% Do not substitute all bursts that merely overlap a population interval.
memberRows = cell(0,6);
for event = 1:numel(NBpattern)
    group = NBpattern{event};
    for member = 1:size(group,1)
        unit = group(member,1);
        sampleStart = group(member,2);
        match = find(BDTrains(:,1)==unit & BDTrains(:,2)==sampleStart);
        assert(isscalar(match),'Native onset-group member is not unique.');
        memberRows(end+1,:) = {event,unit,Regions{unit},Electrodes{unit}, ...
            (BDTrains(match,2)-1)/sf*1000,(BDTrains(match,3)-1)/sf*1000}; %#ok<AGROW>
    end
end
memberTable = cell2table(memberRows,'VariableNames', ...
    {'event_id','unit_id','Region','Electrode','start_ms','end_ms'});
writetable(memberTable,fullfile(outDir,'logisi_native_member_bursts.csv'));

networkThreshold = table(ibeiMax,ibeiFlag,ibeiUsed,numBurstUnits, ...
    networkParam.th,requiredUnits,size(NB,1), ...
    'VariableNames',{'adaptive_ibei_threshold_ms','adaptive_threshold_flag', ...
    'ibei_threshold_used_ms','n_burst_active_units', ...
    'minimum_participating_fraction','minimum_participating_units', ...
    'n_network_bursts'});
writetable(networkThreshold,fullfile(outDir,'logisi_network_threshold.csv'));

settings = table("Pasquale_SPYCODE_original", ...
    "https://github.com/barbaLab/SpyCode.git", ...
    "fb9bc345d562939a4c6f4c90002e5f689de45db1",nU,sf, ...
    isiParam.nBinsPerDec,string(isiParam.smoothMethod),isiParam.smoothSpan, ...
    isiParam.minPeakDistance,isiParam.voidParamTh,isiParam.timeTh, ...
    burstParam.MFRmin,burstParam.ISImaxTh,burstParam.maxISImaxTh, ...
    burstParam.minNumSpikes,burstParam.MBRmin, ...
    ibeiParam.voidParamTh,ibeiParam.timeTh,networkParam.IBeIThDef, ...
    networkParam.th, ...
    'VariableNames',{'detector','source_repository','source_commit','n_input_units', ...
    'sampling_rate_hz','bins_per_decade','smoothing_method','smoothing_span', ...
    'minimum_peak_distance','single_channel_void_threshold', ...
    'single_channel_peak_limit_ms','minimum_firing_rate_hz', ...
    'default_core_isi_ms','maximum_adaptive_isi_ms','minimum_spikes_per_burst', ...
    'minimum_burst_rate_per_min','network_void_threshold', ...
    'network_peak_limit_ms','default_ibei_ms','minimum_participating_fraction'});
writetable(settings,fullfile(outDir,'detector_settings.csv'));

status = struct('n_input_units',nU,'n_analyzed_units',sum(analyzeFlag), ...
    'n_adaptive_threshold_units',sum(adaptiveFlag), ...
    'n_burst_active_units',numBurstUnits,'n_unit_bursts',height(perUnit), ...
    'ibei_threshold_ms',ibeiUsed,'ibei_adaptive',ibeiFlag, ...
    'minimum_participating_units',requiredUnits, ...
    'n_network_bursts',height(network));
fprintf(['Pasquale/SPYCODE: input units %d | analyzed %d | adaptive ISI %d | ' ...
    'burst-active %d | unit bursts %d | IBEI %.3f ms (%s) | ' ...
    'network minimum %d units | network bursts %d\n'], ...
    nU,status.n_analyzed_units,status.n_adaptive_threshold_units, ...
    numBurstUnits,height(perUnit),ibeiUsed, ...
    string(ternary(ibeiFlag,'adaptive','default')),requiredUnits,height(network));
end

function peakTrain = make_peak_train(spikeTimesMs,sf,nSamples)
sample = round(double(spikeTimesMs(:))/1000*sf)+1;
sample = unique(sample(sample>=1 & sample<=nSamples));
peakTrain = sparse(sample,ones(numel(sample),1),ones(numel(sample),1),nSamples,1);
end

function value = ternary(condition,trueValue,falseValue)
if condition, value = trueValue; else, value = falseValue; end
end
