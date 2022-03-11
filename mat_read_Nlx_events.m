clear all


%Set the ID for each patient
%patID       = 'pat_02459_0912';
%patID       = 'pat_02476_0929';
%patID       = 'pat_02495_0949';
%patID       = 'pat_02711_1193';
%patID       = 'pat_02718_1201';
%patID       = 'pat_03083_1527';
patID       = 'pat_03105_1551';

switch patID
    case('pat_02459_0912')
        strRec    = '02459_2017-02-14_13-27';  
    case('pat_02476_0929')
        strRec    = '02476_2017-04-10_10-18';
    case('pat_02495_0949')
        strRec    = '02495_2017-06-20_09-49';
    case('pat_02711_1193')
        strRec    = '02711_2019-04-24_16-39';
    case('pat_02718_1201')
        strRec    = '02718_2019-05-21_10-08';
    case('pat_03083_1527')
        strRec    = '03083_2021-11-15_14-18';
    case('pat_03105_1551')
        strRec    = '03105_2022-01-14_11-05';
end



WhichPC             = 'jules'; % katia'; %'user'
EEGdir              = 'eeg'; %'eegmicromed'; %name of eeg dir if exists, ex: eeg, eegmicromed

% DownSamplingFreq    = 512; %sampling frequency for downsampling

switch WhichPC
    case 'katia'
        OrigDir     = '\\lexport\iss01.epimicro\patients\raw'; %'F:\IR-IHU-ICM\Donnees\Brutes\Epilepsy';
        %OrigDirCmpt = 'F:\IR-IHU-ICM\Donnees\Brutes\Epilepsy';
        OutputDir   = 'F:\IR-IHU-ICM\Donnees\Analyses\Epilepsy';
        ScriptDir   = 'F:\IR-IHU-ICM\Donnees\git_for_gitlab';
        whichOS     = 'windows';
    case 'jules'
        OrigDir     = 'D:\LPPR_CMO_PROJECT\Paris\Data\signals_iEEG'; %path to directory with raw data
        OrigDirCmpt = ''; %path to directory with behavioural data
        OutputDir   = 'D:\LPPR_CMO_PROJECT\Paris\Data\events'; %path to directory for processed data
end



% ft_defaults
addpath('D:\Program Files\MATLAB\R2019b\toolbox\MatlabImportExport_v6.0.0');


    
str_TmpList=[];
str_RecDir=dir(fullfile(OrigDir,patID,'eeg',strRec));
str_TmpList=[str_TmpList;{str_RecDir.name}];
RecList=unique(str_TmpList);
    
clear Header Header_prev EventTable
nbRec = 0;
EventTable = table;
for s_RecDirCount = 1 : numel(RecList)

    if strcmp(RecList{s_RecDirCount},'.') || strcmp(RecList{s_RecDirCount}, '..')
        continue
    end
    str_RecDir = RecList{s_RecDirCount}; %name of patient

    %read event file
    Filename = fullfile(OrigDir,patID,EEGdir,strRec, 'Events.nev');
    if ~exist(Filename)
        continue
    end

    Header = Nlx2MatEV( Filename, [0 0 0 0 0], 1, 1, []);

    if exist('Header_prev', 'var') && isempty(setdiff(Header_prev, Header))
        continue
    end

    if ~exist('Header_prev', 'var') || ~isempty(setdiff(Header_prev, Header))
        [TimeStamps, EventIDs, TTLs, Extras, EventStrings, Header] = Nlx2MatEV( Filename, [1 1 1 1 1], 1, 1, []);
        Header_prev = Header;
        nbRec = nbRec + 1;

        EventTable.filename{nbRec,1}     = str_RecDir;
        EventTable.nbTTL{nbRec,1}        = length(TTLs);
        EventTable.TimeStamps{nbRec,1}   = TimeStamps;
        EventTable.TTL{nbRec,1}          = TTLs;
        EventTable.EventStrings{nbRec,1} = EventStrings;

        NcsFile  = dir(fullfile(OrigDir,patID,EEGdir,strRec, '*.ncs'));
        FilePath = fullfile(OrigDir,patID,EEGdir,strRec, NcsFile(1).name);
        [~, FirstTimestamps, ~, ~] =  Nlx_ReadHeader(FilePath);
        EventTable.FirstTimestamps{nbRec,1} = FirstTimestamps;
    end

end

ts = [EventTable.TimeStamps{:}' EventTable.TTL{:}' (EventTable.TimeStamps{:}' - EventTable.FirstTimestamps{:}) * 1e-6];
ts = [ts [NaN; diff(ts(:,3))]];


save(fullfile(OutputDir, patID, [patID '_events']), 'ts')





