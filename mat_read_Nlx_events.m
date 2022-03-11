todo.whichPC        = 'jules'; %'PCkatia'; PCkatia, knode01 %add your path to scripts with neuralynx mewfiles below in 'user'

%Set the ID for each patient
%patID       = 'pat_02459_0912';
%patID       = 'pat_02476_0929';
%patID       = 'pat_02495_0949';
%patID       = 'pat_02711_1193';
%patID       = 'pat_02718_1201';
%patID       = 'pat_03083_1527';
patID       = 'pat_03105_1551';


switch todo.whichPC
    case 'PCkatia'
        InputDirRespi     = 'F:\IR-IHU-ICM\Bureautique\ownCloud\sEEGresp\Data\respi_iEEG';
        InputDiriEEGpress = 'F:\IR-IHU-ICM\Bureautique\ownCloud\sEEGresp\Data\signals_iEEG';
        OutputDir         = 'F:\IR-IHU-ICM\Bureautique\ownCloud\sEEGresp\Analyses\data_for_analysis';
        eeglabDir         = 'D:\MATLAB\toolboxes\eeglab2019_1';
        path2scripts      = 'F:\IR-IHU-ICM\Bureautique\ownCloud\sEEGresp\Scripts';
        path2NlxMex       = 'F:\IR-IHU-ICM\Donnees\git_for_gitlab\epiShare\core';
    case 'jules'
        InputDirRespi     = 'D:\LPPR_CMO_PROJECT\Paris\Data\respi_iEEG';
        InputDiriEEGpress = 'D:\LPPR_CMO_PROJECT\Paris\Data\signals_iEEG';
        InputDirChanLoc   = fullfile('D:\LPPR_CMO_PROJECT\Paris\Analyses\anatomy\', patID);
        OutputDir         = 'D:\LPPR_CMO_PROJECT\Paris\Analyses\data_for_analysis';
        eeglabDir         = 'D:\Program Files\MATLAB\R2019b\toolbox\eeglab_current\eeglab2019_1';
        nlxopenDir        = 'D:\Program Files\MATLAB\R2019b\toolbox\MatlabImportExport_v6.0.0';
        path2scripts      = 'D:\LPPR_CMO_PROJECT\Paris\Scripts\LPPR_project';
        path2NlxMex       = 'D:\LPPR_CMO_PROJECT\Paris\Scripts\epishare-master\core';
        fieldtripDir          = 'D:\Program Files\MATLAB\fieldtrip';
end
 
addpath(eeglabDir);
addpath(nlxopenDir);
addpath(path2scripts);
addpath(fieldtripDir);
addpath(fullfile(path2NlxMex, 'Nlx2Mat', 'MatlabImportExport_v6.0.0'));



%Set time extraction for the section for each patient
switch patID
    case('pat_02459_0912')
        channels        = {8, 'belt'; 16, 'pres'};
        reference       = {'Cinp_3'};
        PRESiEEGpath    = fullfile(InputDiriEEGpress, patID, 'eeg', '02459_2017-02-14_13-27', '02459_2017-02-14_13-27_PRES1.ncs');
        BELTiEEGpath    = fullfile(InputDiriEEGpress, patID, 'eeg', '02459_2017-02-14_13-27', '02459_2017-02-14_13-27_BELT1.ncs');
        InputDiriEEG    = fullfile(OutputDir, patID, [patID '_allchan.set']);
        iEEGtrig        = fullfile(InputDirRespi, patID, 'SEEGresp_PATIENT_1_H_14022017_iEEGtrig.txt') ; %'D:\\LPPR_CMO_PROJECT\\Paris\\Data\\respi_iEEG\\pat_02459_0912\\SEEGresp_PATIENT_1_H_14022017_iEEGtrig.txt';
        iEEGtrigtest    = fullfile(InputDirRespi, patID, 'SEEGresp_PATIENT_1_H_14022017_iEEGtrig_test.txt');   
    case('pat_02476_0929')
        channels        = {1, 'pres'; 2, 'belt'};
        reference       = {'OPF3'};
        PRESiEEGpath    = fullfile(InputDiriEEGpress, patID, 'eeg', '02476_2017-04-10_10-18', '02476_2017-04-10_10-18_PRES1.ncs');
        BELTiEEGpath    = fullfile(InputDiriEEGpress, patID, 'eeg', '02476_2017-04-10_10-18', '02476_2017-04-10_10-18_BELT1.ncs');
        InputDiriEEG    = 'D:\LPPR_CMO_PROJECT\Paris\Analyses\data_for_analysis\pat_02476_0929\pat_02476_0929_allchan.set';
        iEEGtrig        = 'D:\\LPPR_CMO_PROJECT\\Paris\\Data\\respi_iEEG\\pat_02476_0929\\SEEGresp_PATIENT_2_H_10042017_iEEGtrig.txt';
    case('pat_02495_0949')
        channels        = {1, 'belt'; 2, 'pres'};
        reference       = {'IntT1_1'};
        PRESiEEGpath    = fullfile(InputDiriEEGpress, patID, 'eeg', '02495_2017-06-20_09-49', '02495_2017-06-20_09-49_PRES1.ncs');
        BELTiEEGpath    = fullfile(InputDiriEEGpress, patID, 'eeg', '02495_2017-06-20_09-49', '02495_2017-06-20_09-49_BELT1.ncs');
        InputDiriEEG    = 'D:\LPPR_CMO_PROJECT\Paris\Analyses\data_for_analysis\pat_02495_0949\pat_02495_0949_allchan.set';
        iEEGtrig        = 'D:\\LPPR_CMO_PROJECT\\Paris\\Data\\respi_iEEG\\pat_02495_0949\\SEEGresp_PATIENT_3_H_20062917_iEEGtrig.txt';
    case('pat_02711_1193')
        channels    = {1, 'belt'; 2, 'pres'};
        reference       = {'F2M_6'};
        PRESiEEGpath    = fullfile(InputDiriEEGpress, patID, 'eeg', '02711_2019-04-24_16-39', '02711_2019-04-24_16-39_PRES1.ncs');
        BELTiEEGpath    = fullfile(InputDiriEEGpress, patID, 'eeg', '02711_2019-04-24_16-39', '02711_2019-04-24_16-39_BELT1.ncs');
        InputDiriEEG = 'D:\LPPR_CMO_PROJECT\Paris\Analyses\data_for_analysis\pat_02711_1193\pat_02711_1193_allchan.set';
        %InputDiriEEG = fullfile(OutputDir, patID, 'pat*allchan.set');
        iEEGtrig = 'D:\\LPPR_CMO_PROJECT\\Paris\\Data\\respi_iEEG\\pat_02711_1193\\SEEGresp_PATIENT_4_F_24042019_iEEGtrig.txt';
    case('pat_02718_1201')
        channels    = {1, 'belt'; 2, 'pres'};
        reference       = {'????????'};
        PRESiEEGpath    = fullfile(InputDiriEEGpress, patID, 'eeg', '02718_2019-05-21_10-08', '02718_2019-05-21_10-08_PRES1.ncs');
        BELTiEEGpath    = fullfile(InputDiriEEGpress, patID, 'eeg', '02718_2019-05-21_10-08', '02718_2019-05-21_10-08_BELT1.ncs');
        InputDiriEEG = 'D:\LPPR_CMO_PROJECT\Paris\Analyses\data_for_analysis\pat_02718_1201\pat_02718_1201_allchan.set';
        %InputDiriEEG = fullfile(OutputDir, patID, 'pat*allchan.set');
        iEEGtrig = 'D:\\LPPR_CMO_PROJECT\\Paris\\Data\\respi_iEEG\\pat_02718_1201\\SEEGresp_PATIENT_5_F_21052019_iEEGtrig.txt';
    case('pat_03083_1527')
        dataDir    = fullfile(InputDiriEEGpress, patID, 'eeg', '03083_2021-11-15_14-18');
    case('pat_03105_1551')
        dataDir    = fullfile(InputDiriEEGpress, patID, 'eeg', '03105_2022-01-14_11-05');
        
end

%%

Nlx2MatEV('Events.nev')


Timestamps = [];
EventIDs = [];
TTLs = [];
Extras = [];
EventStrings = [];
Header = [];
[Timestamps, EventIDs, TTLs, Extras, EventStrings, Header] = Nlx2MatEV(fullfile(dataDir, 'Events.nev'), [1 1 1 1 1], 1, 1, [] );



fn = FindFile('*Events.nev');


[EVTimeStamps, EventIDs, TTLs, EVExtras, EventStrings, EVHeader] = Nlx2MatEV('Events.nev',[1 1 1 1 1],1,1,[]);












