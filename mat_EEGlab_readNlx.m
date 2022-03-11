

addpath('D:\Program Files\MATLAB\R2019b\toolbox\eeglab_current\eeglab2021.1');
addpath('D:\LPPR_CMO_PROJECT\Paris\Scripts\epishare-master\core\Nlx2Mat\MatlabImportExport_v6.0.0');
addpath('D:\LPPR_CMO_PROJECT\Paris\Scripts\LPPR_project');

InputDir     = 'D:\LPPR_CMO_PROJECT\Paris\Data\signals_iEEG';
OutputDir   = 'D:\LPPR_CMO_PROJECT\Paris\Analyses\data_for_analysis';
OrigDirTest     = 'D:\LPPR_CMO_PROJECT\Paris\Data\signals_iEEG';

%Set the ID for each patient
%patID       = 'pat_02459_0912';
%patID       = 'pat_02476_0929';
%patID       = 'pat_02495_0949';
%patID       = 'pat_02711_1193';
%patID       = 'pat_02718_1201';
%patID       = 'pat_03083_1527';
patID       = 'pat_03105_1551';

%Set resample rate
srate = 500;

eeglab

%Set time extraction for the section for each patient
switch patID
    case('pat_02459_0912')
        InputDir_ID   = '02459_2017-02-14_13-27';
    case('pat_02476_0929')
        InputDir_ID   = '02476_2017-04-10_10-18';
    case('pat_02495_0949')
        InputDir_ID   = '02495_2017-06-20_09-49';
    case('pat_02711_1193')
        InputDir_ID   = '02711_2019-04-24_16-39';
    case('pat_02718_1201')
        InputDir_ID   = '02718_2019-05-21_10-08';
    case('pat_03083_1527')
        InputDir_ID   = '03083_2021-11-15_14-18';
    case('pat_03105_1551')
        InputDir_ID    = '03105_2022-01-14_11-05';    
end

InputDir = fullfile(InputDir, patID, 'eeg', InputDir_ID);

%Create folder to recieve dataset, and section
mkdir([fullfile(OutputDir, patID)]);
OutputDir = fullfile(OutputDir, patID);

chanlist = dir(fullfile(InputDir, '*.ncs'));
%size(chanList)

%%

%check size concatenation if needed
%before look at data file size
%change sizes
%XXXXXXXXXXXXXXXXXXXXXXXX
%XXXXXXXXXXXXXXXXXXXXXXXX
size_chan_pat.nchan = [];
size_chan_pat.length = [];
for nchan = 1 : numel(chanlist)
    path2file = fullfile(chanlist(nchan).folder, chanlist(nchan).name);
    display(nchan)
    if nchan == 1
        EEG         = pop_fileio(path2file, 'dataformat','auto');
        EEG         = eeg_checkset( EEG );
        
        size_chan_pat(nchan).nchan = nchan;
        size_chan_pat(nchan).length = length(EEG.data);
        
    else
        EEGtmp = pop_fileio(path2file, 'dataformat','auto');
        EEGtmp.setname='temp';
        EEGtmp = eeg_checkset( EEGtmp );
 
        size_chan_pat(nchan).nchan = nchan;
        size_chan_pat(nchan).length = length(EEGtmp.data);
    end
end
%XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
%XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX


%%
%channel concat if no chunk are needed

for nchan = 1 : numel(chanlist)
    path2file = fullfile(chanlist(nchan).folder, chanlist(nchan).name);
    display(nchan)
    
    if nchan == 1
        EEG         = pop_fileio(path2file, 'dataformat','auto');
        EEG.setname = patID;
        EEG         = eeg_checkset( EEG );
        
        %Select data chunk
        %EEG = pop_select( EEG, 'time',[start_sec end_sec] );
        %EEG = eeg_checkset( EEG );
        
        % resample data to SR        
        EEG = pop_resample( EEG, srate);
        EEG = eeg_checkset( EEG );
        
    else
        % load 1 channel with eeglab in temp dataset
        EEGtmp = pop_fileio(path2file, 'dataformat','auto');
        EEGtmp.setname='temp';
        EEGtmp = eeg_checkset( EEGtmp );
        
        %Select data chunk
        %EEGtmp = pop_select( EEGtmp, 'time',[start_sec end_sec] );
        %EEGtmp = eeg_checkset( EEGtmp );
        
        % resample data to SR      
        EEGtmp = pop_resample( EEGtmp, srate);
        EEGtmp = eeg_checkset( EEGtmp );
        
        % eventuellement verifier que vecteurs time et sampling rate
        % identiques et data size aussi
        
        % fill new values in EEG
        EEG.nbchan   = EEG.nbchan + 1;
        EEG.data     = [EEG.data ; EEGtmp.data];
        EEG.chanlocs = [EEG.chanlocs; EEGtmp.chanlocs];
        
        % save dataset (or after loop)
        EEG = pop_saveset( EEG, 'filename', strcat(patID, '_allchan'),'filepath', OutputDir);
        EEG = eeg_checkset( EEG );
        
        
    end
    
    
    
end

%%
%Manual chunk
%size of the template chan

%for pat_02711_1193
% nchan 10 = BELT1, smaller recording
nchan_press = 10;
display(chanlist(nchan_press).name)
path2file = fullfile(chanlist(nchan_press).folder, chanlist(nchan_press).name);
EEG_sizefinal = pop_fileio(path2file, 'dataformat','auto');
EEG_sizefinal = eeg_checkset( EEG_sizefinal );

nchan = 92;
display(chanlist(nchan).name)
path2file = fullfile(chanlist(nchan).folder, chanlist(nchan).name);
EEG_sizechunk = pop_fileio(path2file, 'dataformat','auto');
EEG_sizechunk = eeg_checkset( EEG_sizechunk );

%size to chunk
size_final = size(EEG_sizefinal.data, 2);
size_to_chunk = size(EEG_sizechunk.data, 2);
start_chunk = size_to_chunk - size_final;
end_chunk = size_to_chunk;

for nchan = 1 : numel(chanlist)
    path2file = fullfile(chanlist(nchan).folder, chanlist(nchan).name);
    display(nchan)

    if nchan == 1
        EEG         = pop_fileio(path2file, 'dataformat','auto');
        EEG.setname = patID;
        EEG         = eeg_checkset( EEG );
        
        %Select data chunk
        EEG.data = EEG.data(start_chunk:end_chunk-1);
        EEG.times = EEG.times(start_chunk:end_chunk-1);
        EEG.pnts = length(EEG.data);
        
        % resample data to SR        
        EEG = pop_resample( EEG, srate);
        EEG = eeg_checkset( EEG );
        
    else
        
        
        
        % load 1 channel with eeglab in temp dataset
        EEGtmp = pop_fileio(path2file, 'dataformat','auto');
        EEGtmp.setname='temp';
        EEGtmp = eeg_checkset( EEGtmp );
        
        if chanlist(nchan).bytes == chanlist(nchan_press).bytes
            % resample data to SR      
            EEGtmp = pop_resample( EEGtmp, srate);
            EEGtmp = eeg_checkset( EEGtmp );
            
            % fill new values in EEG
            EEG.nbchan   = EEG.nbchan + 1;
            EEG.data     = [EEG.data ; EEGtmp.data];
            EEG.chanlocs = [EEG.chanlocs; EEGtmp.chanlocs];
            
            % save dataset (or after loop)
            EEG = pop_saveset( EEG, 'filename', strcat(patID, '_allchan'),'filepath', OutputDir);
            EEG = eeg_checkset( EEG )
            
            continue
        end
        
        %Select data chunk
        EEGtmp.data = EEGtmp.data(start_chunk:end_chunk-1);
        EEGtmp.times = EEGtmp.times(start_chunk:end_chunk-1);
        EEGtmp.pnts = length(EEGtmp.data);
        
        % resample data to SR      
        EEGtmp = pop_resample( EEGtmp, srate);
        EEGtmp = eeg_checkset( EEGtmp );
        
        % eventuellement verifier que vecteurs time et sampling rate
        % identiques et data size aussi
        
        % fill new values in EEG
        EEG.nbchan   = EEG.nbchan + 1;
        EEG.data     = [EEG.data ; EEGtmp.data];
        EEG.chanlocs = [EEG.chanlocs; EEGtmp.chanlocs];
        
        % save dataset (or after loop)
        EEG = pop_saveset( EEG, 'filename', strcat(patID, '_allchan'),'filepath', OutputDir);
        EEG = eeg_checkset( EEG );
        
        
    end
    
    
    
end



%%
%if necessary
%chan sig and power monitoring

%set time section
time_section = [1, 1000000];

%all sig in one plot
figure, clf;
hold on
for nchan = 1 : numel(chanlist)
    
    path2file = fullfile(chanlist(nchan).folder, chanlist(nchan).name);
    display(nchan)
    str_title = chanlist(nchan).name(24:end-4);
    if contains(str_title, physio_chan) == 1
        continue
    end
    
    EEG = pop_fileio(path2file, 'dataformat','auto');
    
    %plot(EEG.times(time_section), EEG.data(time_section)); 
    plot(EEG.times, EEG.data); 
    
end

%one plot for every chan
for nchan = 1 : numel(chanlist)
    
    path2file = fullfile(chanlist(nchan).folder, chanlist(nchan).name);
    display(nchan)
    
    EEG = pop_fileio(path2file, 'dataformat','auto');
    
    figure(nchan), clf;
    %plot(EEG.times(time_section), EEG.data(time_section));
    plot(EEG.times, EEG.data);
    str_title = chanlist(nchan).name(24:end-4);
    title(str_title);  
    
    %save if needed
    %cd('C:\Users\jules\Desktop\test pour katia\pat_02476_0929');
    %saveas(figure(nchan), strcat(patID, '_', str_title), 'jpg');
    
end


%power monitoring
%initiate power monitoring
%srate already defined
% create welch window
winsize = 10*srate; % 10-second window
% create Hann window
hannw = .5 - cos(2*pi*linspace(0,1,winsize))./2;
% number of FFT points (frequency resolution) with frequency resolution = srate/nfft
nfft = srate*50;
%srate/nfft

%power monitoring
for nchan = 1 : numel(chanlist)
    
    path2file = fullfile(chanlist(nchan).folder, chanlist(nchan).name);
    display(nchan)
    str_title = chanlist(nchan).name(24:end-4);
    if contains(str_title, physio_chan) == 1
        continue
    end
    
    EEG = pop_fileio(path2file, 'dataformat','auto');
    
    figure(nchan), clf;
    pwelch(EEG.data,hannw,round(winsize/2),nfft,srate);
    %pwelch(EEG.data(time_section),hannw,round(winsize/2),nfft,srate);
    
    title(str_title);

end


%%
% test open eeglab

EEG_loca = pop_loadset('filename', 'pat_03083_1527.set', 'filepath', 'D:\LPPR_CMO_PROJECT\Paris\Analyses\data_for_analysis\pat_03083_1527');
EEG_loca = eeg_checkset( EEG_loca );




