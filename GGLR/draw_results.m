clear all;close all;

colors  = {'y','b','g','m','c','c','r','k','k'};
markers = {'+','o','+','o','>','<','h','v','v','p','h'};
LStyles = {'-','-','-','-','-','-','-','-',':'};
funfcn_batch = {@LPDHG};
funfcn_stoc  = {@STOC_ADMM,@RDA_ADMM,@OPG_ADMM,@Fast_SADMM,@SPDHG_GC};
%{@STOC_ADMM,@RDA_ADMM,@OPG_ADMM,@Fast_SADMM,@Ada_SADMMdiag,@Ada_SADMMfull,@SPDHG_GC};
datasets     = {'cifar-10-bin','cifar-100-bin'};
%{'classic', 'hitech', 'k1b', 'la12', 'la1', 'la2', 'ng3sim', 'ohscal', 'reviews', 'sports'};%;{'covtype.libsvm.binary','gisette_scale','rcv1_train.binary','SUSY','splice','svmguide3','a9a','20news_100word','mushrooms','w8a'};

opts.epochs  = 10;
opts.min_time= 10;
en_subplot   = 1;

nruns_batch  = 1;
showp = 0.1;

size_font_title  = 14; size_font_legend = 12; size_font_data   = 14;
if en_subplot
    size_axis   = 12; size_axis_label   = 14;
else
    size_axis   = 24; size_axis_label   = 28;
end

if en_subplot
    figure(1);
    scrsz = get(0,'ScreenSize');
    set(gcf,'Position',[1 1 scrsz(3) scrsz(4)]);
    x_start = 0.098; x_end = 0.91; num_sub = length(datasets);
    for idx_sub = 1:num_sub
        annotation(gcf,'textbox',[(x_start + (idx_sub*2-1)*(x_end - x_start)/(2*num_sub)) 0.89 0.02 0.06],...
            'VerticalAlignment','middle',...
            'String',strrep(datasets{idx_sub},'_','-'),...
            'HorizontalAlignment','center',...
            'FontSize',14,...
            'FitBoxToText','off',...
            'EdgeColor','none');
    end
end

for idx_dataset = 1:length(datasets)
    dataset_name = datasets{idx_dataset};
    trace_accuracy   =[];
    trace_time       =[];    trace_passes     =[];
    trace_obj_val    =[];    trace_test_loss  =[];
    for idx_method = 1:length(funfcn_stoc)
        stoc_data = load(['results_GGLR_' func2str(funfcn_stoc{idx_method}) '_' dataset_name '.mat'],'stat_data','trace_passes','trace_time','trace_accuracy','trace_obj_val','trace_test_loss');
        num_runs = size(stoc_data.trace_time,2);
        for idx_runs = 1:num_runs
            for idx_trace = 1:length(stoc_data.trace_time(:,idx_runs))
                trace_accuracy(idx_trace,idx_method,idx_runs)   = real(stoc_data.trace_accuracy(idx_trace,idx_runs));
                trace_time(idx_trace,idx_method,idx_runs)       = real(stoc_data.trace_time(idx_trace,idx_runs));
                trace_passes(idx_trace,idx_method,idx_runs)     = real(stoc_data.trace_passes(idx_trace,idx_runs));
                trace_obj_val(idx_trace,idx_method,idx_runs)    = real(stoc_data.trace_obj_val(idx_trace,idx_runs));
                trace_test_loss(idx_trace,idx_method,idx_runs)  = real(stoc_data.trace_test_loss(idx_trace,idx_runs));
            end
        end
    end
    for idx_method = 1:length(funfcn_batch)
        batch_data = load(['results_GGLR_' func2str(funfcn_batch{1}) '_' dataset_name '.mat'],'stat_data','trace_passes','trace_time','trace_accuracy','trace_obj_val','trace_test_loss');
    end
    
    %stochasitc methods
    trace_accuracy_avg   =[];
    trace_accuracy_std   =[];
    trace_time_avg       =[];
    trace_time_std       =[];
    trace_passes_avg     =[];
    trace_passes_std     =[];
    trace_obj_val_avg    =[];
    trace_obj_val_std    =[];
    trace_test_loss_avg  =[];
    trace_test_loss_std  =[];
    for idx_method = 1:length(funfcn_stoc)
        num_traces = length(trace_passes(:,idx_method,1));
        for idx_trace = 1:num_traces
            idx_en = length(length(trace_passes(1,1,:)>0));
            trace_accuracy_avg(idx_trace,idx_method) = mean(trace_accuracy(idx_trace,idx_method,1:idx_en));
            trace_accuracy_std(idx_trace,idx_method) = std(trace_accuracy(idx_trace,idx_method,1:idx_en));
            trace_time_avg(idx_trace,idx_method)     = mean(trace_time(idx_trace,idx_method,1:idx_en));
            trace_time_std(idx_trace,idx_method)     = std(trace_time(idx_trace,idx_method,1:idx_en));
            trace_passes_avg(idx_trace,idx_method)   = mean(trace_passes(idx_trace,idx_method,1:idx_en));
            trace_passes_std(idx_trace,idx_method)   = std(trace_passes(idx_trace,idx_method,1:idx_en));
            trace_obj_val_avg(idx_trace,idx_method)  = mean(trace_obj_val(idx_trace,idx_method,1:idx_en));
            trace_obj_val_std(idx_trace,idx_method)  = std(trace_obj_val(idx_trace,idx_method,1:idx_en));
            trace_test_loss_avg(idx_trace,idx_method)= mean(trace_test_loss(idx_trace,idx_method,1:idx_en));
            trace_test_loss_std(idx_trace,idx_method)= std(trace_test_loss(idx_trace,idx_method,1:idx_en));
        end
    end
    
    %batch methods
    num_runs = nruns_batch;
    batch_data.trace_accuracy_avg   =[];
    batch_data.trace_accuracy_std   =[];
    batch_data.trace_time_avg       =[];
    batch_data.trace_time_std       =[];
    batch_data.trace_passes_avg     =[];
    batch_data.trace_passes_std     =[];
    batch_data.trace_obj_val_avg    =[];
    batch_data.trace_obj_val_std    =[];
    batch_data.trace_test_loss_avg  =[];
    batch_data.trace_test_loss_std  =[];
    for idx_method = 1:length(funfcn_batch)
        num_traces = length(batch_data.stat_data(idx_method,1).trace.times);
        for idx_runs = 1:num_runs
            if num_traces > length(batch_data.stat_data(idx_method,idx_runs).trace.times);
                num_traces = length(batch_data.stat_data(idx_method,idx_runs).trace.times);
            end
        end
        for idx_trace = 1:num_traces
            batch_data.trace_accuracy_avg(idx_trace,idx_method) = mean(batch_data.trace_accuracy(idx_trace,idx_method,:));
            batch_data.trace_accuracy_std(idx_trace,idx_method) = std(batch_data.trace_accuracy(idx_trace,idx_method,:));
            batch_data.trace_time_avg(idx_trace,idx_method)     = mean(batch_data.trace_time(idx_trace,idx_method,:));
            batch_data.trace_time_std(idx_trace,idx_method)     = std(batch_data.trace_time(idx_trace,idx_method,:));
            batch_data.trace_passes_avg(idx_trace,idx_method)   = mean(batch_data.trace_passes(idx_trace,idx_method,:));
            batch_data.trace_passes_std(idx_trace,idx_method)   = std(batch_data.trace_passes(idx_trace,idx_method,:));
            batch_data.trace_obj_val_avg(idx_trace,idx_method)  = mean(batch_data.trace_obj_val(idx_trace,idx_method,:));
            batch_data.trace_obj_val_std(idx_trace,idx_method)  = std(batch_data.trace_obj_val(idx_trace,idx_method,:));
            batch_data.trace_test_loss_avg(idx_trace,idx_method)= mean(batch_data.trace_test_loss(idx_trace,idx_method,:));
            batch_data.trace_test_loss_std(idx_trace,idx_method)= std(batch_data.trace_test_loss(idx_trace,idx_method,:));
        end
    end
    
    
    %     %%%%%%%%%%%%%%%%%%%%%%%%%%%% prediction error vs passes %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %     %prediction error vs passes
    %     figure(1)
    %     subplot(3,4,8+idx_dataset);
    %     min_y = inf;         max_y = -inf;
    %     for idx_method = 1:length(funfcn_stoc)
    %              idx_max = find(opts.epochs < trace_passes_avg(:,idx_method),1);
    %         if isempty(idx_max)
    %             idx_max = find(trace_passes_avg(:,idx_method) == max(trace_passes_avg(:,idx_method)));
    %         end
    %         idx_show = floor(1:idx_max*showp:idx_max);
    %         errorbar(trace_passes_avg(idx_show,idx_method),1-trace_accuracy_avg(idx_show,idx_method),trace_accuracy_std(idx_show,idx_method), ...
    %             colors{idx_method},'Marker',markers{idx_method},'LineWidth',2);     hold on;
    %
    %         temp_min_y = min(1-trace_accuracy_avg(idx_show,idx_method));
    %         temp_max_y = max(1-trace_accuracy_avg(idx_show,idx_method));
    %         if temp_min_y < min_y
    %             min_y = temp_min_y;
    %         end
    %         if temp_max_y > max_y
    %             max_y = temp_max_y;
    %         end
    %
    %     end
    %     for idx_method = 1:length(funfcn_batch)
    %         idx_max = find(opts.epochs < batch_data.trace_passes_avg(:,idx_method),1);
    %         idx_show = floor(1:idx_max*showp:idx_max);
    %         plot(batch_data.trace_passes_avg(idx_show,idx_method),1-batch_data.trace_accuracy_avg(idx_show,idx_method),...
    %             colors{idx_method + length(funfcn_stoc)},'Marker',markers{idx_method + length(funfcn_stoc)},'LineWidth',2);     hold on;
    %
    %         temp_min_y = min(1-batch_data.trace_accuracy_avg(idx_show,idx_method));
    %         temp_max_y = max(1-batch_data.trace_accuracy_avg(idx_show,idx_method));
    %         if temp_min_y < min_y
    %             min_y = temp_min_y;
    %         end
    %         if temp_max_y > max_y
    %             max_y = temp_max_y;
    %         end
    %
    %     end
    %     hold off; grid on; xlim([0 opts.epochs]);ylim([min_y max_y+0.3*(max_y-min_y)]);
    %     %         %set(gca,'fontsize',14); %set(gca,'Xscale','log'); set(gca,'Yscale','log');
    %     xlabel('effective passes','FontSize',size_font_axis); ylabel('prediction error','FontSize',size_font_axis);
    %     %         %         h_legend=legend('STOC-ADMM','RDA-ADMM','OPG-ADMM','SPDHG-GC','LPDHG');        set(h_legend,'FontSize',10);
    %     %         %title(sprintf('eta_1=%g',eta));
    %     %         %saveas(gcf,[dataset_name '_prediction_vs_passes.png']);
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%% time cost vs passes %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %time cost vs passes
    figure(1)
    if en_subplot
        subplot(3,num_sub,2*num_sub+idx_dataset);
    end
    min_y = inf;         max_y = -inf;
    for idx_method = 1:length(funfcn_stoc)
        idx_max = find(opts.epochs < trace_passes_avg(:,idx_method),1);
        if isempty(idx_max)
            idx_max = find(trace_passes_avg(:,idx_method) == max(trace_passes_avg(:,idx_method)));
        end
        idx_show = floor(1:idx_max*showp:idx_max);
        errorbar(trace_passes_avg(idx_show,idx_method),trace_time_avg(idx_show,idx_method),trace_time_std(idx_show,idx_method), ...
            colors{idx_method},'Marker',markers{idx_method},'LineWidth',1.3,'LineStyle',LStyles{idx_method});     hold on;
        
        temp_min_y = min(trace_time_avg(idx_show,idx_method));
        temp_max_y = max(trace_time_avg(idx_show,idx_method));
        if temp_min_y < min_y
            min_y = temp_min_y;
        end
        if temp_max_y > max_y
            max_y = temp_max_y;
        end
        
    end
    hold off;  xlim([0 opts.epochs]);
    ylim([0.6*min_y max_y+5*(max_y-min_y)]);
    %grid on;
    set(gca,'Yscale','log');%set(gca,'Xscale','log'); 
    set(gca,'fontsize',size_axis);
    xlabel('Number of Epochs','FontSize',size_axis_label); 
    if en_subplot
        ylabel('Time Cost','FontSize',size_axis_label);
    end
    %title(sprintf('eta_1=%g',eta));
    if ~en_subplot
        saveas(gca, [dataset_name '_time_vs_epochs.eps'],'psc2');
    end
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%% test loss vs passes %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % test loss vs passes
    figure(1)
    if en_subplot
        subplot(3,num_sub,num_sub+idx_dataset);
    end
    min_y = inf;         max_y = -inf;
    for idx_method = 1:length(funfcn_stoc)
        idx_max = find(opts.epochs < trace_passes_avg(:,idx_method),1);
        if isempty(idx_max)
            idx_max = find(trace_passes_avg(:,idx_method) == max(trace_passes_avg(:,idx_method)));
        end
        idx_show = floor(1:idx_max*showp:idx_max);
        errorbar(trace_passes_avg(idx_show,idx_method),trace_test_loss_avg(idx_show,idx_method),trace_test_loss_std(idx_show,idx_method), ...
            colors{idx_method},'Marker',markers{idx_method},'LineWidth',1.3,'LineStyle',LStyles{idx_method});     hold on;
        
        temp_min_y = min(trace_test_loss_avg(idx_show,idx_method));
        temp_max_y = max(trace_test_loss_avg(idx_show,idx_method));
        if temp_min_y < min_y
            min_y = temp_min_y;
        end
        if temp_max_y > max_y
            max_y = temp_max_y;
        end
        
    end
    for idx_method = 1:length(funfcn_batch)
        idx_max = find(opts.epochs < batch_data.trace_passes_avg(:,idx_method),1);
        idx_show = floor(1:idx_max*showp:idx_max);
        plot(batch_data.trace_passes_avg(idx_show,idx_method),batch_data.trace_test_loss_avg(idx_show,idx_method),...
            colors{idx_method + length(funfcn_stoc)},'Marker',markers{idx_method + length(funfcn_stoc)},'LineWidth',1.3,'LineStyle',LStyles{idx_method});     hold on;
        
        temp_min_y = min(batch_data.trace_test_loss_avg(idx_show,idx_method));
        temp_max_y = max(batch_data.trace_test_loss_avg(idx_show,idx_method));
        if temp_min_y < min_y
            min_y = temp_min_y;
        end
        if temp_max_y > max_y
            max_y = temp_max_y;
        end
        
    end
    hold off; %grid on;
    xlim([0 opts.epochs]);
    switch datasets{idx_dataset}
        case 'a9a'
            ylim([min_y 0.75]);
        case '20news_100word'
            ylim([min_y 0.75]);
        case 'mushrooms'
            ylim([min_y 0.8]);
        case 'w8a'
            ylim([min_y 0.8]);
        case 'splice'
            ylim([min_y 0.8]);
        otherwise
            %ylim([min_y max_y+0.1*(max_y-min_y)]);
    end
    set(gca,'fontsize',size_axis); %set(gca,'Xscale','log'); set(gca,'Yscale','log');    
    xlabel('Number of Epochs','FontSize',size_axis_label); 
    if en_subplot
        ylabel('Test Loss','FontSize',size_axis_label);
    end
    %         %title(sprintf('eta_1=%g',eta));
    if ~en_subplot
        saveas(gca, [dataset_name '_loss_vs_epochs.eps'],'psc2');
    end
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%% objective value vs passes %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %objective value vs passes
    figure(1)
    if en_subplot
        subplot(3,num_sub,idx_dataset);
    end
    min_y = inf;         max_y = -inf;
    for idx_method = 1:length(funfcn_stoc)
        idx_max = find(opts.epochs < trace_passes_avg(:,idx_method),1);
        if isempty(idx_max)
            idx_max = find(trace_passes_avg(:,idx_method) == max(trace_passes_avg(:,idx_method)));
        end
        idx_show = floor(1:idx_max*showp:idx_max);
        errorbar(trace_passes_avg(idx_show,idx_method),trace_obj_val_avg(idx_show,idx_method),trace_obj_val_std(idx_show,idx_method), ...
            colors{idx_method},'Marker',markers{idx_method},'LineWidth',1.3,'LineStyle',LStyles{idx_method});     hold on;
        
        temp_min_y = min(trace_obj_val_avg(idx_show,idx_method));
        temp_max_y = max(trace_obj_val_avg(idx_show,idx_method));
        if temp_min_y < min_y
            min_y = temp_min_y;
        end
        if temp_max_y > max_y
            max_y = temp_max_y;
        end
        
    end
    for idx_method = 1:length(funfcn_batch)
        idx_max = find(opts.epochs < batch_data.trace_passes_avg(:,idx_method),1); idx_start = 1;
        if 0
            [~,idx_start] = max(batch_data.trace_obj_val_avg(:,idx_method));
            if idx_max < idx_start
                idx_max = min(idx_start + 1/showp,length(batch_data.trace_obj_val_avg(:,idx_method)));
            end
        end
        idx_show = floor(idx_start:(idx_max-idx_start)*showp:idx_max);
        plot(batch_data.trace_passes_avg(idx_show,idx_method) - idx_start + 1 ,batch_data.trace_obj_val_avg(idx_show,idx_method),...
            colors{idx_method + length(funfcn_stoc)},'Marker',markers{idx_method + length(funfcn_stoc)},'LineWidth',1.3,'LineStyle',LStyles{idx_method});     hold on;
        
        temp_min_y = min(batch_data.trace_obj_val_avg(idx_show,idx_method));
        temp_max_y = max(batch_data.trace_obj_val_avg(idx_show,idx_method));
        if temp_min_y < min_y
            min_y = temp_min_y;
        end
        if temp_max_y > max_y
            max_y = temp_max_y;
        end
        
    end
    hold off;
    %grid on;
    xlim([0 opts.epochs]);
    switch datasets{idx_dataset}
        case 'a9a'
            ylim([min_y 0.75]);
        case '20news_100word'
            ylim([min_y 0.75]);
        case 'mushrooms'
            ylim([min_y 0.8]);
        case 'w8a'
            ylim([min_y 0.8]);
        case 'splice'
            ylim([min_y 0.8]);
        otherwise
            %ylim([min_y max_y+0.1*(max_y-min_y)]);
    end
   set(gca,'fontsize',size_axis); %set(gca,'Xscale','log'); set(gca,'Yscale','log');   
    xlabel('Number of Epochs','FontSize',size_axis_label);
     if en_subplot
         ylabel('Objective Value','FontSize',size_axis_label);
     end
    %         %title(sprintf('eta_1=%g',eta));
    if ~en_subplot
        saveas(gca, [dataset_name '_obj_vs_epochs.eps'],'psc2');        
    end
end


if en_subplot
    figure(1);
    x= [0.09,0.93]; y= [0.90,0.90];       annotation(gcf,'arrow',x,y);
    x= [0.104,0.104]; y= [0.95,0.06];          annotation(gcf,'arrow',x,y);
    suptitle('The Comparsion of All Methods on Graph-Guided Logistic Regression');
    
    
    figure(1)
    legends = [];
    for idx = 1:length(funfcn_stoc)
        legends{idx} = strrep(func2str(funfcn_stoc{idx}),'_','-');
    end
    for idx = 1:length(funfcn_batch)
        legends{end+1} = func2str(funfcn_batch{idx});
    end
    h_legend=legend(legends);    
    set(h_legend,'FontSize',size_font_legend,'Position',[0.22,0,0.6,0.06],'Box', 'off','Orientation','horizontal');
    
    annotation(gcf,'textarrow',[0.094 0.096875],...
        [0.860025542784166 0.160025542784166],'TextEdgeColor','none',...
        'TextRotation',90,...
        'FontSize',size_font_data,...
        'String','Objective Value',...
        'HeadStyle','none',...
        'LineStyle','none');
    
    annotation(gcf,'textarrow',[0.094 0.0975],...
        [0.557905491698599 0.177905491698599],'TextEdgeColor','none',...
        'TextRotation',90,...
        'FontSize',14,...
        'String','Test Loss',...
        'HeadStyle','none',...
        'LineStyle','none');
    
    annotation(gcf,'textarrow',[0.094 0.0975],...
        [0.26896551724138 0.16896551724138],'TextEdgeColor','none',...
        'TextRotation',90,...
        'FontSize',14,...
        'String','Time Cost',...
        'HeadStyle','none',...
        'LineStyle','none');
end




