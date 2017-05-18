% Graph-Guided Logistic Regression
clear all;close all;

funfcn_batch = {@LPDHG};
funfcn_stoc  = {@STOC_ADMM,@RDA_ADMM,@OPG_ADMM,@Fast_SADMM,@SPDHG_GC};
en_datasets1 = 0;
if en_datasets1
    datasets     = {'covtype.libsvm.binary'};%;{'covtype.libsvm.binary','gisette_scale','rcv1_train.binary','SUSY','splice','svmguide3','a9a','20news_100word','mushrooms','w8a'};
else
    datasets     = {'classic', 'hitech', 'k1b', 'la12', 'la1', 'la2', 'ng3sim', 'ohscal', 'reviews', 'sports'};
end

nruns_stoc   = 1;
n_epochs     = 2;
nruns_batch  = 1;

for idx_datasets = 1:length(datasets)
    dataset_name = datasets{idx_datasets};
    fprintf('\nNow processing (%d/%d) dataset: %s\n',idx_datasets,length(datasets),dataset_name);
    if en_datasets1
        data_path = 'E:\Documents\Datasets\mat_LIBSVM\';
        load([data_path dataset_name '.mat']); 
    else
        data_path = 'E:\Documents\Datasets\mat_docdatasets\';
        dataset = load([data_path dataset_name '']); samples = dataset.dtm'; labels = dataset.classid; 
    end
    
    %Graphical Matrix generation
    if( exist(['temp_F_' dataset_name '.mat'],'file') ==0 )
        S  = cov(samples');
        rho = 0.005; % weighting parameter and the parameters can be tuned
        opts.mxitr = 500; opts.mu0 = 1e-1;  opts.muf = 1e-3; opts.rmu = 1/4;
        opts.tol_gap = 1e-1; opts.tol_frel = 1e-7; opts.tol_Xrel = 1e-7; opts.tol_Yrel = 1e-7;
        opts.numDG = 10; opts.record = 0;opts.sigma = 1e-10;
        out = SICS_ALM(S,rho,opts);
        X = out.X; X(abs(X) > 2.5e-3) = 1; X(abs(X) < 2.5e-3) = 0; F = -tril(X,-1) + triu(X,1);
        save(['temp_F_' dataset_name '.mat'], 'F','S');
    else
        load(['temp_F_' dataset_name '.mat'],'F','S');
    end
    
    %Samples divide
    ratio_train  = 0.8;
    idx_all      = 1:length(labels);
    %     if( exist(['test_data_' dataset_name '.mat'],'file') ==0 )
    %         idx_train    = idx_all(rand(1,length(labels),1)<ratio_train);
    %         idx_test     = setdiff(idx_all,idx_train);
    %         save(['test_data_' dataset_name '.mat'], 'idx_train','idx_test');
    %     else
    %         load(['test_data_' dataset_name '.mat'], 'idx_train','idx_test');
    %     end
    idx_train    = idx_all(rand(1,length(labels),1)<ratio_train);
    idx_test     = setdiff(idx_all,idx_train);
    if sum(sum(samples==0))/(size(samples,1)*size(samples,2)) > 0.8
        s_train      = sparse(double(samples(:,idx_train)));
        s_test       = sparse(double(samples(:,idx_test)));
    else        
        s_train      = full(double(samples(:,idx_train)));
        s_test       = full(double(samples(:,idx_test)));
    end
    l_train      = labels(idx_train);
    l_test       = labels(idx_test);
    
    %Parameters setup
    %Parameters of model
    opts.mu      = 1e-5; %parameter of graph-guided term
    
    %Parameters of algorithms
    opts.F       = F;     %The graph structure
    opts.beta    = 1;     %parameter of STOC-ADMM to balance augmented lagrange term
    opts.gamma   = 1e-2;  %Regularized Logistic Regression term
    opts.epochs  = n_epochs;     %maximum effective passes
    opts.max_it  = length(idx_train)*opts.epochs;
    opts.checkp  = 0.01;  %save the solution very 1% of data processed
    opts.checki  = floor(opts.max_it * opts.checkp);
    opts.a       = 1;     %parameter of Ada_SADMM
    opts.s       = 5e-5;  %parameter of SPDHG to update y;
    
    tempVar = zeros(size(samples,2),1);
    for idx_s = 1:size(samples,2)
        tempVar(idx_s) = samples(:,idx_s)'*samples(:,idx_s);
    end
    opts.L = 0.25*max(tempVar);
    eigFTF = eigs(opts.F'*opts.F, 1);
    opts.L1 = opts.beta*eigFTF + opts.L;
    opts.L2 = sqrt(max(2*opts.L*opts.L+eigFTF, 2*eigFTF));
    opts.L3 = max(8*opts.beta*eigFTF, sqrt(8*opts.L*opts.L + opts.beta*eigFTF))+opts.gamma;
    
    opts.eta = opts.beta*eigFTF;%parameter of RDA-ADMM and OPG-ADMM
    
    %stoc methods
    par_settings = [2.^(-7:1:2) 5:1:16 17:5:32 2.^(6:1:10)];
    for idx_mu = 1%1:length(par_settings)
        %         opts.mu = par_settings(6)*1e-5;
        %         opts.s  = par_settings(idx_mu)*5e-4;
        
        for idx_method = 1:length(funfcn_stoc)
            stat_data    = [];
            trace_accuracy = [];
            trace_test_loss= [];
            trace_obj_val  = [];
            trace_passes   = [];
            trace_time     = [];
            num_train      = length(idx_train);
            num_runs       = nruns_stoc;
            for idx_runs = 1:num_runs
                switch func2str(funfcn_stoc{idx_method})
                    case 'SPDHG_GC'
                        switch datasets{idx_datasets}
                            case 'splice'
                                idx_opt = 8;
                                opts.scaling = par_settings(idx_opt)+0.8;
                            case 'svmguide3'
                                idx_opt = 6;
                                opts.scaling = par_settings(idx_opt);
                                
                            case 'mushrooms'
                                idx_opt = 5;
                                opts.scaling = par_settings(idx_opt);
                            case '20news_100word'
                                idx_opt = 6;
                                opts.scaling = par_settings(idx_opt)-0.0001;
                                
                            case 'a9a'
                                idx_opt = 5;
                                opts.scaling = par_settings(idx_opt);
                            case 'w8a'
                                idx_opt = 5;
                                opts.scaling = par_settings(idx_opt);
                            case 'covtype.libsvm.binary'
                                idx_opt = 7;
                                opts.scaling = par_settings(idx_opt);
                        end
                    case 'STOC_ADMM'
                        switch datasets{idx_datasets}
                            case 'splice'
                                idx_opt = 7;
                                opts.scaling = par_settings(idx_opt);
                            case 'svmguide3'
                                idx_opt = 7;
                                opts.scaling = par_settings(idx_opt);
                            case 'mushrooms'
                                idx_opt = 4;
                                opts.scaling = par_settings(idx_opt);
                            case '20news_100word'
                                idx_opt = 8;
                                opts.scaling = par_settings(idx_opt);
                            case 'a9a'
                                idx_opt = 5;
                                opts.scaling = par_settings(idx_opt);
                            case 'w8a'
                                idx_opt = 2;
                                opts.scaling = par_settings(idx_opt);
                            case 'covtype.libsvm.binary'
                                idx_opt = 7;
                                opts.scaling = 1e-5*par_settings(idx_opt);
                        end
                    case 'RDA_ADMM'
                        switch datasets{idx_datasets}
                            case 'splice'
                                idx_opt = 6;
                                opts.scaling = par_settings(idx_opt);
                            case 'svmguide3'
                                idx_opt = 13;
                                opts.scaling = par_settings(idx_opt);
                            case 'mushrooms'
                                idx_opt = 4;
                                opts.scaling = par_settings(idx_opt);
                            case '20news_100word'
                                idx_opt = 9;
                                opts.scaling = par_settings(idx_opt);
                            case 'a9a'
                                idx_opt = 18;
                                opts.scaling = par_settings(idx_opt);
                            case 'w8a'
                                idx_opt = 5;
                                opts.scaling = par_settings(idx_opt);
                        end
                    case 'OPG_ADMM'
                        switch datasets{idx_datasets}
                            case 'splice'
                                idx_opt = 6;
                                opts.scaling = par_settings(idx_opt) ;
                            case 'svmguide3'
                                idx_opt = 10;
                                opts.scaling = par_settings(idx_opt);
                            case 'mushrooms'
                                idx_opt = 5;
                                opts.scaling = par_settings(idx_opt);
                            case '20news_100word'
                                idx_opt = 9;
                                opts.scaling = par_settings(idx_opt);
                            case 'a9a'
                                idx_opt = 12;
                                opts.scaling = par_settings(idx_opt);
                            case 'w8a'
                                idx_opt = 6;
                                opts.scaling = par_settings(idx_opt);
                        end
                        
                    case 'Ada_SADMMdiag'
                        switch datasets{idx_datasets}
                            case 'splice'
                                idx_opt = 9;
                                opts.scaling = par_settings(idx_opt) ;
                            case 'svmguide3'
                                idx_opt = 8;
                                opts.scaling = par_settings(idx_opt);
                            case 'mushrooms'
                                idx_opt = 8;
                                opts.scaling = par_settings(idx_opt);
                            case '20news_100word'
                                idx_opt = 14;
                                opts.scaling = par_settings(idx_opt);
                            case 'a9a'
                                idx_opt = 12;
                                opts.scaling = par_settings(idx_opt);
                            case 'w8a'
                                idx_opt = 10;
                                opts.scaling = par_settings(idx_opt);
                        end
                    case 'Ada_SADMMfull'
                        switch datasets{idx_datasets}
                            case 'splice'
                                idx_opt = 9;
                                opts.scaling = par_settings(idx_opt) ;
                            case 'svmguide3'
                                idx_opt = 8;
                                opts.scaling = par_settings(idx_opt);
                            case 'mushrooms'
                                idx_opt = 8;
                                opts.scaling = par_settings(idx_opt);
                            case '20news_100word'
                                idx_opt = 14;
                                opts.scaling = par_settings(idx_opt);
                            case 'a9a'
                                idx_opt = 12;
                                opts.scaling = par_settings(idx_opt);
                            case 'w8a'
                                idx_opt = 10;
                                opts.scaling = par_settings(idx_opt);
                        end
                end
                %Trainning
                t = cputime;
                outputs       = funfcn_stoc{idx_method}(s_train, l_train, opts);
                time_solving  = cputime - t;
                time_per_iter = time_solving/outputs.iter;
                fprintf('Method(%d/%d) %s, (%d/%d)runs, time_per_iter:%s\n',idx_method,length(funfcn_stoc),func2str(funfcn_stoc{idx_method}), idx_runs, num_runs, num2str(time_per_iter));
                
                one_run_stat            = [];
                one_run_stat.x          = outputs.x;
                one_run_stat.iter       = outputs.iter;
                one_run_stat.trace      = outputs.trace;
                one_run_stat.time       = time_solving;
                one_run_stat.idx_runs   = idx_runs;
                one_run_stat.idx_method = idx_method;
                one_run_stat.methodname = func2str(funfcn_stoc{idx_method});
                
                stat_data= [stat_data one_run_stat];
                
                num_traces = length(one_run_stat.trace.times);
                for idx_trace = 1:num_traces
                    x = one_run_stat.trace.xs(:,idx_trace);
                    trace_accuracy(idx_trace,idx_runs)  = get_accuracy(s_test, l_test, x);
                    trace_time(idx_trace,idx_runs)      = one_run_stat.trace.times(idx_trace);
                    trace_passes(idx_trace,idx_runs)    = one_run_stat.trace.iters(idx_trace)/num_train;
                    trace_test_loss(idx_trace,idx_runs) = get_test_loss(s_test,l_test,x);
                    trace_obj_val(idx_trace,idx_runs)   = get_obj_val(s_train,l_train,x, opts.F, opts.mu);
                end
                fprintf('idx_opt:%d, opts.mu:%f, time:%.3f, test_loss:%.4f, obj_val:%.4f, accuracy:%.3f\n', idx_mu, opts.mu, trace_time(end,1), trace_test_loss(end,1), trace_obj_val(end,1), trace_accuracy(end,1));
            end
            save(['results_GGLR_' func2str(funfcn_stoc{idx_method}) '_' dataset_name '.mat'],'stat_data','trace_passes','trace_time','trace_accuracy','trace_obj_val','trace_test_loss');
        end
    end
    
    %batch methods
    for idx_method = 1:length(funfcn_batch)
        stat_data    = [];
        trace_accuracy = [];
        trace_test_loss= [];
        trace_obj_val  = [];
        trace_passes   = [];
        trace_time     = [];
        num_train      = length(idx_train);
        num_runs       = nruns_batch;
        for idx_runs = 1:num_runs
            %Trainning
            t = cputime;
            outputs       = funfcn_batch{idx_method}(s_train, l_train, opts);
            time_solving  = cputime - t;
            time_per_iter = time_solving/outputs.iter;
            fprintf('Method(%d/%d) %s, (%d/%d)runs, time_per_iter:%s\n',idx_method,length(funfcn_batch),func2str(funfcn_batch{idx_method}), idx_runs, num_runs, num2str(time_per_iter));
            
            one_run_stat            = [];
            one_run_stat.x          = outputs.x;
            one_run_stat.iter       = outputs.iter;
            one_run_stat.trace      = outputs.trace;
            one_run_stat.time       = time_solving;
            one_run_stat.idx_runs   = idx_runs;
            one_run_stat.idx_method = idx_method;
            one_run_stat.methodname = func2str(funfcn_batch{idx_method});
            
            stat_data= [stat_data one_run_stat];
            
            num_traces = length(stat_data.trace.times);
            for idx_trace = 1:num_traces
                x = stat_data.trace.xs(:,idx_trace);
                trace_accuracy(idx_trace,idx_runs)  = get_accuracy(s_test, l_test, x);
                trace_time(idx_trace,idx_runs)      = stat_data.trace.times(idx_trace);
                trace_passes(idx_trace,idx_runs)    = stat_data.trace.iters(idx_trace);
                trace_test_loss(idx_trace,idx_runs) = get_test_loss(s_test,l_test,x);
                trace_obj_val(idx_trace,idx_runs)   = get_obj_val(s_train,l_train,x, opts.F, opts.mu);
            end
        end
        save(['results_GGLR_' func2str(funfcn_batch{idx_method}) '_' dataset_name '.mat'],'stat_data','trace_passes','trace_time','trace_accuracy','trace_obj_val','trace_test_loss');
    end
end
%end
draw_results
