
#Run the whole iteration for a single 
function run_single(;data::String=data,
             		 cr::Symbol=criterion,
             		 method::String=method,
            	     seed::Int64=2,
             		 gridsearch::Bool=false,
             		 num_tree_restarts::Int64=100,
             		 complexity::Float64= 0.0,
             		 min_bucket::Int64=1,
             		 maxdepth::Int64=3,
             		 datafolderpath::String=datafolderpath,
             		 resultsfolderpath::String=resultsfolderpath)
	#READ THE DATA
	dataset_name = deepcopy(data);
	data_path = joinpath(datafolderpath, data)
	data = readtable(data_path); 
	X = data[1:(end-1)]; 
	y = ones(size(data,1)); 
	truelabels = false; 
	srand(seed)

	#RUN OPTIMAL CLUSTERING TREES BOTH WITH GREEDY AND LOCAL SEARCH
	if method =="localsearch"
		lnr = OptimalTrees.OptimalTreeClassifier(ls_num_tree_restarts = num_tree_restarts, ls_random_seed = seed, cp = complexity, max_depth = maxdepth,
		minbucket = min_bucket, criterion = cr, ls_warmstart_criterion = cr);
		OptimalTrees.fit!(lnr, X, y);
	elseif method =="greedy"
		lnr = OptimalTrees.OptimalTreeClassifier(localsearch = false, cp = complexity, max_depth = maxdepth, ls_warmstart_criterion = cr
		minbucket = min_bucket, criterion = cr);
		OptimalTrees.fit!(lnr, X, y);
	end
	
	#Get the statistics from the local search
	leaf_cnt = leafcount(lnr) ## Defined in tools file
	# optclust_depth = grid.best_params[:max_depth];
	optclust_assignments = OptimalTrees.apply(lnr, X);
	# optclust_score = OptimalTrees.score(lnr, X, y, criterion = cr);
	optclust_score = cluster_score(X, optclust_assignments, cr);

	####### STEP 3: KMEANS. Run k means with the chosen depth, and in the neighborhood of the tree
	# Determine range for k based on leaves of OptClust result
	if truelabels	
			true_k = length(unique(y));	
			min_k = max(min(leaf_cnt,true_k)-2,2); max_k = max(true_k, leaf_cnt)+2;
	else 
			min_k = max(leaf_cnt - 2, 2); max_k = leaf_cnt + 5;
	end
	# Return a dictionary of scores and assignments for each k, as well as the best k value (max score)
	kmeans_scoredict, kmeans_assignmentdict, kmeans_best = eval_kmeans(X, min_k:max_k, seed, cr); ## Defined in tools file
	ari_optclust_kmeans = randindex(optclust_assignments, kmeans_assignmentdict[leaf_cnt])[1];
	
	##### STEP 5: If ground truth available, compare
	if truelabels
		true_assignments = Array{Int64}(y)
		true_k = length(unique(y));
		true_score = cluster_score(X, true_assignments, cr)
		ari_true_kmeans = randindex(true_assignments, kmeans_assignmentdict[true_k])[1]
		if true_k == leaf_cnt 
			ari_true_optclust = randindex(true_assignments, optclust_assignments)[1]
		else ari_true_optclust = -10
		end
	else true_assignments = -10; true_k = -10; true_score = -10; ari_true_kmeans = -10; ari_true_optclust = -10;
	end
	#Results
	### Save results in an array to paste into Excel file 
	results = DataFrame(seed = seed, K_optclust=leaf_cnt, kmeans_k=kmeans_best, true_k=true_k ,
	score_optclust = optclust_score, score_kmeans_bestk=kmeans_scoredict[kmeans_best], score_kmeans_kc=kmeans_scoredict[leaf_cnt], 
	score_true=true_score,
	ari_optclust_kmeans=ari_optclust_kmeans, ari_true_kmeans=ari_true_kmeans, are_true_optclust=ari_true_optclust)

	filepath_lnr = joinpath(resultsfolderpath, "lnr-$dataset_name-$cr-$method-lnr.jld")
	# println(filepath_lnr)
    @save filepath_lnr lnr
    filepath_accuracy = joinpath(resultsfolderpath, "results-$dataset_name-$cr-$method.csv")
  	writetable(filepath_accuracy, results)
end
