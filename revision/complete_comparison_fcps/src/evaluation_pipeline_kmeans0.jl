# data = "Lsun.csv"; cr = :silhouette; method = "ICOT_local"; geom_search = true; threshold = .99; seed = 2; gridsearch = false; num_tree_restarts = 100; complexity = 0.0; min_bucket = 1; maxdepth = 3; datafolderpath = "../data/"; warm_start = :none; resultsfolderpath = "../results/"

# run_single(data = "WingNut.csv", cr = :silhouette, method = "ICOT_local", warm_start = true, geom_search = true,
#  threshold = .99, seed = 1, gridsearch = false, num_tree_restarts = 100, complexity = 0.0, min_bucket = 10, maxdepth = 5, 
#  datafolderpath = "../data/", resultsfolderpath = "..results/")

function run_single(;data=data,
                 cr=criterion,
                 method=clust_method,
                 warm_start = warm_start,
                 geom_search = geom_search,
                 threshold = threshold,
                   seed=seed,               
                 gridsearch=false,
                 num_tree_restarts=100,
                 complexity=0.0,
                 min_bucket=1,
                 maxdepth=3,                
                 datafolderpath=datafolderpath,
                 resultsfolderpath=resultsfolderpath)

  ###### STEP 1: READ THE DATA
  dataset_name = deepcopy(data);
  name_short = split(dataset_name, ".")[1];

  data_path = joinpath(datafolderpath, data)
  data = CSV.read(data_path); 
  X = data[1:(end-1)]; 
  y = data[:label]; 
  truelabels = true; 

  ##### STEP 2: Compare to ground truth
  true_assignments = Array{Int64}(y);
  true_k = length(unique(y));
  true_sil, true_dunn = test_performance_general(X, true_assignments);

  results = DataFrame(seed = seed, data = name_short, criterion = cr,  
    geom_threshold = threshold, warm_start = warm_start,
    method = "True", K = Float64(true_k), silhouette = true_sil, dunn = true_dunn, 
    ari_true = 1.0, runtime = 0.0);


  ####### STEP 3: Set up kmeans warm start 
  println("\nRunning K-Means warm-start")
  @rput seed
  @rput X
  R"""
    library("cluster")
    set.seed($seed)
    gap <- clusGap($X, FUN = kmeans, nstart = 25, iter.max = 50,
                      K.max = 10, B = 50)
      tab <- gap['Tab'][[1]]
      k <- maxSE(tab[, 'gap'], tab[, 'SE.sim'], method='Tibs2001SEmax')
  """

  k_r = @rget k;
  K = max(2, k_r);

  println("Chosen K = ", K);
  X_array = Matrix{Float64}(X);
  X_t = X_array';

  Random.seed!(seed);
  kmeans_result = kmeans(X_t, K);

  y_hat = kmeans_result.assignments;

  ####### STEP 4: RUN ICOT
  println("\nRunning ICOT")

  lnr_icot_warmup = ICOT.InterpretableCluster(ls_num_tree_restarts = num_tree_restarts, ls_random_seed = seed, cp = complexity, max_depth = maxdepth,
  minbucket = min_bucket, criterion = cr, ls_warmstart_criterion = cr, kmeans_warmstart = warm_start,
  geom_search = geom_search, geom_threshold = threshold);
  run_time_icot_warmup = @elapsed ICOT.fit!(lnr_icot_warmup, X, y_hat);
  println("ICOT first run: ", run_time_icot_warmup)

  if method =="ICOT_local" 
    lnr_icot = ICOT.InterpretableCluster(ls_num_tree_restarts = num_tree_restarts, ls_random_seed = seed, 
      cp = complexity, max_depth = maxdepth, minbucket = min_bucket, criterion = cr, 
      ls_warmstart_criterion = cr, kmeans_warmstart = warm_start,
    geom_search = geom_search, geom_threshold = threshold);
    run_time_icot = @elapsed ICOT.fit!(lnr_icot, X, y_hat);
    println("ICOT second run: ", run_time_icot);
  elseif method =="ICOT_greedy"
    lnr_icot = ICOT.OptimalTreeClassifier(localsearch = false, cp = complexity, max_depth = maxdepth,
    minbucket = min_bucket, criterion = cr, kmeans_warmstart = warm_start,
    geom_search = geom_search, geom_threshold = threshold);
    run_time_icot = @elapsed ICOT.fit!(lnr_icot, X, y_hat);
  end
  
  #Get the statistics from the local search
  icot_assignments = ICOT.apply(lnr_icot, X);
  leaf_cnt = length(unique(icot_assignments))
  icot_sil, icot_dunn = test_performance_general(X, icot_assignments);
  
  ari_true_icot = ifelse(true_k == leaf_cnt, randindex(true_assignments, icot_assignments)[1], -10)

  append!(results, DataFrame(seed = seed, data = name_short, criterion = cr,
    geom_threshold = threshold, warm_start = warm_start,
    method = method, K = leaf_cnt, silhouette = icot_sil, dunn = icot_dunn, 
    ari_true = ari_true_icot, runtime = run_time_icot))


  ####### STEP 5: RUN OCT WITH ASSIGNED LABELS
  println("\nRunning OCT with assigned labels")
  oct_k = K; 
  method_oct = string("OCT_", split(method, "_")[2]);


  ### warmup
  grid_OCT_warmup = IAI.GridSearch(IAI.OptimalTreeClassifier(localsearch = true, 
      ls_num_tree_restarts = num_tree_restarts, random_seed = seed, 
      max_depth = maxdepth, minbucket = min_bucket, 
      criterion = :misclassification, ls_warmstart_criterion = :misclassification))
  run_time_oct_warmup = @elapsed IAI.fit!(grid_OCT_warmup, X, y_hat)
  println("OCT first run: ", run_time_oct_warmup)

  if method =="ICOT_local" 
    grid_OCT = IAI.GridSearch(IAI.OptimalTreeClassifier(localsearch = true, 
      ls_num_tree_restarts = num_tree_restarts, random_seed = seed, 
      max_depth = maxdepth, minbucket = min_bucket, 
      criterion = :misclassification, ls_warmstart_criterion = :misclassification))
    run_time_oct = @elapsed IAI.fit!(grid_OCT, X, y_hat)
    println("OCT second run: ", run_time_oct)
    lnr_oct = grid_OCT.lnr
  elseif method =="ICOT_greedy"
    grid_OCT = IAI.GridSearch(IAI.OptimalTreeClassifier(localsearch = false, 
      max_depth = maxdepth, minbucket = min_bucket, 
      criterion = :misclassification))
    run_time_oct = @elapsed IAI.fit!(grid_OCT, X, y_hat)
    lnr_oct = grid_OCT.lnr
  end

  oct_assignments = OptimalTrees.predict(lnr_oct, X);
  oct_sil, oct_dunn = test_performance_general(X, oct_assignments);

  ari_true_oct = ifelse(true_k == oct_k, randindex(true_assignments, oct_assignments)[1], -10)

  append!(results, DataFrame(seed = seed, data = name_short, criterion = cr, 
    geom_threshold = threshold, warm_start = warm_start,
    method = "OCT", K = oct_k, silhouette = oct_sil, dunn = oct_dunn, 
    ari_true = ari_true_oct, runtime = run_time_oct));

  ####### STEP 6: Comparison to alternative clustering methods
  df_assignments = DataFrame(truth = true_assignments, 
    icot = icot_assignments, 
    oct = oct_assignments)

  println("\nRunning Comparison Methods (cross-validate K)")
  # true_k = length(unique(y)); 
  # min_k = max(min(leaf_cnt,true_k)-2,2); max_k = max(true_k, leaf_cnt)+2;
  min_k = 2; max_k = 10;
  eps_range = .1:.1:1.0;

  # Return a dictionary of scores and assignments for each k, as well as the best k value (max score)
  for m in ["kmeans_plus","hclust","gmm","dbscan"]
    println("Method = $m")
    k_range = ifelse(m == "dbscan", eps_range, min_k:max_k)
    run_time = @elapsed best_k, assignments = eval_method(X, k_range, seed, cr, m);
    sil, dunn = test_performance_general(X, assignments);

    ari_true = ifelse(true_k == best_k, randindex(true_assignments, assignments)[1], -10)

    ### Add results to master DF
    to_add = DataFrame(seed = seed, data = name_short, criterion = cr, 
      geom_threshold = threshold, warm_start = warm_start,
      method = m, K = best_k, silhouette = sil, dunn = dunn, 
      ari_true = ari_true, runtime = run_time);
    append!(results, to_add)

    ### Add assignments to master DF
    df_assignments[:temp] = assignments
    rename!(df_assignments, :temp => Symbol(m))
  end

  #Results: Save results in an array to paste into Excel file 
  filepath = "$name_short-$cr-seed$(seed)-geom$(threshold)-ws_$(warm_start)"
  # filepath_lnr = joinpath(resultsfolderpath, "lnr-$dataset_name-$cr-$method-lnr.jld")
 #    @save filepath_lnr lnr

  ICOT.writedot("$(resultsfolderpath)icot-$filepath.dot", lnr_icot);
  run(`dot -Tpng -o  $(resultsfolderpath)icot-$filepath.png $(resultsfolderpath)icot-$filepath.dot`);


  IAI.write_dot("$(resultsfolderpath)oct-$filepath.dot", lnr_oct);
  run(`dot -Tpng -o  $(resultsfolderpath)oct-$filepath.png $(resultsfolderpath)oct-$filepath.dot`);

  CSV.write("$(resultsfolderpath)$filepath.csv", results)
  CSV.write("$(resultsfolderpath)$(filepath)_assignments.csv", df_assignments)

end
