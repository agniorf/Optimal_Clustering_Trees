### same as evaluation_pipeline_kmeans0.jl, but without any ICOT/OCT runs

# data = "Lsun.csv"; cr = :silhouette; method = "ICOT_local"; geom_search = true; threshold = .99; seed = 2; gridsearch = false; num_tree_restarts = 100; complexity = 0.0; min_bucket = 1; maxdepth = 3; datafolderpath = "../data/"; warm_start = :none; resultsfolderpath = "../results/"

# run_single(data = "WingNut.csv", cr = :silhouette, method = "ICOT_local", warm_start = true, geom_search = true,
#  threshold = .99, seed = 1, gridsearch = false, num_tree_restarts = 100, complexity = 0.0, min_bucket = 10, maxdepth = 5, 
#  datafolderpath = "../data/", resultsfolderpath = "..results/")

function run_single_R(;data=data,
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
                 normalize_R = false,             
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


  ####### STEP 6: Comparison to alternative clustering methods
  df_assignments = DataFrame(truth = true_assignments)

  println("\nRunning Comparison Methods (cross-validate K)")
  # true_k = length(unique(y)); 
  # min_k = max(min(leaf_cnt,true_k)-2,2); max_k = max(true_k, leaf_cnt)+2;
  min_k = 2; max_k = 10;
  eps_range = .1:.05:1.0;

  # Return a dictionary of scores and assignments for each k, as well as the best k value (max score)
  for m in ["kmeans_plus","hclust","gmm","dbscan"]
    println("Method = $m")
    k_range = ifelse(m == "dbscan", eps_range, min_k:max_k)
    run_time = @elapsed best_k, assignments = eval_method(X, k_range, seed, cr, m, normalize = normalize_R);
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

  println(results)
  #Results: Save results in an array to paste into Excel file 
  filepath = "$name_short-$cr-seed$(seed)-geom$(threshold)-ws_$(warm_start)"
  # filepath_lnr = joinpath(resultsfolderpath, "lnr-$dataset_name-$cr-$method-lnr.jld")
 #    @save filepath_lnr lnr

  CSV.write("$(resultsfolderpath)$filepath.csv", results)
  CSV.write("$(resultsfolderpath)$(filepath)_assignments.csv", df_assignments)

end
