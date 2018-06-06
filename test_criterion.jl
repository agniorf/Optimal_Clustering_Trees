using RDatasets, MLDataUtils
using Clustering
using Gadfly

# import OptimalTrees

# iris = dataset("datasets", "iris");
# X = iris[1:2]; y = iris[5];
# srand(1);
# (big_X, big_y), (test_X, test_y) = stratifiedobs((X, y), p=0.75);
# (train_X, train_y), (valid_X, valid_y) = stratifiedobs((big_X, big_y), p=0.67);

# lnr = OptimalTrees.OptimalTreeClassifier(max_depth=2, cp=0.01, criterion = :gini);
# OptimalTrees.fit!(lnr, train_X, train_y)

# reload("OptimalTrees")
# lnr2 = OptimalTrees.OptimalTreeClassifier(max_depth=2, cp=0.01, criterion = :cluster, localsearch = false);
# OptimalTrees.fit!(lnr2, train_X, train_y)

######### RUSPINI

# K = 4;
# # initseeds(:rand, ruspini_t, K); 

# ruspini_data = dataset("cluster", "ruspini");
# ruspini_data = convert(Array{Float64}, ruspini_data); ruspini_t = ruspini_data';

# srand(1234)
# rusp_kmeans = kmeans(ruspini_t, K);
# #Apply k-means with 4 classes and get the assignments

# assignments = rusp_kmeans.assignments;
# data = DataFrame(hcat(ruspini_data, assignments));

X = data[1:2]; y = data[3];

reload("OptimalTrees")
lnr2 = OptimalTrees.OptimalTreeClassifier(max_depth=3, cp=0.01, criterion = :cluster, localsearch = true);
OptimalTrees.fit!(lnr2, X, y)
OptimalTrees.showinbrowser(lnr2)

# writetable("ruspini.csv", dataset)

# plot(data, x = :x1, y = :x2, color = :x3)