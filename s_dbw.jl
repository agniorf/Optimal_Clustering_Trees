using RDatasets
using Gadfly
using Distances
using Clustering


########### CREATE SAMPLE DATA FOR TESTING ############

K = 4;

ruspini = dataset("cluster", "ruspini");
ruspini = convert(Array{Float64}, ruspini); ruspini_t = ruspini';

srand(2)
initseeds(:rand, ruspini_t, K); rusp_kmeans = kmeans(ruspini_t, K);

data = ruspini;
assignments = rusp_kmeans.assignments;

plot(DataFrame(hcat(data, assignments)), x = :x1, y = :x2, color = :x3);

############ DEFINE HELPER FUNCTIONS ############

# Calculate centroids for each cluster. Return clusters in array, with cluster in last column
function find_centroids(data, assignments, K)
	dim = size(data,2)
	# result = Array{Float64}(0,dim+1)
	result = Dict{Int64, Array}()
	for k=1:K
		clust = data[assignments.==k,:]
		centroid = mapslices(mean, clust, 1)[1,:];
		result[k] = centroid;
	end
	return result
end

function find_clustvars(data, assignments, K)
	dim = size(data,2)
	result = Dict{Int64, Array{Float64,1}}()
	for k=1:K
		clust = data[assignments.==k,:]
		v = mapslices(var, clust, 1)[1,:];
		result[k] = v;
	end
	return result
end

function clust_stdev(data, assignments, K)
	dim = size(data,2)
	clustvars = find_clustvars(data, assignments, K);
	total = sum(norm(clustvars[k]) for k=1:K)
	return (1/K)*sqrt(total)
end

# Find the density of cluster i
function density(obs, center, stdev)
	dist = Distances.colwise(Euclidean(), center, obs')
	f = ifelse.(dist .> stdev, 0, 1)
	return sum(f)
end

########## MAIN FUNCTION - METRIC DEFINITION ############

function s_dbw(data, assignments, K)

	# First find dens_bw

	centroids = find_centroids(data, assignments, K);
	clustvars = find_clustvars(data, assignments, K);
	stdev = clust_stdev(data, assignments, K);
	Svar = mapslices(var, data, 1);

	pairwise_densities = Dict{Tuple{Int64,Int64}, Float64}()
	for i = 1:K, j = i:K
		centroids_ij = vcat(centroids[i]', centroids[j]')
		u_ij =  mapslices(mean, centroids_ij, 1)[1,:];
		obs_ij = data[(assignments .== i) .| (assignments .== j),:]
		pairwise_densities[(i,j)] = density(obs_ij, u_ij, stdev)

	end

	dens_sum = 0
	for i = 1:K, j = 1:K
		# Only look at i < j (and x2) due to definition of pairwise_densities keys
		if i < j 
			n = pairwise_densities[(i,j)]
			d = max(pairwise_densities[(i,i)], pairwise_densities[(j,j)])
			dens_sum += 2*n/d
		end
	end

	 dens_bw = 1/(K*(K-1))*dens_sum

	 # Now find scat(c)
	 sumvarnorms = sum(norm(clustvars[k]) for k=1:K);
	 scatter = (1/K)*sumvarnorms/norm(Svar)

	 println("dens_bw: ", dens_bw)
	 println("scat: ", scatter)
	 println("S_dbw: ", scatter + dens_bw)

	 return scatter + dens_bw
end


