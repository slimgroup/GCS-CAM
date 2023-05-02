using DrWatson
@quickactivate "GCS-CAM"

# Load package
using JUDI, HDF5, PyPlot, Images, PyCall, Random, JOLI, Statistics, JSON
using AzStorage, AzSessions, SlimOptim, LinearAlgebra, Serialization

using ArgParse

include(srcdir("utils.jl"))

parsed_args = parse_commandline()
nv = parsed_args["nv"]
nsrc = parsed_args["nsrc"]
snr = parsed_args["snr"]
perc = parsed_args["perc"]
leak = parsed_args["leak"]
sim_name = "jrm_"*(leak ? "leakage" : "no_leakage")
file_name = "LeadingEdge_"*(leak ? "Leakage" : "NoLeakage")*".jld2"

# Set paths to credentials + parameters
# Use Standard_F4 or something that has enough memory but cheap

ENV["CREDENTIALS"] = joinpath(pwd(), "../credentials.json")
ENV["PARAMETERS"] = joinpath(pwd(), "../params.json")
ENV["REGISTRYINFO"] = joinpath(pwd(), "../registryinfo.json")

registry_info = JSON.parsefile(ENV["REGISTRYINFO"])

using AzureClusterlessHPC
batch_clear();

# Azure container
batch = pyimport("azure.batch")
container_registry = batch.models.ContainerRegistry(registry_server=registry_info["_REGISTRY_SERVER"],
                                                     user_name=registry_info["_USER_NAME"],
                                                     password=registry_info["_PASSWORD"])

nwk = parse(Int, AzureClusterlessHPC.__params__["_NODE_COUNT_PER_POOL"])
create_pool(container_registry=container_registry)

# Setup AzStorage
session = AzSession(;protocal=AzClientCredentials, resource=registry_info["_RESOURCE"])
container = AzContainer("leading-edge-data-"*(leak ? "leakage" : "no-leakage"); storageaccount=registry_info["_STORAGEACCOUNT"], session=session)
mkpath(container)

@batchdef using JUDI, AzStorage, Distributed, AzureClusterlessHPC, Serialization, LinearAlgebra, Statistics, Images, JOLI, MECurvelets, SlimOptim, Random

using PyPlot, JUDI
using JOLI, JLD2, Images, Random, LinearAlgebra, Statistics

using ArgParse

@batchdef function apply_fwd(n::Integer, L::Integer, z::Vector{T}; γ::T=1f0) where T
    return view(z, n+1:length(z)) .+ T(1)/γ * repeat(view(z, 1:n), L)    # get original components
end 

@batchdef function apply_adj(L::Integer, x::Vector{T}; γ::T=1f0) where T
    return vcat(T(1)/γ * view(sum(reshape(view(x,:), :, L), dims=2), :, 1), x)   # weighted sum on common component
end
# Define the joint reconstruction method function
@batchdef function joJRM(L::Integer, nn::Integer; DDT=Float32, γ::T=1f0, name::String="joJRM") where {T<:Real}
    γ < 0 && throw(joLinearFunctionException("weight on common component must be non-negative"))
    Φ = joLinearFunctionFwd(L*nn, (L+1)*nn,
        v1 -> apply_fwd(nn, L, v1; γ=DDT(γ)),
        v2 -> apply_adj(L, v2; γ=DDT(γ)),
        v3 -> apply_adj(L, v3; γ=DDT(γ)),
        v4 -> apply_fwd(nn, L, v4; γ=DDT(γ)),
        DDT, DDT; name=name)
    return Φ
end

# Create the models directory if it doesn't exist
mkpath(datadir("models"))
file_path = datadir("models", file_name)

# Download the dataset into the data directory if it does not exist
if ~isfile(file_path)
    if leak
        run(`wget https://www.dropbox.com/s/prpbgy6hkbdvoa0/'
            'LeadingEdge_Leakage.jld2 -q -O $file_path`)
    else
        run(`wget https://www.dropbox.com/s/c84qebwxd23yrj8/'
            'LeadingEdge_NoLeakage.jld2 -q -O $file_path`)
    end
end

# Load the data
JLD2.@load datadir("models", file_name) v1 rho1 v2 rho2

nsample = length(v1)
Random.seed!(2023)

# Function to generate random source indices for LSRTM
@batchdef function GenSrcIdxLSRTM(nsrc::Int, batchsize::Int, iter::Int)
    src_list = collect(1:nsrc)
    inds = [zeros(Int, batchsize) for i = 1:iter]

    # Random batch of sources
    for i=1:iter
        length(src_list) < batchsize && (src_list = collect(1:nsrc))
        src_list = src_list[randperm(length(src_list))]
        inds[i] = [pop!(src_list) for b=1:batchsize]
    end

    return inds
end

# Function to calculate the λ threshold value
@batchdef function λfunc(z; quantile=0.995)
    z1 = reshape(z, :, 3)
    threshold = [Statistics.quantile(abs.(z1[:,i]), quantile) for i = 1:size(z1,2)]
    output = maximum(threshold)
    return output
end
# Function to perform joint reconstruction method for a given index
@batchdef function jrm_i(idx_i, v1, rho1, container)
    # Deserialize noisy linear data
    q, dlin = deserialize(container,  "noisy_lindata_$(idx_i)")

    # Set parameters
    batchsize = 4
    niter = 22
    nv = 2
    γ = 1f0
    nsrc = q[1].nsrc

    # Define model geometry
    n = size(v1)
    d = (6.25f0, 6.25f0)
    o = (0f0, 0f0)

    # Apply Gaussian filtering to the velocity and density models
    idx_wb = 37
    v0 = deepcopy(v1)
    v0[:,idx_wb+1:end] = 1f0./imfilter(1f0./v1[:,idx_wb+1:end], Kernel.gaussian(10))
    m0 = (1f0./v0).^2f0
    rho0 = deepcopy(rho1)
    rho0[:,idx_wb+1:end] = 1f0./imfilter(1f0./rho1[:,idx_wb+1:end], Kernel.gaussian(10))

    # Initialize the model
    model0 = Model(n, d, o, m0; rho=rho0, nb = 80)
    opt = JUDI.Options(isic=true)

    # Define operators for the joint reconstruction method
    C = joEye(size(joMECurvelet2D((n), zero_finest=true),1); DDT=ComplexF32)*joMECurvelet2D((n), zero_finest=true)
    Tm = judiTopmute(n, idx_wb, 1)  # Mute water column
    S = judiDepthScaling(model0)  # depth scaling

    # Define the linear operator for joint reconstruction method
    Mr1 = joCoreBlock([S*Tm for i = 1:nv]...)
    C1 = joCoreBlock([C for i = 1:nv+1]...)
    φ = joJRM(nv, size(C,2); γ=γ, DDT=Float32) 
    P = Mr1 * φ

    global iter = 1

    # Generate source indices
    inds = [GenSrcIdxLSRTM(nsrc, batchsize, niter+3), GenSrcIdxLSRTM(nsrc, batchsize, niter+3)]
    inds[1] = [vcat([inds[1][1],inds[1][2],inds[1][3]]...),[inds[1][i] for i = 4:niter+2]...]
    inds[2] = inds[1]

    # Define the objective function
    function obj(x)
        println("computing objective function at iter ", iter)
        img = [(P * x)[(i-1)*prod(n)+1:i*prod(n)] for i = 1:nv]
        q_ = [q[i][inds[i][iter]] for i = 1:nv]
        dlin_ = [dlin[i][inds[i][iter]] for i = 1:nv]
        # Calculate the gradient
        phi, g = lsrtm_objective(model0, q_, dlin_, img; nlind=false, options=opt)
        g = P' * vcat(g...)
        global iter = iter + 1
        return phi, g
    end
    
    # Set up Bregman optimization options
    bregopt = bregman_options(maxIter=niter, verbose=2, alpha=0.6f0, antichatter=false, spg=false, λfunc=λfunc, TD=C1);
    
    # Perform Bregman optimization
    @time sol = bregman(obj, zeros(Float32, size(P,2)), bregopt);

    # Serialize the solution and save it to the container
    serialize(container, "jrm_$(idx_i)", (idx_i=idx_i, bregopt=bregopt, sol=sol))
    return nothing

end

# Print the simulation name
println(sim_name)
Base.flush(stdout)

# Parallel map the joint reconstruction method function to all samples
futures = @batchexec pmap(i -> jrm_i(i, v1[i], rho1[i], container), 1:nsample)

# Fetch results from the futures with specified timeouts and restarts
fetch(futures; num_restart=0, timeout=24000, task_timeout=1000)

# Clean up resources
delete_all_jobs()
delete_pool()
delete_container()
