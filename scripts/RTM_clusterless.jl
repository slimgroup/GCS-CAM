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
leak = parsed_args["leak"]
sim_name = "rtm_"*(leak ? "leakage" : "no_leakage")
file_name = "LeadingEdge_"*(leak ? "Leakage" : "NoLeakage")*".jld2"

# Set paths to credentials + parameters
# Use Standard_F4 or something that has enough memeory but cheap

ENV["CREDENTIALS"] = joinpath(pwd(), "../credentials.json")
ENV["PARAMETERS"] = joinpath(pwd(), "../params.json")
ENV["REGISTRYINFO"] = joinpath(pwd(), "../registryinfo.json")

registry_info = JSON.parsefile(ENV["REGISTRYINFO"])

using AzureClusterlessHPC
batch_clear();

# Azure continer
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

@batchdef using JUDI, AzStorage, Distributed, AzureClusterlessHPC, Serialization, LinearAlgebra, Statistics, Images, JOLI

using PyPlot, JUDI
using JOLI, JLD2, Images, Random, LinearAlgebra, Statistics
using SlimPlotting

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

# Load the model data
JLD2.@load datadir("models", file_name) v1 rho1 v2 rho2

# Set the number of samples
nsample = length(v1)
Random.seed!(2023)

# Function to perform RTM on a given sample
@batchdef function rtm_i(idx_i, v1, rho1, container)
    # Load precomputed linearized data
    q, dlin = deserialize(container,  "noisy_lindata_$(idx_i)")

    # Set model parameters
    n = size(v1)
    d = (6.25f0, 6.25f0)
    o = (0f0, 0f0)
    nv = 2

    # Apply Gaussian filtering to the velocity and density models
    idx_wb = 37
    v0 = deepcopy(v1)
    v0[:,idx_wb+1:end] = 1f0./imfilter(1f0./v1[:,idx_wb+1:end], Kernel.gaussian(10))
    m0 = (1f0./v0).^2f0
    rho0 = deepcopy(rho1)
    rho0[:,idx_wb+1:end] = 1f0./imfilter(1f0./rho1[:,idx_wb+1:end], Kernel.gaussian(10))

    # Create JUDI model
    model0 = Model(n, d, o, m0; rho=rho0, nb = 80)
    opt = JUDI.Options(isic=true)

    # Set up geometries
    srcGeometry = [q[i].geometry for i=1:nv]
    recGeometry = dlin[1].geometry

    # Create operators
    Pr = judiProjection(recGeometry)   # receiver restriction
    Ps = [judiProjection(srcGeometry[i]) for i=1:nv]   # source injection
    F0 = [Pr*judiModeling(model0; options=opt)*Ps[i]' for i=1:nv]
    J = [judiJacobian(F0[i], q[i]) for i=1:nv]

    # Perform RTM
    rtm = [J[i]' * dlin[i] for i = 1:nv]

    # Save the results
    serialize(container, "rtm_$(idx_i)", (idx_i=idx_i, rtm=rtm))
    return nothing
end

# Print simulation name
println(sim_name)
Base.flush(stdout)

# Run the RTM function for all samples in parallel using Azure
futures = @batchexec pmap(i -> rtm_i(i, v1[i], rho1[i], container), 1:nsample)
fetch(futures; num_restart=0, timeout=24000, task_timeout=1000)

# Clean up Azure resources
delete_all_jobs()
delete_pool()
delete_container()
