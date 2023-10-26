using DrWatson
@quickactivate "GCS-CAM"

using JUDI, HDF5, PyPlot, Images, PyCall, Random, JOLI, Statistics, JSON
using AzStorage, AzSessions, SlimOptim, LinearAlgebra, Serialization
using ArgParse

include(srcdir("utils.jl"))

# Parse command-line arguments for various simulation parameters
parsed_args = parse_commandline()
nv = parsed_args["nv"]
nsrc = parsed_args["nsrc"]
snr = parsed_args["snr"]
leak = parsed_args["leak"]
sim_name = "born_"*(leak ? "leakage" : "no_leakage")
file_name = "LeadingEdge_"*(leak ? "Leakage" : "NoLeakage")*".jld2"

# Set paths to credentials and parameters
ENV["CREDENTIALS"] = joinpath(pwd(), "../credentials.json")
ENV["PARAMETERS"] = joinpath(pwd(), "../params.json")
ENV["REGISTRYINFO"] = joinpath(pwd(), "../registryinfo.json")

# Load the registry information
registry_info = JSON.parsefile(ENV["REGISTRYINFO"])

# Import AzureClusterlessHPC library and clear any existing batch jobs
using AzureClusterlessHPC
batch_clear();

# Set up Azure container
batch = pyimport("azure.batch")
container_registry = batch.models.ContainerRegistry(registry_server=registry_info["_REGISTRY_SERVER"],
                                                     user_name=registry_info["_USER_NAME"],
                                                     password=registry_info["_PASSWORD"])

# Create a pool for parallel tasks
nwk = parse(Int, AzureClusterlessHPC.__params__["_NODE_COUNT_PER_POOL"])
create_pool(container_registry=container_registry)

# Set up AzStorage session and create a container for the output data
session = AzSession(;protocal=AzClientCredentials, resource=registry_info["_RESOURCE"])
container = AzContainer("leading-edge-data-"*(leak ? "leakage" : "no-leakage"); storageaccount=registry_info["_STORAGEACCOUNT"], session=session)
mkpath(container)

# Define necessary libraries and packages to be used by batch tasks
@batchdef using JUDI, AzStorage, Distributed, AzureClusterlessHPC, Serialization, LinearAlgebra, Statistics, Images, JOLI, FFTW

using PyPlot, JUDI
using JOLI, JLD2, Images, Random, LinearAlgebra, Statistics
using SlimPlotting

# Set the path for the models
mkpath(datadir("models"))
file_path = datadir("models", file_name)

# Download the dataset into the data directory if it does not exist
if ~isfile(file_path)
    if leak
        run(`wget https://www.dropbox.com/s/nzsrqr1qwd3jsje/'
            'LeadingEdge_Leakage.jld2 -q -O $file_path`)
    else
        run(`wget https://www.dropbox.com/s/1mk4i6j3ljf3xf0/'
            'LeadingEdge_NoLeakage.jld2 -q -O $file_path`)
    end
end

# Load the velocity and density models from the dataset
JLD2.@load datadir("models", file_name) v1 rho1 v2 rho2

# Set up random seed and model dimensions
nsample = length(v1)
Random.seed!(2023)
n = size(v1[1])
d = (6.25f0, 6.25f0)
o = (0f0, 0f0)

# Calculate the number of receivers
nrec = Int(floor(n[1]/4))

# Function to create jittered, equally spaced numbers in the range [0, l]
@batchdef function ContJitter(l::Number, num::Int)
    # l = length, num = number of samples
    interval_width = l/num
    interval_center = range(interval_width/2, stop = l-interval_width/2, length=num)
    randomshift = interval_width .* rand(Float32, num) .- interval_width/2

    return interval_center .+ randomshift
end

# Function to perform born modeling with given parameters
@batchdef function born_modeling_i(idx_i, v1, v2, rho1, rho2, nsrc, snr, container)
    # Define model parameters
    n = size(v1)
    d = (6.25f0, 6.25f0)
    o = (0f0, 0f0)
    nv = 2

    # Calculate the number of receivers
    nrec = Int(floor(n[1]/4))

    # Define the initial model and apply Gaussian smoothing
    idx_wb = 10
    v0 = deepcopy(v1)
    v0[:,idx_wb+1:end] = 1f0./imfilter(1f0./v1[:,idx_wb+1:end], Kernel.gaussian(10))
    m0 = (1f0./v0).^2f0
    rho0 = deepcopy(rho1)
    rho0[:,idx_wb+1:end] = 1f0./imfilter(1f0./rho1[:,idx_wb+1:end], Kernel.gaussian(10))

    # Compute model perturbations
    dimp = [vec(v0.*rho0-v1.*rho1), vec(v0.*rho0-v2.*rho2)]

    # Set up the JUDI model
    model0 = Model(n, d, o, m0; rho=rho0, nb = 80)
    opt = JUDI.Options(isic=true)

    # Calculate the extent of the model
    extentx = (n[1]-1)*d[1];

    # Define source and receiver locations
    xsrc = [ContJitter(extentx, nsrc) for i=1:nv]
    ysrc = range(0f0,stop=0f0,length=nsrc)
    zsrc = range(10f0,stop=10f0,length=nsrc)
    xrec = range(d[1],stop=(n[1]-1)*d[1], length=nrec)
    yrec = 0f0
    zrec = range((idx_wb-1)*d[2]-2f0,stop=(idx_wb-1)*d[2]-2f0,length=nrec)

    # Define recording time and time step
    timeR = 1800f0
    dtR = 4f0
    ntR = Int(floor(timeR/dtR))+1

    # Set up source and receiver geometries
    srcGeometry = [Geometry(convertToCell(xsrc[i]), convertToCell(ysrc), convertToCell(zsrc); dt=dtR, t=timeR) for i=1:nv]
    recGeometry = Geometry(xrec, yrec, zrec; dt=dtR, t=timeR, nsrc=nsrc)

    # Define source wavelet
    f0 = 0.025f0
    wavelet = ricker_wavelet(timeR, dtR, f0)
    # Create JUDI vectors for each source geometry and the wavelet
    q = [judiVector(srcGeometry[i], wavelet) for i=1:nv]

    # Set up JUDI operators
    Pr = judiProjection(recGeometry)   # receiver restriction
    Ps = [judiProjection(srcGeometry[i]) for i=1:nv]   # source injection
    F0 = [Pr*judiModeling(model0; options=opt)*Ps[i]' for i=1:nv] # Forward modeling operators for each source
    J = [judiJacobian(F0[i], q[i]) for i=1:nv] # Jacobian operators for each source

    # Perform Born modeling
    dlin = [J[i] * dimp[i] for i = 1:nv]

    # Serialize linear data to storage
    serialize(container, "lindata_$(idx_i)", (q=q, dlin=dlin))

    # Add noise to the linear data
    noise = deepcopy(dlin)
    for k = 1:nv
        for l = 1:nsrc
            noise[k].data[l] = randn(Float32, size(dlin[k].data[l]))
            noise[k].data[l] = real.(ifft(fft(noise[k].data[l]).*fft(q[k].data[1])))
        end
    end

    # Scale the noise by the desired signal-to-noise ratio (SNR)
    noise = noise/norm(noise) * norm(dlin) * 10f0^(-snr/20f0)
    dlin = dlin + noise

    # Serialize noisy linear data to storage
    serialize(container, "noisy_lindata_$(idx_i)", (q=q, dlin=dlin))

    return nothing
end

# Print simulation name and flush the output buffer
println(sim_name)
Base.flush(stdout)

# Run the born_modeling_i function for all samples in parallel using pmap
futures = @batchexec pmap(i -> born_modeling_i(i, v1[i], v2[i], rho1[i], rho2[i], nsrc, snr, container), 1:nsample)
fetch(futures; num_restart=0, timeout=24000, task_timeout=1000)

# Clean up the resources
delete_all_jobs()
delete_pool()
delete_container()
