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
sim_name = "born_"*(leak ? "leakage" : "no_leakage")
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

@batchdef using JUDI, AzStorage, Distributed, AzureClusterlessHPC, Serialization, LinearAlgebra, Statistics, Images, JOLI, FFTW

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

JLD2.@load datadir("models", file_name) v1 rho1 v2 rho2

nsample = length(v1)
Random.seed!(2023)

n = size(v1[1])
d = (6.25f0, 6.25f0)
o = (0f0, 0f0)

nrec = Int(floor(n[1]/4))

@batchdef function ContJitter(l::Number, num::Int)
    #l = length, num = number of samples
    interval_width = l/num
    interval_center = range(interval_width/2, stop = l-interval_width/2, length=num)
    randomshift = interval_width .* rand(Float32, num) .- interval_width/2

    return interval_center .+ randomshift
end

@batchdef function born_modeling_i(idx_i, v1, v2, rho1, rho2, nsrc, snr, container)

    n = size(v1)
    d = (6.25f0, 6.25f0)
    o = (0f0, 0f0)
    nv = 2

    nrec = Int(floor(n[1]/4))

    idx_wb = 10
    v0 = deepcopy(v1)
    v0[:,idx_wb+1:end] = 1f0./imfilter(1f0./v1[:,idx_wb+1:end], Kernel.gaussian(10))
    m0 = (1f0./v0).^2f0
    rho0 = deepcopy(rho1)
    rho0[:,idx_wb+1:end] = 1f0./imfilter(1f0./rho1[:,idx_wb+1:end], Kernel.gaussian(10))

    dimp = [vec(v0.*rho0-v1.*rho1), vec(v0.*rho0-v2.*rho2)]
    model0 = Model(n, d, o, m0; rho=rho0, nb = 80)
    opt = JUDI.Options(isic=true)

    # extent of the model
    extentx = (n[1]-1)*d[1];

    # source locations
    xsrc = [ContJitter(extentx, nsrc) for i=1:nv]
    ysrc = range(0f0,stop=0f0,length=nsrc)
    zsrc = range(10f0,stop=10f0,length=nsrc)

    # receiver locations
    xrec = range(d[1],stop=(n[1]-1)*d[1], length=nrec)
    yrec = 0f0
    zrec = range((idx_wb-1)*d[2]-2f0,stop=(idx_wb-1)*d[2]-2f0,length=nrec)

    # recording time
    timeR = 1800f0
    dtR = 4f0
    ntR = Int(floor(timeR/dtR))+1

    # set up geometries
    srcGeometry = [Geometry(convertToCell(xsrc[i]), convertToCell(ysrc), convertToCell(zsrc); dt=dtR, t=timeR) for i=1:nv]
    recGeometry = Geometry(xrec, yrec, zrec; dt=dtR, t=timeR, nsrc=nsrc)

    # source
    f0 = 0.025f0     # kHz
    wavelet = ricker_wavelet(timeR, dtR, f0)
    q = [judiVector(srcGeometry[i], wavelet) for i=1:nv]

    # operators
    Pr = judiProjection(recGeometry)   # receiver restriction
    Ps = [judiProjection(srcGeometry[i]) for i=1:nv]   # source injection
    F0 = [Pr*judiModeling(model0; options=opt)*Ps[i]' for i=1:nv]
    J = [judiJacobian(F0[i], q[i]) for i=1:nv]

    #born Modeling
    dlin = [J[i] * dimp[i] for i = 1:nv]

    serialize(container, "lindata_$(idx_i)", (q=q, dlin=dlin))

    # noise
    noise = deepcopy(dlin)
    for k = 1:nv
        for l = 1:nsrc
            noise[k].data[l] = randn(Float32, size(dlin[k].data[l]))
            noise[k].data[l] = real.(ifft(fft(noise[k].data[l]).*fft(q[k].data[1])))
        end
    end

    noise = noise/norm(noise) * norm(dlin) * 10f0^(-snr/20f0)
    dlin = dlin + noise

    serialize(container, "noisy_lindata_$(idx_i)", (q=q, dlin=dlin))

    return nothing
end

println(sim_name)
Base.flush(stdout)

futures = @batchexec pmap(i -> born_modeling_i(i, v1[i], v2[i], rho1[i], rho2[i], nsrc, snr, container), 1:nsample)
fetch(futures; num_restart=0, timeout=24000, task_timeout=1000)

delete_all_jobs()
delete_pool()
delete_container()
