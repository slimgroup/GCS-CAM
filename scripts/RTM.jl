using DrWatson
@quickactivate "GCS-CAM"

using PyPlot, JUDI
using JOLI, JLD2, Images, Random, LinearAlgebra, Statistics
using MECurvelets
using SlimPlotting
using SlimOptim
using ArgParse
matplotlib.use("agg")

include(srcdir("joJRM.jl"))
include(srcdir("utils.jl"))

parsed_args = parse_commandline()
nv = parsed_args["nv"]
nsrc = parsed_args["nsrc"]
snr = parsed_args["snr"]

JLD2.@load datadir("models", "SingleLeakageExample.jld2") v1 rho1 v2 rho2

n = size(v1)
d = (6.25f0, 6.25f0)
o = (0f0, 0f0)

nrec = Int(floor(n[1]/4))

data_dict = @strdict nrec nsrc nv snr
JLD2.@load datadir("lindata_TLE"*savename(data_dict, "jld2"; digits=6)) dlin q

Random.seed!(2022)

n = size(v1)
d = (6.25f0, 6.25f0)
o = (0f0, 0f0)

idx_wb = maximum(find_water_bottom(v1.-v1[1,1]))
v0 = deepcopy(v1)
v0[:,idx_wb+1:end] = 1f0./imfilter(1f0./v1[:,idx_wb+1:end], Kernel.gaussian(10))
m0 = (1f0./v0).^2f0
rho0 = deepcopy(rho1)
rho0[:,idx_wb+1:end] = 1f0./imfilter(1f0./rho1[:,idx_wb+1:end], Kernel.gaussian(10))

dimp = [vec(v0.*rho0-v1.*rho1), vec(v0.*rho0-v2.*rho2)]
model0 = Model(n, d, o, m0; rho=rho0, nb = 80)
opt = Options(isic=true)

Tm = judiTopmute(n, idx_wb, 1)  # Mute water column
S = judiDepthScaling(model0)  # depth scaling

# extent of the model
extentx = (n[1]-1)*d[1];
extentz = (n[2]-1)*d[2];

Mr1 = joCoreBlock([S*Tm for i = 1:nv]...)

sim_name = "RTM_TLE"
plot_path = plotsdir(sim_name)

phi, g = lsrtm_objective(model0, q, dlin, [zeros(Float32, prod(n)) for i = 1:nv]; nlind=false, options=opt)
result_dict = @strdict nsrc nrec nv snr phi g

@tagsave(
    datadir(sim_name,savename(result_dict, "jld2"; digits=6)),
    result_dict;
    safe=true
)

xim = [reshape(-Mr1 * Mr1 * vcat(g...), n[1], n[2], nv)[:,:,i] for i = 1:nv]
fig = figure(figsize=(20,12))
plot_simage(xim[end]'-xim[1]', d; name="RTM", perc=99.99, new_fig=false)
fig_name = @strdict nsrc nrec nv snr
safesave(joinpath(plot_path, savename(fig_name; digits=6)*"_TLE.png"), fig);
