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
batchsize = parsed_args["bs"]
niter = parsed_args["niter"]
γ = parsed_args["gamma"]
perc = parsed_args["perc"]

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
wb = (idx_wb-1)*d[2] * ones(Float32, q[1].nsrc)
v0 = deepcopy(v1)
v0[:,idx_wb+1:end] = 1f0./imfilter(1f0./v1[:,idx_wb+1:end], Kernel.gaussian(10))
m0 = (1f0./v0).^2f0
rho0 = deepcopy(rho1)
rho0[:,idx_wb+1:end] = 1f0./imfilter(1f0./rho1[:,idx_wb+1:end], Kernel.gaussian(10))

dimp = [vec(v0.*rho0-v1.*rho1), vec(v0.*rho0-v2.*rho2)]
model0 = Model(n, d, o, m0; rho=rho0, nb = 80)
opt = Options(isic=true)

C = joEye(size(joMECurvelet2D((n), zero_finest=true),1); DDT=ComplexF32)*joMECurvelet2D((n), zero_finest=true)

Tm = judiTopmute(n, idx_wb, 1)  # Mute water column
S = judiDepthScaling(model0)  # depth scaling

# extent of the model
extentx = (n[1]-1)*d[1];
extentz = (n[2]-1)*d[2];

Mr1 = joCoreBlock([S*Tm for i = 1:nv]...)
C1 = joCoreBlock([C for i = 1:nv+1]...)
φ = joJRM(nv, size(C,2); γ=γ, DDT=Float32) 
P = Mr1 * φ

iter = 1

inds = [GenSrcIdxLSRTM(nsrc, batchsize, niter+3), GenSrcIdxLSRTM(nsrc, batchsize, niter+3)]
inds[1] = [vcat([inds[1][1],inds[1][2],inds[1][3]]...),[inds[1][i] for i = 4:niter+2]...]
inds[2] = [vcat([inds[2][1],inds[2][2],inds[2][3]]...),[inds[2][i] for i = 4:niter+2]...]
inds[2] = inds[1]

sim_name = "JRM_TLE"
plot_path = plotsdir(sim_name)

xsrc = [vcat(q[i].geometry.xloc...) for i = 1:length(q)]

function obj(x)
    println("computing objective function at iter ", iter)
    img = [(P * x)[(i-1)*prod(n)+1:i*prod(n)] for i = 1:nv]
    q_ = [q[i][inds[i][iter]] for i = 1:nv]
    dlin_ = [dlin[i][inds[i][iter]] for i = 1:nv]
    # calculate the gradient
    phi, g = lsrtm_objective(model0, q_, dlin_, img; nlind=false, options=opt)
    g = P' * vcat(g...)
    return phi, g
end

first_callback = true
function callback(sol)
    if first_callback
        global first_callback = false
        return
    end
    println("callback at iter ", iter)
    xim = [reshape((P * sol.x), n[1],n[2],nv)[:,:,i] for i = 1:nv]
    zim = [reshape((P * C1' * sol.z), n[1],n[2],nv)[:,:,i] for i = 1:nv]

    fig = figure(figsize=(20,12))
    subplot(3,3,1)
    plot_simage(xim[1]', d; d_scale=0, name="primal", new_fig=false); colorbar();
    subplot(3,3,4)
    plot_simage(zim[1]', d; d_scale=0, name="dual", new_fig=false); colorbar();
    subplot(3,3,7)
    plot_simage(zim[1]'-xim[1]', d; d_scale=0, name="thresholded", new_fig=false); colorbar();
    subplot(3,3,2)
    plot_simage(xim[1]', d; d_scale=0, name="baseline", new_fig=false); colorbar();
    subplot(3,3,5)
    plot_simage(xim[2]', d; d_scale=0, name="monitor", new_fig=false); colorbar();
    subplot(3,3,8)
    plot_simage(xim[2]'-xim[1]', d; d_scale=0, name="time-lapse", new_fig=false, perc=99.99); colorbar();
    subplot(3,3,3)
    plot_simage(reshape(S*Tm*sol.x[1:size(C,2)], n)', d; d_scale=0, name="common", new_fig=false); colorbar();
    subplot(3,3,6)
    plot_simage(reshape(S*Tm*sol.x[1+size(C,2):2*size(C,2)], n)', d; d_scale=0, name="innov1", new_fig=false); colorbar();
    subplot(3,3,9)
    plot_simage(reshape(S*Tm*sol.x[1+2*size(C,2):3*size(C,2)], n)', d; d_scale=0, name="innov2", new_fig=false); colorbar();
    tight_layout()
    fig_name = @strdict iter nsrc nrec nv snr batchsize niter perc
    safesave(joinpath(plot_path, savename(fig_name; digits=6)*".png"), fig);
    close(fig)
    global iter = iter + 1
end

# Bregman
function λfunc(z; quantile=perc)
    z1 = reshape(z, size(C,1), :)
    threshold = [Statistics.quantile(abs.(z1[:,i]), quantile) for i = 1:size(z1,2)]
    return maximum(threshold)
end
bregopt = bregman_options(maxIter=niter, verbose=2, alpha=0.6f0, antichatter=false, spg=false, λfunc=λfunc, TD=C1);
@time sol = bregman(obj, zeros(Float32, size(P,2)), bregopt; callback=callback);

result_dict = @strdict nsrc nrec nv snr batchsize niter perc bregopt sol

@tagsave(
    datadir(sim_name,savename(result_dict, "jld2"; digits=6)),
    result_dict;
    safe=true
)

xim = [reshape(P * sol.x, n[1], n[2], nv)[:,:,i] for i = 1:nv]
fig = figure(figsize=(20,12))
plot_simage(xim[end]'-xim[1]', d; name="JRM", perc=99.99, new_fig=false)
fig_name = @strdict nsrc nrec nv snr batchsize niter perc
safesave(joinpath(plot_path, savename(fig_name; digits=6)*"_TLE.png"), fig);