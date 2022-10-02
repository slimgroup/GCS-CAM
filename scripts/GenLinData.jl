using DrWatson
@quickactivate "GCS-CAM"

using PyPlot, JUDI
using JOLI, JLD2, Images, Random, LinearAlgebra, Statistics
using SlimPlotting
using MECurvelets
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

Random.seed!(2023)

n = size(v1)
d = (6.25f0, 6.25f0)
o = (0f0, 0f0)

nrec = Int(floor(n[1]/4))

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
dlin = J .* dimp

# band-limited noise
noise = deepcopy(dlin)
for k = 1:nv
    for l = 1:nsrc
        noise[k].data[l] = randn(Float32, size(dlin[k].data[l]))
        noise[k].data[l] = real.(ifft(fft(noise[k].data[l]).*fft(q[k].data[1])))
    end
end

noise = noise/norm(noise) * norm(dlin) * 10f0^(-snr/20f0)
dlin = dlin + noise

data_dict = @strdict nsrc nrec nv snr dlin q

@tagsave(
    datadir("lindata_TLE"*savename(data_dict, "jld2"; digits=6)),
    data_dict;
    safe=true
)

sim_name = "data_TLE"
plot_path = plotsdir(sim_name)

fig = figure(figsize=(20,12))
plot_sdata(dlin[1].data[1], (1f0,1f0); name="shot", new_fig=false)
fig_name = @strdict nsrc nrec nv snr
safesave(joinpath(plot_path, savename(fig_name; digits=6)*"_shot.png"), fig);

fig = figure(figsize=(20,12))
plot_simage(reshape(dimp[2]-dimp[1],n)', d; name="true", perc=99.99, new_fig=false)
fig_name = @strdict nsrc nrec nv snr
safesave(joinpath(plot_path, savename(fig_name; digits=6)*"_TLE.png"), fig);
