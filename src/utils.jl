function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--nv"
            help = "Number of vintages"
            arg_type = Int
            default = 2
        "--nsrc"
            help = "Number of sources per vintage"
            arg_type = Int
            default = 32
        "--niter"
            help = "JRM iterations"
            arg_type = Int
            default = 22
        "--bs"
            help = "batchsize in JRM iterations"
            arg_type = Int
            default = 4
        "--snr"
            help = "SNR of noisy data"
            arg_type = Float64
            default = 8.0
        "--gamma"
            help = "Weighting on common component"
            arg_type = Float64
            default = 1.0
        "--perc"
            help = "thresholding perc at first iteration"
            arg_type = Float64
            default = .995
        "--leak"
            help = "leakage or not"
            arg_type = Bool
            default = true
    end
    return parse_args(s)
end

function ContJitter(l::Number, num::Int)
    #l = length, num = number of samples
    interval_width = l/num
    interval_center = range(interval_width/2, stop = l-interval_width/2, length=num)
    randomshift = interval_width .* rand(Float32, num) .- interval_width/2

    return interval_center .+ randomshift
end

function GenSrcIdxLSRTM(nsrc::Int, batchsize::Int, iter::Int)

    src_list = collect(1:nsrc)

    inds = [zeros(Int, batchsize) for i = 1:iter]
    # random batch of sources
    for i=1:iter
        length(src_list) < batchsize && (src_list = collect(1:nsrc))
        src_list = src_list[randperm(length(src_list))]
        inds[i] = [pop!(src_list) for b=1:batchsize]
    end

    return inds

end