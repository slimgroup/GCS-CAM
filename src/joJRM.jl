# joJRM operator

## helper module
module joJRM_etc

    function apply_fwd(n::Integer, L::Integer, z::Vector{T}; γ::T=1f0) where T
        return view(z, n+1:length(z)) .+ T(1)/γ * repeat(view(z, 1:n), L)    # get original components
    end 

    function apply_adj(L::Integer, x::Vector{T}; γ::T=1f0) where T
        return vcat(T(1)/γ * view(sum(reshape(view(x,:), :, L), dims=2), :, 1), x)   # weighted sum on common component
    end

end
using .joJRM_etc

export joJRM, joTransitionJRM
"""
    julia> op = joJRM(ops;[name=...])
JRM operator of an array of JOLI operators
# Signature
    joJRM(ops::Vector{joAbstractLinearOperator}; γ::Number=1f0, name::String="joJRM")
# Arguments
- `ops`: an array of JOLI operators (subtypes of joAbstractLinearOperator)
- keywords
    - `γ`: weight of common component = 1/γ "Xiaowei Li, A weighted ℓ1-minimization for distributed compressive sensing"
    - `name`: custom name
# Notes
- the domain and range types of joJRM are equal to domain type of any operator in ops
- all operators in the array ops must have the same domain/range types
# Example
define operators
    ops = [joDiag(randn(Float32,3); DDT=Float32, RDT=Float32) for i = 1:4]
    γ = 2f0
define JRM operator
    A=joJRM(ops; γ=γ)
"""


function joJRM(ops::Vector{T1}; γ::T2=1f0, name::String="joJRM") where {T1<:joAbstractOperator, T2<:Real}
    return joJRM(ops...; γ=γ, name=name)
end

function joJRM(ops::Vararg{T1, L}; γ::T2=1f0, name::String="joJRM") where {T1<:joAbstractOperator, T2<:Real, L}

    γ < 0 && throw(joLinearFunctionException("weight on common component must be non-negative"))
    isempty(ops) && throw(joLinearFunctionException("empty argument list"))
    DDT = deltype(ops[1])
    RDT = reltype(ops[1])
    #(T2 == DDT) || throw(joLinearFunctionException("weight on common component type mismatch with domain/range type"))
    opn = ops[1].n
    for i=2:L
        deltype(ops[i])==DDT || throw(joLinearFunctionException("domain type mismatch"))
        reltype(ops[i])==RDT || throw(joLinearFunctionException("range type mismatch"))
        ops[i].n==opn || throw(joLinearFunctionException("domain length mismatch"))
    end
    As = joCoreBlock(ops...)
    Φ = joLinearFunctionFwd(L*ops[1].n, (L+1)*ops[1].n,
        v1 -> joJRM_etc.apply_fwd(ops[1].n, L, v1; γ=DDT(γ)),
        v2 -> joJRM_etc.apply_adj(L, v2; γ=DDT(γ)),
        v3 -> joJRM_etc.apply_adj(L, v3; γ=DDT(γ)),
        v4 -> joJRM_etc.apply_fwd(ops[1].n, L, v4; γ=DDT(γ)),
        DDT, DDT; name=name)
    return As * Φ
end

function joJRM(L::Integer, nn::Integer; DDT=Float32, γ::T=1f0, name::String="joJRM") where {T<:Real}

    γ < 0 && throw(joLinearFunctionException("weight on common component must be non-negative"))
    Φ = joLinearFunctionFwd(L*nn, (L+1)*nn,
        v1 -> joJRM_etc.apply_fwd(nn, L, v1; γ=DDT(γ)),
        v2 -> joJRM_etc.apply_adj(L, v2; γ=DDT(γ)),
        v3 -> joJRM_etc.apply_adj(L, v3; γ=DDT(γ)),
        v4 -> joJRM_etc.apply_fwd(nn, L, v4; γ=DDT(γ)),
        DDT, DDT; name=name)
    return Φ
end
