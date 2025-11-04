"""
Custom Univariate Mixture Model that is `isbits` compatible.
"""

using ConcreteStructs
using Distributions
using StaticArrays
using Random

@concrete struct MyUnivariateMixtureModel <: Distributions.AbstractMixtureModel{Distributions.Univariate, Distributions.Continuous, Distributions.Normal{Float32}}
    components
end

Base.eltype(::MyUnivariateMixtureModel) = Float32
Base.isbits(::MyUnivariateMixtureModel) = true

Distributions.ncomponents(d::MyUnivariateMixtureModel) = length(d.components)
Distributions.component(d::MyUnivariateMixtureModel, k) = d.components[k]
Distributions.probs(d::MyUnivariateMixtureModel) = 1.0f0/ncomponents(d) * @SVector ones(Float32, ncomponents(d))

function Distributions.rand(rng::AbstractRNG, d::MyUnivariateMixtureModel)
    proposals = rand.((rng,), d.components)
    proposals[rand(rng, 1:ncomponents(d))]
end

"""
Create a simple bimodal mixture as a reference distribution.
"""
function create_reference_distribution()
    return MyUnivariateMixtureModel(SA[
        Normal(-2.0f0, 2.0f0),
        Normal(3.0f0, 1.0f0)
    ])
end
