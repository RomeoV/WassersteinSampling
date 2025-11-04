"""
Simple test script to verify package can be loaded.

Run this with:
    julia --project=. test_package.jl
"""

using Pkg
Pkg.activate(".")

println("Testing WassersteinSampling package...")

try
    using WassersteinSampling
    println("✓ Package loaded successfully")

    # Test that exports are available
    @assert isdefined(WassersteinSampling, :VAE)
    @assert isdefined(WassersteinSampling, :VAEWithAuxLoss)
    @assert isdefined(WassersteinSampling, :encoder)
    @assert isdefined(WassersteinSampling, :decoder)
    @assert isdefined(WassersteinSampling, :MyUnivariateMixtureModel)
    @assert isdefined(WassersteinSampling, :WassersteinLossWAux)
    println("✓ All exports available")

    # Test creating a simple model
    using Random
    rng = Random.Xoshiro(1)
    vae = VAE(rng; num_latent_dims=3)
    println("✓ VAE model created successfully")

    vae_aux = VAEWithAuxLoss(rng; num_latent_dims=3)
    println("✓ VAEWithAuxLoss model created successfully")

    # Test reference distribution
    ref_dist = create_reference_distribution()
    println("✓ Reference distribution created: ", typeof(ref_dist))

    # Test loss function creation
    loss = WassersteinLossWAux(;
        nsamples=16,
        nb=16,
        reference_distribution=ref_dist
    )
    println("✓ Loss function created successfully")

    println("\n" * "="^60)
    println("All tests passed! ✓")
    println("="^60)

catch e
    println("\n" * "="^60)
    println("Error during testing: ✗")
    println("="^60)
    showerror(stdout, e, catch_backtrace())
    println()
    exit(1)
end
