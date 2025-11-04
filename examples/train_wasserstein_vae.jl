"""
Example training script for Wasserstein VAE.

This script demonstrates how to train a VAE with Wasserstein-based calibration loss
on a simple distribution learning task.
"""

using Pkg
Pkg.activate(".")

using WassersteinSampling
using Lux, LuxCUDA
using Random
using Optimisers
using Statistics
using Distributions

# Check if CUDA is available
const xdev = CUDA.functional() ? gpu_device() : cpu_device()
const cdev = cpu_device()

println("Using device: ", typeof(xdev))

# Training configuration
const CONFIG = (
    batchsize = 2^10,
    num_latent_dims = 3,
    learning_rate = 1f-3,
    num_epochs = 500,
    seed = 1,

    # Loss function parameters
    nsamples = 256,     # Number of samples from VAE for loss computation
    nb = 256,           # Number of threshold values for Wasserstein distance

    # Logging
    log_interval = 50
)

# Create reference distribution (bimodal mixture)
reference_distribution = create_reference_distribution()
println("Reference distribution: ", reference_distribution)

# Initialize random seed
rng = Random.Xoshiro(CONFIG.seed)

# Create model
println("\nInitializing model...")
vae = VAEWithAuxLoss(rng; num_latent_dims=CONFIG.num_latent_dims)
ps, st = Lux.setup(rng, vae) |> xdev

# Initialize optimizer
opt = Optimisers.Adam(; eta=CONFIG.learning_rate)
train_state = Training.TrainState(vae, ps, st, opt)

# Create loss function
tau = 0.1f0 * std(reference_distribution)
lossfn = WassersteinLossWAux(;
    nsamples = CONFIG.nsamples,
    nb = CONFIG.nb,
    tau = tau,
    reference_distribution = reference_distribution
)

println("Loss function tau: ", tau)

# Training loop
println("\nStarting training...")
println("="^60)

start_time = time()

for epoch in 1:CONFIG.num_epochs
    # Generate training batch
    ys = rand.(reference_distribution, 1, CONFIG.batchsize) |> xdev
    xs = zeros_like(ys)  # Input is just zeros (unconditional generation)

    # Perform training step
    (_, loss, stats, train_state) = Training.single_train_step!(
        AutoZygote(), lossfn, (xs, ys), train_state;
        return_gradients=Val(false)
    )

    # Logging
    if epoch % CONFIG.log_interval == 0 || epoch == 1
        elapsed = time() - start_time
        throughput = (epoch * CONFIG.batchsize) / elapsed

        println("Epoch $epoch/$CONFIG.num_epochs")
        println("  Total Loss: ", round(loss, digits=6))
        println("  KL Div Loss: ", round(stats.kldivloss, digits=6))
        println("  Wasserstein Loss: ", round(stats.wassersteinloss, digits=6))
        println("  NLL Loss: ", round(stats.nllloss, digits=6))
        println("  Weighted NLL Loss: ", round(stats.weighted_nllloss, digits=6))
        println("  Log Var Aux: ", round(sum(train_state.parameters.log_var_aux), digits=6))
        println("  Throughput: ", round(throughput, digits=1), " samples/s")
        println("-"^60)
    end
end

println("\nTraining complete!")
println("Total time: ", round(time() - start_time, digits=2), " seconds")

# Generate samples from trained model
println("\nGenerating samples from trained model...")
n_test_samples = 10_000
test_input = xdev(rand(Float32, 1, n_test_samples))
samples, _ = vae(test_input, train_state.parameters, train_state.states)
samples = samples[1] |> cdev  # Move back to CPU and get first output

# Basic statistics
println("\nSample statistics:")
println("  Mean: ", round(mean(samples), digits=4))
println("  Std: ", round(std(samples), digits=4))
println("  Min: ", round(minimum(samples), digits=4))
println("  Max: ", round(maximum(samples), digits=4))

# Save model parameters
println("\nSaving model...")
using JLD2
JLD2.save("trained_vae.jld2",
    Dict("parameters" => cdev(train_state.parameters),
         "states" => cdev(train_state.states),
         "config" => CONFIG))

println("Model saved to trained_vae.jld2")
println("\nTo visualize results, see examples/visualize_results.jl")
