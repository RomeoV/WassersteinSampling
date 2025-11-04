"""
Visualization script for trained Wasserstein VAE.

This script loads a trained model and creates visualizations comparing
the generated samples to the reference distribution.
"""

using Pkg
Pkg.activate(".")

using WassersteinSampling
using Lux
using Random
using CairoMakie
using JLD2
using Statistics
using StatsBase
using Distributions

# Load trained model
println("Loading trained model...")
model_data = JLD2.load("trained_vae.jld2")
ps = model_data["parameters"]
st = model_data["states"]
config = model_data["config"]

# Recreate model
rng = Random.Xoshiro(config.seed)
vae = VAEWithAuxLoss(rng; num_latent_dims=config.num_latent_dims)

# Generate samples
println("Generating samples...")
n_samples = 10_000
test_input = rand(Float32, 1, n_samples)
samples, _ = vae(test_input, ps, st)
samples = vec(samples[1])  # Flatten to 1D array

# Create reference distribution
reference_distribution = create_reference_distribution()
ref_samples = rand(reference_distribution, n_samples)

# Create visualizations
println("Creating visualizations...")

fig = Figure(size=(1200, 800))

# Histogram comparison
ax1 = Axis(fig[1, 1],
    xlabel = "Value",
    ylabel = "Density",
    title = "Generated vs Reference Distribution"
)

hist!(ax1, samples; normalization=:pdf, label="Generated Samples", bins=50, alpha=0.6)
hist!(ax1, ref_samples; normalization=:pdf, label="Reference Samples", bins=50, alpha=0.6)

# Add theoretical PDF
x_range = range(extrema([samples; ref_samples])..., 200)
y_pdf = [pdf(reference_distribution, x) for x in x_range]
lines!(ax1, x_range, y_pdf; label="Reference PDF", color=:black, linewidth=2)

axislegend(ax1)

# Q-Q plot
ax2 = Axis(fig[1, 2],
    xlabel = "Reference Quantiles",
    ylabel = "Generated Quantiles",
    title = "Q-Q Plot"
)

q_probs = range(0.01, 0.99, 100)
ref_quantiles = quantile.(reference_distribution, q_probs)
gen_quantiles = quantile(samples, q_probs)

scatter!(ax2, ref_quantiles, gen_quantiles; label="Quantiles", markersize=4)
lines!(ax2, ref_quantiles, ref_quantiles; label="Perfect Match",
       color=:red, linestyle=:dash)

axislegend(ax2)

# Probability Integral Transform (PIT) histogram
ax3 = Axis(fig[2, 1],
    xlabel = "PIT Value",
    ylabel = "Density",
    title = "Probability Integral Transform (should be uniform)"
)

pit_values = [cdf(reference_distribution, s) for s in samples]
hist!(ax3, pit_values; normalization=:pdf, bins=20, color=:steelblue)
hlines!(ax3, [1.0]; color=:red, linestyle=:dash, label="Uniform")

axislegend(ax3)

# Summary statistics table
ax4 = Axis(fig[2, 2],
    title = "Summary Statistics"
)
hidedecorations!(ax4)
hidespines!(ax4)

stats_text = """
Generated Samples:
  Mean: $(round(mean(samples), digits=4))
  Std:  $(round(std(samples), digits=4))
  Skew: $(round(skewness(samples), digits=4))
  Kurt: $(round(kurtosis(samples), digits=4))

Reference Distribution:
  Mean: $(round(mean(reference_distribution), digits=4))
  Std:  $(round(std(reference_distribution), digits=4))

Training Config:
  Epochs: $(config.num_epochs)
  Batch Size: $(config.batchsize)
  Learning Rate: $(config.learning_rate)
  Latent Dims: $(config.num_latent_dims)
  N Samples: $(config.nsamples)
"""

text!(ax4, 0.1, 0.5;
    text = stats_text,
    fontsize = 14,
    align = (:left, :center))

# Save figure
output_file = "vae_results.png"
save(output_file, fig)
println("Visualization saved to $output_file")

# Display figure (if in interactive environment)
display(fig)
