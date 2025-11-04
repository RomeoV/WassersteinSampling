### A Pluto.jl notebook ###
# v0.20.20

using Markdown
using InteractiveUtils

# ╔═╡ d0529936-b461-11f0-9c79-dfd31b2afff9
begin
	using Lux, LuxCUDA
	using CairoMakie
	using Optimization, Optim, OptimizationOptimJL, Zygote
	using Distributions
	using Random
	using ConcreteStructs
	using StatsBase
	using MLUtils
	using Optimisers
	using ChainRulesCore
	using StaticArrays
	#using Enzyme, Reactant
	#Reactant.set_default_backend("gpu")
	#AD = AutoEnzyme()
	AD = AutoZygote()
	using Typstry
	using PlutoUI
	using Tullio, CUDA, KernelAbstractions
	LuxCUDA.CUDA.allowscalar(false)
end

# ╔═╡ c85c7530-9550-4d3c-9969-db76505cca53
PlutoUI.TableOfContents()

# ╔═╡ f45610c9-1da2-433d-bf42-7fded6bc88ed
md"""
## Model definition

We define a basic VAE encoder and decoder.
We just have a couple of dense layers for each.
"""

# ╔═╡ f3465a00-2687-4fb4-9454-7b8180f9394b
function decoder(; num_latent_dims::Int,
				   intermediate_dims = [3=>20, 20=>20]
)
    return @compact(;
        decode=Chain(
			[Dense(d, relu) for d in intermediate_dims]...,
			Dense(last(intermediate_dims[end])=>1)
        )
    ) do x
        @return decode(x)
    end
end


# ╔═╡ 96c93256-e22f-4fe9-90ed-fecb31cd952e
md"""
Next we define some wrapper functions around our two model definitions to be able to easily `encode` and `decode`.
"""

# ╔═╡ b4fb7c73-179b-4f25-9d7c-5c606d25b4e1
md"""
## Loss function
Now we need to define the loss function.
As is typical for a VAE we penalize the latent variable with a KL divergence to a 0-1 Gaussian. Further, we use the `calibration_loss` defined below.
"""

# ╔═╡ c9218656-76b2-49a2-96b2-35cd398d0ba5
md"""
Now we define the `calibration_loss`. For this, we essentially want to compute the probability integral transform, which has a binning step. However, binning is non-differentiable, so instead we do "binning" by using Gaussian kernels.
This part was partly Claude generated and I need to review it further to convince myself this is correct.
"""

# ╔═╡ 6584ba83-f40b-43f0-a342-b0961e02d1cf
function soft_histogram(values::AbstractVector{T}, bin_centers, σ) where {T}
    """
    Differentiable soft histogram using Gaussian kernels
    values: Vector of values to bin
    bin_centers: Vector of bin centers
    σ: smoothness parameter
    """
    # Compute distances (N × B matrix)
    delta = values .- bin_centers'
    
    # Gaussian kernel weights
    weights = exp.(-T(0.5) .* (delta ./ σ).^2)
    weights = weights ./ (sum(weights, dims=2) .+ eps(T))
    
    # Sum over samples and normalize
    histogram = sum(weights, dims=1)[:]
    return histogram ./ sum(histogram)
end


# ╔═╡ e1071935-074f-425b-bfa4-5e2bc5a0c110
function calibration_loss(predicted_probs::AbstractVector{T}, num_bins=20; σ=T(1/num_bins)) where {T}
    """
    Calibration loss based on PIT uniformity
    predicted_probs: CDF values (should be uniform if calibrated)
    """
    bin_centers = @ignore_derivatives range(T(0), T(1), length=num_bins)
    
    # Compute soft histogram
    hist = soft_histogram(predicted_probs, bin_centers, σ)
    
    # Target uniform distribution
    uniform = ones_like(hist, T, num_bins) ./ num_bins
    
    # Loss - you can use KL divergence, chi-squared, or L2
    # KL divergence (add small epsilon for numerical stability)
    loss = sum(uniform .* log.((uniform .+ eps(T)) ./ (hist .+ eps(T))))
    
    # Or L2 loss
    # loss = sum((hist .- uniform).^2)
    
    return loss
end


# ╔═╡ d8b335d6-102b-44fa-a85f-dba234820bb7
function loss_function(model, ps, st, (xs, pys))
    (yhat, μ, logσ²), st = model(xs, ps, st)
    calib_loss = calibration_loss(cdf.(pys, yhat[:]))
    kldiv_loss = -sum(1 .+ logσ² .- μ .^ 2 .- exp.(logσ²)) / 2
    loss = calib_loss + kldiv_loss
    return loss, st, (; yhat, μ, logσ², calib_loss, kldiv_loss)
end

# ╔═╡ 907b92ec-4f3f-4548-b543-3e6f849f0485
const czip = collect∘zip

# ╔═╡ ff3b1670-fad1-46f2-bd8a-3fbf91854f83
md"""
## Training and results
"""

# ╔═╡ 26e9b96b-6346-45e3-9a25-a57267f2cb1a
# ╠═╡ disabled = true
#=╠═╡
reference_distribution = Normal(2.0f0, 1.5f0)
  ╠═╡ =#

# ╔═╡ f8d41cbd-3991-46fc-b1ea-5778c5b10dfe
const xdev = gpu_device()
#const xdev = reactant_device()  # enzyme doesn't support 1.12 right now

# ╔═╡ 7a1d7040-99e9-419a-b0b4-e0521d50f94d
const cdev = cpu_device()

# ╔═╡ 0609e1a0-bdb5-409a-a7ff-771c03367ddf
#=╠═╡
(; vae, ps, st) = let
    batchsize=2^10
    num_latent_dims=3
    learning_rate = 1f-3
    num_epochs = 500
    seed=1

	rng = Xoshiro()
    Random.seed!(rng, seed)
	#ys = [Normal(0.f0)]
	#xs = rand.(ys) |> xs->reshape(xs, 1, :)
	vae = VAE(; num_latent_dims)
	ps, st = Lux.setup(rng, vae) |> xdev

	opt = Optimisers.Adam(; eta=learning_rate)
    train_state = Training.TrainState(vae, ps, st, opt)


	start_time = time()

	for epoch in 1:num_epochs
        loss_total = 0.0f0
        total_samples = 0
		#ys = [Normal(0.0f0) for _ in 1:batchsize]
		ys = [reference_distribution for _ in 1:batchsize] |> xdev
		xs = rand.(ys) |> xs->reshape(xs, 1, :) |> xdev

		(_, loss, _, train_state) = Training.single_train_step!(
			AD, loss_function, (xs, ys), train_state; return_gradients=Val(false)
		)

        loss_total += loss
        total_samples += size(xs) |> last

        if epoch % 250 == 0
            throughput = total_samples / (time() - start_time)
            @show "Epoch %d, Loss: %.7f, Throughput: %.6f im/s\n" epoch loss throughput
        end
    end
	(; vae, ps, st)
end

  ╠═╡ =#

# ╔═╡ f9514525-6c62-4b79-92df-562982d2d5ce
md"""
Finally we compare our samples to a reference 0-1 Gaussian.
They should look the same.
"""

# ╔═╡ 1b1bf49b-7cd7-43a9-ac5f-ecf1d38d78e4
#=╠═╡
with_theme(theme_latexfonts()) do 
	samples = vae(rand(Float32, 1, 10_000), cdev(ps), cdev(st))[1][1][:]
	fig = Figure()
	ax = Axis(fig[1,1])
	hist!(ax, samples; normalization=:pdf, label="Samples", bins=30)
	lines!(ax, range(extrema(samples)..., 100), x->pdf(reference_distribution, x); label=L"Reference $\mathcal{N}(0, 1)$", color=Makie.wong_colors()[2])
	axislegend()
	fig
end
  ╠═╡ =#

# ╔═╡ 944e00ae-414a-4018-8973-6f1fb285bf45
md"""
## Custom Univariate Mixture Model
that is `isbits`
"""

# ╔═╡ 2dd01743-0527-48a4-ba61-ee602692b9e2
@concrete struct MyUnivariateMixtureModel <: Distributions.AbstractMixtureModel{Distributions.Univariate, Distributions.Continuous, Distributions.Normal{Float32}}
    components
end

# ╔═╡ 6ed85f96-0640-4c44-995c-13915107cd46
reference_distribution = MyUnivariateMixtureModel(SA[
	Normal(-2.0f0, 2.0f0),
	Normal(3.0f0, 1.0f0)
])

# ╔═╡ fda65a7b-0cee-40f2-8fcd-1da567c00088
Base.isbits(reference_distribution)

# ╔═╡ 8d95a21f-0c17-4a33-aea5-b093d2ea195c
Base.eltype(::MyUnivariateMixtureModel) = Float32

# ╔═╡ 57ffac72-0b1c-4e8e-bd23-c78ac4f71378
function encoder(
    rng=Random.Xoshiro();
    num_latent_dims::Int,
	intermediate_dims = [1=>20, 20=>3]
)
	last_hidden_dim = let (in, out) = intermediate_dims[end]
		out
	end
    return @compact(;
        embed=Chain(
			[Dense(d, relu) for d in intermediate_dims]...
        ),
        proj_mu=Dense(last_hidden_dim=>num_latent_dims;
					  init_bias=zeros32),
        proj_log_var=Dense(last_hidden_dim=>num_latent_dims;
					  init_bias=zeros32),
        rng
    ) do x
        y = embed(x)

        μ = proj_mu(y)
        logσ² = proj_log_var(y)

        T = eltype(logσ²)
        logσ² = clamp.(logσ², -T(20.0f0), T(10.0f0))
        σ = exp.(logσ² .* T(0.5))

        # Generate a tensor of random values from a normal distribution
        ϵ = randn_like(Lux.replicate(rng), σ)

        # Reparameterization trick to backpropagate through sampling
        z = ϵ .* σ .+ μ

        @return z, μ, logσ²
    end
end

# ╔═╡ 933c1fa7-11a9-4ba4-a2f5-dffa3b749a6d
begin # we have to put everything in one Block for Pluto to be happy...
@concrete struct VAE <: AbstractLuxContainerLayer{(:encoder, :decoder)}
    encoder <: AbstractLuxLayer
    decoder <: AbstractLuxLayer
end
function VAE(
    rng=Random.Xoshiro();
    num_latent_dims::Int=3,
)
	enc = encoder(rng; num_latent_dims)
    dec = decoder(; num_latent_dims)
    return VAE(enc, dec)
end
function (vae::VAE)(x, ps, st)
    (z, μ, logσ²), st_enc = vae.encoder(x, ps.encoder, st.encoder)
    x_rec, st_dec = vae.decoder(z, ps.decoder, st.decoder)
    return (x_rec, μ, logσ²), (; encoder=st_enc, decoder=st_dec)
end
end

# ╔═╡ 857dc213-ea03-4465-8e5e-16380b8efcf3
function encode(vae::VAE, x, ps, st)
    (z, μ, logσ²), st_enc = vae.encoder(x, ps.encoder, st.encoder)
    return (; z, μ, logσ²), (; encoder=st_enc, st.decoder)
end


# ╔═╡ a948d08b-5340-49b3-8087-3945dcbde6f7
function decode(vae::VAE, z, ps, st)
    x_rec, st_dec = vae.decoder(z, ps.decoder, st.decoder)
    return x_rec, (; decoder=st_dec, st.encoder)
end

# ╔═╡ 3f99a1cd-546d-4e66-8004-9b741545072b
Distributions.ncomponents(d::MyUnivariateMixtureModel) = length(d.components)

# ╔═╡ d11795ec-f6bd-4b2b-bfbf-47a3afe893aa
Distributions.component(d::MyUnivariateMixtureModel, k) = d.components[k]

# ╔═╡ 10060dc9-eba1-4ccc-b0d6-8a0722869397
Distributions.probs(d::MyUnivariateMixtureModel) = 1.0f0/ncomponents(d) * @SVector ones(Float32, ncomponents(d))

# ╔═╡ 04a07802-4f46-48ef-9336-bb04b5950e1e
Distributions.rand(rng::AbstractRNG, d::MyUnivariateMixtureModel) = let
	proposals = rand.((rng, ), d.components)
	proposals[rand(rng, 1:ncomponents(d))]
end

# ╔═╡ 30fa3bb4-fff0-4e02-a70e-619153a4687e
soft_histogram(cu(rand(Float32, 50)), 0.0f0:0.1f0:1.0f0, 0.1f0)

# ╔═╡ c73f77d0-7d76-463e-866c-400f286108f5
cdf(MyUnivariateMixtureModel(SA[Normal(), Normal()]), 0.5)

# ╔═╡ f909749a-d7c9-4d4b-8e50-06bdcb83a7bd
MixtureModel([
	Normal(0.0f0, 1.0f0),
	Normal(3.0f0, 1.0f0)
]) |> typeof

# ╔═╡ 0ee9d467-77a7-4bdf-83c5-822a8f3ce6ea
md"""
## Loss with point estimates
"""

# ╔═╡ c0b17d51-fb48-400a-a5dc-9287183ae713
# Instead of hard indicator: yhat .<= y'
# Use soft indicator with sigmoid:
function soft_indicator(yhat, y; temperature=0.1)
    sigmoid.((y .- yhat) ./ temperature)
end

# ╔═╡ 09675510-bd19-4a71-812a-092faceb78e2
# ╠═╡ disabled = true
#=╠═╡
(vae2, ps2, st2) = let
	#xdev = cdev
    batchsize=2^10
    num_latent_dims=3
    learning_rate = 1f-3
    num_epochs = 500
    seed=1

	rng = Xoshiro()
    Random.seed!(rng, seed)
	#ys = [Normal(0.f0)]
	#xs = rand.(ys) |> xs->reshape(xs, 1, :)
	vae = VAE(; num_latent_dims)
	ps, st = Lux.setup(rng, vae) |> xdev

	opt = Optimisers.Adam(; eta=learning_rate)
    train_state = Training.TrainState(vae, ps, st, opt)


	start_time = time()

	for epoch in 1:num_epochs
        loss_total = 0.0f0
        total_samples = 0
		#ys = [Normal(0.0f0) for _ in 1:batchsize]
		ys = [reference_distribution for _ in 1:batchsize] |> xdev
		xs = rand.(ys) |> xs->reshape(xs, 1, :) |> xdev

		(_, loss, _, train_state) = Training.single_train_step!(
			AD, loss_function, (xs, xs[:]), train_state; return_gradients=Val(false)
		)

        loss_total += loss
        total_samples += size(xs) |> last

        if epoch % 250 == 0
            throughput = total_samples / (time() - start_time)
            @show "Epoch %d, Loss: %.7f, Throughput: %.6f im/s\n" epoch loss throughput
        end
    end
	(; vae, ps, st)
end

  ╠═╡ =#

# ╔═╡ 8ced94cc-1124-47b5-9a69-51559ef7ab2f
#=╠═╡
with_theme(theme_latexfonts()) do 
	samples = vae2(rand(Float32, 1, 10_000), cdev(ps2), cdev(st2))[1][1][:]
	fig = Figure()
	ax = Axis(fig[1,1])
	hist!(ax, samples; normalization=:pdf, label="Samples", bins=30)
	lines!(ax, range(extrema(samples)..., 100), x->pdf(reference_distribution, x); label=L"Reference $\mathcal{N}(0, 1)$", color=Makie.wong_colors()[2])
	axislegend()
	fig
end
  ╠═╡ =#

# ╔═╡ 63dd8cc3-f892-4732-a0b2-f8e6adbaef00
#=╠═╡
let
	dev = gpu_device()
	testvae = VAE();
	ps, st = Lux.setup(Random.Xoshiro(), testvae) |> dev
	ys = [Normal(0.f0)] |> dev
	xs = rand.(ys) |> xs->reshape(xs, 1, :) |> dev

	loss_function(testvae, ps, st, (xs, xs[:]))
end
  ╠═╡ =#

# ╔═╡ 4fc470d7-8b5f-47f0-a9f7-10e605374158
md"""
# Scratch space
"""

# ╔═╡ df8fbe25-dd78-4371-a14c-7c07f09d6d03
function PIT(xs, ys)
	pit_vals = [cdf(p, x) for (x, p) in zip(xs, ys)]
	fit(Histogram, pit_vals; nbins=10) |> StatsBase.normalize
end

# ╔═╡ de5fbfff-c258-4946-a6f4-89cc3f9649e3
PIT(randn(100), [Normal() for _ in 1:100]).weights

# ╔═╡ c9284099-375b-4e64-a797-f5e00b1db34c
let
	function foo1()
        return (; a=1, b=2), (; c=3, d=4)
	end
	p1, p2 = foo1()
	@show p1, p2
end

# ╔═╡ 0059b566-2ecf-4c30-a1bd-95eafc53d8dd
#=╠═╡
let
	dev = gpu_device()
	testvae = VAE();
	ps, st = Lux.setup(Random.Xoshiro(), testvae) |> dev
	ys = [Normal(0.f0)] |> dev
	xs = rand.(ys) |> xs->reshape(xs, 1, :) |> dev

	loss_function(testvae, ps, st, (xs, ys))
end
  ╠═╡ =#

# ╔═╡ ba8fc908-0061-4c28-a6b3-8277529aaaad
let
	y = [Normal() for _ in 1:10] .|> cu
	yhat = rand.(y)
	calib_loss = calibration_loss(cdf.(y, yhat))
end

# ╔═╡ e27044f1-46b3-4494-b912-bc41e72eafaa
#=╠═╡
eltype(ps.encoder.embed.layer_1.weight)
  ╠═╡ =#

# ╔═╡ 8abfeb60-7e7b-44c1-9105-c4bf2e33eb79
Base.isbits(reference_distribution)

# ╔═╡ c8434618-0eb4-498c-b468-8ac2f800eabf
# ╠═╡ disabled = true
#=╠═╡
let ps = cu([reference_distribution for _ in 1:10]), xs = cu(randn(10))
  Zygote.gradient(x->(cdf.(ps, x .* xs)|>sum), 1.0)
end
  ╠═╡ =#

# ╔═╡ 44a27857-73c2-48b3-9ad4-708461556e7f
md"""
## Half space loss 
After talking to Alex, we can also try to achieve a different property, namely by minimizing the Wasserstein-1 distance, which we can write as

$$\int_{-\infty}^{\infty} |F_x^{-1}(u) - F_y^{-1}(u)| du$$

i.e., we minimize the cdf across all values for $u$. However, if we only have samples we can compute

$$\mathit{mean}_{b \sim p_y}|\mathbb{P}(x > b) - \mathbb{P}(y > b)|$$

Here, $b$ has replaced $u$ and we sample it "reasonably".

However, we have an issue. In our case we have a mathematical setup where we assume there is some process

$$z \mapsto p_y(z)$$

but we only ever get access to one sample $Y_1 \sim p_y(z)$ for each input $z$. However, we can generate many predictions $X_1, \dots, X_k$ which we sample from our VAE model $\hat{p}_y(z)$.

So for our machine learning loss we want to minimize some expactions, maybe like

$$\mathbb{E}_{z \sim p_z, y \sim p_y(z)}  \mathbb{E}_{b \sim p_y(z)} |\mathbb{P}_{X \sim p_x(z)}(X >b) - \mathbb{P}_{Y \sim p_y(z)} (Y > b)|$$

To compute this loss, we replace the first expectation with the dataset of paired data $(z, y)$.
Then, for the inner expectation, we don't know $p_y(z)$, however it's note entire crucial to correctly sample $b$. Instead we can just sample from the marginal $p_y$ (marginalized over all inputs $z$). So that's practical as well.

Finally, we have the two probabilities inside the absolute value. The first probability is not too hard to compute, since we can sample arbitrarily many $X$ from our VAE by running the decoder multiple times.
The right hand side however is only given by the single sample $Y_1 \sim p_y(z)$.
To do something half reasonable here, we have to estimate this over a large batch.
We rely here on the fact that although in a single loss this will be a step function, we rely here on the fact that the model can't know "where" the step will occur and can only do what's probabilistically best.
So we choose a large batch and hope for the best.

Finally we note that we can replace the absolute value by somehting like the Huber loss which should still yield the correct result but have better convergence criteria.
"""

# ╔═╡ 22e2a229-d21d-45fc-9ca9-027eae4a75a6
md"""
## Some extra notes on this
### Using a sigmoid threshold
In order to get everything to be nicely differentiable, we need to make the $1[Y > b]$ property smooth, which we can do with some sigmoid of $Y-b$, perhaps with some temperature scaling $\tau$. Apparently we can set `tau = 0.1f0 * std(training_data)` to get reasonable results.


### Adding an auxiliary loss
We can try to further increase the signal of our approach by introducing an auxiliary loss, the Gaussian NLL loss we have used previously.
For this, we can just take the samples we're already predicting, fit a Gaussian to the first two moments (mean and std), and then use the NLL Loss. This should give us extra signal, especially early on.

Then we need to figure out a way to balance the losses still. We now have the original wasserstein loss, and the new auxiliary loss (and we also have a KL term for the VAE).
We can balance the auxiliary loss "with itself" by further learning a parameter which trades off that loss with its own uncertainty, something like

$$\exp{(-\log \lambda)} \mathcal{l}_{\rm NLL} + \log \lambda.$$

If $\mathcal{l}_{\rm NLL}$ is large the model needs to pick $\log \lambda$ large to downweigh that loss. However, picking the latter large is also penalized. So we have some "self-balancing behaviour".

Another thing here is to make sure the different losses have the same order to magnitude, but I think we can leave that for now.
"""

# ╔═╡ 17f41ed2-49bf-47ac-8fc4-6c7656231086
typst"""
Hello world. Here's an equation:
$
integral_(-inf)^(inf) |F^(-1)_x (u) - F^(-1)_y (u)| "du"
$
"""

# ╔═╡ e99981c3-e017-4eda-8936-db9d36c2d246
md"""
Hello world. Here's an equation:

$$\int_{-\infty}^{\infty} |F_x^{-1}(u) - F_y^{-1}(u)| du$$
"""

# ╔═╡ f2d2489f-fbcc-412d-a25f-74340e70f377
md"""
# Wasserstein VAE
"""

# ╔═╡ 8f92a602-b7ab-470c-828b-2deeeeae3865
begin
	@concrete struct VAEWithAuxLoss <: AbstractLuxContainerLayer{(:vae,)}
	    vae <: AbstractLuxLayer
	end
	
	function VAEWithAuxLoss(rng=Random.Xoshiro(); num_latent_dims::Int=3)
	    vae = VAE(rng; num_latent_dims)
	    return VAEWithAuxLoss(vae)
	end
	
	function LuxCore.initialparameters(rng::AbstractRNG, model::VAEWithAuxLoss)
	    return (vae=LuxCore.initialparameters(rng, model.vae), log_var_aux=zeros(Float32, 1))
	end
	function (vaex::VAEWithAuxLoss)(x, ps, st)
		retval, st = vaex.vae(x, ps.vae, st.vae)
		return (retval, (; vae=st))
	end
end

# ╔═╡ 3130a6c6-878e-4203-983b-75c64b48e5bf
function encode(vaex::VAEWithAuxLoss, x, ps, st)
	    out, st_vae = encode(vaex.vae, x, ps.vae, st.vae)
	    return out, (; vae=st_vae)
	end

# ╔═╡ fd65016d-14eb-4665-868a-0b59df6d2d95
function decode(vaex::VAEWithAuxLoss, z, ps, st)
	    x_rec, st_dec = decode(vaex.vae, z, ps.vae, st.vae)
	    return x_rec, (; vae=st_dec)
	end

# ╔═╡ 0966192d-2cee-4b55-8fea-4f521ddc9a4f
function loss_function(model, ps, st, (X, y)::Tuple{<:AbstractArray, <:AbstractVector{<:Number}}; n_samples=100)
    # y is now a vector of point observations, not distributions
    
    # Get latent parameters
    (_, μ, logσ²), st_encoder = model.encoder(X, ps.encoder, st.encoder) # or however you extract these
	st = (; encoder=st_encoder, st.decoder)
    
    # Sample multiple times from latent space
    batch_size = size(X, 2)  # or appropriate dimension
    #yhat = zeros_like(y, (n_samples, batch_size))

	yhat = map(1:n_samples) do i
		rng = Xoshiro(i)
		ε = randn_like(Lux.replicate(rng), μ)
        z = μ .+ sqrt.(exp.(logσ²)) .* ε
        
        # Decode to prediction
        yhat, st = decode(model, z, ps, st)  # or however decoding works
		yhat
	end |> xs->stack(xs; dims=2)

	# TODO: DOUBLE CHECK DIMENSIONS HERE
    # Compute empirical CDF for each observation
    # For each column (sample), compute fraction of predictions ≤ observation
    pit_values = vec(mean(soft_indicator(yhat, y), dims=1))  # Broadcasting comparison
    
    # Calibration loss on PIT values
    calib_loss = calibration_loss(pit_values)
    
    # KL divergence for VAE (unchanged)
    kldiv_loss = -sum(1 .+ logσ² .- μ .^ 2 .- exp.(logσ²)) / 2
    
    loss = calib_loss + kldiv_loss
    
    return loss, st, (; y, yhat, μ, logσ², calib_loss, kldiv_loss)
end

# ╔═╡ 3e570f5f-8812-4164-8f2f-f887e7cb953d
let
	testvae = VAE();
	ps, st = Lux.setup(Random.Xoshiro(), testvae)
	(; z), st = encode(testvae, rand(Float32, 1,10), ps, st)
	y = decode(testvae, z, ps, st)
	(; z, y)
end

# ╔═╡ 0ffe1fb3-566f-41fd-8c94-c845ebce8c18
abstract type LossFunction end

# ╔═╡ 8ee2f3bb-96a9-43de-9285-040b60ed5282
huberloss(x) = (abs2(x) <= 1 ? abs2(x)/2 : abs(x) - 1//2)

# ╔═╡ 326080b0-cfd7-4e9b-bc7c-609eab19a8b0
weightauxloss(m::VAE, l, ps, st) = l

# ╔═╡ 7050e240-868f-47e6-951e-9d4f2467960d
weightauxloss(m::VAEWithAuxLoss, l, ps, st) = 
  exp(-sum(ps.log_var_aux)) * l + sum(ps.log_var_aux)

# ╔═╡ 9b4ebdb4-bf74-490a-bebe-661a048a15d3
var(rand(3,4,5); dims=2, mean=mean(rand(3,4,5); dims=2)) |> xs->dropdims(xs; dims=2)

# ╔═╡ 2de7ea61-db27-45ad-9398-b2367c654241
begin
@kwdef @concrete struct WassersteinLossWAux <: LossFunction
  nsamples
  nb
  tau=0.1f0
  reducerfn=mean
end
getbvals(l::WassersteinLossWAux) = rand(reference_distribution, (l.nb, 1))
function (l::WassersteinLossWAux)(model, ps, st, (xin, ytrue))
	(; μ, logσ²), st = encode(model, xin, ps, st)
	kldivloss = l.reducerfn(
		@. (-1 + -logσ² + μ^2 + exp(logσ²)) / 2
	)
	
	σz = @. exp(0.5f0 * logσ²)
	ysamples = map(1:l.nsamples) do _
        # Sample from latent distribution
        ε = randn_like(μ)
        z = @. μ + σz * ε
        
        # Decode
		y, st = decode(model, z, ps, st)
        y
    end |> cols->stack(cols; dims=2)  # (outdim, nsamples, batchsize)

	μtotal, vartotal = let
		μtotal = mean(ysamples; dims=2)
		vartotal = var(ysamples; dims=2, mean=μtotal)
		(μtotal, vartotal)
	end .|> xs->dropdims(xs; dims=2)
	
	nllloss = l.reducerfn(
		@. 1//2 * ((ytrue - μtotal)^2 / (vartotal + eps(Float32)) + log(vartotal + eps(Float32)))
	)
	weighted_nllloss = weightauxloss(model, nllloss, ps, st)
	
	#bvals = 10*randn_like(μ, (l.nb))
	bvals = @ignore_derivatives let
		Lux.adapt(typeof(μ), getbvals(l))
	end
	bdirs = ones_like(ysamples, (size(ysamples, 1), l.nb))

	# Project samples
	@tullio xproj[ib, isample, ibatch] := ysamples[idim, isample, ibatch] * bdirs[idim, ib]
	# Project true values
	@tullio yproj[ib, ibatch] := ytrue[idim, ibatch] * bdirs[idim, ib]
	
	# For each threshold and batch element
	prob_x = mean(
		@. sigmoid((xproj - bvals) / l.tau);
	dims=2) |> xs->dropdims(xs; dims=2)
	prob_y = @. sigmoid((yproj - bvals) / l.tau)
	
	# Wasserstein distance
	wassersteinloss = l.reducerfn(
		@. huberloss(prob_x - prob_y)
	)

	totalloss = kldivloss + wassersteinloss + weighted_nllloss
	totalloss, st, (; μ, logσ², ysamples, bvals, bdirs, kldivloss, nllloss, weighted_nllloss, wassersteinloss)
end
end

# ╔═╡ 69df4fe9-4c43-403d-9bf0-cb1f66561abe
# ╠═╡ disabled = true
#=╠═╡
    #bvals = quantile.([Normal(0.0f0), ], rand_like(μ, (l.nb,)))
	#bvals = @ignore_derivatives let
	#	bvals = zeros_like(μ, (l.nb, 1))
	#	bvals .= getbvals(l)
	#	bvals
	#end
	#@tullio wassersteinvals[ib, isample, ibatch] := ysamples[idim, isample, ibatch] * bdirs[idim, ib]
	#wassersteinloss = l.reducerfn(
	#	@. sigmoid((wassersteinvals - bvals) / l.tau)
	#)
	#bvals = 0.0f0; bdirs = 0.0f0
	#wassersteinloss = 0.0f0

  ╠═╡ =#

# ╔═╡ 08a37c44-8e0f-40f4-9178-e0aba75abcb8
let
	dev = cdev
	testvae = VAE();
	ps, st = Lux.setup(Random.Xoshiro(), testvae) |> dev
	py = Normal(0.f0)
	ys = rand.(py, 1, 100) |> dev
	xs = zeros_like(ys)

	
	WassersteinLossWAux(;nsamples=8, nb=12)(testvae, ps, st, (xs, ys))
end

# ╔═╡ b5618124-d702-4dff-b58f-25830c9840c4
let
	dev = cdev
	testvae = VAE();
	ps, st = Lux.setup(Random.Xoshiro(), testvae) |> dev
	py = Normal(0.f0)
	ys = rand.(py, 1, 100) |> dev
	xs = zeros_like(ys)

	testvae(xs, ps, st)
end

# ╔═╡ a8a5aea3-aacf-4c41-bfe5-9960b9575ac0
(vae3, ps3, st3) = let dev = xdev
    batchsize=2^10
    num_latent_dims=3
    learning_rate = 1f-3
    num_epochs = 500
    seed=1

	rng = Xoshiro()
    Random.seed!(rng, seed)
	#ys = [Normal(0.f0)]
	#xs = rand.(ys) |> xs->reshape(xs, 1, :)
	#vae = VAE(; num_latent_dims)
	vae = VAEWithAuxLoss(; num_latent_dims)
	ps, st = Lux.setup(rng, vae) |> dev

	opt = Optimisers.Adam(; eta=learning_rate)
    train_state = Training.TrainState(vae, ps, st, opt)

	lossfn = WassersteinLossWAux(;nsamples=256, nb=256, 
		tau=0.1f0 * std(reference_distribution))

	start_time = time()

	for epoch in 1:num_epochs
        loss_total = 0.0f0
        total_samples = 0
		ys = rand.(reference_distribution, 1, batchsize) |> dev
		xs = zeros_like(ys)
		(_, loss, stats, train_state) = Training.single_train_step!(
			AD, lossfn, (xs, ys), train_state; return_gradients=Val(false)
		)
		#loss = lossfn(vae, ps, st, (xs, ys))

        loss_total += loss
        total_samples += size(xs) |> last

        if epoch % 50 == 0
            throughput = total_samples / (time() - start_time)
            @info epoch loss throughput stats.kldivloss stats.nllloss stats.weighted_nllloss stats.wassersteinloss log_var_aux=sum(train_state.parameters.log_var_aux)
        end
    end
	(; vae, ps=train_state.parameters, st=train_state.states)
end


# ╔═╡ ba0e3b25-8a58-4c4e-b793-8b1bbeecf4ff
with_theme(theme_latexfonts()) do 
	dev = Lux.get_device(ps3)
	samples = vae3(dev(rand(Float32, 1, 1_000)), ps3, st3)[1][1] |> cdev
	fig = Figure()
	ax = Axis(fig[1,1])
	hist!(ax, samples[:]; normalization=:pdf, label="Samples", bins=30)
	lines!(ax, range(extrema(samples[:])..., 100), x->pdf(reference_distribution, x); label=L"Reference $\mathcal{N}(0, 1)$", color=Makie.wong_colors()[2])
	axislegend()
	fig
end

# ╔═╡ ed35f1dc-1996-430a-aeb3-eb4682b36cd5
# ╠═╡ disabled = true
#=╠═╡
function Lux.LuxLib.Impl.matmul_cpu_fallback!(C::AbstractMatrix{Float64}, A::AbstractMatrix{AT}, B::AbstractMatrix{BT}, α::Number, β::Number
) where {T,AT,BT}
throw("")
end
  ╠═╡ =#

# ╔═╡ c0681ed7-3fef-44d5-a7c4-362b47bcd576
# ╠═╡ disabled = true
#=╠═╡
function loss_function(model, ps, st, (x_input, y_true); 
                       num_samples=32, 
                       num_thresholds=10, 
                       tau=0.1f0)
    
    # Encode to get latent distribution parameters
    (_, μ, logσ²), st_enc = model.vae.encoder(x_input, ps.vae.encoder, st.vae.encoder)
    
    # Generate multiple samples from the VAE
    σ = exp.(0.5f0 .* logσ²)
    x_samples = []
    
    for i in 1:num_samples
        # Sample from latent distribution
        ε = randn(eltype(μ), size(μ))
        z = μ .+ σ .* ε
        
        # Decode
        x_pred, st_dec = model.vae.decoder(z, ps.vae.decoder, st.vae.decoder)
        push!(x_samples, x_pred)
    end
    
    # Concatenate samples: shape (output_dim, num_samples * batch_size)
    x_samples_cat = reduce(hcat, x_samples)
    
    # Compute Sliced Wasserstein loss (1D version)
    sw_loss = 0.0f0
    
    # Sample thresholds from the data
    y_flat = vec(y_true)
    x_flat = vec(x_samples_cat)
    
    for _ in 1:num_thresholds
        # Sample threshold from true data
        b = rand(y_flat)
        
        # Compute P(X > b) using sigmoid approximation
        prob_x = mean(σ.((x_flat .- b) ./ tau))
        
        # Compute P(Y > b) using sigmoid approximation  
        prob_y = mean(σ.((y_flat .- b) ./ tau))
        
        # Accumulate absolute difference
        sw_loss += abs(prob_x - prob_y)
    end
    sw_loss = sw_loss / num_thresholds
    
    # Auxiliary Gaussian NLL loss
    # Compute empirical mean and variance over samples
    batch_size = size(y_true, 2)
    x_reshaped = reshape(x_samples_cat, :, num_samples, batch_size)
    
    μ_emp = mean(x_reshaped, dims=2)[:, 1, :]  # Shape: (output_dim, batch_size)
    σ²_emp = var(x_reshaped, dims=2)[:, 1, :] .+ 1f-6  # Add small epsilon for stability
    
    # Gaussian NLL
    aux_loss = mean((y_true .- μ_emp).^2 ./ (2f0 .* σ²_emp) .+ 0.5f0 .* log.(σ²_emp))
    
    # Combine with learned uncertainty weighting
    log_var = ps.log_var_aux[1]
    total_loss = sw_loss + exp(-log_var) * aux_loss + log_var
    
    # Update state
    st_new = (vae=(encoder=st_enc, decoder=st_dec),)
    
    # Stats for monitoring
    stats = (
        sw_loss=sw_loss,
        aux_loss=aux_loss,
        learned_weight=exp(-log_var),
    )
    
    return total_loss, st_new, stats
end
  ╠═╡ =#
