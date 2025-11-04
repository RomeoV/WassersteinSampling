"""
VAE model definitions including encoder, decoder, and main VAE structure.
"""

using Lux
using Random
using ConcreteStructs

"""
Encoder network that maps input to latent distribution parameters.

# Arguments
- `rng`: Random number generator
- `num_latent_dims`: Dimension of latent space
- `intermediate_dims`: List of layer dimensions (input=>output pairs)
"""
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

"""
Decoder network that maps latent variables to output.

# Arguments
- `num_latent_dims`: Dimension of latent space (not used but kept for API consistency)
- `intermediate_dims`: List of layer dimensions (input=>output pairs)
"""
function decoder(;
    num_latent_dims::Int,
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

"""
Variational Autoencoder (VAE) combining encoder and decoder.
"""
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

"""
Encode input to latent space.
"""
function encode(vae::VAE, x, ps, st)
    (z, μ, logσ²), st_enc = vae.encoder(x, ps.encoder, st.encoder)
    return (; z, μ, logσ²), (; encoder=st_enc, st.decoder)
end

"""
Decode latent variables to output.
"""
function decode(vae::VAE, z, ps, st)
    x_rec, st_dec = vae.decoder(z, ps.decoder, st.decoder)
    return x_rec, (; decoder=st_dec, st.encoder)
end

"""
VAE with auxiliary loss parameter for learned loss weighting.
"""
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
    return retval, (; vae=st)
end

function encode(vaex::VAEWithAuxLoss, x, ps, st)
    out, st_vae = encode(vaex.vae, x, ps.vae, st.vae)
    return out, (; vae=st_vae)
end

function decode(vaex::VAEWithAuxLoss, z, ps, st)
    x_rec, st_dec = decode(vaex.vae, z, ps.vae, st.vae)
    return x_rec, (; vae=st_dec)
end
