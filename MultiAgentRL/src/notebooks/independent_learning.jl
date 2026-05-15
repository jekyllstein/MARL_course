### A Pluto.jl notebook ###
# v0.20.24

using Markdown
using InteractiveUtils

# ╔═╡ 6fa3e0d9-a755-4cec-974a-7a5c5e762ff6
using PlutoDevMacros

# ╔═╡ cd9d50a8-032d-11f1-a111-edabac2b8ec3
using DataFrames, CSV, JuMP, HiGHS, DataStructures, Dates

# ╔═╡ b97e16fa-db92-4fd2-a39f-f6c06595d7b1
# ╠═╡ skip_as_script = true
#=╠═╡
begin
	using PlutoUI, PlutoPlotly, PlutoProfile, BenchmarkTools, LaTeXStrings, HypertextLiteral
	TableOfContents()
end
  ╠═╡ =#

# ╔═╡ 50abaf3a-af49-438b-9fcb-5d36d15aac9f
@only_in_nb begin
	@frompackage @raw_str(joinpath(@__DIR__, "..", "RL_Module")) import *
	include(joinpath(@__DIR__, "multi_agent_types.jl"))
	include(joinpath(@__DIR__, "joint_action_learning.jl"))
end

# ╔═╡ e20ed61f-887c-4f2e-b4fd-4ae6b8dfd874
md"""
# Stochastic Game Utilities
"""

# ╔═╡ d496f057-85d4-42ca-b97d-df333284195e
function update_random_dist!(v::Vector{T}) where T<:Real
	l = length(v)
	p = one(T) / l
	v .= p
end

# ╔═╡ be9b8bc5-68ae-4033-b91f-97fc7bcb52be
function make_random_policies(game::StateStochasticGame{T, S, A, N, P, F1, F2}) where {T<:Real, S, A, N, P, F1, F2}
	ntuple(N) do i
	 	s -> rand(1:length(game.agent_actions[i]))
	 end 
end

# ╔═╡ df3c233d-791e-4f0f-9163-ca63bcc455d0
function sample_joint_action(πs::NTuple{N, F}, s::S) where {S, N, F<:Function} 
	NTuple{N, Int64}(πs[n](s) for n in 1:N)
end

# ╔═╡ 548659da-1099-4f2f-ada7-c943cea23a66
function (ptf::AbstractGameTransition{T, N})(s, πs::NTuple{N, F}) where {T<:Real, N, F<:Function}
	a = sample_joint_action(πs, s)
	(r, s′) = ptf(s, a)
	return (r, s′, a)
end

# ╔═╡ e6f47cea-5b67-467d-a7cb-f2fc30f8e358
begin
	function TabularRL.runepisode!((states, joint_actions, rewards)::Tuple{Vector{S}, Vector{NTuple{N, Int64}}, R}, game::StateStochasticGame{T, S, A, N, P, F1, F2}; s0::S = game.initialize_state(), πs::NTuple{N, F3} = make_random_policies(game), a0::NTuple{N, Int64} = sample_joint_action(πs, s0), max_steps = Inf) where {T<:Real, S, A, P, F1<:Function, F2<:Function, F3<:Function, N, R<:Union{Vector{NTuple{N, T}}, Vector{T}}}
		s = s0
		l = length(states)
		@assert l == length(joint_actions) == length(rewards)
	
		function add_value!(v, x, i) 
			if i > l
				push!(v, x)
			else
				v[i] = x
			end
		end 
		add_value!(states, s, 1)
		a = a0
		(r, s′) = game.ptf(s, a0)
		add_value!(joint_actions, a, 1)
		add_value!(rewards, r, 1)
		step = 2
		sterm = s
		if game.isterm(s′)
			sterm = s′
		else
			sterm = s
		end
		s = s′
	
		#note that the terminal state will not be added to the state list
		while !game.isterm(s) && (step <= max_steps)
			add_value!(states, s, step)
			(r, s′, a) = game.ptf(s, πs)
			add_value!(joint_actions, a, step)
			add_value!(rewards, r, step)
			s = s′
			step += 1
			if game.isterm(s′)
				sterm = s′
			end
		end
		return states, joint_actions, rewards, sterm, step-1
	end

	initialize_episode_rewards(ptf::AbstractStateGameTransition{T, S, F, N}) where {T<:Real, S, F, N} = Vector{NTuple{N, T}}()
	initialize_episode_rewards(ptf::StateCommonRewardGameTransitionDeterministic{T, S, F, N}) where {T<:Real, S, F, N} = Vector{T}()
	initialize_episode_rewards(ptf::StateZeroSumGameTransitionDeterministic{T, S, F}) where {T<:Real, S, F} = Vector{T}()
	
	function TabularRL.runepisode(game::StateStochasticGame{T, S, A, N, P, F1, F2}; kwargs...) where {T<:Real, S, A, P<:AbstractStateGameTransition, F1, F2, N}
		states = Vector{S}()
		actions = Vector{NTuple{N, Int64}}()
		rewards = initialize_episode_rewards(game.ptf)
		runepisode!((states, actions, rewards), game; kwargs...)
	end
end

# ╔═╡ 3d5a1285-d243-45ad-a2b9-3edd6b2b577b
begin
	ReinforcementLearning.form_feature_matrix(game::StateStochasticGame{T, S, A, N, P, F1, F2}, feature_vector::Vector{T}, batch_size::Integer) where {T<:Real, S, A, N, P, F1, F2} = zeros(T, length(feature_vector), batch_size)

	function ReinforcementLearning.form_feature_matrix(game::StateStochasticGame{T, S, A, N, P, F1, F2}, feature_vector::V, batch_size::Integer)  where {T<:Real, S, A, N, P, F1, F2, V<:AbstractBinaryFeatures}
		output = Vector{V}(undef, batch_size)
		for i in 1:batch_size
			output[i] = deepcopy(feature_vector)
		end
		return output
	end
end

# ╔═╡ 40a4c2a0-47e7-4f32-9927-683304c1c66e
begin
	function ReinforcementLearning.form_action_value_kwargs(num_actions::Integer, feature_vector, parameters::Array{T, N}) where {T<:Real, N} 
		(action_values = zeros(T, num_actions), feature_vector = deepcopy(feature_vector), parameters = parameters)
	end

	function ReinforcementLearning.form_action_value_kwargs(num_actions::Integer, feature_vector, parameters1::Array{T, N}, parameters2::Array{T, N}) where {T<:Real, N} 
		(action_values1 = zeros(T, num_actions), action_values2 = zeros(T, num_actions), feature_vector = deepcopy(feature_vector), parameters1 = parameters1, parameters2 = parameters2)
	end

	function ReinforcementLearning.form_action_value_kwargs(num_actions::Integer, feature_vector, parameters::FCANNParams{T}) where {T<:Real} 
		(action_values = zeros(T, num_actions), feature_vector = deepcopy(feature_vector), parameters = parameters, activations = FCANN.form_activations(parameters.weights[1]))
	end

	function ReinforcementLearning.form_action_value_kwargs(num_actions::Integer, feature_vector::Vector{T}, cpu_params::FCANNParams{T}, gpu_params::FCANNParamsGPU) where {T<:Real} 
		activations = FCANN.form_activations(gpu_params.weights[1])
		d_x = FCANN.cuda_allocate(feature_vector)
		function cleanup_vars()
			FCANN.clear_gpu_data(activations)
			FCANN.clear_gpu_data([d_x])
		end
		gpu_kwargs = (activations = activations, d_x = d_x, cleanup_vars = cleanup_vars)
		(action_values = zeros(T, num_actions), feature_vector = deepcopy(feature_vector), parameters = cpu_params, activations = FCANN.form_activations(cpu_params.weights[1]), gpu_kwargs = gpu_kwargs)
	end

	function ReinforcementLearning.form_action_value_kwargs(num_actions::Integer, feature_vector, parameters1::FCANNParams{T}, parameters2::FCANNParams{T}) where {T<:Real} 
		(action_values1 = zeros(T, num_actions), action_values2 = zeros(T, num_actions), feature_vector = deepcopy(feature_vector), parameters1 = parameters1, parameters2 = parameters2, activations = FCANN.form_activations(parameters1.weights[1]))
	end
end

# ╔═╡ ecc7b7e2-deca-4cea-ab4f-3fadc7ba1f3d
begin
	function form_agent_value_functions(game::StateStochasticGame{T, S, A, N, P, F1, F2}, update_feature_vectors!::NTuple{N, Function}, update_action_values!::NTuple{N, Function}, feature_vectors::NTuple{N, V}, value_params::NTuple{N, W}) where {T<:Real, S, A, N, P, F1, F2, V, W}
		q̂s = ntuple(N) do i
			function q̂(s::S; action_values::Vector{T} = zeros(T, length(game.agent_actions[i])), feature_vector::V = deepcopy(feature_vectors[i]), parameters::W = value_params[i], kwargs...)
				update_feature_vectors![i](feature_vector, s)
				maxq, i_a_max = update_action_values![i](action_values, feature_vector, parameters; kwargs...)
				(action_values = action_values, maximizing_action = i_a_max, maximizing_value = maxq)
			end
		end

		form_kwargs = ntuple(N) do i
			form_kwargs() = ReinforcementLearning.form_action_value_kwargs(length(game.agent_actions[i]), feature_vectors[i], value_params[i])
		end

		return q̂s, form_kwargs
	end
end

# ╔═╡ ce02448f-19b9-4426-b2b8-1db1aa6161f0
function extract_joint_policies(game::StateStochasticGame{T, S, A, N, P, F1, F2}, value_functions::NTuple{N, F}; ϵ::T = zero(T)) where {T<:Real, S, A, N, F<:Function, P, F1, F2}
	if iszero(ϵ)
		ntuple(N) do i
			f(s; kwargs...) = value_functions[i](s; kwargs...).maximizing_action
		end
	else
		ntuple(N) do i
			f(s; kwargs...) = rand() ≤ ϵ ? rand(eachindex(game.agent_actions[i])) : value_functions[i](s; kwargs...).maximizing_action
		end
	end
end

# ╔═╡ ae63619a-3136-4ee6-b262-2d9242262bd2
md"""
# Value-Based Learning
"""

# ╔═╡ bed9f5da-e034-4d65-9019-6ed4186a7404
md"""
## Independent DQN
"""

# ╔═╡ 51e2b6e9-21a6-44df-adf1-52d222e2acde
md"""
### Utility Functions
"""

# ╔═╡ 0f8158bc-d373-43e0-80e5-197cd85e450b
function check_bad_vector(x::BinaryFeatureVector)
	any(iszero, view(x.active_features, 1:x.num_features))
end

# ╔═╡ 23ae726b-fd9b-499f-b8e8-8b243bbe68f7
#case where the indices span multiple replay buffers for experience sharing
function ReinforcementLearning.update_nstep_returns!(targets::Vector{T}, target_const::Vector{T}, feature_matrix, γ::T, replay_buffers::NTuple{N, CircularBuffer}, batch_inds::Vector{Int64}, nstep::Integer) where {N, T<:Real}
	l = length(first(replay_buffers)) - nstep
	for i in eachindex(batch_inds)
		j = batch_inds[i]
		buffer_select = ceil(Int64, j / l)
		buffer_index = iszero(j % l) ? l : j % l 
		(x, i_a, r, x′, terminated) = replay_buffers[buffer_select][buffer_index]
		g = r
		k = buffer_index+1
		while !terminated && (k <= buffer_index+nstep)
			(x, i_a, r, x′, terminated) = replay_buffers[buffer_select][k]
			g += r * γ^(k - buffer_index)
			k += 1
		end
		ReinforcementLearning.update_feature_matrix!(feature_matrix, x′, i)
		#populate target values with the reward 
		targets[i] = g
		target_const[i] = terminated ? zero(T) : γ^(k-buffer_index) #update constant to be used to multiply the target values.  Depending on the number of future steps, the discount rate is used but if the N-step window ends in termination then the output value is ignored
	end
end

# ╔═╡ 8aed40da-2cc8-4f02-8660-7f8c6190a624
#case where the indices span multiple replay buffers for experience sharing
begin
	#linear function approximation with a binary feature vector
	function ReinforcementLearning.update_targets!(targets::Vector{T}, γ::T, replay_buffers::NTuple{N, CircularBuffer}, batch_inds::Vector{Int64}, nstep::Integer, target_const::Vector{T}, target_params::Matrix{T}, feature_matrix::Vector{V}, action_values::Vector{T}, output_matrix::Matrix{T}) where {T<:Real, N, V<:AbstractBinaryFeatures}
		l = length(first(replay_buffers)) - nstep
		#update feature matrix with replay buffer
		for i in eachindex(batch_inds)
			j = batch_inds[i]
			buffer_select = ceil(Int64, j / l)
			buffer_index = iszero(j % l) ? l : j % l 
			(x, i_a, r, x′, terminated) = replay_buffers[buffer_select][buffer_index]
			g = r
			k = buffer_index+1
			while !terminated && (k <= buffer_index+nstep)
				(x, i_a, r, x′, terminated) = replay_buffers[buffer_select][k]
				g += r * γ^(k-buffer_index)
				k += 1
			end
			targets[i] = g
			if !terminated
				ReinforcementLearning.update_linear_action_values!(action_values, x′, target_params)
				targets[i] += γ^(k-buffer_index) * maximum(action_values)
			end
		end
	end

	#linear function approximation with a binary feature vector
	function ReinforcementLearning.update_targets!(targets::Vector{T}, γ::T, replay_buffers::NTuple{N, CircularBuffer}, batch_inds::Vector{Int64}, nstep::Integer, target_const::Vector{T}, target_params::Matrix{T}, value_params::Matrix{T}, feature_matrix::Vector{V}, action_values::Vector{T}, target_output::Matrix{T}, value_output::Matrix{T}) where {N, T<:Real, V<:AbstractBinaryFeatures}
		l = length(first(replay_buffers)) - nstep
		#update feature matrix with replay buffer
		for i in eachindex(batch_inds)
			j = batch_inds[i]
			buffer_select = ceil(Int64, j / l)
			buffer_index = iszero(j % l) ? l : j % l 
			(x, i_a, r, x′, terminated) = replay_buffers[buffer_select][buffer_index]
			g = r
			k = buffer_index+1
			while !terminated && (k <= buffer_index+nstep)
				(x, i_a, r, x′, terminated) = replay_buffers[buffer_select][k]
				g += r * γ^(k-buffer_index)
				k += 1
			end
			targets[i] = g
			if !terminated
				ReinforcementLearning.update_linear_action_values!(action_values, x′, value_params)
				i_a_max = argmax(action_values)
				ReinforcementLearning.update_linear_action_values!(action_values, x′, target_params)
				targets[i] += γ^(k-buffer_index) * action_values[i_a_max]
			end
		end
	end
	
end

# ╔═╡ 7eefc3ad-a4ce-4ba6-9393-86b29ab5ca40
md"""
### Algorithm
"""

# ╔═╡ 141cf16f-2994-492a-a526-4d29fd944de5
function independent_dqn!(value_params::NTuple{N, Q}, target_params::NTuple{N, Q}, game::StateStochasticGame{T, S, A, N, P, F1, F2}, γ::T, max_episodes::Integer, max_steps::Integer, feature_vectors::NTuple{N, V}, update_feature_vectors!::NTuple{N, Function}, update_action_values!::NTuple{N, Function}, update_value_gradients!::NTuple{N, Function}; target_args::NTuple{N, Tuple} = ntuple(i -> (), N), α = one(T)/10, ϵ = one(T) / 10, buffer_size::Integer = 10_000, batch_size::Integer = 512, target_update_interval::Integer = 100, α_decay = one(T), decay_step = typemax(Int64), save_step_rewards::Bool = false, use_double_q::Bool = false, nstep::Integer = 0, ∇q̂s::NTuple{N, Q} = deepcopy(value_params), share_experience::Bool = false, kwargs...) where {Q, T<:Real, S, A, N, P<:AbstractStateGameTransition, F1<:Function, F2<:Function, V}
	#with experience sharing, all feature vectors must be identical
	share_experience && @assert all(i -> length(feature_vectors[i]) == length(feature_vectors[1]), 2:N)
	
	#initialize memory
	action_values = ntuple(i -> zeros(T, length(game.agent_actions[i])), N)
	policies = deepcopy(action_values)
	replay_buffers = ntuple(i -> CircularBuffer{Tuple{V, Int64, T, V, Bool}}(buffer_size), N)
	# @info "replay buffers are of type $(typeof(replay_buffers))"
	targets = Vector{T}(undef, batch_size)
	target_const = Vector{T}(undef, batch_size)
	batch_inds = Vector{Int64}(undef, batch_size)
	feature_matrices = ntuple(i -> ReinforcementLearning.form_feature_matrix(game, feature_vectors[i], batch_size), N)
	output_matrices = ntuple(i -> zeros(T, batch_size, length(game.agent_actions[i])), N)
	output_args = !use_double_q ? ntuple(i -> (output_matrices[i],), N) : ntuple(i -> (output_matrices[i], copy(output_matrices[i])), N)
	param_args = !use_double_q ? ntuple(i -> (target_params[i],), N) : ntuple(i -> (target_params[i], value_params[i]), N)
	output_inds = Vector{Int64}(undef, batch_size)
	feature_vectors2 = deepcopy(feature_vectors)

	#initialize episode
	s = game.initialize_state()

	joint_actions = ntuple(N) do i
		update_feature_vectors![i](feature_vectors[i], s)
		update_action_values![i](action_values[i], feature_vectors[i], value_params[i])
		policies[i] .= action_values[i]
		make_ϵ_greedy_policy!(policies[i]; ϵ = ϵ)
		sample_action(policies[i])
	end

	ep = 1
	step = 1
	eprewards = zeros(T, N)
	episode_rewards = ntuple(i -> Vector{T}(), N)
	episode_steps = Vector{Int64}()
	step_rewards = ntuple(i -> Vector{T}(), N)
	decay = one(T)

	while (ep ≤ max_episodes) && (step ≤ max_steps)
		(rewards, s′) = game.ptf(s, joint_actions)
		
		terminated = game.isterm(s′)

		for i in 1:N
			update_feature_vectors![i](feature_vectors2[i], s′)
			r = rewards[i]
			i_a = joint_actions[i]
			# check_bad_vector(feature_vectors[i]) && @info "Found bad feature vector $(feature_vectors[i]) on step $step for state $s"
			# check_bad_vector(feature_vectors2[i]) && @info "Found bad feature vector $(feature_vectors2[i]) on step $step for state $s′"
			push!(replay_buffers[i], (deepcopy(feature_vectors[i]), i_a, r, deepcopy(feature_vectors2[i]), terminated))
			save_step_rewards && push!(step_rewards[i], r)	
			eprewards[i] += r
		end

		if terminated
			s′ = game.initialize_state()
			for i in 1:N
				update_feature_vectors![i](feature_vectors2[i], s′)
				push!(episode_rewards[i], eprewards[i])	
				eprewards[i] = zero(T)
			end
			push!(episode_steps, step)
			ep += 1
		end

		joint_actions′ = ntuple(N) do i
			update_action_values![i](action_values[i], feature_vectors2[i], value_params[i])
			policies[i] .= action_values[i]
			ReinforcementLearning.make_ϵ_greedy_policy!(policies[i]; ϵ = ϵ)
			sample_action(policies[i])
		end

		# @info "Sampled the following joint actions: $joint_actions′"

		decay *= (step > decay_step)*α_decay + (step <= decay_step)

		buffer_effective_length = length(replay_buffers[1]) - nstep

		if share_experience
			buffer_effective_length *= N
		end

		#only perform gradient parameter update once the replay buffer is large enough to fill up an entire batch
		if buffer_effective_length ≥ batch_size
			# @info "Performing batch gradient updates"
			for i in 1:N
				# @info "replay_buffer is of type $(typeof(replay_buffer))"
				# @info "Performing batch gradient update for agent $i"
				if !share_experience
					ReinforcementLearning.update_batch_inds!(batch_inds, step, buffer_size, nstep)
				else #with experience sharing we effectively have N times more examples
					ReinforcementLearning.update_batch_inds!(batch_inds, N*step, N*buffer_size, N*nstep)
				end
				# @info "Selecting batch indices: $batch_inds"
				# @info param_args[i]
				# @info "Updating target values with $nstep step returns output arguments of $(output_args[i]) and the replay buffer $(replay_buffer)"

				# @info "On step $step feature matrix is $(feature_matrices[i])"

				# any(check_bad_vector, [replay_buffer[i][1] for i in batch_inds]) && @info "Found bad feature vector"
				# any(check_bad_vector, [replay_buffer[i][4] for i in batch_inds]) && @info "Found bad feature vector"
				ReinforcementLearning.update_targets!(targets, γ, share_experience ? replay_buffers : replay_buffers[i], batch_inds, nstep, target_const, param_args[i]..., feature_matrices[i], action_values[i], output_args[i]..., target_args[i]...)
				# update_targets_test!(targets, γ, replay_buffer, batch_inds, nstep, target_const, param_args[i]..., feature_matrices[i], action_values[i], output_args[i]..., target_args[i]...)

				# @info "Updating feature matrix"
				#update_feature_matrix
				for j in eachindex(batch_inds)
					(x_k, i_a_k, _, _, _) = if !share_experience
						replay_buffers[i][batch_inds[j]]
					else
						l = length(first(replay_buffers)) - nstep
						raw_buffer_index = batch_inds[j]
						buffer_select = ceil(Int64, raw_buffer_index / l)
						buffer_index = iszero(raw_buffer_index % l) ? l : raw_buffer_index % l
						replay_buffers[buffer_select][buffer_index]
					end
						
					ReinforcementLearning.update_feature_matrix!(feature_matrices[i], x_k, j)
					output_inds[j] = i_a_k
				end

				update_value_gradients![i](∇q̂s[i], value_params[i], targets, output_inds, feature_matrices[i], output_matrices[i])
				ReinforcementLearning.update_params_with_gradient!(value_params[i], α*decay, ∇q̂s[i])

				if iszero(step % target_update_interval)
					copy!(target_params[i], value_params[i])
				end
			end
		end

		# @info "Preparing for next step"
		# @info "Updating state to $s′"
		s = s′
		# @info "Updating feature vector to $feature_vector2"
		feature_vectors = deepcopy(feature_vectors2)
		# @info "Updating joint actions to $joint_actions′"
		joint_actions = joint_actions′
		step += 1
		# @info "One step $step of $max_steps"
	end

	for i in 1:N
		ReinforcementLearning.cleanup_gradient!(∇q̂s[i])
	end
	
	q̂s, form_kwargs = form_agent_value_functions(game, update_feature_vectors!, update_action_values!, feature_vectors, value_params)

	return (value_functions = q̂s, episode_rewards = episode_rewards, episode_steps = episode_steps, final_parameters = deepcopy(value_params), form_kwargs = form_kwargs)
end

# ╔═╡ 90713eff-9d6b-476c-a41d-45819476b695
begin
	convert_to_ntuple(x::Function, N::Integer) = ntuple(i -> x, N)
	convert_to_ntuple(x, n::Integer) = ntuple(i -> deepcopy(x), n)
	convert_to_ntuple(x::NTuple{N, T}, n::Val{N}) where {N, T} = x
	convert_to_ntuple(x::NTuple{N, T}, n::Integer) where {N, T} = convert_to_ntuple(x, Val(n))
end

# ╔═╡ 1af06b2c-ee8a-497e-8d0e-e6a6127e48bf
md"""
### Linear Approximation
"""

# ╔═╡ 1f8873d6-6f7e-449e-b2ea-17fa489ac2a5
begin
	independent_dqn_linear(game::StateStochasticGame{T, S, A, N, P, F1, F2}, γ::T, max_episodes::Integer, max_steps::Integer, feature_vectors::NTuple{N, V}, update_feature_vectors!::NTuple{N, Function}; init_value::T = zero(T), value_params::NTuple{N, Matrix{T}} = ntuple(i -> initialize_linear_parameters(feature_vectors[i], length(game.agent_actions[i]), init_value), N), target_params::NTuple{N, Matrix{T}} = deepcopy(value_params), kwargs...) where {T<:Real, S, A, N, P, F1<:Function, F2<:Function, V} = independent_dqn!(value_params, target_params, game, γ, max_episodes, max_steps, feature_vectors, update_feature_vectors!, convert_to_ntuple(ReinforcementLearning.update_linear_action_values!, N), convert_to_ntuple(ReinforcementLearning.update_linear_value_gradient!, N); kwargs...)
	
	independent_dqn_linear(game::StateStochasticGame{T, S, A, N, P, F1, F2}, γ::T, max_episodes::Integer, max_steps::Integer, feature_vector, update_feature_vector!::Function; kwargs...) where {T<:Real, S, A, N, P, F1<:Function, F2<:Function} = independent_dqn_linear(game, γ, max_episodes, max_steps, convert_to_ntuple(feature_vector, N), convert_to_ntuple(update_feature_vector!, N); kwargs...)
end	

# ╔═╡ 5f7d0c18-ca88-4057-a6e6-632f562c720c
md"""
### Non-Linear Approximation
"""

# ╔═╡ 311ae6ed-9594-49dd-8e4e-bce501b9c4b0
begin
	function independent_dqn_fcann(game::StateStochasticGame{T, S, A, N, P, F1, F2}, γ::T, max_episodes::Integer, max_steps::Integer, feature_vectors::NTuple{N, V}, update_feature_vectors!::NTuple{N, Function}, hidden_layers::Vector{Int64}; batch_size::Integer = 512, reslayers::Int64 = 0, use_μP::Bool = true, value_params::NTuple{N, FCANNParams{T}} = ntuple(i -> initialize_fcann_params(feature_vectors[i], hidden_layers, length(game.agent_actions[i]), reslayers, use_μP), N), target_params::NTuple{N, FCANNParams{T}} = deepcopy(value_params), dropout = zero(T), activation_list = fill(true, length(hidden_layers)), l2 = zero(T), use_gpu::Bool = false, kwargs...) where {T<:Real, S, A, N, P, F1<:Function, F2<:Function, V} 
		setups = ntuple(i -> setup_fcann_action_value_arguments(value_params[i], target_params[i], batch_size, l2, dropout, use_μP, activation_list; use_gpu = use_gpu), N)
		
		independent_dqn!(value_params, target_params, game, γ, max_episodes, max_steps, feature_vectors, update_feature_vectors!, ntuple(i -> setups[i].update_action_values!, N), ntuple(i -> setups[i].update_value_gradient!, N); target_args = ntuple(i -> setups[i].target_args, N), batch_size = batch_size, kwargs...)
	end

	independent_dqn_fcann(game::StateStochasticGame{T, S, A, N, P, F1, F2}, γ::T, max_episodes::Integer, max_steps::Integer, feature_vector, update_feature_vector!::Function, hidden_layers::Vector{Int64}; kwargs...) where {T<:Real, S, A, N, P, F1<:Function, F2<:Function} = independent_dqn_fcann(game, γ, max_episodes, max_steps, convert_to_ntuple(feature_vector, N), convert_to_ntuple(update_feature_vector!, N), hidden_layers; kwargs...)
end

# ╔═╡ 1c4ad5ee-e29f-4550-8aa6-6d002c304338
md"""
## Value Decomposition Networks

For common reward games, we could solve the task with a central value function.  For these problems, the goal matches that of single agent reinforcement learning: to maximize the discounted reward sum over the course of an episode.  A central value function that solves this task would need to learn an action value function over the space of joint action values for all of the agents.  The goal of value decomposition is to learn a simpler set of value functions on a per agent basis that still matches the properties of the true joint action-value function.

Linear value decomposition is an approach where the joint action-value function is modeled as a sum of individual agent value functions over the actions of just that agent.  

```math
\mathcal{L}(\theta) \leftarrow \frac{1}{B} \sum_{k=1}^B \left ( y^k - \sum_{i \in I} Q(h_i^k, a_i^k; \theta_i) \right )^2

```

The loss function is written in terms of all of the parameters $\theta$ composed of the parameters for the individual value functions.  If we calculate the gradient for each set of parameters individually, we need to use target values that respect the shared loss function since the forward pass will only compute the output of one of the value functions.  The other parameters only affect the gradient through the target value and the overall delta value.  We can replicate this gradient by calculating the delta value first for each example and then using the gradient of the output at each action index multiplied by the delta.

```math
\frac{\partial \mathcal{L}(\theta)}{\partial \theta_i} = \frac{2}{B} \sum_{k=1}^B \left ( y^k - \sum_{i \in I} Q(h_i^k, a_i^k; \theta_i) \right ) \left ( - \frac{\partial Q(h_i^k, a_i^k; \theta_i)}{\partial \theta_i} \right )

```
"""

# ╔═╡ 7c504736-ddbe-4a0a-af0b-fc7306693269
md"""
### Individual-Global-Max Property

The greedy joint actions with respect to the centralized action-value function should be equal to the joint actions composed of the greedy individual actions of all agents that maximize the respective individual utilities.  Below we define what is meant by the greedy action selection in each scenario.

```math
A^*(h, z; \theta) = \mathrm{argmax}_{a \in A} Q(h, z, a; \theta)

A^*_i(h_i; \theta_i) = \mathrm{argmax}_{a_i \in A_i} Q(h_i, a_i; \theta_i)
```

Note that $Q(h, z, a; \theta)$ is a function of the joint histories and actions of all agents whereas $Q(h_i, a_i; \theta_i)$ is a utility function over the history and actions for an individual agent.  The IGM property is satisfied if the following holds for all full histories $\hat h$ with joint-observation histories $h = \sigma(\hat h)$, individual observation histories, $h_i = \sigma_i (\hat h)$, and centralized information $z$:

```math
\forall a = (a_1, \dots, a_n) \in A : a \in A^* (h, z; \theta) \iff \forall i \in I: a_i \in A_i^* (h_i; \theta_i)
```

Each agent can follow the greedy policy with respect to its individual utility function for decentralized execution, and all agents together will select the greedy *joint* action with respect to the decomposed centralized action-value function.  Also, the greedy joint action with respect to the decomposed centralized action-value function, needed to compute the target value during training, can be efficiently obtained by computing the greedy individual actions of all agents with respect to their individual utility.  If the individual utility functions satisfy the IGM property for the centralized action-value function, we also say that the utility functions *factorize* the centralized value function.

Individual utility functions are conditioned on the observation history of that agent, so if it correctly factorizes the centralized value function, then it should predict the contribution of that agent to the reward.  These functions can then be useful in solving the multi agent credit assignment problem.  Note that for some environments there may not exist a decomposition that satisfies the IGM property in all cases, but often we can learn a decomposition that works in most cases.
"""

# ╔═╡ 745b4c73-7245-4d0a-ad21-52de88fedb9c
md"""
### Linear Value Decomposition

A simple method of decomposing $Q(h, z, a; \theta)$ that satisfies the IGM property is to assume a linear decomposition of common rewards, that is, the sum of the individual utilities of agents equals the common reward

```math
r_t = \bar r^t_1 + \cdots \bar r^t_n
```

where $\bar r^t_i$ denotes the utility of agent $i$ at time step $t$.  The bar over the reward symbol denotes that these utilities are obtained by the decomposition, and do not represent true rewards received by the environment.  Using this assumption, the centralized action-value function of agents can be decomposed as follows, with the expectation being defined over the probability distribution of full histories $\hat h^t$ defined in Equation 4.3

Linear decomposition defines the approach of *value decomposition networks* (VDN).  VDN maintains a replay buffer $\mathcal{D}$ containing the experience of all agents and jointly optimizes the loss defined below over the approximate centralized value function for all agents.  Here the centralized value function is factorized as a sum of utility functions for each agent.

```math
\mathcal{L}(\theta) = \frac{1}{B} \sum_{(h^t, a^t, r^t, h^{t+1}) \in \mathcal{B}} \left ( r^t + \gamma \max_{a \in A} Q(h^{t+1}, a; \bar \theta) - Q(h^t, a^t; \theta) \right )^2 \\

Q(h^t, a^t; \theta) = \sum_{i \in I} Q(h_i^t, a_i^t; \theta_i) \\

\max_{a \in A} Q(h^{t+1}, a; \bar \theta) = \sum_{i \in A_i} Q(h_i^{t+1}, a_i; \bar \theta_i)
```

Minimizing this loss function forces all agents to learn their individual utility functions and each agent can simply choose greedy actions with respect to those functions.  The VDN algorithm follows the same structure as IDQN but all agents sample the same examples from the replay buffer and the loss function is computed differently as shown above.
"""

# ╔═╡ aeeb8126-367c-4e44-bf2a-c76a0b97cd88
md"""
### Monotonic Value Decomposition

Sometimes the contribution of each agent to the reward is better represented by a non-linear relation.  *QMIX* is one approach to handle such decompositions of the value function.  The IGM property required can be ensured if (strict) *monotonicity* of the centralized action-value function with respect to the individual utilities holds, that is the derivative of the centralized action-value function with respect to the agent utilities is positive:

```math
```


Intuitively, this means that an increase in the utility of any agent $i$ for its action $a_i$ must lead to an increase in the decomposed centralized action-value function for joint actions containing $a_i$.

Similar to VDN, QMIX builds on top of IDQN and represents each agent's individual utility function as a deep Q-network.  In order ot be able to represent any monotonic decomposition of the centralized action-value function into these individual utilities, QMIX defines a mixing network $f_{\text{mix}}$ given by a feedforward neural network that combines individual utilities to approximate the centralized action-value function:

```math
Q(h, z, a, \theta) = f_{\text{mix}}(Q(h_1, a_1; \theta_1), \dots, Q(h_n, a_n; \theta_n); \theta_{\text{mix}})
```

This decomposition ensures the monotonicity property from equation 9.46 if the mixing function is monotonic with respect to the utilities of all agents.  We can prove that the monotonic decomposition is a sufficient condition to ensure the IGM property:


In practice, the monotonicity assumption is satisfied if the mixing network $f_{\text{mix}}$ is a network with only positive weights for the utility inputs.  Note, the same constraint does not need to be imposed on the bias vectors in $\theta_{\text{mix}}$.  The parameters $\theta_{\text{mix}}$ of the mixing function are obtained through a separate hypernetwork $f_{\text{hyper}}$ parameterized by $\theta_{\text{hyper}}$, which receives additional centralized information $z$ as its input and outputs the parameters $\theta_{\text{mix}}$ of the mixing network (hence the name "hyper").  The ensure positive weights, the hypernetwork $f_{\text{hyper}}$ applies an absolute value function as activation function to the outputs corresponding to the weight matrix of the mising network $f_{\text{mix}}$ and, thus, ensures monotonicity.  Whenever $Q(h, z, a; \theta)$ is needed for optimization, the individual utilities are computed nad the mixing network parameters are obtained by feeding centralized information to the hypernetwork.  The utilities are then aggregated using hte mixing network with the parameters received by the hypernetwork.

During optimization, all parameters $\theta$ of the decomposed centralized action-value function, including the parameters of individual utility networks and parameters of the hypernetwork are jointly optimized by minimizing the value loss given by:

```math
```


The parameters of the mixing network are not optimized by gradient-based optimization, but instead are always obtained as an output of the optimized hypernetwork.  The set of centralized action-value functions that can be represented with QMIX is a superset of the centralized action-value functions that an be decomposed with VDN. 
"""

# ╔═╡ c594d260-67b6-47a7-8768-b8f3cba7ea8b
md"""
### Utility Functions
"""

# ╔═╡ d7b7a9e0-6d23-4fd4-8189-9ae36d16b095
#note that the first three arguments are modified inside this function
function update_shared_nstep_returns!(targets::Vector{T}, target_const::Vector{T}, feature_matrices::NTuple{N, M}, γ::T, replay_buffer::CircularBuffer, batch_inds::Vector{Int64}, nstep::Integer) where {T<:Real, N, M}
	for i in eachindex(batch_inds)
		j = batch_inds[i]
		(xs, i_a, r, xs′, terminated) = replay_buffer[j]
		g = r
		k = j+1
		while !terminated && (k <= j+nstep)
			(xs, i_a, r, xs′, terminated) = replay_buffer[k]
			g += r * γ^(k - j)
			k += 1
		end

		for agent_index in 1:N
			ReinforcementLearning.update_feature_matrix!(feature_matrices[agent_index], xs′[agent_index], i)
		end
		#populate target values with the reward 
		targets[i] = g
		target_const[i] = terminated ? zero(T) : γ^(k-j) #update constant to be used to multiply the target values.  Depending on the number of future steps, the discount rate is used but if the N-step window ends in termination then the output value is ignored
	end
end

# ╔═╡ d2390d6d-bd81-4b88-a855-260a5d74de4a
#update shared target values using parameters, action_value computation function and batch_args which will vary depending on the type of network
begin
	#linear function approximation with a dense feature vector
	function update_shared_targets!(targets::Vector{T}, γ::T, replay_buffer::CircularBuffer, batch_inds::Vector{Int64}, nstep::Integer, target_const::Vector{T}, target_params::NTuple{N, Matrix{T}}, feature_matrices::NTuple{N, Matrix{T}}, action_values::NTuple{N, Vector{T}}, output_matrices::NTuple{N, Matrix{T}}, target_args) where {T<:Real, N}
		#update feature matrix with replay buffer
		update_shared_nstep_returns!(targets, target_const, feature_matrices, γ, replay_buffer, batch_inds, nstep)

		for k in 1:N
			#perform forward pass to fill in target values with function output
			LinearAlgebra.BLAS.gemm!('T', 'N', one(T), feature_matrices[k], target_params[k], zero(T), output_matrices[k])
	
			ReinforcementLearning.maximize_output_matrix!(output_matrices[k])
	
			#for non terminal states add to target discounted future function value
			@inbounds @simd for i in eachindex(batch_inds)
				targets[i] += target_const[i] * output_matrices[k][i, 1]
			end
		end
	end

	#linear function approximation with a binary feature vector
	function update_shared_targets!(targets::Vector{T}, γ::T, replay_buffer::CircularBuffer, batch_inds::Vector{Int64}, nstep::Integer, target_const::Vector{T}, target_params::NTuple{N, Matrix{T}}, feature_matrices::NTuple{N, M}, action_values::NTuple{N, Vector{T}}, output_matrices::NTuple{N, Matrix{T}}, target_args) where {T<:Real, V<:AbstractBinaryFeatures, M<:Vector{V}, N}
		#update feature matrix with replay buffer
		for i in eachindex(batch_inds)
			j = batch_inds[i]
			(xs, joint_actions, r, xs′, terminated) = replay_buffer[j]
			g = r
			k = j+1
			while !terminated && (k <= j+nstep)
				(xs, i_a, r, xs′, terminated) = replay_buffer[k]
				g += r * γ^(k-j)
				k += 1
			end

			targets[i] = g

			if !terminated
				for agent in 1:N
					ReinforcementLearning.update_linear_action_values!(action_values[agent], xs′[agent], target_params[agent])
					targets[i] += γ^(k-j) * maximum(action_values[agent])
				end
			end
		end
	end

	#nonlinear gpu function approximation with a dense feature vector
	function update_shared_targets!(targets::Vector{T}, γ::T, replay_buffer::CircularBuffer, batch_inds::Vector{Int64}, nstep::Integer, target_const::Vector{T}, target_params::NTuple{N, FCANNParamsGPU}, feature_matrices::NTuple{N, Matrix{T}}, action_values::NTuple{N, Vector{T}}, output_matrices::NTuple{N, Matrix{T}}, target_args) where {T<:Real, N}
		#update feature matrix with replay buffer
		update_shared_nstep_returns!(targets, target_const, feature_matrices, γ, replay_buffer, batch_inds, nstep)
		
		for k in 1:N
			(activations, gpu_input) = target_args[i]
			feature_matrix = feature_matrices[i]
			input_orientation = ReinforcementLearning.get_input_orientation(feature_matrix)
			FCANN.memcpy!(gpu_input, feature_matrix)
			#perform forward pass to fill in target values with function output
			FCANN.forwardNOGRAD_base!(activations, target_params[k].weights..., gpu_input, target_params[k].reslayers; input_orientation = input_orientation)
			FCANN.memcpy!(output_matrices[k], activations[end])
			ReinforcementLearning.maximize_output_matrix!(output_matrices[k])
	
			#for non terminal states add to target discounted future function value
			@inbounds @simd for i in eachindex(batch_inds)
				targets[i] += target_const[i] * output_matrices[k][i, 1]
			end
		end
	end

	#nonlinear function approximation
	function update_shared_targets!(targets::Vector{T}, γ::T, replay_buffer::CircularBuffer, batch_inds::Vector{Int64}, nstep::Integer, target_const::Vector{T}, target_params::NTuple{N, FCANNParams{T}}, feature_matrices::NTuple{N, M}, action_values::NTuple{N, Vector{T}}, output_matrices::NTuple{N, Matrix{T}}, target_args) where {T<:Real, N, M}
		#update feature matrix with replay buffer
		update_shared_nstep_returns!(targets, target_const, feature_matrices, γ, replay_buffer, batch_inds, nstep)

		for k in 1:N
			feature_matrix = feature_matrices[k]
			(activations,) = target_args[k]
			
			input_orientation = ReinforcementLearning.get_input_orientation(feature_matrix)
			#perform forward pass to fill in target values with function output
			FCANN.forwardNOGRAD_base!(activations, target_params[k].weights..., feature_matrix, target_params[k].reslayers; input_orientation = input_orientation)
			ReinforcementLearning.maximize_output_matrix!(activations[end])
	
			#for non terminal states add to target discounted future function value
			@inbounds @simd for i in eachindex(batch_inds)
				targets[i] += target_const[i] * activations[end][i, 1]
			end
		end
	end
end

# ╔═╡ ba5cfa57-8c29-4425-bc3e-5306605f19a2
#update shared target values using parameters, action_value computation function and batch_args which will vary depending on the type of network
begin
	#linear function approximation with a dense feature vector
	function update_shared_values!(values::Vector{T}, utility_params::NTuple{N, Matrix{T}}, feature_matrices::NTuple{N, Matrix{T}}, output_inds::NTuple{N, Vector{Int64}}, action_values::NTuple{N, Vector{T}}, output_matrices::NTuple{N, Matrix{T}}, value_args) where {T<:Real, N}
		values .= zero(T)

		for k in 1:N
			feature_matrix = feature_matrices[k]
			#perform forward pass to fill in target values with function output
			LinearAlgebra.BLAS.gemm!('T', 'N', one(T), feature_matrix, utility_params[k], zero(T), output_matrices[k])
	
			#for non terminal states add to target discounted future function value
			@inbounds @simd for i in eachindex(values)
				values[i] += output_matrices[k][i, output_inds[k][i]]
			end
		end
	end

	#linear function approximation with a binary feature vector
	function update_shared_values!(values::Vector{T}, utility_params::NTuple{N, Matrix{T}}, feature_matrices::NTuple{N, M}, output_inds::NTuple{N, Vector{Int64}}, action_values::NTuple{N, Vector{T}}, output_matrices::NTuple{N, Matrix{T}}, value_args) where {T<:Real, V<:AbstractBinaryFeatures, M <: Vector{V}, N}
		values .= zero(T)

		for k in 1:N
			feature_matrix = feature_matrices[k]
			for i in eachindex(values)
				x′ = feature_matrix[i]
				ReinforcementLearning.update_linear_action_values!(action_values[k], x′, utility_params[k])
				values[i] += action_values[k][output_inds[k][i]]
			end
		end
	end

	#nonlinear gpu function approximation with a dense feature vector
	function update_shared_values!(values::Vector{T}, utility_params::NTuple{N, FCANNParamsGPU}, feature_matrices::NTuple{N, Matrix{T}}, output_inds::NTuple{N, Vector{Int64}}, action_values::NTuple{N, Vector{T}}, output_matrices::NTuple{N, Matrix{T}}, value_args) where {T<:Real, N}
		values .= zero(T)

		for k in 1:N
			feature_matrix = feature_matrices[k]
			(activations, gpu_input) = value_args[k]
			input_orientation = ReinforcementLearning.get_input_orientation(feature_matrix)
			FCANN.memcpy!(gpu_input, feature_matrix)
			#perform forward pass to fill in target values with function output
			FCANN.forwardNOGRAD_base!(activations, utility_params[k].weights..., gpu_input, utility_params[k].reslayers; input_orientation = input_orientation)
			FCANN.memcpy!(output_matrices[k], activations[end])
	
			#for non terminal states add to target discounted future function value
			@inbounds @simd for i in eachindex(values)
				values[i] += output_matrices[k][i, output_inds[k][i]]
			end
		end
	end

	#nonlinear function approximation
	function update_shared_values!(values::Vector{T}, utility_params::NTuple{N, FCANNParams{T}}, feature_matrices, output_inds::NTuple{N, Vector{Int64}}, action_values::NTuple{N, Vector{T}}, output_matrices::NTuple{N, Matrix{T}}, value_args) where {T<:Real, N}
		values .= zero(T)

		for k in 1:N
			(activations,) = value_args[k]
			feature_matrix = feature_matrices[k]
			input_orientation = ReinforcementLearning.get_input_orientation(feature_matrix)
			#perform forward pass to fill in target values with function output
			FCANN.forwardNOGRAD_base!(activations, utility_params[k].weights..., feature_matrix, utility_params[k].reslayers; input_orientation = input_orientation)
			
			#for non terminal states add to target discounted future function value
			@inbounds @simd for i in eachindex(values)
				values[i] += activations[end][i, output_inds[k][i]]
			end
		end
	end
end

# ╔═╡ b437f801-0f37-4c76-bb49-db426802d341
md"""
### Algorithm
"""

# ╔═╡ 1a28b9c0-ca64-4f71-bba9-a1e37a4507f8
function reward_value(rewards::T, reducer::Function) where {T<:Real}
	return rewards
end

function reward_value(rewards, reducer::Function)
	return reduce(reducer, rewards)
end

function independent_vdn!(utility_params::NTuple{N, Q}, target_params::NTuple{N, Q}, game::StateStochasticGame{T, S, A, N, P, F1, F2}, γ::T, max_episodes::Integer, max_steps::Integer, feature_vectors::NTuple{N, V}, update_feature_vectors!::NTuple{N, Function}, update_action_values!::NTuple{N, Function}, update_utility_gradients!::NTuple{N, Function}; target_args::NTuple{N, Tuple} = ntuple(i -> (), N), α = one(T)/10, ϵ = one(T) / 10, buffer_size::Integer = 10_000, batch_size::Integer = 512, target_update_interval::Integer = 100, α_decay = one(T), decay_step = typemax(Int64), save_step_rewards::Bool = false, nstep::Integer = 0, ∇q̂s::NTuple{N, Q} = deepcopy(utility_params), reducer::Function = +, kwargs...) where {Q, T<:Real, S, A, N, P<:AbstractStateGameTransition, F1<:Function, F2<:Function, V}
	#initialize memory
	action_values = ntuple(i -> zeros(T, length(game.agent_actions[i])), N)
	policies = deepcopy(action_values)
	replay_buffer = CircularBuffer{Tuple{NTuple{N, V}, NTuple{N, Int64}, T, NTuple{N, V}, Bool}}(buffer_size)
	# @info "replay buffers are of type $(typeof(replay_buffers))"
	targets = Vector{T}(undef, batch_size)
	target_const = Vector{T}(undef, batch_size)
	batch_inds = Vector{Int64}(undef, batch_size)
	feature_matrices = ntuple(i -> ReinforcementLearning.form_feature_matrix(game, feature_vectors[i], batch_size), N)
	values = Vector{T}(undef, batch_size)
	δs = Vector{T}(undef, batch_size)
	output_matrices = ntuple(i -> zeros(T, batch_size, length(game.agent_actions[i])), N)
	output_inds = ntuple(i -> Vector{Int64}(undef, batch_size), N)
	feature_vectors2 = deepcopy(feature_vectors)

	#initialize episode
	s = game.initialize_state()
	
	joint_actions = ntuple(N) do i
		update_feature_vectors![i](feature_vectors[i], s)
		update_action_values![i](action_values[i], feature_vectors[i], utility_params[i])
		policies[i] .= action_values[i]
		make_ϵ_greedy_policy!(policies[i]; ϵ = ϵ)
		sample_action(policies[i])
	end

	ep = 1
	step = 1
	epreward = zero(T)
	episode_rewards = Vector{T}()
	episode_steps = Vector{Int64}()
	step_rewards = Vector{T}()
	decay = one(T)

	while (ep ≤ max_episodes) && (step ≤ max_steps)
		(rewards, s′) = game.ptf(s, joint_actions)
		for i in 1:N
			update_feature_vectors![i](feature_vectors2[i], s′)
		end
		terminated = game.isterm(s′)

		#create shared reward with reduction function if needed
		r = reward_value(rewards, reducer)
		push!(replay_buffer, (deepcopy(feature_vectors), joint_actions, r, deepcopy(feature_vectors2), terminated))
		save_step_rewards && push!(step_rewards, r)
		epreward += r

		if terminated
			s′ = game.initialize_state()
			for i in 1:N
				update_feature_vectors![i](feature_vectors2[i], s′)
			end
			push!(episode_rewards, epreward)
			epreward = zero(T)
			push!(episode_steps, step)
			ep += 1
		end

		joint_actions′ = ntuple(N) do i
			update_action_values![i](action_values[i], feature_vectors2[i], utility_params[i])
			policies[i] .= action_values[i]
			ReinforcementLearning.make_ϵ_greedy_policy!(policies[i]; ϵ = ϵ)
			sample_action(policies[i])
		end

		# @info "Sampled the following joint actions: $joint_actions′"

		decay *= (step > decay_step)*α_decay + (step <= decay_step)

		#only perform gradient parameter update once the replay buffer is large enough to fill up an entire batch
		if step ≥ (batch_size + nstep)
			# @info "Performing batch gradient updates"
			ReinforcementLearning.update_batch_inds!(batch_inds, step, buffer_size, nstep)
			update_shared_targets!(targets, γ, replay_buffer, batch_inds, nstep, target_const, target_params, feature_matrices, action_values, output_matrices, target_args)


			#update_feature_matrix and action indices for gradient and δ
			for j in eachindex(batch_inds)
				(xs_k, joint_actions, _, _, _) = replay_buffer[batch_inds[j]]
				
				for k in 1:N
					ReinforcementLearning.update_feature_matrix!(feature_matrices[k], xs_k[k], j)
					output_inds[k][j] = joint_actions[k]
				end
			end

			update_shared_values!(values, utility_params, feature_matrices, output_inds, action_values, output_matrices, target_args)

			δs .= targets .- values

			for k in 1:N
				
				#need to replace this with something that computes the forward pass for all the networks ahead of time and computes the output value as a sum so it can be subtracted from the targets and used as a δ vector.  Then that δ can be used in combination with the action indices to compute the gradient
				update_utility_gradients![k](∇q̂s[k], utility_params[k], output_inds[k], δs, feature_matrices[k], output_matrices[k])

				#the gradient of the output should be multiplied by -1 but we are minimizing the loss so adding the gradient here is effectively subtracting the gradient w.r.t. the loss function.
				ReinforcementLearning.update_params_with_gradient!(utility_params[k], α*decay, ∇q̂s[k])

				if iszero(step % target_update_interval)
					copy!(target_params[k], utility_params[k])
				end
			end
		end

		# @info "Preparing for next step"
		# @info "Updating state to $s′"
		s = s′
		# @info "Updating feature vector to $feature_vector2"
		feature_vectors = deepcopy(feature_vectors2)
		# @info "Updating joint actions to $joint_actions′"
		joint_actions = joint_actions′
		step += 1
		# @info "One step $step of $max_steps"
	end

	for k in 1:N
		ReinforcementLearning.cleanup_gradient!(∇q̂s[k])
	end
	
	q̂s, form_kwargs = form_agent_value_functions(game, update_feature_vectors!, update_action_values!, feature_vectors, utility_params)

	return (value_functions = q̂s, episode_rewards = episode_rewards, episode_steps = episode_steps, final_parameters = deepcopy(utility_params), form_kwargs = form_kwargs)
end

function independent_vdn_common_reward!(utility_params::NTuple{N, Q}, target_params::NTuple{N, Q}, game::StateStochasticGame{T, S, A, N, StateCommonRewardGameTransitionDeterministic{T, S, F, N}, F2}, γ::T, max_episodes::Integer, max_steps::Integer, feature_vectors::NTuple{N, V}, update_feature_vectors!::NTuple{N, Function}, update_action_values!::NTuple{N, Function}, update_utility_gradients!::NTuple{N, Function}; target_args::NTuple{N, Tuple} = ntuple(i -> (), N), α = one(T)/10, ϵ = one(T) / 10, buffer_size::Integer = 10_000, batch_size::Integer = 512, target_update_interval::Integer = 100, α_decay = one(T), decay_step = typemax(Int64), save_step_rewards::Bool = false, nstep::Integer = 0, ∇q̂s::NTuple{N, Q} = deepcopy(utility_params), kwargs...) where {Q, T<:Real, S, A, N, F<:Function, F2<:Function, V}
	return independent_vdn!(utility_params, target_params, game, γ, max_episodes, max_steps, feature_vectors, update_feature_vectors!, update_action_values!, update_utility_gradients!; target_args = target_args, α = α, ϵ = ϵ, buffer_size = buffer_size, batch_size = batch_size, target_update_interval = target_update_interval, α_decay = α_decay, decay_step = decay_step, save_step_rewards = save_step_rewards, nstep = nstep, ∇q̂s = ∇q̂s, reducer = identity, kwargs...)
end

# ╔═╡ 12fa36af-4a3d-421a-ac3d-b3c20b6f8498
md"""
### Linear Approximation
"""

# ╔═╡ 47df7c4d-f29d-4422-8a80-d057225e2ad9
function update_linear_utility_gradient!(∇q̂::Matrix{T}, utility_params::Matrix{T}, output_indices::Vector{I}, δs::Vector{T}, feature_matrix, output_matrix::Matrix{T}) where {T<:Real, I<:Integer}
	#reset gradient to 0
	∇q̂ .= zero(T)

	#initialize batch size in order to calculate constant for average
	batch_size = length(δs)
	c = T(2 / batch_size)

	#accumulate gradient of loss function per example
	for i in eachindex(δs)
		δ = δs[i]
		ReinforcementLearning.accumulate_linear_gradient!(∇q̂, c*δ, i, output_indices[i], feature_matrix)
	end
end

# ╔═╡ 2e3d6f58-541e-4856-98d9-5794449faab1
begin
	independent_vdn_linear(game::StateStochasticGame{T, S, A, N, P, F1, F2}, γ::T, max_episodes::Integer, max_steps::Integer, feature_vectors::NTuple{N, V}, update_feature_vectors!::NTuple{N, Function}; init_value::T = zero(T), utility_params::NTuple{N, Matrix{T}} = ntuple(i -> initialize_linear_parameters(feature_vectors[i], length(game.agent_actions[i]), init_value), N), target_params::NTuple{N, Matrix{T}} = deepcopy(utility_params), kwargs...) where {T<:Real, S, A, N, P, F1<:Function, F2<:Function, V} = independent_vdn!(utility_params, target_params, game, γ, max_episodes, max_steps, feature_vectors, update_feature_vectors!, convert_to_ntuple(ReinforcementLearning.update_linear_action_values!, N), convert_to_ntuple(update_linear_utility_gradient!, N); kwargs...)

	independent_vdn_linear(game::StateStochasticGame{T, S, A, N, P, F1, F2}, γ::T, max_episodes::Integer, max_steps::Integer, feature_vector, update_feature_vector!::Function; kwargs...) where {T<:Real, S, A, N, P, F1<:Function, F2<:Function} = independent_vdn_linear(game, γ, max_episodes, max_steps, convert_to_ntuple(feature_vector, N), convert_to_ntuple(update_feature_vector!, N); kwargs...)
end

# ╔═╡ 5286ea3f-9615-4201-938a-2647d84caa20
md"""
### Non-Linear Approximation
"""

# ╔═╡ 4536d83d-c4e9-41ba-8b13-4f206bec518e
function setup_fcann_utility_arguments(utility_params::FCANNParams{T}, target_params::FCANNParams{T}, batch_size::Integer, l2::T, dropout::T, use_μP::Bool, activation_list; use_gpu = false) where {T<:Real}
	input_length, hidden_layers, num_hidden = get_network_dimensions(utility_params)
	input_length2, hidden_layers2, num_hidden2 = get_network_dimensions(target_params)
	@assert input_length == input_length2 "Value and target networks don't share the same input dimension"
	@assert hidden_layers == hidden_layers2 "Value and target networks don't share the same hidden layers"
	@assert utility_params.reslayers == target_params.reslayers "Value and target networks don't share the same skip connections"
	
	#form activations for network
	activations_batch = FCANN.form_activations(utility_params.weights[1], batch_size)
	activations = FCANN.form_activations(utility_params.weights[1])
	tanh_grad_z = deepcopy(activations_batch)
	deltas = deepcopy(activations_batch)
	onesvec = ones(T, batch_size)

	#note that the scales are multiplied by -1 to minimize loss in gradient update
	scales = fill(one(T), length(utility_params.weights[1]))
	if use_μP
		for i in eachindex(hidden_layers)
			i′ = i + 1
			scales[i′] /= size(utility_params.weights[1][i′], 2)
		end
	end

	function update_action_values!(action_values::Vector{T}, x, params; activations::FCANNActivations{T} = activations, kwargs...) 
		ReinforcementLearning.fcann_value_function!(activations, x, params)
		action_values .= activations[end]
		val, index = findmax(action_values)
		isnan(val) && error("Got NaN action value inside $action_values")
		isinf(val) && error("Got Inf action value inside $action_values")
		return (val, index)
	end
	
	function update_utility_gradient!(∇q̂::FCANNParams{T}, params::FCANNParams{T}, output_inds::Vector{I}, δs::Vector{T}, feature_matrix, output_matrix::Matrix{T}) where I<:Integer
		FCANN.nnCostFunction(params.weights..., hidden_layers, feature_matrix, output_inds, δs, l2, ∇q̂.weights..., tanh_grad_z, activations_batch, deltas, onesvec, dropout; resLayers = params.reslayers, activation_list = activation_list, input_orientation = 'T')
		ReinforcementLearning.scale_fcann_params!(∇q̂, scales) #note that this also multiplies the gradient by -1 to account for minimization
		return ∇q̂
	end

	output = (update_action_values! = update_action_values!, update_utility_gradient! = update_utility_gradient!, target_args = (activations_batch,))

	!use_gpu && return output
	
	error("No GPU implementation yet")
	
	#dont' have gpu implementation yet for this cost function

	# if in(:GPU, backendList)
	# 	d_activations = FCANN.device_allocate(activations)
	# 	d_activations_batch = FCANN.device_allocate(activations_batch)
	# 	d_tanh_grad_z = FCANN.device_allocate(tanh_grad_z)
	# 	d_deltas = FCANN.device_allocate(deltas)
	# 	d_value_params = initialize_gpu_params(utility_params)
	# 	d_target_params = initialize_gpu_params(target_params)
	# 	d_x = FCANN.cuda_allocate(zeros(T, input_length))
	# 	d_feature_matrix = FCANN.cuda_allocate(zeros(T, input_length, batch_size))
	# 	d_targets = FCANN.cuda_allocate(zeros(T, batch_size))
	# 	d_output_inds = FCANN.cuda_allocate(zeros(Cint, batch_size))
	# 	output_inds2 = zeros(Cint, batch_size)
	# 	d_onesvec = FCANN.cuda_allocate(onesvec)

	# 	gpu_feature_update! = setup_gpu_feature(zeros(T, input_length), update_feature_vector!)

	# 	#x is always going to come from the replay buffer and hence will be an ordinary vector
	# 	function update_action_values!(action_values::Vector{T}, x::Vector{T}, params::FCANNParamsGPU; d_x::FCANN.CUDAArray = d_x, d_activations::FCANNActivationsGPU = d_activations, kwargs...)		
	# 		FCANN.memcpy!(d_x, x)
	# 		fcann_value_function!(d_activations, d_x, params)
	# 		FCANN.memcpy!(action_values, d_activations[end])
	# 		val, index = findmax(action_values)
	# 		isnan(val) && error("Got NaN action value inside $action_values")
	# 		isinf(val) && error("Got Inf action value inside $action_values")
	# 		return (val, index)
	# 	end

	# 	function update_value_gradient!(∇q̂::FCANNParamsGPU, params::FCANNParamsGPU, targets::Vector{T}, output_inds::Vector{I}, feature_matrix::Matrix{T}, output_matrix::Matrix{T}) where {T<:Real, I<:Integer}
	# 		FCANN.memcpy!(d_feature_matrix, feature_matrix)
	# 		FCANN.memcpy!(d_targets, targets)
	# 		output_inds2 .= Cint.(output_inds .- 1) #note that the GPU uses zero indexing and 32 bit integers
	# 		FCANN.memcpy!(d_output_inds, output_inds2)
	# 		update_fcann_value_gradient!(∇q̂, params, d_feature_matrix, d_targets, d_output_inds, hidden_layers, l2, d_tanh_grad_z, d_activations_batch, d_deltas, d_onesvec, dropout, activation_list)
	# 		scale_fcann_params!(∇q̂, scales)
	# 		return ∇q̂
	# 	end

	# 	function cleanup_vars()
	# 		FCANN.clear_gpu_data(d_value_params.weights[1])
	# 		FCANN.clear_gpu_data(d_value_params.weights[2])
	# 		FCANN.clear_gpu_data(d_target_params.weights[1])
	# 		FCANN.clear_gpu_data(d_target_params.weights[2])
	# 		FCANN.clear_gpu_data(d_deltas)
	# 		FCANN.clear_gpu_data(d_tanh_grad_z)
	# 		FCANN.clear_gpu_data([d_x])
	# 		FCANN.clear_gpu_data([d_feature_matrix])
	# 		FCANN.clear_gpu_data([d_targets])
	# 		FCANN.clear_gpu_data([d_output_inds])
	# 		FCANN.clear_gpu_data(d_activations)
	# 		FCANN.clear_gpu_data(d_activations_batch)
	# 		FCANN.clear_gpu_data([d_onesvec])
	# 	end

	# 	gpu_args = (utility_params = d_value_params, target_params = d_target_params, target_args = (d_activations_batch, d_feature_matrix), cleanup_vars = cleanup_vars)
	# else
	# 	gpu_args = ()
	# end

	# return (;output..., gpu_args = gpu_args)
end

# ╔═╡ ad5c6fd6-cf80-452d-beb2-018936849846
begin
	function independent_vdn_fcann(game::StateStochasticGame{T, S, A, N, P, F1, F2}, γ::T, max_episodes::Integer, max_steps::Integer, feature_vectors::NTuple{N, V}, update_feature_vectors!::NTuple{N, Function}, hidden_layers::Vector{Int64}; batch_size::Integer = 512, reslayers::Int64 = 0, use_μP::Bool = true, utility_params::NTuple{N, FCANNParams{T}} = ntuple(i -> initialize_fcann_params(feature_vectors[i], hidden_layers, length(game.agent_actions[i]), reslayers, use_μP), N), target_params::NTuple{N, FCANNParams{T}} = deepcopy(utility_params), dropout = zero(T), activation_list = fill(true, length(hidden_layers)), l2 = zero(T), use_gpu::Bool = false, kwargs...) where {T<:Real, S, A, N, P, F1<:Function, F2<:Function, V} 
		setups = ntuple(i -> setup_fcann_utility_arguments(utility_params[i], target_params[i], batch_size, l2, dropout, use_μP, activation_list; use_gpu = use_gpu), N)
		
		independent_vdn!(utility_params, target_params, game, γ, max_episodes, max_steps, feature_vectors, update_feature_vectors!, ntuple(i -> setups[i].update_action_values!, N), ntuple(i -> setups[i].update_utility_gradient!, N); target_args = ntuple(i -> setups[i].target_args, N), batch_size = batch_size, kwargs...)
	end

	independent_vdn_fcann(game::StateStochasticGame{T, S, A, N, P, F1, F2}, γ::T, max_episodes::Integer, max_steps::Integer, feature_vector, update_feature_vector!::Function, hidden_layers::Vector{Int64}; kwargs...) where {T<:Real, S, A, N, P, F1<:Function, F2<:Function} = independent_vdn_fcann(game, γ, max_episodes, max_steps, convert_to_ntuple(feature_vector, N), convert_to_ntuple(update_feature_vector!, N), hidden_layers; kwargs...)
end

# ╔═╡ ad469f86-c30b-4988-8d2e-5694e5270bc9
md"""
# Matrix Game Tests

Consider the following trivial examples consisting of non-repeated matrix games with a single playable state.  We consider common reward games, so a single matrix is needed to define the entire game rewards.  After the joint action selection, the game terminates.  The goal is to learn the maximizing joint action, but through decomposition where each agent only maximizes over its own action space.
"""

# ╔═╡ ffc4a8e6-46aa-4034-bf60-cae7a95b9cc2
md"""
## Example 1: Linear Game
"""

# ╔═╡ 1ba8f790-1e33-42d9-840a-9cb8ca727439
# ╠═╡ skip_as_script = true
#=╠═╡
const matrix_actions_1 = [1, 2]
  ╠═╡ =#

# ╔═╡ 66206149-5cdf-4eb8-b08f-c4280091192d
# ╠═╡ skip_as_script = true
#=╠═╡
const reward_matrix_1 = [1f0 5f0; 5f0 9f0]
  ╠═╡ =#

# ╔═╡ 5109c85d-e379-4afd-a16b-1e5a87b145b7
#=╠═╡
const tabular_matrix_game_1 = create_non_repeated_game(ntuple(i -> reward_matrix_1 ./ 2, 2), ntuple(i -> matrix_actions_1, 2))
  ╠═╡ =#

# ╔═╡ b16f5471-7027-4c81-be1f-c53c53cf454c
#=╠═╡
const tabular_matrix_mdp_1 = TabularMDP(tabular_matrix_game_1, sum) #creates an MDP out of the game by assigning the sum of rewards as the shared common reward for the whole environment
  ╠═╡ =#

# ╔═╡ 3a585fae-1056-4473-bab6-f08ccfbe9cac
#=╠═╡
const state_matrix_game_1 = StateStochasticGame(tabular_matrix_game_1)
  ╠═╡ =#

# ╔═╡ 1bdf1d35-897d-4b99-aef4-05d377215cbc
md"""
### Exact Solution
"""

# ╔═╡ a1cc64f0-8aa6-48d5-b091-bc377bdaf56a
#=╠═╡
const game_1_exact = value_iteration_q(tabular_matrix_mdp_1, 1f0)
  ╠═╡ =#

# ╔═╡ f1cd0fc3-9eaf-4ab0-955a-253dfe6dc6f9
#=╠═╡
function display_exact_solution_1(output::NamedTuple)
	output_values = zeros(Float32, 2, 2)
	output_policy = zeros(Float32, 2, 2)
	for i_a in eachindex(tabular_matrix_mdp_1.actions)
		a1, a2 = tabular_matrix_mdp_1.actions[i_a]
		output_values[a1, a2] = output.final_value[i_a, 1]
		output_policy[a1, a2] = output.optimal_policy[i_a, 1]
	end
	return (joint_action_values = output_values, optimal_policy = output_policy)
end
  ╠═╡ =#

# ╔═╡ 8075559d-2651-4a79-a435-9082724e4892
#=╠═╡
display_exact_solution_1(game_1_exact)
  ╠═╡ =#

# ╔═╡ 45ee2102-5543-4416-8772-a1902319921c
md"""
### Tabular MARL Solutions

These techniques store values as a table and compute updates through averaging.  Practically for simple matrix games, we do not need to use approximation techniques that use gradients, but we can use the tabular solution to verify what type of behavior we should expect from the equivalent gradient method such as IQL vs IDQN.
"""

# ╔═╡ 189c1b0d-80fc-43fa-8700-0cf8e44e46ae
md"""
#### Independent Q-learning

Since this game is already linear, IQL learns the correct solution despite not being explicitely trained to decomone the joint action value function.  Although the learned values are not correct, the ordinal ranking between the individual and joint values produces the same optimal policy.
"""

# ╔═╡ 583df59f-8f67-484b-9cd2-9a0e9236df4b
#=╠═╡
const game_1_iql = independent_q_learning(tabular_matrix_game_1, 1f0; α = 3f-4, max_steps = 1_000_000, ϵ = 0.01f0, α_decay = 0.999999f0)
  ╠═╡ =#

# ╔═╡ fe49260a-8369-455e-8c06-70de4486cb68
# ╠═╡ skip_as_script = true
#=╠═╡
function display_joint_tabular_solution_1(output::NamedTuple)
	v1 = round.(Float64.(output.value_estimates[1][:, 1]); sigdigits = 3)
	v2 = round.(Float64.(output.value_estimates[2][:, 1]); sigdigits = 3)

	p1 = round.(Float64.(output.policies[1][:, 1]); sigdigits = 3)
	p2 = round.(Float64.(output.policies[2][:, 1]); sigdigits = 3)
	
	values = md"""
	Learned Values
	
	||$(v1[1])|$(v1[2])|
	|---|---|---|
	|$(v2[1])|$(round(v1[1] + v2[1], sigdigits = 3))|$(round(v1[1] + v2[2]; sigdigits = 3))|
	|$(v2[2])|$(round(v1[2] + v2[1]; sigdigits = 3))|$(round(v1[2] + v2[2], sigdigits = 3))|
	"""

	policies = md"""
	Learned Policies
	
	||$(p1[1])|$(p1[2])|
	|---|---|---|
	|$(p2[1])|$(round(p1[1] * p2[1], sigdigits = 3))|$(round(p1[1] * p2[2]; sigdigits = 3))|
	|$(p2[2])|$(round(p1[2] * p2[1]; sigdigits = 3))|$(round(p1[2] * p2[2], sigdigits = 3))|
	"""

	(values, policies)
end
  ╠═╡ =#

# ╔═╡ f093884c-fe6c-4785-9808-2b88ef20ce85
#=╠═╡
display_joint_tabular_solution_1(game_1_iql)
  ╠═╡ =#

# ╔═╡ 132d8d08-f347-4ded-aef6-947b653cadf4
md"""
#### Joint-Action Learning with Agent Modeling

Since this technique explicitely models the value functions across the joint action space, it has no problem learning the correct values both individually and their sum.  The policies reflect the learned policy of each agent during the training process so it includes the effect of the random exploration parameter ϵ.
"""

# ╔═╡ bbd6a798-0851-4e8b-93ac-b00fdd43a4c5
#=╠═╡
const game_1_jal = jal_am(tabular_matrix_game_1, 1f0, typemax(Int64), 1_000_000; ϵ = 0.01f0)
  ╠═╡ =#

# ╔═╡ 4f41c31d-335a-4c1d-b59c-f9ed0d555e58
# ╠═╡ skip_as_script = true
#=╠═╡
function display_jal_tabular_solution_1(output::NamedTuple)
	joint_values = round.(Float64.(output.joint_action_values[1][:, :, 1] .+ output.joint_action_values[2][:, :, 1]); sigdigits = 3)

	p1 = round.(Float64.(output.policies[1][:, 1]); sigdigits = 3)
	p2 = round.(Float64.(output.policies[2][:, 1]); sigdigits = 3)
	
	values = md"""
	Learned Values
	
	||||
	|---|---|---|
	||$(joint_values[1, 1])|$(joint_values[1, 2])|
	||$(joint_values[2, 1])|$(joint_values[2, 2])|
	"""

	policies = md"""
	Empirial Policies
	
	||$(p1[1])|$(p1[2])|
	|---|---|---|
	|$(p2[1])|$(round(p1[1] * p2[1], sigdigits = 3))|$(round(p1[1] * p2[2]; sigdigits = 3))|
	|$(p2[2])|$(round(p1[2] * p2[1]; sigdigits = 3))|$(round(p1[2] * p2[2], sigdigits = 3))|
	"""

	(values, policies)
end
  ╠═╡ =#

# ╔═╡ a2f10a2a-7520-41f0-bddb-2410ac2edce1
#=╠═╡
display_jal_tabular_solution_1(game_1_jal)
  ╠═╡ =#

# ╔═╡ e4c23a21-5a3b-4377-b819-a1c39a17f7ab
md"""
### Approximation MARL Solutions

"""

# ╔═╡ 656d25ed-e59d-42a1-9e66-bf626cb8eef0
md"""
#### Feature Vector Setup
"""

# ╔═╡ c99bab73-518e-4c2f-817e-4e57a9b934f4
function setup_matrix_game_features(tabular_game::TabularStochasticGame)
	s0 = tabular_game.states[tabular_game.initialize_state_index()]
	num_groups = length(tabular_game.states)
	f(s) = tabular_game.state_index[s]
	state_aggregation_feature_setup(s0, num_groups, f)
end

# ╔═╡ 0930fa3d-409a-45a3-b074-520acaf7fae6
function setup_matrix_game_features_dense(tabular_game::TabularStochasticGame{T, S, A, P, F}) where {T<:Real, S, A, P, F}
	num_groups = length(tabular_game.states)
	x = zeros(T, num_groups)
	function f!(x, s::S)
		x .= zero(T)
		i_s = tabular_game.state_index[s]
		x[i_s] = one(T)
		return x
	end
	(x, f!)
end

# ╔═╡ 7efbb914-75b0-4692-a557-97a9b80bfc39
#=╠═╡
const game_1_feature_setup = setup_matrix_game_features(tabular_matrix_game_1)
  ╠═╡ =#

# ╔═╡ 7a29339e-8dd5-4ce7-a273-6204217e918e
#=╠═╡
const game_1_dense_feature_setup = setup_matrix_game_features_dense(tabular_matrix_game_1)
  ╠═╡ =#

# ╔═╡ 92e30dc7-2b9f-453b-a9bb-8ac2831d0503
md"""
#### Independent DQN
"""

# ╔═╡ 3a02d212-c162-48cd-85bf-f0a9987c2b24
#=╠═╡
const game_1_idqn = independent_dqn_linear(state_matrix_game_1, 1f0, typemax(Int64), 20_000, game_1_feature_setup...; batch_size = 128, buffer_size = 1_000, α = 3f-4, ϵ = 0.01f0)
  ╠═╡ =#

# ╔═╡ 868c7277-7d31-4175-bb8e-9580d22571ea
#=╠═╡
const game_1_idqn2 = independent_dqn_linear(state_matrix_game_1, 1f0, typemax(Int64), 20_000, game_1_dense_feature_setup...; batch_size = 128, buffer_size = 1_000, α = 3f-4, ϵ = 1.0f0)
  ╠═╡ =#

# ╔═╡ 6d0448d3-c81c-480e-bc79-7df7ca20a059
#=╠═╡
const game_1_idqn3 = independent_dqn_fcann(state_matrix_game_1, 1f0, typemax(Int64), 20_000, game_1_dense_feature_setup..., [64, 64]; batch_size = 128, buffer_size = 1_000, α = 3f-4, ϵ = 1.0f0)
  ╠═╡ =#

# ╔═╡ 8634cf9c-66ef-4fd2-a862-459476fa0b26
# ╠═╡ skip_as_script = true
#=╠═╡
function display_joint_solution_1(output::NamedTuple)
	f1, f2 = output.value_functions

	v1 = round.(Float64.(f1(:play).action_values); sigdigits = 3)
	v2 = round.(Float64.(f2(:play).action_values); sigdigits = 3)

	p1 = copy(v1)
	p2 = copy(v2)
	make_greedy_policy!(p1)
	make_greedy_policy!(p2)
	
	values = md"""
	Learned Values
	
	||$(v1[1])|$(v1[2])|
	|---|---|---|
	|$(v2[1])|$(round(v1[1] + v2[1], sigdigits = 3))|$(round(v1[1] + v2[2]; sigdigits = 3))|
	|$(v2[2])|$(round(v1[2] + v2[1]; sigdigits = 3))|$(round(v1[2] + v2[2], sigdigits = 3))|
	"""

	policies = md"""
	Learned Policies
	
	||$(p1[1])|$(p1[2])|
	|---|---|---|
	|$(p2[1])|$(round(p1[1] * p2[1], sigdigits = 3))|$(round(p1[1] * p2[2]; sigdigits = 3))|
	|$(p2[2])|$(round(p1[2] * p2[1]; sigdigits = 3))|$(round(p1[2] * p2[2], sigdigits = 3))|
	"""

	(values, policies)
end
  ╠═╡ =#

# ╔═╡ dbe66895-0734-4ffc-96cc-9ae97d7bbb27
#=╠═╡
display_joint_solution_1(game_1_idqn)
  ╠═╡ =#

# ╔═╡ 52a29eb4-bb87-40ad-a740-918ae9ee0e7f
#=╠═╡
display_joint_solution_1(game_1_idqn2)
  ╠═╡ =#

# ╔═╡ 1056e7e0-c28f-492d-9f3f-5d1cdbaeed20
#=╠═╡
display_joint_solution_1(game_1_idqn3)
  ╠═╡ =#

# ╔═╡ 5a1c6606-6593-40dd-8ed2-0285af060c25
md"""
#### Value Decomposition Networks

Since this game matrix can be represented by a linear decomposition, VDN has no problem converging to the correct value function with all methods.
"""

# ╔═╡ a4a5ffcf-fea3-482d-aa30-b41637826453
#=╠═╡
const game_1_vdn = independent_vdn_linear(state_matrix_game_1, 1f0, typemax(Int64), 20_000, game_1_feature_setup...; batch_size = 128, buffer_size = 1_000, α = 3f-4, ϵ = 1.0f0)
  ╠═╡ =#

# ╔═╡ 1ec05cdb-dbef-4a8a-b582-9bc3b3ef34b8
#=╠═╡
const game_1_vdn2 = independent_vdn_linear(state_matrix_game_1, 1f0, typemax(Int64), 20_000, game_1_dense_feature_setup...; batch_size = 128, buffer_size = 1_000, α = 3f-4, ϵ = 1.0f0)
  ╠═╡ =#

# ╔═╡ abd189c8-0ca9-4eec-9497-d2e3392d385c
#=╠═╡
const game_1_vdn3 = independent_vdn_fcann(state_matrix_game_1, 1f0, typemax(Int64), 20_000, game_1_dense_feature_setup..., [64, 64]; batch_size = 128, buffer_size = 1_000, α = 3f-4, ϵ = 1.0f0)
  ╠═╡ =#

# ╔═╡ 3130e1c7-332c-4d61-9a56-465ea966b03d
#=╠═╡
display_joint_solution_1(game_1_vdn)
  ╠═╡ =#

# ╔═╡ d3598c9f-c9e4-419a-948f-ed2f837a0d53
#=╠═╡
display_joint_solution_1(game_1_vdn2)
  ╠═╡ =#

# ╔═╡ 1c90f846-8705-4cbe-afbc-1f86d58c955d
#=╠═╡
display_joint_solution_1(game_1_vdn3)
  ╠═╡ =#

# ╔═╡ 10f07638-d6d3-4a7a-839e-595d7526f7af
md"""
## Example 2: Monotonic Game

This matrix game can be represented by a composition of monotonic functions of each agents individual utility function, but not by a linear composition like the previous game.
"""

# ╔═╡ 51411837-d94a-478a-b1f2-f2776e983858
# ╠═╡ skip_as_script = true
#=╠═╡
const reward_matrix_2 = [0f0 0f0; 0f0 10f0]
  ╠═╡ =#

# ╔═╡ 84b09d35-1df7-4aea-8e5d-b7516d545ab3
#=╠═╡
const tabular_matrix_game_2 = create_non_repeated_game(ntuple(i -> reward_matrix_2 ./ 2, 2), ntuple(i -> matrix_actions_1, 2))
  ╠═╡ =#

# ╔═╡ 1fc576b2-5eae-4910-968a-ec888e8f53b3
#=╠═╡
const tabular_matrix_mdp_2 = TabularMDP(tabular_matrix_game_2, sum) #creates an MDP out of the game by assigning the sum of rewards as the shared common reward for the whole environment
  ╠═╡ =#

# ╔═╡ 6be2ddbe-8040-4e26-9f52-7b4d297fe6b1
#=╠═╡
const state_matrix_game_2 = StateStochasticGame(tabular_matrix_game_2)
  ╠═╡ =#

# ╔═╡ 3d21d8bb-3c8c-468d-b9f4-588b014fbab6
md"""
### Exact Solution
"""

# ╔═╡ d13e0602-dd22-4fd5-bd40-02ef92d71cbd
#=╠═╡
const game_2_exact = value_iteration_q(tabular_matrix_mdp_2, 1f0)
  ╠═╡ =#

# ╔═╡ b1240d14-8c60-48a1-9bd6-0b513cb9441e
#=╠═╡
display_exact_solution_1(game_2_exact)
  ╠═╡ =#

# ╔═╡ 4518ec9c-6391-429a-9580-461402ec33d1
md"""
### Tabular MARL Solutions
"""

# ╔═╡ d23f1861-4624-4d72-821b-93db1f44f87c
md"""
#### Independent Q-learning

Since this game is not linear, IQL learns individual value functions which correctly estimate the optimal action value of (2, 2) but are unable to represent the mixed values.  Despite this failure, the optimal policy is still correct because the function is monotonic.

We can see the learned values change depending on the value of ϵ but still preserve the ordinal ranking of joint actions.  When ϵ is close to 1, the samples include mixed selections (1, 2) and (2, 1) in which case the agents will receive rewards of 0.  This pushes the individual values of action selection 2 down for each agent since the value function does not reflect the optimal policy and instead reflects the random policy.
"""

# ╔═╡ ab698e42-290e-4be0-9021-c0c1c32c9191
#=╠═╡
md"""
``\epsilon`` select: $(@bind game_2_iql_ϵ Slider(0.1f0:0.1f0:1f0; show_value = true))
"""
  ╠═╡ =#

# ╔═╡ 9ad3b7fb-e92c-4782-b8d7-6313ba7d3cda
#=╠═╡
const game_2_iql = independent_q_learning(tabular_matrix_game_2, 1f0; α = 1f-3, max_steps = 1_000_000, ϵ = game_2_iql_ϵ, α_decay = 0.999999f0)
  ╠═╡ =#

# ╔═╡ 7ac7fb89-4e82-4c8a-a7f5-377e2ecc07f0
#=╠═╡
display_joint_tabular_solution_1(game_2_iql)
  ╠═╡ =#

# ╔═╡ 13559698-a13b-4f22-8809-c24a5826db18
md"""
#### Joint-Action Learning with Agent Modeling

Since this technique explicitely models the value functions across the joint action space, it has no problem learning the correct values regardless of what decompositions exist.  The policies reflect the learned policy of each agent during the training process so it includes the effect of the random exploration parameter ϵ.

Normally, we need agent modeling in order to compute equilibrium solutions that depend on the joint action value function in a non-trivial way.  Since this is equivalent to a common reward game, the agent models are not necessary since each the optimal joint action is just that which maximizes the centralized joint action value function.  We could also solve this as an MDP over the joint action space which would result in the same solution.

Note that regardless of the value of ϵ, the correct joint action value function is learned because here the random action selection does not affect the objective function which learns a separate value for each joint action.
"""

# ╔═╡ 00faac3f-314d-4974-a857-6edba5ddb724
#=╠═╡
md"""
``\epsilon`` select: $(@bind game_2_jal_ϵ Slider(0.1f0:0.1f0:1f0; show_value = true))
"""
  ╠═╡ =#

# ╔═╡ 96fae564-bdc9-40fd-9cf5-30620693237f
#=╠═╡
const game_2_jal = jal_am(tabular_matrix_game_2, 1f0, typemax(Int64), 1_000_000; ϵ = game_2_jal_ϵ)
  ╠═╡ =#

# ╔═╡ 418b9570-9d55-4a14-bff4-127ce096f8dd
#=╠═╡
display_jal_tabular_solution_1(game_2_jal)
  ╠═╡ =#

# ╔═╡ d9a1cdf1-3261-4b4f-8aae-ea8c2fde5893
md"""
### Approximation MARL Solutions

"""

# ╔═╡ dd25a1d7-62d5-4280-8ce2-abfa185a39c6
md"""
#### Independent DQN

Similar to IQL, IDQN cannot represent the joint value function due to the lack of a linear decomposition.  We can see the same dependence on the exploration parameter ``\epsilon`` on the learned solution as we saw in IQL.
"""

# ╔═╡ 77a9b132-0694-4051-a363-05d98a1f271f
#=╠═╡
md"""
``\epsilon`` select: $(@bind game_2_dqn_ϵ Slider(0.1f0:0.1f0:1f0; show_value = true))
"""
  ╠═╡ =#

# ╔═╡ 5d570d21-c334-46cc-a4d1-59073c698a7e
#=╠═╡
const game_2_idqn = independent_dqn_linear(state_matrix_game_2, 1f0, typemax(Int64), 20_000, game_1_feature_setup...; batch_size = 128, buffer_size = 1_000, α = 3f-4, ϵ = game_2_dqn_ϵ)
  ╠═╡ =#

# ╔═╡ 12654916-76a2-4658-b08b-b8a27b8db7b6
#=╠═╡
const game_2_idqn2 = independent_dqn_linear(state_matrix_game_2, 1f0, typemax(Int64), 20_000, game_1_dense_feature_setup...; batch_size = 128, buffer_size = 1_000, α = 3f-4, ϵ = game_2_dqn_ϵ)
  ╠═╡ =#

# ╔═╡ f1b23597-0adc-4487-851e-24e83c55096c
#=╠═╡
const game_2_idqn3 = independent_dqn_fcann(state_matrix_game_2, 1f0, typemax(Int64), 20_000, game_1_dense_feature_setup..., [64, 64]; batch_size = 128, buffer_size = 1_000, α = 3f-4, ϵ = game_2_dqn_ϵ)
  ╠═╡ =#

# ╔═╡ 730bc4c4-dcb0-443d-91e9-49097d006007
#=╠═╡
display_joint_solution_1(game_2_idqn)
  ╠═╡ =#

# ╔═╡ 91cf05d3-2d82-43a6-adbb-c7643b5d3353
#=╠═╡
display_joint_solution_1(game_2_idqn2)
  ╠═╡ =#

# ╔═╡ 6f1ee1cc-1a45-4414-8fbf-f84227533a4d
#=╠═╡
display_joint_solution_1(game_2_idqn3)
  ╠═╡ =#

# ╔═╡ 9cd2fd7d-164b-45cd-b6fc-91aa16138c32
md"""
#### Value Decomposition Networks

VDN learns a different decomposition than IDQN but still preserves the ordinal ranking of joint action values.  It also is affected by the exploration parameter for the learned values but not the learned policy.  The effect of ``\epsilon`` seems to be less so than for IDQN possibly due to the explicit use of decomposition in the loss function.
"""

# ╔═╡ ad25bc92-927d-46c0-af03-528f9c77f7d3
#=╠═╡
md"""
``\epsilon`` select: $(@bind game_2_vdn_ϵ Slider(0.1f0:0.1f0:1f0; show_value = true, default = 1f0))
"""
  ╠═╡ =#

# ╔═╡ 269d96ac-529e-4d2f-b069-59283c80133f
#=╠═╡
const game_2_vdn = independent_vdn_linear(state_matrix_game_2, 1f0, typemax(Int64), 20_000, game_1_feature_setup...; batch_size = 128, buffer_size = 1_000, α = 3f-4, ϵ = game_2_vdn_ϵ)
  ╠═╡ =#

# ╔═╡ bd9b4d61-04f3-4092-aea5-5e1e259632b8
#=╠═╡
const game_2_vdn2 = independent_vdn_linear(state_matrix_game_2, 1f0, typemax(Int64), 20_000, game_1_dense_feature_setup...; batch_size = 128, buffer_size = 1_000, α = 3f-4, ϵ = 1.0f0)
  ╠═╡ =#

# ╔═╡ 269eddb2-e45a-4c31-808b-53496d16ca6c
#=╠═╡
const game_2_vdn3 = independent_vdn_fcann(state_matrix_game_2, 1f0, typemax(Int64), 20_000, game_1_dense_feature_setup..., [64, 64]; batch_size = 128, buffer_size = 1_000, α = 3f-4, ϵ = 1.0f0)
  ╠═╡ =#

# ╔═╡ 4e76bb8e-67b7-4c39-8401-a431ff2080c9
#=╠═╡
display_joint_solution_1(game_2_vdn)
  ╠═╡ =#

# ╔═╡ 08e473bd-8e08-4c54-b42f-31609b870712
#=╠═╡
display_joint_solution_1(game_2_vdn2)
  ╠═╡ =#

# ╔═╡ d6ec5d03-7f7d-48a8-8b2b-4eee61290c94
#=╠═╡
display_joint_solution_1(game_2_vdn3)
  ╠═╡ =#

# ╔═╡ dba11945-938d-46eb-905f-b09c229d2f13
md"""
## Example 3: Climbing Game

This game's joint-action-value function cannot be represented by a linear decomposition or a monotonic decomposition, thus it may present unique challenges for techniques that do not explicitely model joint-action-values.
"""

# ╔═╡ 6c36b446-bd18-47e8-95cb-915c7c28f24e
# ╠═╡ skip_as_script = true
#=╠═╡
const matrix_actions_3 = [1, 2, 3]
  ╠═╡ =#

# ╔═╡ 5b856518-01b1-4346-bec1-9605d9c152f3
# ╠═╡ skip_as_script = true
#=╠═╡
const reward_matrix_3 = [11f0 -30f0 0f0; -30f0 7f0 0f0; 0f0 6f0 5f0]
  ╠═╡ =#

# ╔═╡ 64e55266-3f69-49f8-90cc-afe267c5f0a3
#=╠═╡
const tabular_matrix_game_3 = create_non_repeated_game(ntuple(i -> reward_matrix_3 ./ 2, 2), ntuple(i -> matrix_actions_3, 2))
  ╠═╡ =#

# ╔═╡ 8585c505-f9df-4838-9be5-c0d22563cc99
#=╠═╡
const tabular_matrix_mdp_3 = TabularMDP(tabular_matrix_game_3, sum) #creates an MDP out of the game by assigning the sum of rewards as the shared common reward for the whole environment
  ╠═╡ =#

# ╔═╡ 3ef242ab-4b8f-4ab5-ad7b-f50f6475eb59
#=╠═╡
const state_matrix_game_3 = StateStochasticGame(tabular_matrix_game_3)
  ╠═╡ =#

# ╔═╡ 969f5700-9713-47de-8739-bebea01c192c
md"""
### Exact Solution
"""

# ╔═╡ 3587e8d3-cf10-4dc9-a413-cd62fc829c50
#=╠═╡
const game_3_exact = value_iteration_q(tabular_matrix_mdp_3, 1f0)
  ╠═╡ =#

# ╔═╡ 9668706d-9c23-49ae-994b-a807371df430
#=╠═╡
function display_exact_solution_3(output::NamedTuple)
	output_values = zeros(Float32, 3, 3)
	output_policy = zeros(Float32, 3, 3)
	for i_a in eachindex(tabular_matrix_mdp_3.actions)
		a1, a2 = tabular_matrix_mdp_3.actions[i_a]
		output_values[a1, a2] = output.final_value[i_a, 1]
		output_policy[a1, a2] = output.optimal_policy[i_a, 1]
	end
	return (joint_action_values = output_values, optimal_policy = output_policy)
end
  ╠═╡ =#

# ╔═╡ c0cb489a-c0a6-4bb7-bf83-b57ff83d28f7
#=╠═╡
display_exact_solution_3(game_3_exact)
  ╠═╡ =#

# ╔═╡ 78bcaadf-f11e-4ea3-b1c4-7d8f6e675a01
md"""
### Tabular MARL Solutions
"""

# ╔═╡ 2defab11-b88d-4b19-b012-8e14c0183fdb
md"""
#### Independent Q-learning

We can see depending on the value of ``\epsilon`` the value functions converge to different optimal policies and the learning process is not stable.
"""

# ╔═╡ cb1ac0db-8b20-4160-b73c-ef9c3602ab16
#=╠═╡
md"""
``\epsilon`` select: $(@bind game_3_iql_ϵ Slider(0.1f0:0.1f0:1f0; show_value = true))
"""
  ╠═╡ =#

# ╔═╡ d0df8f15-d070-43a4-b77c-caacd8d57cac
#=╠═╡
const game_3_iql = independent_q_learning(tabular_matrix_game_3, 1f0; α = 3f-4, max_steps = 1_000_000, ϵ = game_3_iql_ϵ)
  ╠═╡ =#

# ╔═╡ 1ec77176-f122-41a7-bf02-bdf7c6c02868
# ╠═╡ skip_as_script = true
#=╠═╡
function display_joint_tabular_solution_3(output::NamedTuple)
	v1 = round.(Float64.(output.value_estimates[1][:, 1]); sigdigits = 3)
	v2 = round.(Float64.(output.value_estimates[2][:, 1]); sigdigits = 3)

	p1 = round.(Float64.(output.policies[1][:, 1]); sigdigits = 3)
	p2 = round.(Float64.(output.policies[2][:, 1]); sigdigits = 3)
	
	values = md"""
	Learned Values
	
	||$(v2[1])|$(v2[2])|$(v2[3])|
	|---|---|---|---|
	|$(v1[1])|$(round(v1[1] + v2[1], sigdigits = 3))|$(round(v1[1] + v2[2]; sigdigits = 3))|$(round(v1[1] + v2[3]; sigdigits = 3))|
	|$(v1[2])|$(round(v1[2] + v2[1]; sigdigits = 3))|$(round(v1[2] + v2[2], sigdigits = 3))|$(round(v1[2] + v2[3], sigdigits = 3))|
	|$(v1[3])|$(round(v1[3] + v2[1]; sigdigits = 3))|$(round(v1[3] + v2[2], sigdigits = 3))|$(round(v1[3] + v2[3], sigdigits = 3))|
	"""

	policies = md"""
	Learned Policies
	
	||$(p2[1])|$(p2[2])|$(p2[3])|
	|---|---|---|---|
	|$(p1[1])|$(round(p1[1] * p2[1], sigdigits = 3))|$(round(p1[1] * p2[2]; sigdigits = 3))|$(round(p1[1] * p2[3]; sigdigits = 3))|
	|$(p1[2])|$(round(p1[2] * p2[1]; sigdigits = 3))|$(round(p1[2] * p2[2], sigdigits = 3))|$(round(p1[2] * p2[3], sigdigits = 3))|
	|$(p1[3])|$(round(p1[3] * p2[1]; sigdigits = 3))|$(round(p1[3] * p2[2], sigdigits = 3))|$(round(p1[3] * p2[3], sigdigits = 3))|
	"""

	(values, policies)
end
  ╠═╡ =#

# ╔═╡ c81112e5-f568-4be3-9fb8-50087d6d8ce2
#=╠═╡
display_joint_tabular_solution_3(game_3_iql)
  ╠═╡ =#

# ╔═╡ 1b779c48-d617-4a3e-9cc6-2476ed959b26
md"""
#### Joint-Action Learning

Despite the complexity of the matrix game, joint-action learning can represent the exact value function and thus the optimal policy for the case of a common reward game.  The values are also unaffected by the value of ``\epsilon``
"""

# ╔═╡ 5022a5ef-364b-4d08-8113-df217705e05d
#=╠═╡
md"""
``\epsilon`` select: $(@bind game_3_jal_ϵ Slider(0.1f0:0.1f0:1f0; show_value = true))
"""
  ╠═╡ =#

# ╔═╡ a85cfb03-ded6-44d7-9996-452729dd4f0b
#=╠═╡
const game_3_jal = jal_am(tabular_matrix_game_3, 1f0, typemax(Int64), 1_000_000; ϵ = game_3_jal_ϵ)
  ╠═╡ =#

# ╔═╡ 24e7ab3c-ed28-4a26-914a-9112eb6905b4
# ╠═╡ skip_as_script = true
#=╠═╡
function display_jal_tabular_solution_3(output::NamedTuple)
	joint_values = round.(Float64.(output.joint_action_values[1][:, :, 1] .+ output.joint_action_values[2][:, :, 1]); sigdigits = 3)

	p1 = round.(Float64.(output.policies[1][:, 1]); sigdigits = 3)
	p2 = round.(Float64.(output.policies[2][:, 1]); sigdigits = 3)
	
	values = md"""
	Learned Values
	
	|||||
	|---|---|---|---|
	||$(joint_values[1, 1])|$(joint_values[1, 2])|$(joint_values[1, 3])|
	||$(joint_values[2, 1])|$(joint_values[2, 2])|$(joint_values[2, 3])|
	||$(joint_values[3, 1])|$(joint_values[3, 2])|$(joint_values[3, 3])|
	"""

	policies = md"""
	Empirial Policies
	
	||$(p2[1])|$(p2[2])|$(p2[3])|
	|---|---|---|---|
	|$(p1[1])|$(round(p1[1] * p2[1], sigdigits = 3))|$(round(p1[1] * p2[2]; sigdigits = 3))|$(round(p1[1] * p2[3]; sigdigits = 3))|
	|$(p1[2])|$(round(p1[2] * p2[1]; sigdigits = 3))|$(round(p1[2] * p2[2], sigdigits = 3))|$(round(p1[2] * p2[3], sigdigits = 3))|
	|$(p1[3])|$(round(p1[3] * p2[1]; sigdigits = 3))|$(round(p1[3] * p2[2], sigdigits = 3))|$(round(p1[3] * p2[3], sigdigits = 3))|
	"""

	(values, policies)
end
  ╠═╡ =#

# ╔═╡ 5a1cac0b-81fb-4393-89cc-ffc380b9e1d1
#=╠═╡
display_jal_tabular_solution_3(game_3_jal)
  ╠═╡ =#

# ╔═╡ f4a6e254-5718-4b40-890e-4cea24075093
md"""
### Approximation MARL Solutions

"""

# ╔═╡ ad4b9d1e-275f-4f84-a0d7-ca37813cc25f
md"""
#### Independent DQN

Similar to IQL, IDQN cannot represent the joint value function due to the lack of a linear decomposition.  We can see the same dependence on the exploration parameter ``\epsilon`` on the learned solution as we saw in IQL.
"""

# ╔═╡ ccded9dc-5698-4452-89cd-aa4ba9c91705
#=╠═╡
md"""
``\epsilon`` select: $(@bind game_3_dqn_ϵ Slider(0.1f0:0.1f0:1f0; show_value = true))
"""
  ╠═╡ =#

# ╔═╡ 2070630a-8e3b-4c3f-9bb5-a545b7075f0f
#=╠═╡
const game_3_idqn = independent_dqn_linear(state_matrix_game_3, 1f0, typemax(Int64), 20_000, game_1_feature_setup...; batch_size = 128, buffer_size = 1_000, α = 3f-4, ϵ = game_3_dqn_ϵ)
  ╠═╡ =#

# ╔═╡ fc128165-6050-436f-83be-34a05914b7ea
md"""
#### Value Decomposition Networks

VDN like IDQN cannot learn an accurate value representation due to the lack of any monotonic or linear decomposition.  The learned policy is also not correct.
"""

# ╔═╡ ec532cb3-460a-4aa1-b735-a7d629ec93ac
#=╠═╡
md"""
``\epsilon`` select: $(@bind game_3_vdn_ϵ Slider(0.1f0:0.1f0:1f0; show_value = true, default = 1f0))
"""
  ╠═╡ =#

# ╔═╡ ee566b9c-e117-4c48-bc77-154a4b2c24e4
#=╠═╡
const game_3_vdn = independent_vdn_linear(state_matrix_game_3, 1f0, typemax(Int64), 20_000, game_1_feature_setup...; batch_size = 128, buffer_size = 1_000, α = 3f-4, ϵ = game_3_vdn_ϵ)
  ╠═╡ =#

# ╔═╡ 8ec95abd-cd0e-4b82-99e1-611ce282c792
# ╠═╡ skip_as_script = true
#=╠═╡
function display_joint_solution_3(output::NamedTuple)
	f1, f2 = output.value_functions

	v1 = round.(Float64.(f1(:play).action_values); sigdigits = 3)
	v2 = round.(Float64.(f2(:play).action_values); sigdigits = 3)

	p1 = copy(v1)
	p2 = copy(v2)
	make_greedy_policy!(p1)
	make_greedy_policy!(p2)
	
	values = md"""
	Learned Values
	
	||$(v2[1])|$(v2[2])|$(v2[3])|
	|---|---|---|---|
	|$(v1[1])|$(round(v1[1] + v2[1], sigdigits = 3))|$(round(v1[1] + v2[2]; sigdigits = 3))|$(round(v1[1] + v2[3]; sigdigits = 3))|
	|$(v1[2])|$(round(v1[2] + v2[1]; sigdigits = 3))|$(round(v1[2] + v2[2], sigdigits = 3))|$(round(v1[2] + v2[3], sigdigits = 3))|
	|$(v1[3])|$(round(v1[3] + v2[1]; sigdigits = 3))|$(round(v1[3] + v2[2], sigdigits = 3))|$(round(v1[3] + v2[3], sigdigits = 3))|
	"""

	policies = md"""
	Learned Policies
	
	||$(p2[1])|$(p2[2])|$(p2[3])|
	|---|---|---|---|
	|$(p1[1])|$(round(p1[1] * p2[1], sigdigits = 3))|$(round(p1[1] * p2[2]; sigdigits = 3))|$(round(p1[1] * p2[3]; sigdigits = 3))|
	|$(p1[2])|$(round(p1[2] * p2[1]; sigdigits = 3))|$(round(p1[2] * p2[2], sigdigits = 3))|$(round(p1[2] * p2[3], sigdigits = 3))|
	|$(p1[3])|$(round(p1[3] * p2[1]; sigdigits = 3))|$(round(p1[3] * p2[2], sigdigits = 3))|$(round(p1[3] * p2[3], sigdigits = 3))|
	"""

	(values, policies)
end
  ╠═╡ =#

# ╔═╡ 5201f52a-dd7f-4627-896d-569ef101dd7d
#=╠═╡
display_joint_solution_3(game_3_idqn)
  ╠═╡ =#

# ╔═╡ 3d92bdb9-3514-4b15-80f4-dd2cde565723
#=╠═╡
display_joint_solution_3(game_3_vdn)
  ╠═╡ =#

# ╔═╡ ea3ccd3b-b675-4cd6-8370-92fc830b07a2
md"""
# Policy-Based Learning
"""

# ╔═╡ fe612451-789b-48d1-8929-1b07a80eea12
md"""
## Utility Functions
"""

# ╔═╡ 8519a80f-02e5-4a63-808c-dcbac04ab186
begin
	function link_params(params::NTuple{N, Array{T, M}}) where {T<:Real, N, M}
		
	end
end

# ╔═╡ 980975ca-d28f-4169-8d61-739d7109501c
begin
	function form_policy_and_value_functions(game::StateStochasticGame{T, S, A, N, PTF, F1, F2}, feature_vectors::NTuple{N, V}, update_feature_vectors!::NTuple{N, Function}, policy_params::NTuple{N, Θ}, value_params::NTuple{N, W}) where {T<:Real, S, A, N, PTF, F1, F2, V, Θ, W}
		function π!(policy::Vector{T}, x::V, params::Θ, args...)
			ReinforcementLearning.update_policy_dist!(policy, x, params, args...)
			return policy
		end

		value_functions = [form_state_value_function(feature_vectors[i], update_feature_vectors![i], value_params[i]) for i in 1:N]

		v̂s = ntuple(i -> value_functions[i][1], N)
		form_value_kwargs = ntuple(i -> value_functions[2], N)
	
		form_policy_kwargs = ntuple(i -> () -> (feature_vector = deepcopy(feature_vectors[i]), policy = zeros(T, length(game.agent_actions[i])), policy_args = form_policy_args(policy_params[i])), N)
	
		πs = ntuple(N) do i
			function π(s::S; feature_vector::V = deepcopy(feature_vectors[i]), policy::Vector{T} = zeros(T, length(game.agent_actions[i])), policy_parameters::Θ = policy_params[i], policy_args = form_policy_args(policy_parameters), kwargs...) 
				update_feature_vectors![i](feature_vector, s)
				π!(policy, feature_vector, policy_parameters, policy_args...)
			end
		end
	
		π_samples = ntuple(N) do i
			function π_sample(s::S; kwargs...) 
				policy = πs[i](s; kwargs...)
				sample_action(policy)
			end
		end

		policies_and_values = ntuple(N) do i
			function policy_and_value(s::S; feature_vector::V = deepcopy(feature_vectors[i]), policy::Vector{T} = zeros(T, length(game.agent_actions[i])), policy_parameters::Θ = policy_params[i], value_parameters::W = value_params[i], policy_args = form_policy_args(policy_parameters), kwargs...)
				update_feature_vectors![i](feature_vector, s)
				ReinforcementLearning.update_policy_dist!(policy, feature_vector, policy_parameters, policy_args...)
				v = v̂s[i](feature_vector, value_parameters; kwargs...)
				return (value = v, policy_dist = policy)
			end
		end
	
		form_policy_and_value_kwargs = ntuple(i -> () -> (;form_value_kwargs[i]()..., form_policy_kwargs[i]()...), N)
	
		return (policy_functions = πs, form_policy_kwargs = form_policy_kwargs, value_functions = v̂s, form_value_kwargs = form_value_kwargs, policy_sample_actions = π_samples, policies_and_values = policies_and_values, form_policy_and_value_kwargs = form_policy_and_value_kwargs)
	end

	# function form_policy_and_value_function(mdp::StateMDP{T, S, A, PTF, F1, F2, F3}, feature_vector::Vector{T}, update_feature_vector!::Function, policy_parameters::FCANNParamsGPU, value_parameters::FCANNParamsGPU) where {T<:Real, S, A, PTF, F1, F2, F3}
	# 	cpu_policy_params = initialize_cpu_params(policy_parameters)
	# 	cpu_value_params = initialize_cpu_params(value_parameters)
	# 	gpu_policy_params = initialize_gpu_params(cpu_policy_params)
	# 	gpu_value_params = initialize_gpu_params(cpu_value_params)

	# 	v̂, form_value_kwargs = form_state_value_function(feature_vector, update_feature_vector!, value_parameters)


	# 	function π!(policy::Vector{T}, x, params, args...)
	# 		update_policy_dist!(policy, x, params, args...)
	# 		return policy
	# 	end
		
	# 	function π(s::S, params::FCANNParams{T}; feature_vector::Vector{T} = copy(feature_vector), policy::Vector{T} = zeros(T, length(mdp.actions)), policy_args_cpu = form_policy_args(params)) 
	# 		update_feature_vector!(feature_vector, s)
	# 		π!(policy, feature_vector, params, policy_args_cpu...)
	# 	end

	# 	function π(s::S, params::FCANNParamsGPU; feature_vector::Vector{T} = copy(feature_vector), d_x::FCANN.CUDAArray = FCANN.cuda_allocate(feature_vector), policy::Vector{T} = zeros(T, length(mdp.actions)), policy_args_gpu = form_policy_args(params)) 
	# 		update_feature_vector!(feature_vector, s)
	# 		FCANN.memcpy!(d_x, feature_vector)
	# 		π!(policy, d_x, params, policy_args_gpu...)
	# 	end

	# 	function π(s::S; policy_parameters::FCANNParams{T} = cpu_policy_params, policy_parameters_gpu::FCANNParamsGPU = gpu_policy_params, use_gpu::Bool = false, policy_kwargs_cpu::NamedTuple = NamedTuple(), policy_kwargs_gpu::NamedTuple = NamedTuple(), kwargs...) 
	# 		if !use_gpu
	# 			π(s, policy_parameters; policy_kwargs_cpu...)
	# 		else
	# 			π(s, policy_parameters_gpu; policy_kwargs_gpu...)
	# 		end
	# 	end

	# 	form_policy_kwargs_cpu() = (feature_vector = copy(feature_vector), policy = zeros(T, length(mdp.actions)), policy_args_cpu = form_policy_args(cpu_policy_params))
	# 	form_policy_kwargs_gpu() = (feature_vector = copy(feature_vector), d_x = FCANN.cuda_allocate(feature_vector), policy = zeros(T, length(mdp.actions)), policy_args_gpu = form_policy_args(gpu_policy_params))

	# 	form_policy_kwargs() = (policy_kwargs_cpu = form_policy_kwargs_cpu(), policy_kwargs_gpu = form_policy_kwargs_gpu())

	# 	function π_sample(s::S; kwargs...) 
	# 		policy = π(s; kwargs...)
	# 		sample_action(policy)
	# 	end

	# 	function policy_and_value(s::S, policy_parameters::FCANNParams{T}, value_parameters::FCANNParams{T}; feature_vector::Vector{T} = copy(feature_vector), policy::Vector{T} = zeros(T, length(mdp.actions)), policy_args_cpu = form_policy_args(policy_parameters), kwargs...)
	# 		update_feature_vector!(feature_vector, s)
	# 		update_policy_dist!(policy, feature_vector, policy_parameters, policy_args_cpu...)
	# 		v = v̂(feature_vector, value_parameters; kwargs...)
	# 		return (value = v, policy_dist = policy)
	# 	end

	# 	function policy_and_value(s::S, policy_parameters::FCANNParamsGPU, value_parameters::FCANNParamsGPU; feature_vector::Vector{T} = copy(feature_vector), d_x::FCANN.CUDAArray = FCANN.cuda_allocate(feature_vector), policy::Vector{T} = zeros(T, length(mdp.actions)), policy_args_gpu = form_policy_args(policy_parameters), kwargs...)
	# 		update_feature_vector!(feature_vector, s)
	# 		FCANN.memcpy!(d_x, feature_vector)
	# 		update_policy_dist!(policy, d_x, policy_parameters, policy_args_gpu...)
	# 		v = v̂(d_x, value_parameters; kwargs...)
	# 		return (value = v, policy_dist = policy)
	# 	end

	# 	function policy_and_value(s::S; policy_parameters::FCANNParams{T} = cpu_policy_params, policy_parameters_gpu::FCANNParamsGPU = gpu_policy_params, value_parameters::FCANNParams{T} = cpu_value_params, value_parameters_gpu::FCANNParamsGPU = gpu_value_params, use_gpu::Bool = false, kwargs_cpu::NamedTuple = NamedTuple(), kwargs_gpu::NamedTuple = NamedTuple(), kwargs...)
	# 		if !use_gpu
	# 			policy_and_value(s, policy_parameters, value_parameters; kwargs_cpu...)
	# 		else
	# 			policy_and_value(s, policy_parameters_gpu, value_parameters_gpu; kwargs_gpu...)
	# 		end
	# 	end

	# 	form_policy_and_value_kwargs_cpu() = (;form_policy_kwargs_cpu()..., value_parameters = cpu_value_params, value_activations = FCANN.form_activations(cpu_value_params.weights[1]))
		
	# 	form_policy_and_value_kwargs_gpu() = (;form_policy_kwargs_gpu()..., value_parameters = gpu_value_params, value_activations = FCANN.form_activations(gpu_value_params.weights[1]))

	# 	form_policy_and_value_kwargs() = (kwargs_cpu = form_policy_and_value_kwargs_cpu(), kwargs_gpu = form_policy_and_value_kwargs_gpu())

	# 	return (policy_function = π, form_policy_kwargs = form_policy_kwargs, value_function = v̂, form_value_kwargs = form_value_kwargs, policy_sample_action = π_sample, policy_and_value = policy_and_value, form_policy_and_value_kwargs = form_policy_and_value_kwargs)
	# end

	# function form_policy_and_value_function(mdp::StateMDP{T, S, A, PTF, F1, F2, F3}, feature_vector::FCANN.CUDAArray, update_feature_vector!::Function, policy_parameters::FCANNParamsGPU, value_parameters::FCANNParamsGPU) where {T<:Real, S, A, PTF, F1, F2, F3}
	# 	cpu_feature = FCANN.host_allocate(feature_vector)
	# 	form_policy_and_value_function(mdp, cpu_feature, update_feature_vector!, policy_parameters, value_parameters)
	# end
end

# ╔═╡ 441bc9b0-7971-40f8-bb01-cda7f941a806
md"""
## Independent Actor-Critic with Synchronous Environments
"""

# ╔═╡ 838f0029-4a62-44cf-a203-9b5d950cdf8a
function synchronous_independent_actor_critic!(policy_params::NTuple{N, PP}, value_params::NTuple{N, VP}, game::StateStochasticGame{T, S, A, N, PTF, F1, F2}, γ::T, max_steps::Integer, num_env::Integer, feature_vectors::NTuple{N, V}, update_feature_vectors!::NTuple{N, Function}, value_args::Tuple, value_gradient_args::Tuple, policy_args::Tuple, policy_gradient_args::Tuple; α_w::T = one(T)/10, α_θ::T = one(T)/10, nstep::Integer = 0, ∇v̂::NTuple{N, VP} = deepcopy(value_params), ∇lnπ::NTuple{N, PP} = deepcopy(policy_params)) where {T<:Real, S, A, N, PTF, F1, F2, VP, PP, V}

	# iszero(N) && return synchronous_actor_critic!(policy_params, value_params, mdp, γ, max_steps, num_env, feature_vector, update_feature_vector!, value_args, value_gradient_args, policy_args, policy_gradient_args; α_w = α_w, α_θ = α_θ, ∇v̂ = ∇v̂, ∇lnπ = ∇lnπ)
	
	episode_steps = ntuple(_ -> Vector{Int64}(), N)
	episode_rewards = ntuple(_ -> Vector{T}(), N)
	avg_step_rewards = ntuple(_ -> Vector{T}(), N)

	#initialize variables
	batch_episodes = ones(Int64, num_env)
	batch_episode_steps = [Vector{Int64}() for _ in 1:num_env]
	batch_episode_rewards = ntuple(_ -> [Vector{T}() for _ in 1:num_env], N)
	rtots = zeros(T, N, num_env)
	cs = ones(T, num_env)
	batch_states = [game.initialize_state() for _ in 1:num_env]
	current_feature_vectors = ntuple(i -> ReinforcementLearning.initialize_synchronous_features(feature_vectors[i], num_env), N) #should store the feature vectors of the current time state for that environment
	update_feature_vectors = ntuple(i -> ReinforcementLearning.initialize_synchronous_features(feature_vectors[i], num_env), N) #should store the feature vectors of the state being updated
	policy_matrix = ntuple(i -> zeros(T, num_env, length(game.agent_actions[i])), N)
	batch_actions = ntuple(_ -> ones(Int64, num_env), N)
	batch_state_values = ntuple(_ -> zeros(T, num_env), N)
	batch_targets = ntuple(_ -> zeros(T, num_env), N)
	δs = ntuple(_ -> zeros(T, num_env), N)
	row_sums = zeros(T, num_env)
	row_mins = zeros(T, num_env)
	row_maxes = zeros(T, num_env)

	batch_nstep_rewards = ntuple(_ -> [CircularBuffer{T}(N+1) for _ in 1:num_env], N)
	batch_nstep_states = ntuple(_ -> [CircularBuffer{S}(N+1) for _ in 1:num_env], N)
	batch_nstep_actions = ntuple(_ -> [CircularBuffer{Int64}(N+1) for _ in 1:num_env], N)
	batch_bootstrap_discount = ones(T, N, num_env)
	batch_ready = fill(false, num_env) #tracks for each environment if it is ready for a batch update.  initially this will not be true for any before not enough n-step data has been accumulated yet
	batch_terminal_check = fill(false, num_env) #tracks for each environment if the current episode has terminated or not
	update_actions = ntuple(_ -> fill(0, num_env), N)

	for (i, s) in enumerate(batch_states)
		for k in 1:N
			update_feature_vectors![k](feature_vectors[k], s)
			ReinforcementLearning.update_feature_matrix!(current_feature_vectors[k], feature_vectors[k], i)
		end
	end

	num_updates = 0
	batch_steps = fill(0, num_env)
	
	while num_updates < max_steps
		# @info "Current batch states: $batch_states"
		# @info "Using a policy matrix of $policy_matrix sampled the following actions: $batch_actions"

		#for each environment update the policy distribution on a per row basis and then sample an action from each environment
		if !all(batch_ready) && !all(batch_terminal_check) #only envs that are NOT ready perform a step update so if all are ready we can just proceed straight to gradient updates
			for i in 1:N
				ReinforcementLearning.update_batch_policy_dist!(policy_matrix[i], current_feature_vectors[i], policy_params[i], row_sums, row_mins, row_maxes, policy_args[i]...)
				ReinforcementLearning.sample_batch_actions!(batch_actions[i], policy_matrix[i])
			end

			#each environment has a tuple of joint actions for all of the agents in that environment
			env_joint_actions = ntuple(num_env) do k
				ntuple(i -> batch_actions[i][k], N)
			end
		end
		
		r_avg = zeros(T, N) 
		#perform transitions for entire batch
		for k in 1:num_env
			if !batch_ready[k] #only update if the batch is not ready from the previous step
				if !batch_terminal_check[k] #only take a new step if the environment hasn't terminated yet
					(r, s′) = game.ptf(batch_states[k], env_joint_actions[k])
					batch_steps[k] += 1
					for i in 1:N
						push!(batch_nstep_actions[i][k], batch_actions[i][k])
						push!(batch_nstep_states[i][k], batch_states[k])
						push!(batch_nstep_rewards[i][k], r[i])
						rtots[i, k] += r[i]
						r_avg[i] += r[i]
					end
					batch_states[k] = s′
					terminal = game.isterm(s′)

					batch_terminal_check[k] = terminal
					
					if !terminal
						for i in 1:N
							update_feature_vectors![i](feature_vectors[i], s′)
							ReinforcementLearning.update_feature_matrix!(current_feature_vectors[i], feature_vectors[i], k)
							batch_bootstrap_discount[i, k] = γ^(length(batch_nstep_rewards[i][k]))
						end
					else
						batch_ready[k] = true
						batch_bootstrap_discount[k] = zero(T)
					end
						
					#if the current buffer is full or the current state is terminal then this environment is ready for a batch update
					if (length(batch_nstep_rewards[1][k]) == N + 1) || terminal
						batch_ready[k] = true
					end
				elseif length(batch_nstep_rewards[1][k]) > 1 # the environment is needed for the next gradient update, but it has already terminated, so we need to remove items from the buffer and update the feature vector for the gradient update 
					for i in 1:N
						popfirst!(batch_nstep_rewards[i][k])
						popfirst!(batch_nstep_states[i][k])
						popfirst!(batch_nstep_actions[i][k])
						batch_bootstrap_discount[i, k] = zero(T)
					end
					batch_ready[k] = true
				else #the environment is needed for the next gradient update but the buffers are empty so we need to initialize a new episode
					for i in 1:N
						popfirst!(batch_nstep_rewards[i][k])
						popfirst!(batch_nstep_states[i][k])
						popfirst!(batch_nstep_actions[i][k])
					end
					s′ = game.initialize_state()
					
					batch_episodes[k] += 1
					push!(batch_episode_steps[k], batch_steps[k])
					for i in 1:N
						update_feature_vectors![i](feature_vectors[i], s′)
						ReinforcementLearning.update_feature_matrix!(current_feature_vectors[i], feature_vectors[i], k)
						push!(batch_episode_rewards[i][k], rtots[i, k])
						rtots[i, k] = zero(T)
						batch_bootstrap_discount[i, k] = zero(T)
					end
					cs[k] = one(T)
					batch_ready[k] = false
					batch_terminal_check[k] = false
					batch_states[k] = s′
				end
			end
		end

		#only update targets and gradient when the entire batch is ready
		if all(batch_ready)
			r_avg .= zero(T)
			for k in 1:num_env
				for i in 1:N
					update_feature_vectors![i](feature_vectors[i], first(batch_nstep_states[1][k]))
					ReinforcementLearning.update_feature_matrix!(update_feature_vectors[i], feature_vectors[i], k)
					update_actions[i][k] = first(batch_nstep_actions[i][k])
					batch_targets[i][k] = sum(batch_nstep_rewards[i][k][t]*γ^(t-1) for t in eachindex(batch_nstep_rewards[i][k]); init = zero(T)) #update batch_targets with discounted reward sum for up to the previous N+1 rewards
					r_avg[i] += first(batch_nstep_rewards[i][k])
				end
			end

			for i in 1:N
				push!(avg_step_rewards[i], r_avg[i] / num_env)
			end
			
			for i in 1:N
				#calculate state values for current states
				ReinforcementLearning.update_batch_state_values!(batch_state_values[i], current_feature_vectors[i], value_params[i], value_args[i]...)
		
				#zero out prediction values for terminal states and add discounted value to reward
				batch_targets[i] .+= batch_bootstrap_discount[i] .* batch_state_values[i]
		
				#updates value gradient with the loss function and updates δs with the states values minus the target values for use later in the policy gradient calculation
				ReinforcementLearning.update_batch_value_gradient!(∇v̂[i], δs[i], value_params[i], batch_targets[i], update_feature_vectors[i], value_gradient_args[i]...)	
		
				#updates batch advantage values to use in policy gradient by multiplying by γ^n where n is the number of steps since the episode started
				δs[i] .*= cs
				
				#update value parameters using the value gradient
				ReinforcementLearning.update_params_with_gradient!(value_params[i], α_w, ∇v̂[i])
		
				# @info "Updating policy_params with the following information: δs = $δs, policy_matrix = $policy_matrix"
				#update policy parameters using the policy distribution, batch actions, and advantage values
				ReinforcementLearning.update_batch_policy_gradient!(∇lnπ[i], policy_params[i], δs[i], policy_matrix[i], update_actions[i], update_feature_vectors[i], policy_gradient_args[i]...)
				
				ReinforcementLearning.update_params_with_gradient!(policy_params[i], α_θ, ∇lnπ[i])
			end
	
			batch_ready .= false #once a gradient update has occured, inform all environments they need to perform a new step
			cs .*= γ
			num_updates += 1
		end

		
	end
	
	policy_and_value_components = form_policy_and_value_functions(game, feature_vectors, update_feature_vectors!, policy_params, value_params)

	#note that this step is a noop unless the gradients are gpu objects in which case they get deallocated
	for i in 1:N
		ReinforcementLearning.cleanup_gradient!(∇v̂[i])
		ReinforcementLearning.cleanup_gradient!(∇lnπ[i])
	end

	return (;avg_step_rewards = avg_step_rewards, batch_episodes = batch_episodes, batch_episode_steps = batch_episode_steps, batch_episode_rewards = batch_episode_rewards, policy_parameters = policy_params, value_parameters = value_params, policy_and_value_components...)
end

# ╔═╡ 211f5eb9-3494-4cd1-b113-d6a2fa64e3fe
md"""
### Linear Approximation
"""

# ╔═╡ 1ac8f069-76cb-4058-8a03-bbf317d25985
begin
	function synchronous_independent_actor_critic_linear(game::StateStochasticGame{T, S, A, N, PTF, F1, F2}, γ::T, max_steps::Integer, feature_vectors::NTuple{N, V}, update_feature_vectors!::NTuple{N, Function}; 
		init_value::T = zero(T), 
		share_value_params::Bool = false, 
		value_params::NTuple{N, VP} = if share_value_params
			@assert all(i -> length(feature_vectors[i]) == length(feature_vectors[1]), 2:N)
			let
				params = initialize_linear_parameters(feature_vectors[1], init_value)
				ntuple(i -> params, N)
			end
		else
			ntuple(i -> initialize_linear_parameters(feature_vectors[i], init_value), N)
		end,
		share_policy_params::Bool = false,
		policy_params::NTuple{N, PP} = if share_policy_params
			let
				params = initialize_linear_parameters(feature_vectors[1], length(game.agent_actions[1]), init_value)
				ntuple(i -> params, N)
			end
		else
			ntuple(i -> initialize_linear_parameters(feature_vectors[i], length(game.agent_actions[i]), init_value), N)
		end, 
		num_env::Integer = 8, 
		kwargs...) where {T<:Real, S, A, N, PTF, F1, F2, VP, PP, V} 
			#if using parameter sharing, the input and output dimensions must match for each agent
			share_policy_params || share_value_params && @assert all(i -> length(feature_vectors[i]) == length(first(feature_vectors)), 2:N)
			share_policy_params && @assert all(i -> length(game.agent_actions[i]) == length(game.agent_actions[1]), 2:N)
			synchronous_independent_actor_critic!(policy_params, value_params, game, γ, max_steps, num_env, feature_vectors, update_feature_vectors!, ntuple(_ -> (), N), ntuple(_ -> (), N), ntuple(_ -> (), N), ntuple(_ -> (), N); kwargs...)
	end

	synchronous_independent_actor_critic_linear(game::StateStochasticGame{T, S, A, N, PTF, F1, F2}, γ::T, max_steps::Integer, feature_vector, update_feature_vector!::Function; kwargs...) where {T<:Real, S, A, N, PTF, F1, F2} = synchronous_independent_actor_critic_linear(game, γ, max_steps, convert_to_ntuple(feature_vector, N), convert_to_ntuple(update_feature_vector!, N); kwargs...)
end

# ╔═╡ af61484b-c682-4da6-acbf-d073568398de
md"""
### Non-Linear Approximation
"""

# ╔═╡ c141e214-dd85-4035-aa1f-018e5d806856
begin
	function synchronous_independent_actor_critic_fcann(game::StateStochasticGame{T, S, A, N, PTF, F1, F2}, γ::T, max_steps::Integer, feature_vectors::NTuple{N, V}, update_feature_vectors!::NTuple{N, Function}, hidden_layers::Vector{Int64}; 
		reslayers::Integer = 0, 
		use_μP::Bool = true,
		share_policy_params::Bool = false,												
		policy_params::NTuple{N, FCANNParams{T}} = if share_policy_params
			let
				params = initialize_fcann_params(feature_vectors[1], hidden_layers, length(game.agent_actions[1]), reslayers, use_μP)
				ntuple(i -> params, N)
			end
		else
			ntuple(i -> initialize_fcann_params(feature_vectors[i], hidden_layers, length(game.agent_actions[i]), reslayers, use_μP), N)
		end, 
		share_value_params::Bool = false, 
		value_params::NTuple{N, FCANNParams{T}} = if share_value_params
			let
				params = initialize_fcann_value_params(policy_params[1], use_μP)
				ntuple(i -> params, N)
			end
		else
			ntuple(i -> initialize_fcann_value_params(policy_params[i], use_μP), N)
		end, 
		l2::T = zero(T), 
		dropout::T = zero(T), 
		num_env::Integer = 8, 
		activation_list::Vector{Bool} = fill(true, length(hidden_layers)), 
		use_gpu::Bool = false, 
		kwargs...) where {T<:Real, S, A, N, PTF, F1, F2, V}
		
		#if using parameter sharing, the input and output dimensions must match for each agent
		share_policy_params || share_value_params && @assert all(i -> length(feature_vectors[i]) == length(first(feature_vectors)), 2:N)
		share_policy_params && @assert all(i -> length(game.agent_actions[i]) == length(game.agent_actions[1]), 2:N)
		
		#add setup functions
		policy_setups = ntuple(i -> ReinforcementLearning.setup_fcann_batch_policy_arguments(policy_params[i], num_env, l2, dropout, use_μP, activation_list; use_gpu = use_gpu), N)
		value_setups = ntuple(i -> ReinforcementLearning.setup_fcann_batch_value_arguments(policy_setups[i], value_params[i], num_env, l2, dropout, use_μP, activation_list; use_gpu = use_gpu), N)
	
		!use_gpu && return synchronous_independent_actor_critic!(policy_params, value_params, game, γ, max_steps, num_env, feature_vectors, update_feature_vectors!, ntuple(i -> value_setups[i].value_args, N), ntuple(i -> value_setups[i].value_gradient_args, N), ntuple(i -> policy_setups[i].policy_args, N), ntuple(i -> policy_setups[i].policy_gradient_args, N); kwargs...)
	
		isempty(value_setups[1].gpu_args) && error("GPU backend is not available")
		isempty(policy_setups[1].gpu_args) && error("GPU backend is not available")

		#if using parameter sharing then replace all of the parameters for each agent with the first one
		if share_policy_params
			for i in 2:N
				policy_setups[i].gpu_args.params = policy_setups[1].gpu_args.params
			end
		end

		if share_value_params
			for i in 2:N
				value_setups[i].gpu_args.params = value_setups[1].gpu_args.params
			end
		end
	
		output = synchronous_independent_actor_critic!(ntuple(i -> policy_setups[i].gpu_args.params, N), ntuple(i -> value_setups[i].gpu_args.params, N), game, γ, max_steps, num_env, feature_vector, update_feature_vector!, ntuple(i -> value_setups[i].gpu_args..value_args, N), ntuple(i -> value_setups[i].gpu_args.value_gradient_args, N), ntuple(i -> policy_setups[i].gpu_args.policy_args, N), ntuple(i -> policy_setups[i].gpu_args.policy_gradient_args, N); kwargs...)
	
		for i in 1:N
			FCANN.GPU2Host(value_params[i].weights, value_setups[i].gpu_args.params.weights)
			FCANN.GPU2Host(policy_params[i].weights, policy_setups[i].gpu_args.params.weights)
	
			value_setups[i].gpu_args.cleanup_vars()
			policy_setups[i].gpu_args.cleanup_vars()
		end
	
		return (;output..., policy_parameters = deepcopy(policy_params), value_parameters = deepcopy(value_params))
	end

	synchronous_independent_actor_critic_fcann(game::StateStochasticGame{T, S, A, N, PTF, F1, F2}, γ::T, max_steps::Integer, feature_vector::V, update_feature_vector!::Function, hidden_layers::Vector{Int64}; kwargs...) where {T<:Real, S, A, N, PTF, F1, F2, V} = synchronous_independent_actor_critic_fcann(game, γ, max_steps, convert_to_ntuple(feature_vector, N), convert_to_ntuple(update_feature_vector!, N), hidden_layers; kwargs...)
end

# ╔═╡ 9128bc9b-55cb-48e1-afd4-91e9c70cc0fb
md"""
# Level-Based Foraging Example
"""

# ╔═╡ a92ce21c-b793-4b83-9312-77df54fa5ac8
md"""
## Performance Benchmark
"""

# ╔═╡ d44e801d-0789-4edf-82e1-66694e444cc8
# ╠═╡ skip_as_script = true
#=╠═╡
const lbf_env = LevelBasedForaging.make_environment(;num_agents = 2, width = 15, height = 15)
  ╠═╡ =#

# ╔═╡ e7626310-cf72-461b-ac31-6c61d56dcb53
#=╠═╡
const ep_output_test = runepisode(lbf_env; max_steps = 10_000)
  ╠═╡ =#

# ╔═╡ 04fab2e6-e685-4fee-afbe-73d013512c1d
#=╠═╡
@btime runepisode!($((ep_output_test[1], ep_output_test[2], ep_output_test[3])), $lbf_env; max_steps = 10_000)
  ╠═╡ =#

# ╔═╡ e0560b85-6e67-4ee1-a363-9b0e14367acd
10_000 / 110e-6

# ╔═╡ 3b9ea3c6-3623-493f-b7e8-6616c7f42506
#=╠═╡
const ep_output = runepisode(lbf_env; max_steps = 10_000)
  ╠═╡ =#

# ╔═╡ 146052ca-1d6e-4db3-b951-04283f35bae6
#=╠═╡
@bind epindex Slider(1:ep_output[5]; show_value=true)
  ╠═╡ =#

# ╔═╡ c8dbc747-7867-40e8-8c70-8355d5f6505d
md"""
## Test Environments

We will evaluate two environments, each with a grid size of 15 by 15.  Environment 1 will have two agents and two items with a maximum agent level of 2 and a maximum item level of 4.  Environment 2 will have three agents and three items with the same level constraints.
"""

# ╔═╡ b1dd46bf-0f0a-45f6-bf90-d1d7a328e7a1
# ╠═╡ skip_as_script = true
#=╠═╡
const lbf_task0 = LevelBasedForaging.make_environment(;num_agents = 2, num_items = 2, width = 8, height = 8, reset_chance = 0.01f0)
  ╠═╡ =#

# ╔═╡ 760a8733-a63a-4301-95be-94602fd98410
# ╠═╡ skip_as_script = true
#=╠═╡
const lbf_task1 = LevelBasedForaging.make_environment(;num_agents = 2, num_items = 2, width = 15, height = 15, reset_chance = 0.01f0)
  ╠═╡ =#

# ╔═╡ 45f45798-af9a-4cac-8f89-5b582cecbff1
# ╠═╡ skip_as_script = true
#=╠═╡
const lbf_task2 = LevelBasedForaging.make_environment(;num_agents = 3, num_items = 3, width = 15, height = 15, reset_chance = 0.01f0)
  ╠═╡ =#

# ╔═╡ 578e3515-b7cf-4a51-b120-d1091c4f20a1
#=╠═╡
const task1_ep_example = runepisode(lbf_task1)
  ╠═╡ =#

# ╔═╡ 12900bdd-6e0e-46c1-8272-5b395a402eaa
md"""
### Average Performance of Random Policies
"""

# ╔═╡ 5bd98306-d8b9-4cbf-b9c0-45aa7d730ff1
function eval_task(task; num_trials = 10_000, kwargs...)
	summarystats(1:num_trials |> Map(_ -> sum(a -> sum(a), runepisode(task; kwargs...)[3])) |> tcollect)
end

# ╔═╡ b6c965c8-db71-4004-968c-95cc4ea8f94e
#=╠═╡
eval_task(lbf_task0)
  ╠═╡ =#

# ╔═╡ db819d7b-c790-435e-a166-0c6093518ee4
#=╠═╡
eval_task(lbf_task1)
  ╠═╡ =#

# ╔═╡ 82f9e878-02ed-4cc6-b00a-2602df94a656
#=╠═╡
eval_task(lbf_task2; max_steps = 1_000_000)
  ╠═╡ =#

# ╔═╡ a1f90ae7-b08e-4517-ad90-634821bf7bd8
md"""
## Feature Vector Representation

We could use (X, Y) coordinates as a feature, but given the limited grid size, it may be more convenient to use sparse one-hot encoding for all the information including the positions of all the agents, items, and whether or not the items have been collected.  We can even represent the item and agent levels this way since there is a maximium value for both.
"""

# ╔═╡ 711befa2-ac12-4f96-8883-54a4228cad7b
md"""
### Sparse Feature Vector
"""

# ╔═╡ 8ca28ae4-ba1a-4b96-b2a8-e9e910b80a63
function make_lbf_sparse_feature(width::Integer, height::Integer, agent_levels::NTuple{N, Int64}, item_levels::NTuple{M, Int64}, max_agent_level::Integer, max_item_level::Integer) where {N, M}
	#first calculate the number of features and the number of active features
	active_features = N*3 + M*3 #we need an indicator for both agents and items for their x-position, y-position, and level
	num_features = N*(width+height+max_agent_level) + M*(width+height+max_item_level) #the number of indicators span the length, width, and max level for all agents and items

	#initialize feature vector and populate with placeholder active features, note that the number of active features will always be the same for this representation
	feature_vector = BinaryFeatureVector(num_features)
	for i in 1:active_features
		push!(feature_vector.active_features, 0)
	end

	function calc_agent_position_features(s::LevelBasedForaging.ForagingState{N, M, X, Y}, agent_index::Integer) where {X, Y}
		(x, y) = s.agent_positions[agent_index]
		base_ind = (agent_index - 1)*(X+Y)
		f1 = x + base_ind
		f2 = y + base_ind + X
		return f1, f2 #feature for the x and y position of a single agent
	end

	function calc_agent_level_features(s::LevelBasedForaging.ForagingState{N, M, X, Y}, agent_index::Integer) where {X, Y}
		level = s.agent_levels[agent_index]
		base_ind = N*(X + Y) + (agent_index - 1)*max_agent_level
		return base_ind + level #level feature for a single agent
	end

	function calc_item_position_features(s::LevelBasedForaging.ForagingState{N, M, X, Y}, item_index::Integer) where {X, Y}
		(x, y) = s.item_positions[item_index]
		base_ind = N*(X + Y + max_agent_level) + (item_index - 1)*(X+Y)
		f1 = x + base_ind
		f2 = y + base_ind + X
		return f1, f2 #feature for the x and y position of a single item
	end

	function calc_item_level_features(s::LevelBasedForaging.ForagingState{N, M, X, Y}, item_index::Integer) where {X, Y}
		level = s.item_levels[item_index]
		base_ind = N*(X + Y + max_agent_level) + M*(X + Y) + (item_index - 1)*max_item_level
		return base_ind + level #level feature for a single item
	end

	function update_feature_vector!(x::BinaryFeatureVector{Int64, F}, s::LevelBasedForaging.ForagingState{N, M, X, Y}) where {X, Y, F}
		active_index = 1
		for i in 1:N
			f1, f2 = calc_agent_position_features(s, i)
			x.active_features[active_index] = f1
			active_index += 1
			x.active_features[active_index] = f2
			active_index += 1
		end

		for i in 1:N
			f = calc_agent_level_features(s, i)
			x.active_features[active_index] = f
			active_index += 1
		end

		for i in 1:M
			if !s.item_collect[i]
				f1, f2 = calc_item_position_features(s, i)
				x.active_features[active_index] = f1
				active_index += 1
				x.active_features[active_index] = f2
				active_index += 1
			end
		end

		for i in 1:M
			if !s.item_collect[i]
				f = calc_item_level_features(s, i)
				x.active_features[active_index] = f
				active_index += 1
			end
		end
		x.num_features = active_index - 1
		return x
	end

	return (feature_vector, update_feature_vector!)
end

# ╔═╡ d92d3dcc-e792-493e-aa5c-4c73e93f1fda
function make_lbf_sparse_feature2(width::Integer, height::Integer, agent_levels::NTuple{N, Int64}, item_levels::NTuple{M, Int64}, max_agent_level::Integer, max_item_level::Integer) where {N, M}
	#first calculate the number of features and the number of active features
	active_features = N*3 + M*3 #we need an indicator for both agents and items for their x-position, y-position, and level
	num_features = N*(width+height+max_agent_level) + M*(width+height+max_item_level) #the number of indicators span the length, width, and max level for all agents and items

	#initialize feature vector 
	feature_vector = zeros(Float32, num_features)


	function calc_agent_position_features(s::LevelBasedForaging.ForagingState{N, M, X, Y}, agent_index::Integer) where {X, Y}
		(x, y) = s.agent_positions[agent_index]
		base_ind = (agent_index - 1)*(X+Y)
		f1 = x + base_ind
		f2 = y + base_ind + X
		return f1, f2 #feature for the x and y position of a single agent
	end

	function calc_agent_level_features(s::LevelBasedForaging.ForagingState{N, M, X, Y}, agent_index::Integer) where {X, Y}
		level = s.agent_levels[agent_index]
		base_ind = N*(X + Y) + (agent_index - 1)*max_agent_level
		return base_ind + level #level feature for a single agent
	end

	function calc_item_position_features(s::LevelBasedForaging.ForagingState{N, M, X, Y}, item_index::Integer) where {X, Y}
		(x, y) = s.item_positions[item_index]
		base_ind = N*(X + Y + max_agent_level) + (item_index - 1)*(X+Y)
		f1 = x + base_ind
		f2 = y + base_ind + X
		return f1, f2 #feature for the x and y position of a single item
	end

	function calc_item_level_features(s::LevelBasedForaging.ForagingState{N, M, X, Y}, item_index::Integer) where {X, Y}
		level = s.item_levels[item_index]
		base_ind = N*(X + Y + max_agent_level) + M*(X + Y) + (item_index - 1)*max_item_level
		return base_ind + level #level feature for a single item
	end

	function update_feature_vector!(x::Vector{T}, s::LevelBasedForaging.ForagingState{N, M, X, Y}) where {T<:Real, X, Y}
		x .= zero(T)
		for i in 1:N
			f1, f2 = calc_agent_position_features(s, i)
			x[f1] = one(T)
			x[f2] = one(T)
		end

		for i in 1:N
			f = calc_agent_level_features(s, i)
			x[f] = one(T)
		end

		for i in 1:M
			if !s.item_collect[i]
				f1, f2 = calc_item_position_features(s, i)
				x[f1] = one(T)
				x[f2] = one(T)
			end
		end

		for i in 1:M
			if !s.item_collect[i]
				f = calc_item_level_features(s, i)
				x[f] = one(T)
			end
		end
		return x
	end

	return (feature_vector, update_feature_vector!)
end

# ╔═╡ e631ed5a-1d49-4fc5-b57a-c0f74e33fdfb
function make_lbf_sparse_feature(s::LevelBasedForaging.ForagingState{N, M, X, Y}, max_agent_level::Integer, max_item_level::Integer) where {N, M, X, Y}
	make_lbf_sparse_feature(X, Y, s.agent_levels, s.item_levels, max_agent_level, max_item_level)
end

# ╔═╡ d3cb1724-0098-47a5-a7fe-2d6e456a4ce2
function make_lbf_sparse_feature2(s::LevelBasedForaging.ForagingState{N, M, X, Y}, max_agent_level::Integer, max_item_level::Integer) where {N, M, X, Y}
	make_lbf_sparse_feature2(X, Y, s.agent_levels, s.item_levels, max_agent_level, max_item_level)
end

# ╔═╡ d5e2603e-82ab-4f56-a990-ee07d6c704c4
#=╠═╡
x, f! = make_lbf_sparse_feature(lbf_task1.initialize_state(), 2, 4)
  ╠═╡ =#

# ╔═╡ 4fd570f9-0ee8-43d9-a0dc-687835c1ed34
#=╠═╡
x_sparse2, f_sparse2! = make_lbf_sparse_feature2(lbf_task1.initialize_state(), 2, 4)
  ╠═╡ =#

# ╔═╡ 9829e745-7fd3-42f3-bbd3-951250f98ee4
#=╠═╡
const task1_test_episode = runepisode(lbf_task1)
  ╠═╡ =#

# ╔═╡ a8cc486d-3608-4070-ab5b-26d3f2a24759
#=╠═╡
#notice the 12 features corresponding to the x position, y position, and level of the 2 agents and 2 items = 4*3 = 12
f!(x, task1_test_episode[1][1])
  ╠═╡ =#

# ╔═╡ 5ac88229-5547-4f2c-854d-c7868ad252e2
#=╠═╡
f_sparse2!(x_sparse2, task1_test_episode[1][1])
  ╠═╡ =#

# ╔═╡ f22173d1-b5c6-42bb-bf59-4d0ea25a43f5
#=╠═╡
sum(x_sparse2)
  ╠═╡ =#

# ╔═╡ c36c5629-c5ea-4c36-89f8-14fcb4fb3385
#=╠═╡
#in the terminal state both items have been collected so we can see that there are now only 6 active features.  the item features which are no longer active are still the same as before since they should not change throughout the episode other than becoming inactive
f!(x, task1_test_episode[4])
  ╠═╡ =#

# ╔═╡ 75e01ce3-4fee-4082-94bf-9a4fef46ab50
#=╠═╡
f_sparse2!(x_sparse2, task1_test_episode[4])
  ╠═╡ =#

# ╔═╡ be9dd54a-0460-47c8-9a0b-edd23abbe351
#=╠═╡
sum(x_sparse2)
  ╠═╡ =#

# ╔═╡ efcd3e16-5a30-4e56-9f3a-96adb9579b4e
#=╠═╡
function test_sparse_feature_vector()
	(states, _) = runepisode(lbf_task1; max_steps = 100_000)
	for s in states
		f!(x, s)
	end
end
  ╠═╡ =#

# ╔═╡ 295cfadc-06cd-4bcd-84a2-4afc37142896
#=╠═╡
test_sparse_feature_vector()
  ╠═╡ =#

# ╔═╡ 2017a885-c570-48d9-a75e-2e9abc5c9955
md"""
### Dense Feature Vector
"""

# ╔═╡ 5c6aced7-f987-4245-a04d-16df75fc1ebe
function make_lbf_dense_feature(width::Integer, height::Integer, agent_levels::NTuple{N, Int64}, item_levels::NTuple{M, Int64}, max_agent_level::Integer, max_item_level::Integer) where {N, M}
	#store the x and y coordinate for everything as well as the level as a value normalized to between 0 and 1
	l = N*3 + M*3

	maxlevel = max(max_agent_level, max_item_level)

	normalize_level(l) = Float32((l-1) / (maxlevel-1))

	normalize_x(x) = Float32(2*(x - (width / 2))/width)
	normalize_y(y) = Float32(2*(y - (height / 2))/height)

	feature_vector = zeros(Float32, l)

	function update_feature_vector!(feature_vector::Vector{Float32}, s::LevelBasedForaging.ForagingState{N, M, X, Y}) where {X, Y}
		index = 1
		for i in 1:N
			(x, y) = s.item_positions[i]
			feature_vector[index] = normalize_x(x)
			index += 1
			feature_vector[index] = normalize_y(y)
			index += 1
			l = s.agent_levels[i]
			feature_vector[index] = normalize_level(l)
			index += 1
		end

		for i in 1:M
			if !s.item_collect[i]
				(x, y) = s.item_positions[i]
				feature_vector[index] = normalize_x(x)
				index += 1
				feature_vector[index] = normalize_y(y)
				index += 1
				l = s.item_levels[i]
				feature_vector[index] = normalize_level(l)
				index += 1
			else
				feature_vector[index] = 0f0
				index += 1
				feature_vector[index] = 0f0
				index += 1
				feature_vector[index] = 0f0
				index += 1
			end
		end
		return feature_vector
	end
	return (feature_vector, update_feature_vector!)
end

# ╔═╡ 040dadb9-2e64-458c-a513-61bcf735d6b5
function make_lbf_dense_feature(s::LevelBasedForaging.ForagingState{N, M, X, Y}, max_agent_level::Integer, max_item_level::Integer) where {N, M, X, Y}
	make_lbf_dense_feature(X, Y, s.agent_levels, s.item_levels, max_agent_level, max_item_level)
end

# ╔═╡ 8322e025-efdb-49a2-9f75-069a3141c781
#=╠═╡
x2, f2! = make_lbf_dense_feature(lbf_task1.initialize_state(), 2, 4)
  ╠═╡ =#

# ╔═╡ a9a5d50b-ba31-4dcb-b3af-dfe5c2075cb4
#=╠═╡
#notice the 12 features corresponding to the x position, y position, and level of the 2 agents and 2 items = 4*3 = 12
f2!(x2, task1_test_episode[1][1])
  ╠═╡ =#

# ╔═╡ 904e1718-ea13-48fc-bfd6-183438ec480f
#=╠═╡
#in the terminal state both items have been collected so we can see that the 6 ending features corresponding to the items are all 0
f2!(x2, task1_test_episode[4])
  ╠═╡ =#

# ╔═╡ b4aaa02d-a4fb-42f7-bc46-416cec534cd6
md"""
## Central Learning Reductions
"""

# ╔═╡ daad86bb-0de9-4d04-a7ff-f338dad502fd
md"""
### Value Function Methods
"""

# ╔═╡ 69292ea5-f513-46b4-b6a2-6c5533d0e555
#=╠═╡
const task0_central = StateMDP(lbf_task0, sum)
  ╠═╡ =#

# ╔═╡ 0cbd429b-c674-4802-916a-d3aeef4632db
#=╠═╡
const task1_central = StateMDP(lbf_task1, sum)
  ╠═╡ =#

# ╔═╡ 07d3d939-6c64-46b3-9281-25e91d64174e
#=╠═╡
runepisode(task1_central)
  ╠═╡ =#

# ╔═╡ 703fd9fa-c323-4667-9a6d-a49d56c52ece
# ╠═╡ disabled = true
#=╠═╡
BLAS.set_num_threads(1)
  ╠═╡ =#

# ╔═╡ ad649468-40d4-4426-918f-d1e883ef4af7
#=╠═╡
const lbf_task1_nonlinear_value_setup_sparse = setup_episodic_value_nonlinear_training(task1_central, make_lbf_sparse_feature2(lbf_task1.initialize_state(), 2, 4)...; min_reward = 0f0)
  ╠═╡ =#

# ╔═╡ 9691981d-87b3-408c-8e47-b4e17bcd82b7
#=╠═╡
const lbf_task1_nonlinear_value_setup_dense = setup_episodic_value_nonlinear_training(task1_central, make_lbf_dense_feature(lbf_task1.initialize_state(), 2, 4)...; min_reward = 0f0)
  ╠═╡ =#

# ╔═╡ b32ccdf0-00b0-47ca-adc2-b5e53a9652c3
#=╠═╡
lbf_task1_nonlinear_value_setup_sparse.train([64, 64], 1, 0.99f0, 1f-3, 0.99f0, 100_000; use_dp = true)
  ╠═╡ =#

# ╔═╡ aa4c1f17-efe8-4d61-95a4-a31770b3631c
#=╠═╡
lbf_task1_nonlinear_value_setup_dense.train([64, 64], 1, 0.99f0, 1f-3, 0.99f0, 100_000; use_dp = true)
  ╠═╡ =#

# ╔═╡ 129c6306-598a-4e67-82ef-269037f201c3
#=╠═╡
const lbf_task1_dp_fcann_sparse = lbf_task1_nonlinear_value_setup_sparse.train_rate_decay([64, 64], 1, 0.99f0, 1f-2, 0.5f0, 100_000; use_dp = true, ϵ = 0.05f0)
  ╠═╡ =#

# ╔═╡ a840c721-d8d1-4d95-95c2-b9eaf41d6594
#=╠═╡
const lbf_task1_dp_fcann_dense = lbf_task1_nonlinear_value_setup_dense.train_rate_decay([64, 64], 1, 0.99f0, 1f-2, 0.5f0, 100_000; use_dp = true, ϵ = 0.05f0)
  ╠═╡ =#

# ╔═╡ 9cf6d7fc-32db-40f6-9a27-3ee1f8c8013c
# ╠═╡ disabled = true
#=╠═╡
const lbf_task1_dp_params = dp_λ_fcann(task1_central, 0.99f0, 0.9f0, typemax(Int64), 1_000, make_lbf_sparse_feature2(lbf_task1.initialize_state(), 2, 4)..., [64, 64]; reslayers = 1, α = 1f-3)
  ╠═╡ =#

# ╔═╡ 34321c84-3f64-478b-a6f0-e816f871dda9
# ╠═╡ disabled = true
#=╠═╡
const lbf_task1_dp = dp_λ_fcann(task1_central, 0.99f0, 0.95f0, typemax(Int64), 10_000_000, make_lbf_sparse_feature2(lbf_task1.initialize_state(), 2, 4)..., [64, 64]; reslayers = 1, α = 1f-2, ϵ = 0.25f0, params = lbf_task1_dp_params)
  ╠═╡ =#

# ╔═╡ aec8b1a3-16ff-45d9-a30e-5119acd8bc45
#=╠═╡
const lbf_task1_dp_ep = runepisode(task1_central; π = s -> lbf_task1_dp_fcann_sparse.value_function(s).maximizing_action, max_steps = 10_000)
  ╠═╡ =#

# ╔═╡ 113c2dab-d0e2-4318-b5a6-78c0868bfd77
#=╠═╡
const lbf_task1_dp_ep2 = runepisode(task1_central; π = s -> lbf_task1_dp_fcann_dense.value_function(s).maximizing_action, max_steps = 10_000)
  ╠═╡ =#

# ╔═╡ 0e4d5ce3-0695-4a87-bf9d-d182048df426
#=╠═╡
s0_test = task1_central.initialize_state()
  ╠═╡ =#

# ╔═╡ 7564cd5f-3df6-482b-8532-07294091db37
#=╠═╡
lbf_task1_dp_fcann_dense.value_function(s0_test)
  ╠═╡ =#

# ╔═╡ e8b848c7-a9a0-45b6-9599-7bfff194f6ff
#=╠═╡
test_av = lbf_task1_dp_fcann_dense.value_function(s0_test).action_values
  ╠═╡ =#

# ╔═╡ 0482bf4e-915c-4861-8a87-4c4cd6ce4fa0


# ╔═╡ 217fdcdd-69bd-425c-b305-7d9f4262377a
#=╠═╡
make_greedy_policy!(test_av)
  ╠═╡ =#

# ╔═╡ 6bf1d985-71ec-4e80-b3a9-b0d5e19745bc
#=╠═╡
const lbf_task1_linear_value_setup = setup_episodic_value_linear_training(task1_central, make_lbf_sparse_feature2(lbf_task1.initialize_state(), 2, 4)...; min_reward = 0f0)
  ╠═╡ =#

# ╔═╡ ce7f4ffa-7b0b-409f-bf41-9f32f6f8c6f8
#=╠═╡
const lbf_task1_dp_linear = lbf_task1_linear_value_setup.train(0.99f0, 1f-4, 0.95f0, 100_000; trace_type = DutchTrace(), use_dp = true)
  ╠═╡ =#

# ╔═╡ 89908964-28d9-4562-b17a-505fe5496636
#=╠═╡
const lbf_task1_dp_linear2 = lbf_task1_linear_value_setup.train_rate_decay(0.99f0, 1f-2, 0.99f0, 100_000; trace_type = DutchTrace(), use_dp = true)
  ╠═╡ =#

# ╔═╡ e248cbbd-e37e-4913-8800-f29490ab1f66
# ╠═╡ disabled = true
#=╠═╡
const lbf_task1_dp_linear = dp_λ_linear(task1_central, 0.99f0, 0.5f0, typemax(Int64), 100_000, make_lbf_sparse_feature2(lbf_task1.initialize_state(), 2, 4)...; α = 1f-4)
  ╠═╡ =#

# ╔═╡ 5c3e8713-f951-42f4-89a4-cd37b169b714
#=╠═╡
const lbf_task1_dp_ep_linear = runepisode(task1_central; π = s -> lbf_task1_dp_linear.value_function(s).maximizing_action, max_steps = 10_000)
  ╠═╡ =#

# ╔═╡ b1389ca7-9c87-4e6f-adcd-11dd079b9caf
#=╠═╡
sum(lbf_task1_dp_ep_linear[3])
  ╠═╡ =#

# ╔═╡ f1bbfc95-121f-4219-9f8f-1647037f59d1
#=╠═╡
const lbf_task1_sarsa_linear = sarsa_λ_linear(task1_central, 0.99f0, 0.9f0, typemax(Int64), 100_000, make_lbf_dense_feature(lbf_task1.initialize_state(), 2, 4)...; α = 1f-2)
  ╠═╡ =#

# ╔═╡ 911c3790-2e23-4036-be72-bb64ba6e4bbb
#=╠═╡
const lbf_task1_sarsa_ep_linear = runepisode(task1_central; π = s -> lbf_task1_sarsa_linear.value_function(s).maximizing_action, max_steps = 100_000)
  ╠═╡ =#

# ╔═╡ 55b8f267-8b6a-4a59-926a-e44e2b25fff7
#=╠═╡
const lbf_task1_sarsa = sarsa_λ_fcann(task1_central, 0.99f0, 0.95f0, typemax(Int64), 100_000, make_lbf_dense_feature(lbf_task1.initialize_state(), 2, 4)..., [64, 64]; reslayers = 1, α = 6f-4)
  ╠═╡ =#

# ╔═╡ 73efc1bb-4786-4ca2-8084-db08956ac82a
#=╠═╡
const lbf_task1_sarsa_ep = runepisode(task1_central; π = s -> lbf_task1_sarsa.value_function(s).maximizing_action, max_steps = 10_000)
  ╠═╡ =#

# ╔═╡ 1e2ddcbb-f2af-4555-b516-959ae647b343
md"""
### Policy Gradient Methods
"""

# ╔═╡ 65c9422c-3bb0-452b-bfc7-8cac4e4dc353
#=╠═╡
const lbf_task0_linear_policy_setup_sparse = setup_episodic_policy_linear_training(task0_central, make_lbf_sparse_feature2(lbf_task0.initialize_state(), 2, 4)...; min_reward = 0f0)
  ╠═╡ =#

# ╔═╡ 90a4b613-16dc-46b2-817a-bbbb83c17a32
#=╠═╡
const lbf_task0_nonlinear_policy_setup_sparse = setup_episodic_policy_nonlinear_training(task0_central, make_lbf_sparse_feature2(lbf_task0.initialize_state(), 2, 4)...; min_reward = 0f0)
  ╠═╡ =#

# ╔═╡ a0116b94-b01d-466b-93e3-96476d19fe47
#=╠═╡
const lbf_task1_linear_policy_setup_sparse = setup_episodic_policy_linear_training(task1_central, make_lbf_sparse_feature2(lbf_task1.initialize_state(), 2, 4)...; min_reward = 0f0)
  ╠═╡ =#

# ╔═╡ fbc1e086-18f9-4303-8e0c-6061cc4321a4
#=╠═╡
const lbf_task1_nonlinear_policy_setup_sparse = setup_episodic_policy_nonlinear_training(task1_central, make_lbf_sparse_feature2(lbf_task1.initialize_state(), 2, 4)...; min_reward = 0f0)
  ╠═╡ =#

# ╔═╡ 3067dfdc-a21f-4b0e-81bd-aee416934359
#=╠═╡
const lbf_task1_nonlinear_policy_setup_dense = setup_episodic_policy_nonlinear_training(task1_central, make_lbf_dense_feature(lbf_task1.initialize_state(), 2, 4)...; min_reward = 0f0)
  ╠═╡ =#

# ╔═╡ 1197f21f-58b4-4a65-a284-bce2c416a035
#=╠═╡
lbf_task0_linear_policy_setup_sparse.train_rate_decay(0.99f0, 1f-2, 1f-2, 0.95f0, 0.5f0, 100_000)
  ╠═╡ =#

# ╔═╡ 7d341dfd-9c8f-446d-b270-e461af194c3b
#=╠═╡
lbf_task0_linear_policy_setup_sparse.sync_train_rate_decay(0.99f0, 1f-2, 1f-2, 100_000; num_env = 8, N = 10)
  ╠═╡ =#

# ╔═╡ 71c8a440-4292-4e6d-a0c2-0bfb872775eb
#=╠═╡
lbf_task0_nonlinear_policy_setup_sparse.train_rate_decay([64, 64], 1, 0.99f0, 1f-2, 1f-2, 0.95f0, 0.5f0, 100_000)
  ╠═╡ =#

# ╔═╡ 213a681f-d1a4-4031-a138-769b42ae533a
#=╠═╡
lbf_task0_nonlinear_policy_setup_sparse.sync_train_rate_decay([64, 64], 1, 0.99f0, 1f-2, 1f-2, 100_000; num_env = 8, N = 10)
  ╠═╡ =#

# ╔═╡ f67dbf1c-7976-4dec-b6fb-d92cd740966c
#=╠═╡
lbf_task1_nonlinear_policy_setup_sparse.train([64, 64], 1, 0.99f0, 1f-3, 1f-3, 0.99f0, 0.99f0, 10_000)
  ╠═╡ =#

# ╔═╡ 384f9170-d63a-4b9e-9a9a-a3a7439aed3a
#=╠═╡
lbf_task1_nonlinear_policy_setup_dense.train([64, 64], 1, 0.99f0, 1f-3, 1f-3, 0.99f0, 0.99f0, 10_000)
  ╠═╡ =#

# ╔═╡ 29d9f218-e083-474b-af0b-64a3927e4db7
#=╠═╡
const lbf_task1_policy_gradient_fcann_sparse = lbf_task1_nonlinear_policy_setup_sparse.train_rate_decay([64, 64], 1, 0.99f0, 4f-1, 1f-1, 0.95f0, 0.95f0, 1_000_000)
  ╠═╡ =#

# ╔═╡ 89a63603-84a8-4e8c-92c2-de220da3b6ae
#=╠═╡
lbf_task1_nonlinear_policy_setup_sparse.sync_train_rate_decay([64, 64], 1, 0.99f0, 4f-1, 1f-1, 100_000; N = 10, num_env = 8)
  ╠═╡ =#

# ╔═╡ cac747d2-cd16-4988-8798-b2328daa162f
#=╠═╡
const lbf_task1_policy_gradient_fcann_dense = lbf_task1_nonlinear_policy_setup_dense.train_rate_decay([64, 64], 1, 0.99f0, 1f-3, 1f-3, 0.5f0, 0.5f0, 100_000)
  ╠═╡ =#

# ╔═╡ 3058ed0c-68ab-4f7e-a57e-11195f5e3887
#=╠═╡
eval_task(task1_central; π = lbf_task1_policy_gradient_fcann_sparse.policy_sample_action, max_steps = 100_000)
  ╠═╡ =#

# ╔═╡ 8f8df220-88a5-41d9-a838-270fbd5320af
#=╠═╡
eval_task(task1_central; π = lbf_task1_policy_gradient_fcann_dense.policy_sample_action, max_steps = 100_000)
  ╠═╡ =#

# ╔═╡ 7c7db0a7-bc44-428f-8566-9374d8ebe165
#=╠═╡
eval_task(task1_central)
  ╠═╡ =#

# ╔═╡ 864fb287-032c-4b65-839a-d5ddd4ef0a29
#=╠═╡
const lbf_task1_policy_gradient_fcann_sparse_ep = runepisode(task1_central; π = lbf_task1_policy_gradient_fcann_sparse.policy_sample_action, max_steps = 100_000)
  ╠═╡ =#

# ╔═╡ 3b2e0ecd-6f49-486e-95e0-00c743ae7fba
#=╠═╡
@bind lbf_task1_policy_gradient_fcann_sparse_ep_index Slider(1:min(100, length(lbf_task1_policy_gradient_fcann_sparse_ep[1])); show_value=true)
  ╠═╡ =#

# ╔═╡ 209e227d-d009-44c0-8568-cd08400d04fb
#=╠═╡
const lbf_task1_policy_gradient_fcann_dense_ep = runepisode(task1_central; π = lbf_task1_policy_gradient_fcann_dense.policy_sample_action, max_steps = 100_000)
  ╠═╡ =#

# ╔═╡ c96ac8c4-23a1-4a00-95ca-9e543206a8f4
#=╠═╡
@bind lbf_task1_policy_gradient_fcann_dense_ep_index Slider(1:min(100, length(lbf_task1_policy_gradient_fcann_dense_ep[1])); show_value=true)
  ╠═╡ =#

# ╔═╡ ae3964cc-6a80-4371-a3f3-bd712b47029c
# ╠═╡ disabled = true
#=╠═╡
const lbf_task1_policy_gradient_sync_fcann_dense = synchronous_nstep_actor_critic_fcann(task1_central, 0.99f0, 10_000_000, 64, make_lbf_dense_feature(task1_central.initialize_state(), 2, 4)..., [64, 64]; α_θ = 1f-2, α_w = 1f-2, reslayers = 1, N = 10, policy_params = lbf_task1_policy_gradient_sync_params)
  ╠═╡ =#

# ╔═╡ 728199b2-e239-4a75-b921-6ade95a45b78
#=╠═╡
eval_task(task1_central; π = lbf_task1_policy_gradient_sync_fcann_dense.policy_sample_action, max_steps = 100_000)
  ╠═╡ =#

# ╔═╡ fe44dcd6-5f08-49c4-9e3d-a77ce6d1dfa1
# ╠═╡ disabled = true
#=╠═╡
const lbf_task1_policy_gradient_sync_fcann_dense_ep = runepisode(task1_central; π = lbf_task1_policy_gradient_sync_fcann_dense.policy_sample_action, max_steps = 100_000)
  ╠═╡ =#

# ╔═╡ 2de57998-ed42-4882-8afd-2d067ca47fa3
#=╠═╡
@bind lbf_task1_policy_gradient_sync_fcann_dense_ep_index Slider(1:min(100, length(lbf_task1_policy_gradient_sync_fcann_dense_ep[1])); show_value=true)
  ╠═╡ =#

# ╔═╡ 31b5b269-8e71-4be6-94dd-4e4a80690792
#=╠═╡
plot_foraging_state_policy_and_value(lbf_task1_policy_gradient_sync_fcann_dense_ep, lbf_task1_policy_gradient_sync_fcann_dense, lbf_task1_policy_gradient_sync_fcann_dense_ep_index)
  ╠═╡ =#

# ╔═╡ 76f134bd-ec74-48d4-9562-e245a80234ff
#=╠═╡
const lbf_task1_policy_gradient_sync_sparse_params = synchronous_nstep_actor_critic_fcann(task1_central, 0.99f0, 100, 256, make_lbf_sparse_feature2(task1_central.initialize_state(), 2, 4)..., [64, 64]; α_θ = 1f-2, α_w = 1f-3, reslayers = 1, N = 10).policy_parameters
  ╠═╡ =#

# ╔═╡ 9b2c0f63-ff25-4f00-be47-33bb0bf6b335
# ╠═╡ disabled = true
#=╠═╡
const lbf_task1_policy_gradient_sync_fcann_sparse = synchronous_nstep_actor_critic_fcann(task1_central, 0.99f0, 10_000_000, 64, make_lbf_sparse_feature2(task1_central.initialize_state(), 2, 4)..., [64, 64]; α_θ = 8f-2, α_w = 8f-2, reslayers = 1, N = 100, policy_params = lbf_task1_policy_gradient_sync_sparse_params)
  ╠═╡ =#

# ╔═╡ b9039f56-ae1c-43f6-a847-37d862e47166
#=╠═╡
eval_task(task1_central; π = lbf_task1_policy_gradient_sync_fcann_sparse.policy_sample_action, max_steps = 1_000_000)
  ╠═╡ =#

# ╔═╡ d2e4b338-6726-4604-8775-332fc644b472
#=╠═╡
eval_task(task1_central; max_steps = 1_000_000)
  ╠═╡ =#

# ╔═╡ 99127002-79dc-4215-9acb-9a409f1314cb
#=╠═╡
const lbf_task1_policy_gradient_sync_fcann_sparse_ep = runepisode(task1_central; π = lbf_task1_policy_gradient_sync_fcann_sparse.policy_sample_action, max_steps = 1_000_000)
  ╠═╡ =#

# ╔═╡ 95fc4fe7-db30-4222-94d7-cbc573ec186f
#=╠═╡
@bind lbf_task1_policy_gradient_sync_fcann_sparse_ep_index Slider(1:length(lbf_task1_policy_gradient_sync_fcann_sparse_ep[1]); show_value=true)
  ╠═╡ =#

# ╔═╡ 3b616e5a-dcf0-49ae-a5b4-4d7079065b79
#=╠═╡
plot_foraging_state_policy_and_value(lbf_task1_policy_gradient_sync_fcann_sparse_ep, lbf_task1_policy_gradient_sync_fcann_sparse, lbf_task1_policy_gradient_sync_fcann_sparse_ep_index)
  ╠═╡ =#

# ╔═╡ 69fedc29-0202-486f-97e9-79b1507fc8dc
# ╠═╡ disabled = true
#=╠═╡
const lbf_task1_policy_gradient_init = actor_critic_with_eligibility_traces_fcann(task1_central, 0.99f0, 0.9f0, 0.9f0, typemax(Int64), 10, make_lbf_dense_feature(lbf_task1.initialize_state(), 2, 4)..., [64, 64]; reslayers = 1, α_θ = 6f-3, α_w = 6f-3)
  ╠═╡ =#

# ╔═╡ 59e46620-2e27-4eca-99c2-492b4211bb29
#=╠═╡
const lbf_task1_policy_gradient_policy_params = lbf_task1_policy_gradient_init.policy_parameters
  ╠═╡ =#

# ╔═╡ 01dad2d3-3911-4549-95de-8bf33ba860c1
#=╠═╡
const lbf_task1_policy_gradient_value_params = ReinforcementLearning.initialize_fcann_value_params(lbf_task1_policy_gradient_policy_params, true)
  ╠═╡ =#

# ╔═╡ 410fc812-baad-4a38-8899-e5ed2a78583e
#=╠═╡
begin
	lbf_task1_policy_gradient_value_params.weights[1][end] .= lbf_task1_policy_gradient_init.value_parameters.weights[1][end]
	lbf_task1_policy_gradient_value_params.weights[2][end] .= lbf_task1_policy_gradient_init.value_parameters.weights[2][end]
end
  ╠═╡ =#

# ╔═╡ 42391b1d-d4fd-438d-baf9-f17c83d5acab
# ╠═╡ disabled = true
#=╠═╡
const lbf_task1_policy_gradient1 = actor_critic_with_eligibility_traces_fcann(task1_central, 0.99f0, 0.9f0, 0.9f0, typemax(Int64), 10_000, make_lbf_dense_feature(lbf_task1.initialize_state(), 2, 4)..., [64, 64]; reslayers = 1, α_θ = 4f-2, α_w = 4f-2, policy_params = lbf_task1_policy_gradient_policy_params, value_params = lbf_task1_policy_gradient_value_params)
  ╠═╡ =#

# ╔═╡ 8f86f808-37d9-483b-94fd-55b222b51151
#=╠═╡
lbf_task1_policy_gradient1.episode_steps[2:end] .- lbf_task1_policy_gradient1.episode_steps[1:end-1] |> plot
  ╠═╡ =#

# ╔═╡ c1e87b5a-50f5-441a-9f31-2dad14e7005a
# ╠═╡ skip_as_script = true
#=╠═╡
function plot_foraging_state_policy_and_value(ep::Tuple, output::NamedTuple, index::Integer)
	s = ep[1][index]
	f = output.policy_and_value
	plot_foraging_state_policy_and_value(s, f)
end
  ╠═╡ =#

# ╔═╡ 9704ce53-01da-40ae-89a9-1c446fa4a0d7
# ╠═╡ skip_as_script = true
#=╠═╡
function plot_foraging_state_value(ep::Tuple, output::NamedTuple, index::Integer)
	s = ep[1][index]
	f = output.value_function
	plot_foraging_state_value(s, f)
end
  ╠═╡ =#

# ╔═╡ 7e4ff070-1cf1-4f9c-b3bc-fcc1a6e6f31e
#=╠═╡
const lbf_task1_dqn_linear = dqn_linear(task1_central, 0.99f0, typemax(Int64), 100_000, make_lbf_dense_feature(lbf_task1.initialize_state(), 2, 4)...; α = 1f-2, batch_size = 16, buffer_size = 1_000, nstep = 10, use_double_q = true)
  ╠═╡ =#

# ╔═╡ 3cbf1a5d-3e82-47a2-be83-4446551d1060
function BLAS.gemm!(O1::Char, O2::Char, α::T, A::Matrix{T}, B::Vector{V}, c::T, C::Matrix{T}) where {T<:Real, I<:Integer, F, V<:BinaryFeatureVector{I, F}}
	l = length(B)
	if !isone(c)
		C .*= c
	end
	@assert F == size(C, 2)
	if O1 == 'T'
		@assert size(A, 2) == size(C, 1)
		@assert size(A, 1) == l
		for l in 1:length(B)
			x = B[l]
			for i in 1:size(C, 1)
				for j in 1:x.num_features
					@inbounds @simd for k in 1:size(A, 1)
						C[i, x.active_features[j]] += α*A[k, i]
					end
				end
			end
		end
	elseif O1 == 'N'
		@assert size(A, 1) == size(C, 1)
		@assert size(A, 2) == l
		for l in 1:length(B)
			x = B[l]
			for j in 1:x.num_features
				for k in 1:size(A, 2)
					@inbounds @simd for i in 1:size(C, 1)
						C[i, x.active_features[j]] += α*A[i, k]
					end
				end
			end
		end
	end
	return C
end

# ╔═╡ afdd0004-f272-4be5-9d7b-2a0410b43aec
#i for C controls which row of A we are in
#j for C controls which col of B we are in
#for sparse features the cols of B are going to be sparse with only the active features for each row of B
#we can also iterate through the rows of B

#matrix A has one column for each example and one row for each feature of X so I need to take the dot product of 

# ╔═╡ c2efa04e-0688-40b9-8635-a0b8dfd7f089
# ╠═╡ disabled = true
#=╠═╡
const lbf_task1_dqn_fcann = dqn_fcann(task1_central, 0.99f0, typemax(Int64), 100_000, make_lbf_dense_feature(lbf_task1.initialize_state(), 2, 4)..., [64, 64]; reslayers = 1, α = 1f-3, batch_size = 16, ϵ = 0.25f0, buffer_size = 1_000, nstep = 10, use_double_q = true)
  ╠═╡ =#

# ╔═╡ 89e38e04-a704-4d38-8f27-2e0e966897f9
# ╠═╡ disabled = true
#=╠═╡
const lbf_task1_dqn_fcann2 = dqn_fcann(task1_central, 0.99f0, typemax(Int64), 100_000, make_lbf_sparse_feature2(lbf_task1.initialize_state(), 2, 4)..., [64, 64]; reslayers = 1, α = 1f-3, batch_size = 16, ϵ = 0.25f0, buffer_size = 1_000, nstep = 10, use_double_q = true)
  ╠═╡ =#

# ╔═╡ 69601769-d646-4af4-90cf-8c06eeedbc4d
# ╠═╡ disabled = true
#=╠═╡
const lbf_task1_dqn_fcann3 = dqn_fcann(task1_central, 0.99f0, typemax(Int64), 10_000, make_lbf_sparse_feature2(lbf_task1.initialize_state(), 2, 4)..., [64, 64]; reslayers = 1, α = 1f-3, batch_size = 16, ϵ = 0.25f0, buffer_size = 1_000, nstep = 10, use_double_q = true)
  ╠═╡ =#

# ╔═╡ 942d2d73-10ad-4a35-a619-0cbb957a8fd8
md"""
## Independent DQN
"""

# ╔═╡ f0903e44-4443-4751-bc45-9c62a7bde1d6
md"""
### Linear Approximation
"""

# ╔═╡ fb91e135-4474-41cf-bac0-936b124ffd27
#=╠═╡
const lbf_task1_idqn_linear = independent_dqn_linear(lbf_task1, 0.99f0, typemax(Int64), 100_000, make_lbf_sparse_feature2(lbf_task1.initialize_state(), 2, 4)...; α = 1f-3, ϵ = 0.1f0, batch_size = 64, buffer_size = 100, nstep = 10, use_double_q = true)
  ╠═╡ =#

# ╔═╡ 5a30701d-078f-4170-a7f8-75095248e5ca
#=╠═╡
const lbf_task1_idqn_ep = runepisode(lbf_task1; πs = extract_joint_policies(lbf_task1, lbf_task1_idqn_linear.value_functions; ϵ = 0.1f0), max_steps = 100_000)
  ╠═╡ =#

# ╔═╡ 9f9de138-6324-4740-b926-3172df10eb78
#=╠═╡
[sum(a[i] for a in lbf_task1_idqn_ep[3]) for i in 1:2]
  ╠═╡ =#

# ╔═╡ 7989fb06-dbd2-4a1f-bbc4-c36e3c79a639
md"""
### Non-linear Approximation
"""

# ╔═╡ 21a6c62f-76e6-424b-b373-9faf160d976d
#=╠═╡
const lbf_task0_idqn_fcann_params = independent_dqn_fcann(lbf_task0, 0.99f0, typemax(Int64), 1_000, make_lbf_sparse_feature2(lbf_task0.initialize_state(), 2, 4)..., [64, 64]; reslayers = 1, α = 1f-2, ϵ = 0.05f0, batch_size = 16, buffer_size = 1_000, nstep = 10, use_double_q = true).final_parameters
  ╠═╡ =#

# ╔═╡ 0f91b108-8de4-4811-ac4d-a30b318c836d
#=╠═╡
independent_dqn_fcann(lbf_task0, 0.99f0, typemax(Int64), 1_000, make_lbf_sparse_feature2(lbf_task0.initialize_state(), 2, 4)..., [64, 64]; reslayers = 1, α = 1f-2, ϵ = 0.05f0, batch_size = 16, buffer_size = 1_000, nstep = 10, use_double_q = true, share_experience = true).final_parameters
  ╠═╡ =#

# ╔═╡ 4fb82746-1b35-40a8-8953-bf62389d8673
#=╠═╡
const lbf_task1_idqn_fcann_params = independent_dqn_fcann(lbf_task1, 0.99f0, typemax(Int64), 1_000, make_lbf_sparse_feature2(lbf_task1.initialize_state(), 2, 4)..., [64, 64]; reslayers = 1, α = 1f-2, ϵ = 0.05f0, batch_size = 16, buffer_size = 1_000, nstep = 10, use_double_q = true).final_parameters
  ╠═╡ =#

# ╔═╡ d52dbcd7-7d99-4380-b538-8cca211b95a9
#=╠═╡
const lbf_task0_idqn_fcann = independent_dqn_fcann(lbf_task0, 0.99f0, typemax(Int64), 10_000, make_lbf_sparse_feature2(lbf_task0.initialize_state(), 2, 4)..., [64, 64]; reslayers = 1, α = 4f-4, ϵ = 0.25f0, batch_size = 128, buffer_size = 100_000, nstep = 10, use_double_q = true, value_params = lbf_task0_idqn_fcann_params)
  ╠═╡ =#

# ╔═╡ 1b2db9c2-8a28-4797-9c89-ea4cbcc25eb1
#=╠═╡
independent_dqn_fcann(lbf_task0, 0.99f0, typemax(Int64), 10_000, make_lbf_sparse_feature2(lbf_task0.initialize_state(), 2, 4)..., [64, 64]; reslayers = 1, α = 4f-4, ϵ = 0.25f0, batch_size = 128, buffer_size = 100_000, nstep = 10, use_double_q = true, value_params = lbf_task0_idqn_fcann_params, share_experience = true)
  ╠═╡ =#

# ╔═╡ 839f5283-5319-4ff9-8ec0-605f17803cea
#=╠═╡
const lbf_task0_idqn_ep2 = runepisode(lbf_task0; πs = extract_joint_policies(lbf_task0, lbf_task0_idqn_fcann.value_functions; ϵ = 0.05f0), max_steps = 100_000)
  ╠═╡ =#

# ╔═╡ 929f620b-955c-4f1a-8d1b-bc371f064113
#=╠═╡
const lbf_task1_idqn_fcann = independent_dqn_fcann(lbf_task1, 0.99f0, typemax(Int64), 10_000, make_lbf_sparse_feature2(lbf_task1.initialize_state(), 2, 4)..., [64, 64]; reslayers = 1, α = 1f-2, ϵ = 0.25f0, batch_size = 64, buffer_size = 100_000, nstep = 10, use_double_q = true, value_params = lbf_task1_idqn_fcann_params)
  ╠═╡ =#

# ╔═╡ 0ff27f26-4294-4661-84aa-690a1ff360ba
#=╠═╡
const lbf_task1_idqn_ep2 = runepisode(lbf_task1; πs = extract_joint_policies(lbf_task1, lbf_task1_idqn_fcann.value_functions; ϵ = 0.05f0), max_steps = 100_000)
  ╠═╡ =#

# ╔═╡ 780ce8ec-4efa-4132-862c-847c1b438859
#=╠═╡
[sum(a[i] for a in lbf_task1_idqn_ep2[3]) for i in 1:2]
  ╠═╡ =#

# ╔═╡ 4290fcb1-1ec8-4656-8290-d997ac5aceb5
md"""
## Value Decomposition Networks
"""

# ╔═╡ d5524ca0-4714-4701-bd30-3cb395c4820a
md"""
### Linear Approximation
"""

# ╔═╡ 017dfa67-14e8-4e17-82d4-799753f18591
#=╠═╡
const lbf_task0_vdn_linear = independent_vdn_linear(lbf_task0, 0.99f0, typemax(Int64), 1_000_000, make_lbf_sparse_feature2(lbf_task0.initialize_state(), 2, 4)...; α = 3f-4, ϵ = 0.05f0, batch_size = 128, buffer_size = 100_000, nstep = 10)
  ╠═╡ =#

# ╔═╡ 7bdd73ef-6f54-4cb3-9a88-9b0c7f58c93b
#=╠═╡
const lbf_task0_vdn_linear_ep = runepisode(lbf_task0; πs = extract_joint_policies(lbf_task0, lbf_task0_vdn_linear.value_functions; ϵ = 0.25f0), max_steps = 100_000)
  ╠═╡ =#

# ╔═╡ 80fd5886-2240-4fd3-a167-dd1741cf7e66
# ╠═╡ disabled = true
#=╠═╡
eval_task(lbf_task0; πs = extract_joint_policies(lbf_task0, lbf_task0_vdn_linear.value_functions; ϵ = 0.25f0))
  ╠═╡ =#

# ╔═╡ d225deb8-62d4-40a9-8da0-ba7c5a4d1058
md"""
### Non-linear Approximation
"""

# ╔═╡ caf32e13-4c35-4d69-8078-156ca8a12d74
#=╠═╡
const lbf_task0_vdn_fcann_params = independent_vdn_fcann(lbf_task0, 0.99f0, typemax(Int64), 1_000, make_lbf_sparse_feature2(lbf_task0.initialize_state(), 2, 4)..., [64, 64]; reslayers = 1, α = 1f-2, ϵ = 0.1f0, batch_size = 128, buffer_size = 1_000, nstep = 10).final_parameters
  ╠═╡ =#

# ╔═╡ fedb0c5e-7850-4dbc-bd10-e30b8a7c5160
#=╠═╡
const lbf_task0_vdn_nonlinear = independent_vdn_fcann(lbf_task0, 0.99f0, typemax(Int64), 1_000_000, make_lbf_sparse_feature2(lbf_task0.initialize_state(), 2, 4)..., [64, 64]; reslayers = 1, α = 3f-4, ϵ = 0.05f0, batch_size = 128, buffer_size = 100_000, nstep = 10, utility_params = lbf_task0_vdn_fcann_params)
  ╠═╡ =#

# ╔═╡ 7340f0bb-512c-4825-a263-ea264c5929bd
#=╠═╡
const lbf_task0_vdn_ep = runepisode(lbf_task0; πs = extract_joint_policies(lbf_task0, lbf_task0_vdn_nonlinear.value_functions; ϵ = 0.05f0), max_steps = 100_000)
  ╠═╡ =#

# ╔═╡ 9a718593-d9f5-419d-8d37-7f1a44b5f9cb
#=╠═╡
lbf_task0_vdn_nonlinear.value_functions[1](lbf_task0_vdn_ep[1][1])
  ╠═╡ =#

# ╔═╡ 566b34e2-d643-4eca-9867-89752949a018
md"""
## Independent Actor Critic
"""

# ╔═╡ 842523a6-91eb-4803-9a27-e46be1fcc402
md"""
### Linear Approximation
"""

# ╔═╡ 9ffbc773-6307-4c6a-b6bb-f9af72d06d97
#=╠═╡
const lbf_task1_iac_linear0 = synchronous_independent_actor_critic_linear(lbf_task0, 0.99f0, 100_000, make_lbf_sparse_feature2(lbf_task0.initialize_state(), 2, 4)...; α_θ = 5f-1, α_w = 1f-1, num_env = 8, nstep = 10)
  ╠═╡ =#

# ╔═╡ 1bd41b66-c2bc-4c70-98a5-5b24d5cb27da
#=╠═╡
lbf_task1_iac_linear0.avg_step_rewards |> t -> (t[1] .+ t[2]) ./ 2 |> mean
  ╠═╡ =#

# ╔═╡ 50b6330d-dd07-4261-850c-0d6a1d293238
#=╠═╡
const lbf_task1_iac_linear1 = synchronous_independent_actor_critic_linear(lbf_task1, 0.99f0, 100_000, make_lbf_sparse_feature2(lbf_task1.initialize_state(), 2, 4)...; α_θ = 5f-1, α_w = 1f-1, num_env = 8, nstep = 10)
  ╠═╡ =#

# ╔═╡ be126cde-b0ae-4b6f-bef4-0b4ecf4087a7
#=╠═╡
lbf_task1_iac_linear1.avg_step_rewards |> t -> (t[1] .+ t[2]) ./ 2 |> mean
  ╠═╡ =#

# ╔═╡ b619a8e2-2cda-47ab-bc77-24256e807b47
md"""
### Non-linear Approximation
"""

# ╔═╡ a7ff876d-3c77-42a4-bf62-37b4d53b0713
#=╠═╡
const lbf_task1_iac_nonlinear0 = synchronous_independent_actor_critic_fcann(lbf_task0, 0.99f0, 100_000, make_lbf_sparse_feature2(lbf_task0.initialize_state(), 2, 4)..., [64, 64]; reslayers = 1, α_θ = 8f-2, α_w = 4f-2, num_env = 8, nstep = 10)
  ╠═╡ =#

# ╔═╡ 5cc5fbbc-e982-4ebb-83e9-0eb632d40bb3
#=╠═╡
lbf_task1_iac_nonlinear0.avg_step_rewards |> t -> (t[1] .+ t[2]) ./ 2 |> mean
  ╠═╡ =#

# ╔═╡ 34d81780-75e0-4446-803f-8bc58b4f1c70
#=╠═╡
const lbf_task0_iac_fcann_ep = runepisode(lbf_task0; πs = lbf_task1_iac_nonlinear0.policy_sample_actions, max_steps = 100_000)
  ╠═╡ =#

# ╔═╡ 5dbc6867-cff6-4340-9316-0c76621530d9
#=╠═╡
const lbf_task1_iac_nonlinear1 = synchronous_independent_actor_critic_fcann(lbf_task1, 0.99f0, 100_000, make_lbf_sparse_feature2(lbf_task1.initialize_state(), 2, 4)..., [64, 64]; reslayers = 1, α_θ = 8f-2, α_w = 4f-2, num_env = 8, nstep = 10)
  ╠═╡ =#

# ╔═╡ 4aef6b1a-5fa9-47c0-8510-ca8e44845f8d
#=╠═╡
lbf_task1_iac_nonlinear1.avg_step_rewards |> t -> (t[1] .+ t[2]) ./ 2 |> mean
  ╠═╡ =#

# ╔═╡ f2ac6bef-4b36-4c58-946d-ea5a241a23e5
#=╠═╡
const lbf_task1_iac_fcann_ep = runepisode(lbf_task1; πs = lbf_task1_iac_nonlinear1.policy_sample_actions, max_steps = 100_000)
  ╠═╡ =#

# ╔═╡ 9dc092ea-9e74-4b28-b913-51c479b61cfd
#=╠═╡
lbf_task1_iac_nonlinear1.policy_functions[1](lbf_task1_iac_fcann_ep[1][2])
  ╠═╡ =#

# ╔═╡ 6317f0a0-1424-411b-bd46-4993cb28a55b
md"""
# Visualization Tools
"""

# ╔═╡ a39de0cb-c5a4-44bf-ab4b-2d394a00cac8
md"""
## Level-Based Foraging
"""

# ╔═╡ a1c545f6-04f3-4b6e-9b26-4b520d7bb6ef
#=╠═╡
function plot_foraging_state(s::LevelBasedForaging.ForagingState{N, M, X, Y}) where {N, M, X, Y}
	bottom_border = scatter(x = [0, X+1], y = [0, 0], mode = "lines", line_color = "black", showlegend = false, name = "Edge", line_width = 3)
	top_border = scatter(x = [0, X+1], y = [Y+1, Y+1], mode = "lines", line_color = "black", showlegend = false, name = "Edge", line_width = 3)
	left_border = scatter(x = [0, 0], y = [0, Y+1], mode = "lines", line_color = "black", showlegend = false, name = "Edge", line_width = 3)
	right_border = scatter(x = [X+1, X+1], y = [0, Y+1], mode = "lines", line_color = "black", showlegend = false, name = "Edge", line_width = 3)
	agent_xs = [s.agent_positions[i][1] for i in 1:N]
	agent_ys = [s.agent_positions[i][2] for i in 1:N]
	agent_trace = scatter(x = agent_xs, y = agent_ys, name = "Agents", mode = "markers")

	#finds the items that have not yet been collected
	item_inds = findall(.!s.item_collect)

	item_xs = [s.item_positions[i][1] for i in item_inds]
	item_ys = [s.item_positions[i][2] for i in item_inds]
	item_trace = scatter(x = item_xs, y = item_ys, name = "Items", mode = "markers")

	plot([agent_trace, item_trace, bottom_border, top_border, left_border, right_border], Layout(xaxis_title = "x", yaxis_title = "y", xaxis_range = [0, X+1], yaxis_range = [0, Y+1], yaxis_scaleanchor="x", width = 450, xaxis_dtick = 1, yaxis_dtick = 1, xaxis_tickvals = 1:X, xaxis_ticktext = string.(1:X), yaxis_tickvals = 1:Y, yaxis_ticktext = string.(1:Y), annotations = vcat([attr(x = s.agent_positions[i][1], y = s.agent_positions[i][2], text = s.agent_levels[i], showarrow = true,  font = attr(color = "blue", size = 14, weight = 1000, showdow = "auto")) for i in 1:N], [attr(x = s.item_positions[i][1], y = s.item_positions[i][2], text = s.item_levels[i], showarrow = true,  font = attr(color = "orange", size = 14, weight = 1000, showdow = "auto")) for i in item_inds])))
end
  ╠═╡ =#

# ╔═╡ 5b185f70-5a36-4404-ace7-32dc62b47452
#=╠═╡
plot_foraging_state(ep_output[1][epindex])
  ╠═╡ =#

# ╔═╡ e6b4e689-b567-4d8e-bc82-7ebec987e9fb
#=╠═╡
[plot_foraging_state(lbf_task1.initialize_state()), plot_foraging_state(lbf_task2.initialize_state())]
  ╠═╡ =#

# ╔═╡ 70ac1377-0624-4880-9f3b-a247bf904683
#=╠═╡
plot_foraging_state(task1_ep_example[1][1])
  ╠═╡ =#

# ╔═╡ 485de1c8-4614-4c0e-b514-4c3ecc9046e6
#=╠═╡
function plot_foraging_state_policy_and_value(s::LevelBasedForaging.ForagingState, f::Function)
	(plot_foraging_state(s), f(s))
end
  ╠═╡ =#

# ╔═╡ 7417aad8-2b02-4812-a244-af8fb638796a
#=╠═╡
plot_foraging_state_policy_and_value(lbf_task1_policy_gradient_fcann_sparse_ep, lbf_task1_policy_gradient_fcann_sparse, lbf_task1_policy_gradient_fcann_sparse_ep_index)
  ╠═╡ =#

# ╔═╡ 5a86d878-01c8-427e-9dbe-724e967c50b6
#=╠═╡
plot_foraging_state_policy_and_value(lbf_task1_policy_gradient_fcann_dense_ep, lbf_task1_policy_gradient_fcann_dense, lbf_task1_policy_gradient_fcann_dense_ep_index)
  ╠═╡ =#

# ╔═╡ 45d71799-e2bf-4664-8f8d-b2779bbd2737
#=╠═╡
function plot_foraging_state_value(s::LevelBasedForaging.ForagingState, f::Function)
	(plot_foraging_state(s), f(s))
end
  ╠═╡ =#

# ╔═╡ 9eae0c07-d09d-48a4-b1cc-63351171468f
#=╠═╡
plot_foraging_state_value(lbf_task1_dp_ep, lbf_task1_dp_fcann_sparse, 1)
  ╠═╡ =#

# ╔═╡ 5567e751-7562-4dc8-806e-17ad2b334be0
#=╠═╡
plot_foraging_state_value(lbf_task1_dp_ep2, lbf_task1_dp_fcann_dense, 1)
  ╠═╡ =#

# ╔═╡ c0cfcd9b-afef-4a50-a1d0-20f8a4425fbc
#=╠═╡
plot_foraging_state_value(lbf_task1_dp_ep_linear, lbf_task1_dp_linear, 1)
  ╠═╡ =#

# ╔═╡ 3bba2d87-232d-4d0d-9db8-e9ea540617a4
#=╠═╡
plot_foraging_state_value(lbf_task1_sarsa_ep, lbf_task1_sarsa, 1)
  ╠═╡ =#

# ╔═╡ cabaedf9-b357-4a98-b71f-97268f0f85cb
#=╠═╡
begin
	function plot_foraging_state_values(s::LevelBasedForaging.ForagingState, fs)
		(plot_foraging_state(s), [f(s) for f in fs])
	end

	function plot_foraging_state_values(ep::Tuple, output::NamedTuple, index::Integer)
		s = ep[1][index]
		f = output.value_functions
		plot_foraging_state_values(s, f)
	end
end
  ╠═╡ =#

# ╔═╡ 6c790648-5d37-44a2-ac18-5adfc7feafaa
#=╠═╡
plot_foraging_state_values(lbf_task1_idqn_ep2, lbf_task1_idqn_fcann, 1)
  ╠═╡ =#

# ╔═╡ 12aee9ee-616c-4360-94c8-dd12c9353bd3
#=╠═╡
function plot_lbf_episode(timestep)
	@htl("""
		 <div style = "display: flex; height: 420px;">
		 <div>
		 State Value Function: $(round(value_iter.final_value[vcat(value_iter_episode[1], value_iter_episode[4])[timestep]]; sigdigits = 3)), Numerical Annotations = Level
		 $(plot_foraging_state(tab_5_3.states[vcat(value_iter_episode[1], value_iter_episode[4])[timestep]]))
		 </div>
		 <div>
		 <div style = "width: 600px;">
		 Agent Rewards per Time Step
		 $(plot([scatter(x = [timestep], y = 0, mode = "markers", color = "red", name = "Current Timestep"), bar(y = [0; 0; value_iter_episode[3]], name = "Agent Reward Sum")], Layout(yaxis_title = "Reward", xaxis_title = "Episode Step", xaxis_range = [0, length(value_iter_episode[1])+2])))
		 </div>
		 </div>
		 """)
end
  ╠═╡ =#

# ╔═╡ 5acdbed5-a2a3-4455-abe6-83865fe63322
md"""
# Dependencies
"""

# ╔═╡ 283ce916-a355-44e4-b437-b4c262e35c36
html"""
<style>
	main {
		margin: 0 auto;
		max-width: min(1600px, 90%);
		padding-left: max(50px, 5%);
		padding-right: max(200px, 5%);
		font-size: max(10px, min(24px, 2vw));
	}
</style>
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
CSV = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
DataStructures = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
Dates = "ade2ca70-3891-5945-98fb-dc099432e06a"
HiGHS = "87dc4568-4c63-4d18-b0c0-bb2238e4078b"
HypertextLiteral = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
JuMP = "4076af6c-e467-56ae-b986-b466b2749572"
LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
PlutoDevMacros = "a0499f29-c39b-4c5c-807c-88074221b949"
PlutoPlotly = "8e989ff0-3d88-8e9f-f020-2b208a939ff0"
PlutoProfile = "ee419aa8-929d-45cd-acf6-76bd043cd7ba"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"

[compat]
BenchmarkTools = "~1.6.3"
CSV = "~0.10.16"
DataFrames = "~1.8.1"
DataStructures = "~0.19.3"
HiGHS = "~1.18.2"
HypertextLiteral = "~0.9.5"
JuMP = "~1.29.4"
LaTeXStrings = "~1.4.0"
PlutoDevMacros = "~0.9.2"
PlutoPlotly = "~0.6.5"
PlutoProfile = "~0.4.0"
PlutoUI = "~0.7.79"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.12.6"
manifest_format = "2.0"
project_hash = "d9e1e7191d00b7c8678ccf496e84f0c616dba773"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "6e1d2a35f2f90a4bc7c2ed98079b2ba09c35b83a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.3.2"

[[deps.AbstractTrees]]
git-tree-sha1 = "03e0550477d86222521d254b741d470ba17ea0b5"
uuid = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
version = "0.3.4"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.BenchmarkTools]]
deps = ["Compat", "JSON", "Logging", "Printf", "Profile", "Statistics", "UUIDs"]
git-tree-sha1 = "7fecfb1123b8d0232218e2da0c213004ff15358d"
uuid = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
version = "1.6.3"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1b96ea4a01afe0ea4090c5c8039690672dd13f2e"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.9+0"

[[deps.CSV]]
deps = ["CodecZlib", "Dates", "FilePathsBase", "InlineStrings", "Mmap", "Parsers", "PooledArrays", "PrecompileTools", "SentinelArrays", "Tables", "Unicode", "WeakRefStrings", "WorkerUtilities"]
git-tree-sha1 = "8d8e0b0f350b8e1c91420b5e64e5de774c2f0f4d"
uuid = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
version = "0.10.16"

[[deps.CodeTracking]]
deps = ["InteractiveUtils", "UUIDs"]
git-tree-sha1 = "b7231a755812695b8046e8471ddc34c8268cbad5"
uuid = "da1fd8a2-8d9e-5ec2-8556-3022fb5608a2"
version = "3.0.0"

[[deps.CodecBzip2]]
deps = ["Bzip2_jll", "TranscodingStreams"]
git-tree-sha1 = "84990fa864b7f2b4901901ca12736e45ee79068c"
uuid = "523fee87-0ab8-5b00-afb7-3ecf72e48cfd"
version = "0.8.5"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "962834c22b66e32aa10f7611c08c8ca4e20749a9"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.8"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "b0fd3f56fa442f81e0a47815c92245acfaaa4e34"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.31.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "b10d0b65641d57b8b4d5e234446582de5047050d"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.5"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Requires", "Statistics", "TensorCore"]
git-tree-sha1 = "a1f44953f2382ebb937d60dafbe2deea4bd23249"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.10.0"
weakdeps = ["SpecialFunctions"]

    [deps.ColorVectorSpace.extensions]
    SpecialFunctionsExt = "SpecialFunctions"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "362a287c3aa50601b0bc359053d5c2468f0e7ce0"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.11"

[[deps.CommonSubexpressions]]
deps = ["MacroTools"]
git-tree-sha1 = "cda2cfaebb4be89c9084adaca7dd7333369715c5"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.1"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "9d8a54ce4b17aa5bdce0ea5c34bc5e7c340d16ad"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.18.1"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.3.0+1"

[[deps.Crayons]]
git-tree-sha1 = "249fe38abf76d48563e2f4556bebd215aa317e15"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.1.1"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataFrames]]
deps = ["Compat", "DataAPI", "DataStructures", "Future", "InlineStrings", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrecompileTools", "PrettyTables", "Printf", "Random", "Reexport", "SentinelArrays", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "d8928e9169ff76c6281f39a659f9bca3a573f24c"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.8.1"

[[deps.DataStructures]]
deps = ["OrderedCollections"]
git-tree-sha1 = "e357641bb3e0638d353c4b29ea0e40ea644066a6"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.19.3"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.DiffResults]]
deps = ["StaticArraysCore"]
git-tree-sha1 = "782dd5f4561f5d267313f23853baaaa4c52ea621"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.1.0"

[[deps.DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "23163d55f885173722d1e4cf0f6110cdbaf7e272"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.15.1"

[[deps.DocStringExtensions]]
git-tree-sha1 = "7442a5dfe1ebb773c29cc2962a8980f47221d76c"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.5"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.7.0"

[[deps.FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "6522cfb3b8fe97bec632252263057996cbd3de20"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.18.0"

    [deps.FileIO.extensions]
    HTTPExt = "HTTP"

    [deps.FileIO.weakdeps]
    HTTP = "cd3eb016-35fb-5094-929b-558a96fad6f3"

[[deps.FilePathsBase]]
deps = ["Compat", "Dates"]
git-tree-sha1 = "3bab2c5aa25e7840a4b065805c0cdfc01f3068d2"
uuid = "48062228-2e41-5def-b9a4-89aafe57970f"
version = "0.9.24"
weakdeps = ["Mmap", "Test"]

    [deps.FilePathsBase.extensions]
    FilePathsBaseMmapExt = "Mmap"
    FilePathsBaseTestExt = "Test"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "05882d6995ae5c12bb5f36dd2ed3f61c98cbb172"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.5"

[[deps.FlameGraphs]]
deps = ["AbstractTrees", "Colors", "FileIO", "FixedPointNumbers", "IndirectArrays", "LeftChildRightSiblingTrees", "Profile"]
git-tree-sha1 = "d9eee53657f6a13ee51120337f98684c9c702264"
uuid = "08572546-2f56-4bcf-ba4e-bab62c3a3f89"
version = "0.2.10"

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions"]
git-tree-sha1 = "eef4c86803f47dcb61e9b8790ecaa96956fdd8ae"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "1.3.2"

    [deps.ForwardDiff.extensions]
    ForwardDiffStaticArraysExt = "StaticArrays"

    [deps.ForwardDiff.weakdeps]
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"
version = "1.11.0"

[[deps.HashArrayMappedTries]]
git-tree-sha1 = "2eaa69a7cab70a52b9687c8bf950a5a93ec895ae"
uuid = "076d061b-32b6-4027-95e0-9a2c6f6d7e74"
version = "0.2.0"

[[deps.HiGHS]]
deps = ["HiGHS_jll", "MathOptInterface", "PrecompileTools", "SparseArrays"]
git-tree-sha1 = "2aab56d4e161be93c44fc16dfc95940996f84a12"
uuid = "87dc4568-4c63-4d18-b0c0-bb2238e4078b"
version = "1.18.2"

[[deps.HiGHS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "f56d423e3f583e26ceaef08a15a270b28723c89a"
uuid = "8fd58aa0-07eb-5a78-9b36-339c94fd15ea"
version = "1.11.0+1"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "179267cfa5e712760cd43dcae385d7ea90cc25a4"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.5"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "7134810b1afce04bbc1045ca1985fbe81ce17653"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.5"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "0ee181ec08df7d7c911901ea38baf16f755114dc"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "1.0.0"

[[deps.IndirectArrays]]
git-tree-sha1 = "012e604e1c7458645cb8b436f8fba789a51b257f"
uuid = "9b13fd28-a010-5f03-acff-a1bbcff69959"
version = "1.0.0"

[[deps.InlineStrings]]
git-tree-sha1 = "8f3d257792a522b4601c24a577954b0a8cd7334d"
uuid = "842dd82b-1e85-43dc-bf29-5d0ee9dffc48"
version = "1.4.5"

    [deps.InlineStrings.extensions]
    ArrowTypesExt = "ArrowTypes"
    ParsersExt = "Parsers"

    [deps.InlineStrings.weakdeps]
    ArrowTypes = "31f734f8-188a-4ce0-8406-c8a06bd891cd"
    Parsers = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.InvertedIndices]]
git-tree-sha1 = "6da3c4316095de0f5ee2ebd875df8721e7e0bdbe"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.3.1"

[[deps.IrrationalConstants]]
git-tree-sha1 = "b2d91fe939cae05960e760110b328288867b5758"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.6"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "0533e564aae234aff59ab625543145446d8b6ec2"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.7.1"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.JuMP]]
deps = ["LinearAlgebra", "MacroTools", "MathOptInterface", "MutableArithmetics", "OrderedCollections", "PrecompileTools", "Printf", "SparseArrays"]
git-tree-sha1 = "8e4088727b5a130c12b1fedbc316306b6bbf2b9d"
uuid = "4076af6c-e467-56ae-b986-b466b2749572"
version = "1.29.4"

    [deps.JuMP.extensions]
    JuMPDimensionalDataExt = "DimensionalData"

    [deps.JuMP.weakdeps]
    DimensionalData = "0703355e-b756-11e9-17c0-8b28908087d0"

[[deps.JuliaInterpreter]]
deps = ["CodeTracking", "InteractiveUtils", "Random", "UUIDs"]
git-tree-sha1 = "f5e59455236d8269b7868665c3665e8477af0e37"
uuid = "aa1ae85d-cabe-5617-a682-6adf51b2e16a"
version = "0.10.10"

[[deps.JuliaSyntaxHighlighting]]
deps = ["StyledStrings"]
uuid = "ac6e5ff7-fb65-4e79-a425-ec3bc9c03011"
version = "1.12.0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "dda21b8cbd6a6c40d9d02a73230f9d70fed6918c"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.4.0"

[[deps.LeftChildRightSiblingTrees]]
deps = ["AbstractTrees"]
git-tree-sha1 = "b864cb409e8e445688bc478ef87c0afe4f6d1f8d"
uuid = "1d6d02ad-be62-4b6b-8a6d-2f90e265016e"
version = "0.1.3"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "OpenSSL_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.15.0+0"

[[deps.LibGit2]]
deps = ["LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"
version = "1.11.0"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "OpenSSL_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.9.0+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "OpenSSL_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.3+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
version = "1.11.0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.12.0"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "13ca9e2586b89836fd20cccf56e57e2b9ae7f38f"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.29"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"
version = "1.11.0"

[[deps.MIMEs]]
git-tree-sha1 = "c64d943587f7187e751162b3b84445bbbd79f691"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "1.1.0"

[[deps.MacroTools]]
git-tree-sha1 = "1e0228a030642014fe5cfe68c2c0a818f9e3f522"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.16"

[[deps.Markdown]]
deps = ["Base64", "JuliaSyntaxHighlighting", "StyledStrings"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.MathOptInterface]]
deps = ["BenchmarkTools", "CodecBzip2", "CodecZlib", "ForwardDiff", "JSON", "LinearAlgebra", "MutableArithmetics", "NaNMath", "OrderedCollections", "PrecompileTools", "Printf", "SparseArrays", "SpecialFunctions", "Test"]
git-tree-sha1 = "7fb98657926ccb4de8f9bb96cda453700ca39a8b"
uuid = "b8f27783-ece8-5eb3-8dc8-9495eed66fee"
version = "1.49.0"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "ec4f7fbeab05d7747bdf98eb74d130a2a2ed298d"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.2.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"
version = "1.11.0"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2025.11.4"

[[deps.MutableArithmetics]]
deps = ["LinearAlgebra", "SparseArrays", "Test"]
git-tree-sha1 = "22df8573f8e7c593ac205455ca088989d0a2c7a0"
uuid = "d8a4904e-b15c-11e9-3269-09a3773c0cb0"
version = "1.6.7"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "9b8215b1ee9e78a293f99797cd31375471b2bcae"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.1.3"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.3.0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.29+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.7+0"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.5.4+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1346c9208249809840c91b26703912dff463d335"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.6+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "05868e21324cede2207c6f0f466b4bfef6d5e7ee"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.8.1"

[[deps.Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "7d2f8f21da5db6a806faf7b9b292296da42b2810"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.3"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "Random", "SHA", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.12.1"
weakdeps = ["REPL"]

    [deps.Pkg.extensions]
    REPLExt = "REPL"

[[deps.PlotlyBase]]
deps = ["ColorSchemes", "Colors", "Dates", "DelimitedFiles", "DocStringExtensions", "JSON", "LaTeXStrings", "Logging", "Parameters", "Pkg", "REPL", "Requires", "Statistics", "UUIDs"]
git-tree-sha1 = "6256ab3ee24ef079b3afa310593817e069925eeb"
uuid = "a03496cd-edff-5a9b-9e67-9cda94a718b5"
version = "0.8.23"

    [deps.PlotlyBase.extensions]
    DataFramesExt = "DataFrames"
    DistributionsExt = "Distributions"
    IJuliaExt = "IJulia"
    JSON3Ext = "JSON3"

    [deps.PlotlyBase.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
    IJulia = "7073ff75-c697-5162-941a-fcdaad2a7d2a"
    JSON3 = "0f8b85d8-7281-11e9-16c2-39a750bddbf1"

[[deps.PlutoDevMacros]]
deps = ["JuliaInterpreter", "Logging", "MacroTools", "Pkg", "TOML"]
git-tree-sha1 = "709c36a806ec0af91840184f3052bb3c6cc60915"
uuid = "a0499f29-c39b-4c5c-807c-88074221b949"
version = "0.9.2"

[[deps.PlutoPlotly]]
deps = ["AbstractPlutoDingetjes", "Artifacts", "ColorSchemes", "Colors", "Dates", "Downloads", "HypertextLiteral", "InteractiveUtils", "LaTeXStrings", "Markdown", "Pkg", "PlotlyBase", "PrecompileTools", "Reexport", "ScopedValues", "Scratch", "TOML"]
git-tree-sha1 = "8acd04abc9a636ef57004f4c2e6f3f6ed4611099"
uuid = "8e989ff0-3d88-8e9f-f020-2b208a939ff0"
version = "0.6.5"

    [deps.PlutoPlotly.extensions]
    PlotlyKaleidoExt = "PlotlyKaleido"
    UnitfulExt = "Unitful"

    [deps.PlutoPlotly.weakdeps]
    PlotlyKaleido = "f2990250-8cf9-495f-b13a-cce12b45703c"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.PlutoProfile]]
deps = ["AbstractTrees", "FlameGraphs", "Profile", "ProfileCanvas"]
git-tree-sha1 = "154819e606ac4205dd1c7f247d7bda0bf4f215c4"
uuid = "ee419aa8-929d-45cd-acf6-76bd043cd7ba"
version = "0.4.0"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Downloads", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "3ac7038a98ef6977d44adeadc73cc6f596c08109"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.79"

[[deps.PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "36d8b4b899628fb92c2749eb488d884a926614d3"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.3"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "07a921781cab75691315adc645096ed5e370cb77"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.3.3"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "8b770b60760d4451834fe79dd483e318eee709c4"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.5.2"

[[deps.PrettyTables]]
deps = ["Crayons", "LaTeXStrings", "Markdown", "PrecompileTools", "Printf", "REPL", "Reexport", "StringManipulation", "Tables"]
git-tree-sha1 = "211530a7dc76ab59087f4d4d1fc3f086fbe87594"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "3.2.3"

    [deps.PrettyTables.extensions]
    PrettyTablesTypstryExt = "Typstry"

    [deps.PrettyTables.weakdeps]
    Typstry = "f0ed7684-a786-439e-b1e3-3b82803b501e"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.Profile]]
deps = ["StyledStrings"]
uuid = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"
version = "1.11.0"

[[deps.ProfileCanvas]]
deps = ["FlameGraphs", "JSON", "Pkg", "Profile", "REPL"]
git-tree-sha1 = "41fd9086187b8643feda56b996eef7a3cc7f4699"
uuid = "efd6af41-a80b-495e-886c-e51b0c7d77a3"
version = "0.1.0"

[[deps.REPL]]
deps = ["InteractiveUtils", "JuliaSyntaxHighlighting", "Markdown", "Sockets", "StyledStrings", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"
version = "1.11.0"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "62389eeff14780bfe55195b7204c0d8738436d64"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.1"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.ScopedValues]]
deps = ["HashArrayMappedTries", "Logging"]
git-tree-sha1 = "ac4b837d89a58c848e85e698e2a2514e9d59d8f6"
uuid = "7e506255-f358-4e82-b7e4-beb19740aa63"
version = "1.6.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "9b81b8393e50b7d4e6d0a9f14e192294d3b7c109"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.3.0"

[[deps.SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "ebe7e59b37c400f694f52b58c93d26201387da70"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.4.9"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"
version = "1.11.0"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "64d974c2e6fdf07f8155b5b2ca2ffa9069b608d9"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.2"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.12.0"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "5acc6a41b3082920f79ca3c759acbcecf18a8d78"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.7.1"

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

    [deps.SpecialFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"

[[deps.StaticArraysCore]]
git-tree-sha1 = "6ab403037779dae8c514bad259f32a447262455a"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.4"

[[deps.Statistics]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "ae3bb1eb3bba077cd276bc5cfc337cc65c3075c0"
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.11.1"
weakdeps = ["SparseArrays"]

    [deps.Statistics.extensions]
    SparseArraysExt = ["SparseArrays"]

[[deps.StringManipulation]]
deps = ["PrecompileTools"]
git-tree-sha1 = "d05693d339e37d6ab134c5ab53c29fce5ee5d7d5"
uuid = "892a3eda-7b42-436c-8928-eab12a02cf0e"
version = "0.4.4"

[[deps.StyledStrings]]
uuid = "f489334b-da3d-4c2e-b8f0-e476e12c162b"
version = "1.11.0"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.8.3+2"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "OrderedCollections", "TableTraits"]
git-tree-sha1 = "f2c1efbc8f3a609aadf318094f8fc5204bdaf344"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.12.1"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
version = "1.11.0"

[[deps.TranscodingStreams]]
git-tree-sha1 = "0c45878dcfdcfa8480052b6ab162cdd138781742"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.11.3"

[[deps.Tricks]]
git-tree-sha1 = "311349fd1c93a31f783f977a71e8b062a57d4101"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.13"

[[deps.URIs]]
git-tree-sha1 = "bef26fb046d031353ef97a82e3fdb6afe7f21b1a"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.6.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"
version = "1.11.0"

[[deps.UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
version = "1.11.0"

[[deps.WeakRefStrings]]
deps = ["DataAPI", "InlineStrings", "Parsers"]
git-tree-sha1 = "b1be2855ed9ed8eac54e5caff2afcdb442d52c23"
uuid = "ea10d353-3f73-51f8-a26c-33c1cb351aa5"
version = "1.4.2"

[[deps.WorkerUtilities]]
git-tree-sha1 = "cd1659ba0d57b71a464a29e64dbc67cfe83d54e7"
uuid = "76eceee3-57b5-4d4a-8e66-0e911cebbf60"
version = "1.6.1"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.3.1+2"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.15.0+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.64.0+1"

[[deps.p7zip_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.7.0+0"
"""

# ╔═╡ Cell order:
# ╟─e20ed61f-887c-4f2e-b4fd-4ae6b8dfd874
# ╠═d496f057-85d4-42ca-b97d-df333284195e
# ╠═be9b8bc5-68ae-4033-b91f-97fc7bcb52be
# ╠═df3c233d-791e-4f0f-9163-ca63bcc455d0
# ╠═548659da-1099-4f2f-ada7-c943cea23a66
# ╠═e6f47cea-5b67-467d-a7cb-f2fc30f8e358
# ╠═3d5a1285-d243-45ad-a2b9-3edd6b2b577b
# ╠═40a4c2a0-47e7-4f32-9927-683304c1c66e
# ╠═ecc7b7e2-deca-4cea-ab4f-3fadc7ba1f3d
# ╠═ce02448f-19b9-4426-b2b8-1db1aa6161f0
# ╟─ae63619a-3136-4ee6-b262-2d9242262bd2
# ╟─bed9f5da-e034-4d65-9019-6ed4186a7404
# ╟─51e2b6e9-21a6-44df-adf1-52d222e2acde
# ╠═0f8158bc-d373-43e0-80e5-197cd85e450b
# ╠═23ae726b-fd9b-499f-b8e8-8b243bbe68f7
# ╠═8aed40da-2cc8-4f02-8660-7f8c6190a624
# ╟─7eefc3ad-a4ce-4ba6-9393-86b29ab5ca40
# ╠═141cf16f-2994-492a-a526-4d29fd944de5
# ╠═90713eff-9d6b-476c-a41d-45819476b695
# ╟─1af06b2c-ee8a-497e-8d0e-e6a6127e48bf
# ╠═1f8873d6-6f7e-449e-b2ea-17fa489ac2a5
# ╟─5f7d0c18-ca88-4057-a6e6-632f562c720c
# ╠═311ae6ed-9594-49dd-8e4e-bce501b9c4b0
# ╟─1c4ad5ee-e29f-4550-8aa6-6d002c304338
# ╟─7c504736-ddbe-4a0a-af0b-fc7306693269
# ╟─745b4c73-7245-4d0a-ad21-52de88fedb9c
# ╟─aeeb8126-367c-4e44-bf2a-c76a0b97cd88
# ╟─c594d260-67b6-47a7-8768-b8f3cba7ea8b
# ╠═d7b7a9e0-6d23-4fd4-8189-9ae36d16b095
# ╠═d2390d6d-bd81-4b88-a855-260a5d74de4a
# ╠═ba5cfa57-8c29-4425-bc3e-5306605f19a2
# ╟─b437f801-0f37-4c76-bb49-db426802d341
# ╠═1a28b9c0-ca64-4f71-bba9-a1e37a4507f8
# ╟─12fa36af-4a3d-421a-ac3d-b3c20b6f8498
# ╠═47df7c4d-f29d-4422-8a80-d057225e2ad9
# ╠═2e3d6f58-541e-4856-98d9-5794449faab1
# ╟─5286ea3f-9615-4201-938a-2647d84caa20
# ╠═4536d83d-c4e9-41ba-8b13-4f206bec518e
# ╠═ad5c6fd6-cf80-452d-beb2-018936849846
# ╟─ad469f86-c30b-4988-8d2e-5694e5270bc9
# ╟─ffc4a8e6-46aa-4034-bf60-cae7a95b9cc2
# ╠═1ba8f790-1e33-42d9-840a-9cb8ca727439
# ╠═66206149-5cdf-4eb8-b08f-c4280091192d
# ╠═5109c85d-e379-4afd-a16b-1e5a87b145b7
# ╠═b16f5471-7027-4c81-be1f-c53c53cf454c
# ╠═3a585fae-1056-4473-bab6-f08ccfbe9cac
# ╟─1bdf1d35-897d-4b99-aef4-05d377215cbc
# ╠═a1cc64f0-8aa6-48d5-b091-bc377bdaf56a
# ╠═8075559d-2651-4a79-a435-9082724e4892
# ╠═f1cd0fc3-9eaf-4ab0-955a-253dfe6dc6f9
# ╟─45ee2102-5543-4416-8772-a1902319921c
# ╟─189c1b0d-80fc-43fa-8700-0cf8e44e46ae
# ╠═583df59f-8f67-484b-9cd2-9a0e9236df4b
# ╠═f093884c-fe6c-4785-9808-2b88ef20ce85
# ╠═fe49260a-8369-455e-8c06-70de4486cb68
# ╟─132d8d08-f347-4ded-aef6-947b653cadf4
# ╠═bbd6a798-0851-4e8b-93ac-b00fdd43a4c5
# ╠═a2f10a2a-7520-41f0-bddb-2410ac2edce1
# ╠═4f41c31d-335a-4c1d-b59c-f9ed0d555e58
# ╟─e4c23a21-5a3b-4377-b819-a1c39a17f7ab
# ╟─656d25ed-e59d-42a1-9e66-bf626cb8eef0
# ╠═c99bab73-518e-4c2f-817e-4e57a9b934f4
# ╠═0930fa3d-409a-45a3-b074-520acaf7fae6
# ╠═7efbb914-75b0-4692-a557-97a9b80bfc39
# ╠═7a29339e-8dd5-4ce7-a273-6204217e918e
# ╟─92e30dc7-2b9f-453b-a9bb-8ac2831d0503
# ╠═3a02d212-c162-48cd-85bf-f0a9987c2b24
# ╠═868c7277-7d31-4175-bb8e-9580d22571ea
# ╠═6d0448d3-c81c-480e-bc79-7df7ca20a059
# ╠═dbe66895-0734-4ffc-96cc-9ae97d7bbb27
# ╠═52a29eb4-bb87-40ad-a740-918ae9ee0e7f
# ╠═1056e7e0-c28f-492d-9f3f-5d1cdbaeed20
# ╠═8634cf9c-66ef-4fd2-a862-459476fa0b26
# ╟─5a1c6606-6593-40dd-8ed2-0285af060c25
# ╠═a4a5ffcf-fea3-482d-aa30-b41637826453
# ╠═1ec05cdb-dbef-4a8a-b582-9bc3b3ef34b8
# ╠═abd189c8-0ca9-4eec-9497-d2e3392d385c
# ╠═3130e1c7-332c-4d61-9a56-465ea966b03d
# ╠═d3598c9f-c9e4-419a-948f-ed2f837a0d53
# ╠═1c90f846-8705-4cbe-afbc-1f86d58c955d
# ╟─10f07638-d6d3-4a7a-839e-595d7526f7af
# ╠═51411837-d94a-478a-b1f2-f2776e983858
# ╠═84b09d35-1df7-4aea-8e5d-b7516d545ab3
# ╠═1fc576b2-5eae-4910-968a-ec888e8f53b3
# ╠═6be2ddbe-8040-4e26-9f52-7b4d297fe6b1
# ╟─3d21d8bb-3c8c-468d-b9f4-588b014fbab6
# ╠═d13e0602-dd22-4fd5-bd40-02ef92d71cbd
# ╠═b1240d14-8c60-48a1-9bd6-0b513cb9441e
# ╟─4518ec9c-6391-429a-9580-461402ec33d1
# ╟─d23f1861-4624-4d72-821b-93db1f44f87c
# ╟─ab698e42-290e-4be0-9021-c0c1c32c9191
# ╠═9ad3b7fb-e92c-4782-b8d7-6313ba7d3cda
# ╠═7ac7fb89-4e82-4c8a-a7f5-377e2ecc07f0
# ╟─13559698-a13b-4f22-8809-c24a5826db18
# ╟─00faac3f-314d-4974-a857-6edba5ddb724
# ╠═96fae564-bdc9-40fd-9cf5-30620693237f
# ╠═418b9570-9d55-4a14-bff4-127ce096f8dd
# ╟─d9a1cdf1-3261-4b4f-8aae-ea8c2fde5893
# ╟─dd25a1d7-62d5-4280-8ce2-abfa185a39c6
# ╟─77a9b132-0694-4051-a363-05d98a1f271f
# ╠═5d570d21-c334-46cc-a4d1-59073c698a7e
# ╠═12654916-76a2-4658-b08b-b8a27b8db7b6
# ╠═f1b23597-0adc-4487-851e-24e83c55096c
# ╠═730bc4c4-dcb0-443d-91e9-49097d006007
# ╠═91cf05d3-2d82-43a6-adbb-c7643b5d3353
# ╠═6f1ee1cc-1a45-4414-8fbf-f84227533a4d
# ╟─9cd2fd7d-164b-45cd-b6fc-91aa16138c32
# ╟─ad25bc92-927d-46c0-af03-528f9c77f7d3
# ╠═269d96ac-529e-4d2f-b069-59283c80133f
# ╠═bd9b4d61-04f3-4092-aea5-5e1e259632b8
# ╠═269eddb2-e45a-4c31-808b-53496d16ca6c
# ╠═4e76bb8e-67b7-4c39-8401-a431ff2080c9
# ╠═08e473bd-8e08-4c54-b42f-31609b870712
# ╠═d6ec5d03-7f7d-48a8-8b2b-4eee61290c94
# ╟─dba11945-938d-46eb-905f-b09c229d2f13
# ╠═6c36b446-bd18-47e8-95cb-915c7c28f24e
# ╠═5b856518-01b1-4346-bec1-9605d9c152f3
# ╠═64e55266-3f69-49f8-90cc-afe267c5f0a3
# ╠═8585c505-f9df-4838-9be5-c0d22563cc99
# ╠═3ef242ab-4b8f-4ab5-ad7b-f50f6475eb59
# ╟─969f5700-9713-47de-8739-bebea01c192c
# ╠═3587e8d3-cf10-4dc9-a413-cd62fc829c50
# ╠═c0cb489a-c0a6-4bb7-bf83-b57ff83d28f7
# ╠═9668706d-9c23-49ae-994b-a807371df430
# ╟─78bcaadf-f11e-4ea3-b1c4-7d8f6e675a01
# ╟─2defab11-b88d-4b19-b012-8e14c0183fdb
# ╟─cb1ac0db-8b20-4160-b73c-ef9c3602ab16
# ╠═d0df8f15-d070-43a4-b77c-caacd8d57cac
# ╠═c81112e5-f568-4be3-9fb8-50087d6d8ce2
# ╠═1ec77176-f122-41a7-bf02-bdf7c6c02868
# ╟─1b779c48-d617-4a3e-9cc6-2476ed959b26
# ╟─5022a5ef-364b-4d08-8113-df217705e05d
# ╠═a85cfb03-ded6-44d7-9996-452729dd4f0b
# ╠═5a1cac0b-81fb-4393-89cc-ffc380b9e1d1
# ╠═24e7ab3c-ed28-4a26-914a-9112eb6905b4
# ╟─f4a6e254-5718-4b40-890e-4cea24075093
# ╟─ad4b9d1e-275f-4f84-a0d7-ca37813cc25f
# ╟─ccded9dc-5698-4452-89cd-aa4ba9c91705
# ╠═2070630a-8e3b-4c3f-9bb5-a545b7075f0f
# ╠═5201f52a-dd7f-4627-896d-569ef101dd7d
# ╟─fc128165-6050-436f-83be-34a05914b7ea
# ╟─ec532cb3-460a-4aa1-b735-a7d629ec93ac
# ╠═ee566b9c-e117-4c48-bc77-154a4b2c24e4
# ╠═3d92bdb9-3514-4b15-80f4-dd2cde565723
# ╠═8ec95abd-cd0e-4b82-99e1-611ce282c792
# ╟─ea3ccd3b-b675-4cd6-8370-92fc830b07a2
# ╠═fe612451-789b-48d1-8929-1b07a80eea12
# ╠═8519a80f-02e5-4a63-808c-dcbac04ab186
# ╠═980975ca-d28f-4169-8d61-739d7109501c
# ╟─441bc9b0-7971-40f8-bb01-cda7f941a806
# ╠═838f0029-4a62-44cf-a203-9b5d950cdf8a
# ╟─211f5eb9-3494-4cd1-b113-d6a2fa64e3fe
# ╠═1ac8f069-76cb-4058-8a03-bbf317d25985
# ╟─af61484b-c682-4da6-acbf-d073568398de
# ╠═c141e214-dd85-4035-aa1f-018e5d806856
# ╟─9128bc9b-55cb-48e1-afd4-91e9c70cc0fb
# ╟─a92ce21c-b793-4b83-9312-77df54fa5ac8
# ╠═d44e801d-0789-4edf-82e1-66694e444cc8
# ╠═e7626310-cf72-461b-ac31-6c61d56dcb53
# ╠═04fab2e6-e685-4fee-afbe-73d013512c1d
# ╠═e0560b85-6e67-4ee1-a363-9b0e14367acd
# ╠═3b9ea3c6-3623-493f-b7e8-6616c7f42506
# ╟─146052ca-1d6e-4db3-b951-04283f35bae6
# ╟─5b185f70-5a36-4404-ace7-32dc62b47452
# ╟─c8dbc747-7867-40e8-8c70-8355d5f6505d
# ╠═b1dd46bf-0f0a-45f6-bf90-d1d7a328e7a1
# ╠═760a8733-a63a-4301-95be-94602fd98410
# ╠═45f45798-af9a-4cac-8f89-5b582cecbff1
# ╠═e6b4e689-b567-4d8e-bc82-7ebec987e9fb
# ╠═578e3515-b7cf-4a51-b120-d1091c4f20a1
# ╠═70ac1377-0624-4880-9f3b-a247bf904683
# ╟─12900bdd-6e0e-46c1-8272-5b395a402eaa
# ╠═b6c965c8-db71-4004-968c-95cc4ea8f94e
# ╠═db819d7b-c790-435e-a166-0c6093518ee4
# ╠═82f9e878-02ed-4cc6-b00a-2602df94a656
# ╠═5bd98306-d8b9-4cbf-b9c0-45aa7d730ff1
# ╟─a1f90ae7-b08e-4517-ad90-634821bf7bd8
# ╟─711befa2-ac12-4f96-8883-54a4228cad7b
# ╠═8ca28ae4-ba1a-4b96-b2a8-e9e910b80a63
# ╠═d92d3dcc-e792-493e-aa5c-4c73e93f1fda
# ╠═e631ed5a-1d49-4fc5-b57a-c0f74e33fdfb
# ╠═d3cb1724-0098-47a5-a7fe-2d6e456a4ce2
# ╠═d5e2603e-82ab-4f56-a990-ee07d6c704c4
# ╠═4fd570f9-0ee8-43d9-a0dc-687835c1ed34
# ╠═9829e745-7fd3-42f3-bbd3-951250f98ee4
# ╠═a8cc486d-3608-4070-ab5b-26d3f2a24759
# ╠═5ac88229-5547-4f2c-854d-c7868ad252e2
# ╠═f22173d1-b5c6-42bb-bf59-4d0ea25a43f5
# ╠═c36c5629-c5ea-4c36-89f8-14fcb4fb3385
# ╠═75e01ce3-4fee-4082-94bf-9a4fef46ab50
# ╠═be9dd54a-0460-47c8-9a0b-edd23abbe351
# ╠═efcd3e16-5a30-4e56-9f3a-96adb9579b4e
# ╠═295cfadc-06cd-4bcd-84a2-4afc37142896
# ╟─2017a885-c570-48d9-a75e-2e9abc5c9955
# ╠═5c6aced7-f987-4245-a04d-16df75fc1ebe
# ╠═040dadb9-2e64-458c-a513-61bcf735d6b5
# ╠═8322e025-efdb-49a2-9f75-069a3141c781
# ╠═a9a5d50b-ba31-4dcb-b3af-dfe5c2075cb4
# ╠═904e1718-ea13-48fc-bfd6-183438ec480f
# ╟─b4aaa02d-a4fb-42f7-bc46-416cec534cd6
# ╟─daad86bb-0de9-4d04-a7ff-f338dad502fd
# ╠═69292ea5-f513-46b4-b6a2-6c5533d0e555
# ╠═0cbd429b-c674-4802-916a-d3aeef4632db
# ╠═07d3d939-6c64-46b3-9281-25e91d64174e
# ╠═703fd9fa-c323-4667-9a6d-a49d56c52ece
# ╠═ad649468-40d4-4426-918f-d1e883ef4af7
# ╠═9691981d-87b3-408c-8e47-b4e17bcd82b7
# ╠═b32ccdf0-00b0-47ca-adc2-b5e53a9652c3
# ╠═aa4c1f17-efe8-4d61-95a4-a31770b3631c
# ╠═129c6306-598a-4e67-82ef-269037f201c3
# ╠═a840c721-d8d1-4d95-95c2-b9eaf41d6594
# ╠═9cf6d7fc-32db-40f6-9a27-3ee1f8c8013c
# ╠═34321c84-3f64-478b-a6f0-e816f871dda9
# ╠═aec8b1a3-16ff-45d9-a30e-5119acd8bc45
# ╠═9eae0c07-d09d-48a4-b1cc-63351171468f
# ╠═113c2dab-d0e2-4318-b5a6-78c0868bfd77
# ╠═0e4d5ce3-0695-4a87-bf9d-d182048df426
# ╠═7564cd5f-3df6-482b-8532-07294091db37
# ╠═e8b848c7-a9a0-45b6-9599-7bfff194f6ff
# ╠═0482bf4e-915c-4861-8a87-4c4cd6ce4fa0
# ╠═217fdcdd-69bd-425c-b305-7d9f4262377a
# ╠═5567e751-7562-4dc8-806e-17ad2b334be0
# ╠═6bf1d985-71ec-4e80-b3a9-b0d5e19745bc
# ╠═ce7f4ffa-7b0b-409f-bf41-9f32f6f8c6f8
# ╠═89908964-28d9-4562-b17a-505fe5496636
# ╠═e248cbbd-e37e-4913-8800-f29490ab1f66
# ╠═5c3e8713-f951-42f4-89a4-cd37b169b714
# ╠═b1389ca7-9c87-4e6f-adcd-11dd079b9caf
# ╠═c0cfcd9b-afef-4a50-a1d0-20f8a4425fbc
# ╠═f1bbfc95-121f-4219-9f8f-1647037f59d1
# ╠═911c3790-2e23-4036-be72-bb64ba6e4bbb
# ╠═55b8f267-8b6a-4a59-926a-e44e2b25fff7
# ╠═73efc1bb-4786-4ca2-8084-db08956ac82a
# ╠═3bba2d87-232d-4d0d-9db8-e9ea540617a4
# ╟─1e2ddcbb-f2af-4555-b516-959ae647b343
# ╠═65c9422c-3bb0-452b-bfc7-8cac4e4dc353
# ╠═90a4b613-16dc-46b2-817a-bbbb83c17a32
# ╠═a0116b94-b01d-466b-93e3-96476d19fe47
# ╠═fbc1e086-18f9-4303-8e0c-6061cc4321a4
# ╠═3067dfdc-a21f-4b0e-81bd-aee416934359
# ╠═1197f21f-58b4-4a65-a284-bce2c416a035
# ╠═7d341dfd-9c8f-446d-b270-e461af194c3b
# ╠═71c8a440-4292-4e6d-a0c2-0bfb872775eb
# ╠═213a681f-d1a4-4031-a138-769b42ae533a
# ╠═f67dbf1c-7976-4dec-b6fb-d92cd740966c
# ╠═384f9170-d63a-4b9e-9a9a-a3a7439aed3a
# ╠═29d9f218-e083-474b-af0b-64a3927e4db7
# ╠═89a63603-84a8-4e8c-92c2-de220da3b6ae
# ╠═cac747d2-cd16-4988-8798-b2328daa162f
# ╠═3058ed0c-68ab-4f7e-a57e-11195f5e3887
# ╠═8f8df220-88a5-41d9-a838-270fbd5320af
# ╠═7c7db0a7-bc44-428f-8566-9374d8ebe165
# ╠═864fb287-032c-4b65-839a-d5ddd4ef0a29
# ╟─3b2e0ecd-6f49-486e-95e0-00c743ae7fba
# ╟─7417aad8-2b02-4812-a244-af8fb638796a
# ╠═209e227d-d009-44c0-8568-cd08400d04fb
# ╟─c96ac8c4-23a1-4a00-95ca-9e543206a8f4
# ╠═5a86d878-01c8-427e-9dbe-724e967c50b6
# ╠═ae3964cc-6a80-4371-a3f3-bd712b47029c
# ╠═728199b2-e239-4a75-b921-6ade95a45b78
# ╠═fe44dcd6-5f08-49c4-9e3d-a77ce6d1dfa1
# ╟─2de57998-ed42-4882-8afd-2d067ca47fa3
# ╠═31b5b269-8e71-4be6-94dd-4e4a80690792
# ╠═76f134bd-ec74-48d4-9562-e245a80234ff
# ╠═9b2c0f63-ff25-4f00-be47-33bb0bf6b335
# ╠═b9039f56-ae1c-43f6-a847-37d862e47166
# ╠═d2e4b338-6726-4604-8775-332fc644b472
# ╠═99127002-79dc-4215-9acb-9a409f1314cb
# ╟─95fc4fe7-db30-4222-94d7-cbc573ec186f
# ╠═3b616e5a-dcf0-49ae-a5b4-4d7079065b79
# ╠═69fedc29-0202-486f-97e9-79b1507fc8dc
# ╠═59e46620-2e27-4eca-99c2-492b4211bb29
# ╠═01dad2d3-3911-4549-95de-8bf33ba860c1
# ╠═410fc812-baad-4a38-8899-e5ed2a78583e
# ╠═42391b1d-d4fd-438d-baf9-f17c83d5acab
# ╠═8f86f808-37d9-483b-94fd-55b222b51151
# ╠═485de1c8-4614-4c0e-b514-4c3ecc9046e6
# ╠═c1e87b5a-50f5-441a-9f31-2dad14e7005a
# ╠═45d71799-e2bf-4664-8f8d-b2779bbd2737
# ╠═9704ce53-01da-40ae-89a9-1c446fa4a0d7
# ╠═7e4ff070-1cf1-4f9c-b3bc-fcc1a6e6f31e
# ╠═3cbf1a5d-3e82-47a2-be83-4446551d1060
# ╠═afdd0004-f272-4be5-9d7b-2a0410b43aec
# ╠═c2efa04e-0688-40b9-8635-a0b8dfd7f089
# ╠═89e38e04-a704-4d38-8f27-2e0e966897f9
# ╠═69601769-d646-4af4-90cf-8c06eeedbc4d
# ╟─942d2d73-10ad-4a35-a619-0cbb957a8fd8
# ╟─f0903e44-4443-4751-bc45-9c62a7bde1d6
# ╠═fb91e135-4474-41cf-bac0-936b124ffd27
# ╠═5a30701d-078f-4170-a7f8-75095248e5ca
# ╠═9f9de138-6324-4740-b926-3172df10eb78
# ╟─7989fb06-dbd2-4a1f-bbc4-c36e3c79a639
# ╠═21a6c62f-76e6-424b-b373-9faf160d976d
# ╠═0f91b108-8de4-4811-ac4d-a30b318c836d
# ╠═4fb82746-1b35-40a8-8953-bf62389d8673
# ╠═d52dbcd7-7d99-4380-b538-8cca211b95a9
# ╠═1b2db9c2-8a28-4797-9c89-ea4cbcc25eb1
# ╠═839f5283-5319-4ff9-8ec0-605f17803cea
# ╠═929f620b-955c-4f1a-8d1b-bc371f064113
# ╠═0ff27f26-4294-4661-84aa-690a1ff360ba
# ╠═780ce8ec-4efa-4132-862c-847c1b438859
# ╠═6c790648-5d37-44a2-ac18-5adfc7feafaa
# ╠═cabaedf9-b357-4a98-b71f-97268f0f85cb
# ╟─4290fcb1-1ec8-4656-8290-d997ac5aceb5
# ╟─d5524ca0-4714-4701-bd30-3cb395c4820a
# ╠═017dfa67-14e8-4e17-82d4-799753f18591
# ╠═7bdd73ef-6f54-4cb3-9a88-9b0c7f58c93b
# ╠═80fd5886-2240-4fd3-a167-dd1741cf7e66
# ╟─d225deb8-62d4-40a9-8da0-ba7c5a4d1058
# ╠═caf32e13-4c35-4d69-8078-156ca8a12d74
# ╠═fedb0c5e-7850-4dbc-bd10-e30b8a7c5160
# ╠═7340f0bb-512c-4825-a263-ea264c5929bd
# ╠═9a718593-d9f5-419d-8d37-7f1a44b5f9cb
# ╟─566b34e2-d643-4eca-9867-89752949a018
# ╟─842523a6-91eb-4803-9a27-e46be1fcc402
# ╠═9ffbc773-6307-4c6a-b6bb-f9af72d06d97
# ╠═1bd41b66-c2bc-4c70-98a5-5b24d5cb27da
# ╠═50b6330d-dd07-4261-850c-0d6a1d293238
# ╠═be126cde-b0ae-4b6f-bef4-0b4ecf4087a7
# ╟─b619a8e2-2cda-47ab-bc77-24256e807b47
# ╠═a7ff876d-3c77-42a4-bf62-37b4d53b0713
# ╠═5cc5fbbc-e982-4ebb-83e9-0eb632d40bb3
# ╠═34d81780-75e0-4446-803f-8bc58b4f1c70
# ╠═5dbc6867-cff6-4340-9316-0c76621530d9
# ╠═4aef6b1a-5fa9-47c0-8510-ca8e44845f8d
# ╠═f2ac6bef-4b36-4c58-946d-ea5a241a23e5
# ╠═9dc092ea-9e74-4b28-b913-51c479b61cfd
# ╟─6317f0a0-1424-411b-bd46-4993cb28a55b
# ╟─a39de0cb-c5a4-44bf-ab4b-2d394a00cac8
# ╠═a1c545f6-04f3-4b6e-9b26-4b520d7bb6ef
# ╠═12aee9ee-616c-4360-94c8-dd12c9353bd3
# ╠═5acdbed5-a2a3-4455-abe6-83865fe63322
# ╠═6fa3e0d9-a755-4cec-974a-7a5c5e762ff6
# ╠═cd9d50a8-032d-11f1-a111-edabac2b8ec3
# ╠═50abaf3a-af49-438b-9fcb-5d36d15aac9f
# ╠═b97e16fa-db92-4fd2-a39f-f6c06595d7b1
# ╠═283ce916-a355-44e4-b437-b4c262e35c36
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
