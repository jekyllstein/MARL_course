### A Pluto.jl notebook ###
# v0.20.24

using Markdown
using InteractiveUtils

# ╔═╡ bb0f0b2d-575c-43dd-ad5e-11902144b3e5
using PlutoDevMacros

# ╔═╡ c87289df-243d-4ee9-a1e6-d6a677730f99
using DataFrames, CSV, JuMP, HiGHS, DataStructures, Dates

# ╔═╡ 4ba61675-1b58-46fd-a5d5-ec260a061bf7
# ╠═╡ skip_as_script = true
#=╠═╡
begin
	using PlutoUI, PlutoPlotly, PlutoProfile, BenchmarkTools, LaTeXStrings, HypertextLiteral
	TableOfContents()
end
  ╠═╡ =#

# ╔═╡ b08b4238-b047-485a-81fe-8670399b2342
@only_in_nb begin
	@frompackage @raw_str(joinpath(@__DIR__, "..", "RL_Module")) import *
	include(joinpath(@__DIR__, "multi_agent_types.jl"))
	include(joinpath(@__DIR__, "joint_action_learning.jl"))
	include(joinpath(@__DIR__, "independent_learning.jl"))
end

# ╔═╡ 54fd8229-4611-4cda-a36f-9b8df8ddb2da
md"""
## Deep Joint-Action Learning
"""

# ╔═╡ 1e978fa9-e664-4a84-9451-ade2d09216ad
md"""
### Utility Functions
"""

# ╔═╡ 672157a7-a7d4-4476-a34d-a192420daac0
function get_other_agent_inds(N::Integer, agent_index::Integer)
	isone(agent_index) && return (2:N)
	agent_index == N && return (1:N-1)
	return vcat(1:agent_index - 1, agent_index+1:N)
end

# ╔═╡ ea2647b7-4141-464b-8a30-0390831e0975
# ╠═╡ disabled = true
#=╠═╡
begin
	update_action_value_targets!(targets, γ, replay_buffer, batch_inds, nstep, target_const, param_args[i]..., feature_matrices[i], action_values[i], output_args[i]..., target_args[i]...)
	
	function update_action_value_targets!(targets::Vector{T}, γ::T, replay_buffer::CircularBuffer, batch_inds::Vector{Int64}, nstep::Integer, target_const::Vector{T}, target_params::Array{Matrix{T}, N}, feature_vectors::Vector{V}, action_values::Vector{T}, output_matrix::Matrix{T}) where {T<:Real, N, V}
		for i in eachindex(batch_inds)
			j = batch_inds[i]
			(x, i_a, r, x′, terminated) = replay_buffer[j]
			g = r
			k = j+1
			while !terminated && (k <= j+N)
				(x, i_a, r, x′, terminated) = replay_buffer[k]
				g += r * γ^(k-j)
				k += 1
			end
			targets[i] = g
			if !terminated
				update_linear_action_values!(action_values, x′, target_params)
				targets[i] += γ^(k-j) * maximum(action_values)
			end
		end

		
end
  ╠═╡ =#

# ╔═╡ 1f416406-bcc6-4cee-bc2e-4def87e7caec
md"""
### Algorithm
"""

# ╔═╡ bb19a7d2-70d7-4412-9fff-df0e2becc6ad
# ╠═╡ disabled = true
#=╠═╡
function joint_action_learning!(value_params::NTuple{N, Q}, target_params::NTuple{N, Q}, policy_params::NTuple{N, Π}, game::StateStochasticGame{T, S, A, N, P, F1, F2}, γ::T, max_episodes::Integer, max_steps::Integer, feature_vectors::NTuple{N, V}, update_feature_vectors!::NTuple{N, Function}, update_action_values!::NTuple{N, Function}, update_value_gradients!::NTuple{N, Function}, update_policy_distributions!::NTuple{N, Function}; target_args::NTuple{N, Tuple} = ntuple(i -> (), N), α = one(T)/10, ϵ = one(T) / 10, buffer_size::Integer = 10_000, batch_size::Integer = 512, target_update_interval::Integer = 100, α_decay = one(T), decay_step = typemax(Int64), save_step_rewards::Bool = false, nstep::Integer = 0, ∇q̂s::NTuple{N, Q} = deepcopy(value_params), ∇lnπ::NTuple{N, Π} = deepcopy(policy_params), K = 10, kwargs...) where {Q, T<:Real, S, A, N, P<:AbstractStateGameTransition, F1<:Function, F2<:Function, V, Π}
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
	output_args = ntuple(i -> (output_matrices[i],), N)
	param_args = ntuple(i -> (target_params[i],), N)
	output_inds = Vector{Int64}(undef, batch_size)
	feature_vectors2 = deepcopy(feature_vectors)
	other_actions = Vector{Vector{Int64}}(undef, batch_size)
	action_values2 = ntuple(i -> zeros(T, length(game.agent_actions[i])), N)

	#initialize episode
	s = game.initialize_state()

	#compute agent policy functions
	for i in 1:N
		update_feature_vectors![i](feature_vectors[i], s)
		update_policy_distributions![i](policies[i], feature_vectors[i], policy_params[i])
	end

	#update action values for all agents by sampling joint actions from model distributions
	for agent in 1:N
		other_agent_inds = get_other_agent_inds(N, agent)
		for other_agent in other_agent_inds
			update_policy_distributions![i](policies[other_agent], feature_vectors[other_agent], policy_params[agent][other_agent])
		joint_actions = [NTuple{N-1, Int64}([sample_action(policies[other_agent_index]) for other_agent_index in get_other_agent_inds(N, agent)]) for _ in 1:K]
		update_action_values![agent](action_values[agent], feature_vectors[agent], joint_actions, value_params[agent]) #the action value function now needs to take as input the joint actions sampled from the other agents

		#the linear version of this should maybe have a separate set of parameters for every joint action of the remaining agents while the non-linear version can form a one-hot encoding of the other action selection and use that as input
	end

	#get joint action selection for next step based on following ϵ greedy policy from action values
	joint_actions = ntuple(N) do i
		policies[i] .= action_values[i]
		ReinforcementLearning.make_ϵ_greedy_policy!(policies[i]; ϵ = ϵ)
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

		
		#compute agent policy functions
		for i in 1:N
			update_policy_distributions![i](policies[i], feature_vectors2[i], policy_params[i])
		end
	
		#update action values for all agents by sampling joint actions from model distributions
		for agent in 1:N
			joint_actions = [NTuple{N-1, Int64}([sample_action(policies[other_agent_index]) for other_agent_index in get_other_agent_inds(N, agent)]) for _ in 1:K]
			update_action_values![agent](action_values[agent], feature_vectors2[agent], joint_actions, value_params[agent]) #the action value function now needs to take as input the joint actions sampled from the other agents
		end
	
		#get joint action selection for next step based on following ϵ greedy policy from action values
		joint_actions′ = ntuple(N) do i
			policies[i] .= action_values[i]
			ReinforcementLearning.make_ϵ_greedy_policy!(policies[i]; ϵ = ϵ)
			sample_action(policies[i])
		end

		# @info "Sampled the following joint actions: $joint_actions′"

		decay *= (step > decay_step)*α_decay + (step <= decay_step)

		#only perform gradient parameter update once the replay buffer is large enough to fill up an entire batch
		if step ≥ (batch_size + nstep)
			# @info "Performing batch gradient updates"
			for i in 1:N
				replay_buffer = replay_buffers[i]
				# @info "replay_buffer is of type $(typeof(replay_buffer))"
				# @info "Performing batch gradient update for agent $i"
				ReinforcementLearning.update_batch_inds!(batch_inds, step, buffer_size, nstep)
				# @info "Selecting batch indices: $batch_inds"
				# @info param_args[i]
				# @info "Updating target values with $nstep step returns output arguments of $(output_args[i]) and the replay buffer $(replay_buffer)"

				#need to replace this with a target value calculation that uses the joint-action values and the policy outputs of the other agents at the batch index future transition states
				update_action_value_targets!(targets, γ, replay_buffer, batch_inds, nstep, target_const, param_args[i]..., feature_matrices[i], action_values[i], output_args[i]..., target_args[i]...)

				other_agent_inds = get_other_agent_inds(N, i)

				# @info "Updating feature matrix"
				#update_feature_matrix
				for j in eachindex(batch_inds)
					(x_k, i_a_k, _, _, _) = replay_buffer[batch_inds[j]]
					other_agent_actions = [begin
						(_, i_a, _, _, _) = replay_buffer[batch_inds[j]]
						i_a
					end
					for other_agent_index in other_agent_inds]
					ReinforcementLearning.update_feature_matrix!(feature_matrices[i], x_k, j)
					output_inds[j] = i_a_k
					other_actions[j] = other_agent_actions
				end

				update_value_gradients![i](∇q̂s[i], value_params[i], targets, output_inds, feature_matrices[i], other_agent_actions, output_matrices[i])
				ReinforcementLearning.update_params_with_gradient!(value_params[i], α*decay, ∇q̂s[i])

				ReinforcementLearning.update_batch_policy_gradient!(∇lnπ[i], policy_params[i], policy_matrix[i], output_inds, update_feature_vectors[i], policy_gradient_args[i]...)
				
				ReinforcementLearning.update_params_with_gradient!(policy_params[i], α_θ, ∇lnπ[i])

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
		ReinforcementLearning.cleanup_gradient!(∇lnπ[i])
	end

		#maybe should change this to policy function instead
	q̂s, form_kwargs = form_agent_value_functions(game, update_feature_vectors!, update_action_values!, feature_vectors, value_params)

	return (value_functions = q̂s, episode_rewards = episode_rewards, episode_steps = episode_steps, final_parameters = deepcopy(value_params), form_kwargs = form_kwargs)
end
  ╠═╡ =#

# ╔═╡ 559d8ec0-61ce-4837-bd3e-2e19acfbcfbf
md"""
### Linear Approximation
"""

# ╔═╡ 5a29909f-2268-412e-ade5-bb8ca1a6aba3
function update_linear_action_values!(action_values::Vector{T}, feature_vector, joint_action_samples::Vector{NTuple{Nm1, Int64}}, value_params::Array{Matrix{T}, Nm1}, action_values2::Vector{T}) where {T<:Real, Nm1}
	action_values .= zero(T)
	for joint_action in joint_action_samples
		ReinforcementLearning.update_linear_action_values!(action_values2, feature_vector, value_params[joint_action...])
		action_values .+= action_values2
	end
	action_values ./= length(joint_action_samples)
end

# ╔═╡ 8075e662-04a8-46ae-9d49-82ee2a465303
function update_linear_value_gradient!(∇q̂::Array{Matrix{T}, Nm1}, value_params::Array{Matrix{T}, Nm1}, targets::Vector{T}, output_inds::Vector{I}, feature_vectors::Vector{V}, other_agent_actions::Vector{Vector{Int64}}, output_matrix::Matrix{T}) where {T<:Real, Nm1, I<:Integer, V}
	
	
end

# ╔═╡ 2ff33994-2142-4c87-87ff-d585be2a5a0f
#create an array of value parameters that stores a matrix for every joint-action value index of the other agents
function initialize_linear_jal_params(game::StateStochasticGame{T, S, A, N, P, F1, F2}, feature_vector::V, agent_index::Integer, init_value::T) where {T<:Real, S, A, N, P, F1, F2, V}
	num_actions = length(game.agent_actions[agent_index])
	other_agents = if N == 2
		isone(agent_index) ? (2:2) : (1:1)
	else
		vcat(1:agent_index - 1:agent_index+1:N)
	end
	other_actions = [length(game.agent_actions[i]) for i in other_agents]
	params = Array{Matrix{T}, N-1}(undef, Tuple(other_actions)...)
	for i in eachindex(params)
		params[i] = initialize_linear_parameters(feature_vector, num_actions, init_value)
	end
	return params
end

# ╔═╡ 7c167873-55b0-4a61-a1f0-a0c9d5573967
begin
	function joint_action_learning_linear(game::StateStochasticGame{T, S, A, N, P, F1, F2}, γ::T, max_episodes::Integer, max_steps::Integer, feature_vectors::NTuple{N, V}, update_feature_vectors!::NTuple{N, Function};
		init_value::T = zero(T),
		value_params::NTuple{N, Q} = ntuple(i -> initialize_linear_jal_params(game, feature_vectors[i], i, init_value), N), 
		target_params::NTuple{N, Q} = deepcopy(value_params), 
		policy_params::NTuple{N, Π} = ntuple(i -> initialize_linear_parameters(feature_vectors[i], length(game.agent_actions[i]), init_value), N),
		kwargs...) where {T<:Real, Nm1, Q <: Array{Matrix{T}, Nm1}, Π <: Matrix{T}, S, A, N, P<:AbstractStateGameTransition, F1<:Function, F2<:Function, V} 

		action_values2 = ntuple(i -> zeros(T, length(game.agent_actions[i])), N)
		update_action_values! = ntuple(N) do i
			f!(args...) = update_linear_action_values!(args..., action_values2[i])
		end
				
		joint_action_learning!(value_params, target_params, policy_params, game, γ, max_episodes, max_steps, feature_vectors, update_feature_vectors!, update_action_values!, ntuple(i -> update_linear_value_gradient!, N), ntuple(i -> ReinforcementLearning.update_policy_dist!, N); target_args = ntuple(i -> (), N), kwargs...)
	end

	joint_action_learning_linear(game::StateStochasticGame{T, S, A, N, P, F1, F2}, γ::T, max_episodes::Integer, max_steps::Integer, feature_vector::V, update_feature_vector!::Function; kwargs...) where {T<:Real, S, A, N, P<:AbstractStateGameTransition, F1<:Function, F2<:Function, V} = joint_action_learning_linear(game, γ, max_episodes, max_steps, convert_to_ntuple(feature_vector, N), convert_to_ntuple(update_feature_vector!, N); kwargs...)
end

# ╔═╡ 413d6cd9-2b0d-404e-849e-d0ad18cde7f8
(1-.03)^25

# ╔═╡ 01f46865-4d1c-491f-aa59-812bf0434b48
1/25

# ╔═╡ 92a76486-76fb-4ef6-a5f4-16c7add8ede7
test_array = Array{Int64, 3}(undef, (2, 2, 2)...)

# ╔═╡ 0482822e-8275-49c7-87a6-db08e6b46895
test_array[(1, 1, 1)...]

# ╔═╡ 651d1dd4-7fe1-4e2c-98ae-744cf346932f


# ╔═╡ a9c977ae-8fef-4e6d-b4fe-220422f589a3
md"""
## Policy Representations

We can extend algorithms like independent actor-critic to include training an encoded representation of the other agents actions.  That information can then be passed along as an input to the policy functions of an agent.
"""

# ╔═╡ c43e0fa7-3fc6-4af8-94b3-4be49faeb0a9
md"""
### Utility Functions
"""

# ╔═╡ 318c9efc-fe93-49ed-93c2-9c98b0a91a04
begin
	function make_combined_inputs(feature_matrix::Matrix{T}, encodings::NTuple{N, Matrix{T}}) where {T<:Real, N}
		(n, m1) = size(feature_matrix)
		for i in eachindex(encodings)
			@assert size(encodings[i], 1) == n
		end
		ms = [size(e, 2) for e in encodings]
		mtot = m1 + sum(ms)
		return zeros(T, n, mtot)
	end
end

# ╔═╡ a997aa3f-b11c-42eb-96fc-daa518384f39
#concatenate the feature inputs with the compact encodings learned for the other agents
begin
	function update_combined_inputs!(combined_inputs::Matrix{T}, feature_input::Matrix{T}, encodings::NTuple{N, Matrix{T}}) where {T<:Real, N}
		@inbounds @simd for i in eachindex(feature_input)
			combined_inputs[i] = feature_input[i]
		end

		for j in 1:N
			base_ind = length(feature_input) + sum(k -> length(encodings[k]), 1:j-1; init = zero(T))
			@inbounds @simd for i in base_ind+1:base_ind+length(encodings[j])
				combined_inputs[i] = encodings[j][i]
			end
		end
		return combined_inputs
	end
end

# ╔═╡ 199a483d-1eaf-4c90-b931-2359b0bee30c
#initialize feature matrix that includes the feature vectors as well as the encoding features for other agents
begin
	initialize_synchronous_encoding_features(feature_vectors::NTuple{N, Vector{T}}, l::Integer, num_env::Integer) where {N, T<:Real} = ntuple(i -> zeros(T, length(x)+(l*(N-1)), num_env), N)
end

# ╔═╡ 5cc2328f-e346-400c-b7e8-297a6b090baf
#update one column of a feature matrix with the feature vector and other encoding features
begin
	function update_combined_features!(combined_matrix::Matrix{T}, feature_matrix::Matrix{T}, encoding_matrix::Matrix{T}) where T<:Real
		(n, m) = size(feature_matrix)
		encoding_size = size(encoding_matrix, 1)
		for col in 1:m
			@inbounds @simd for i in 1:n
				combined_matrix[i, col] = feature_matrix[i, col]
			end
			@inbounds @simd for i in 1:encoding_size
				combined_matrix[i+n, col] = encoding_matrix[i, col]
			end
		end
		# rand() < 0.0001 && @info "Combined Matrix has dimensions $(size(combined_matrix)) and col 1: $(combined_matrix[:, 1])"
		return combined_matrix
	end

	function update_combined_features!(combined_matrix::Matrix{T}, feature_vector::Vector{T}, encoding_vector::Vector{T}, k::Integer) where T<:Real
		n = length(feature_vector)
		encoding_size = length(encoding_vector)
		col = k
		@inbounds @simd for i in 1:n
			combined_matrix[i, col] = feature_vector[i]
		end
		@inbounds @simd for i in 1:encoding_size
			combined_matrix[i+n, col] = encoding_vector[i]
		end
		# rand() < 0.0001 && @info "Combined Matrix has dimensions $(size(combined_matrix)) and col 1: $(combined_matrix[:, 1])"
		return combined_matrix
	end

	function update_combined_features!(combined_vector::Vector{T}, feature_vector::Vector{T}, encoding_vector::Vector{T}) where T<:Real
		l1 = length(feature_vector)
		l2 = length(encoding_vector)
		l3 = length(combined_vector)
		# @assert l3 == l1 + l2
		@inbounds @simd for i in 1:l1
		# for i in 1:l1
			combined_vector[i] = feature_vector[i]
		end

		@inbounds @simd for i in 1:l2
		# for i in 1:l2
			combined_vector[l1+i] = encoding_vector[i]
		end

		return combined_vector
	end
end

# ╔═╡ 70dd59f1-5dca-4598-8620-8042728a7561
#=╠═╡
#needed for computing the value function for a single example at a time
function update_encoding!(encoding_vector::Vector{T}, params::FCANNParams{T}, feature_vector::Vector{T}; activations = FCANN.form_activations(params.weights[1])) where T<:Real
	FCANN.forwardNOGRAD_base!(activations, params.weights..., feature_vector, params.reslayers)
	encoding_vector .= last(activations)
	encoding_vector .-= mean(encoding_vector)
	if length(encoding_vector) > 1
		encoding_vector ./= (std(encoding_vector) .+ one(T)/1_000_000)
	end
end
  ╠═╡ =#

# ╔═╡ 3784e659-5963-4c09-9035-f61745ce2512
#=╠═╡
#create value function that uses an encoding vector as an additional input that is produced by another neural network contained in `encoding_setup`
function ReinforcementLearning.form_state_value_function(feature_vector::Vector{T}, update_feature_vector!::Function, parameters::FCANNParams{T}, encoding_size::Integer, encode_params::FCANNParams{T}) where {T<:Real}
	combined_size = length(feature_vector) + encoding_size
	function v̂(s; feature_vector::Vector{T} = deepcopy(feature_vector), encoding_vector::Vector{T} = zeros(T, encoding_size), combined_vector::Vector{T} = zeros(T, combined_size), parameters::FCANNParams{T} = parameters, encode_params::FCANNParams{T} = encode_params, activations = FCANN.form_activations(parameters.weights[1]), encoder_activations = FCANN.form_activations(encode_params.weights[1]), kwargs...)
		update_feature_vector!(feature_vector, s)
		update_encoding!(encoding_vector, encode_params, feature_vector; activations = encoder_activations)
		update_combined_features!(combined_vector, feature_vector, encoding_vector)
		ReinforcementLearning.fcann_value_function!(activations, combined_vector, parameters)
		return first(last(activations))
	end

	#also return a method that acts on the feature vector itself which has already been updated
	function v̂(x::Vector{T}, parameters; activations = FCANN.form_activations(parameters.weights[1]), kwargs...) 
		ReinforcementLearning.fcann_value_function!(activations, x, parameters)
		return first(last(activations))
	end

	form_kwargs() = (feature_vector = deepcopy(feature_vector), parameters = parameters, encoding_vector = zeros(T, encoding_size), combined_vector = zeros(T, combined_size), activations = FCANN.form_activations(parameters.weights[1]), encode_params = encode_params, encoder_activations = FCANN.form_activations(encode_params.weights[1]))
	
	return (v̂, form_kwargs)
end
  ╠═╡ =#

# ╔═╡ cb4eca54-3d51-4b4c-b2a9-b70ef1d90e1d
#=╠═╡
function form_policy_and_value_functions(game::StateStochasticGame{T, S, A, N, PTF, F1, F2}, feature_vectors::NTuple{N, V}, update_feature_vectors!::NTuple{N, Function}, policy_params::NTuple{N, Θ}, value_params::NTuple{N, W}, encoding_size::Integer, encode_params::NTuple{N, FCANNParams{T}}) where {T<:Real, S, A, N, PTF, F1, F2, V, Θ, W}
	function π!(policy::Vector{T}, x::V, params::Θ, args...)
		ReinforcementLearning.update_policy_dist!(policy, x, params, args...)
		return policy
	end

	value_functions = [form_state_value_function(feature_vectors[i], update_feature_vectors![i], value_params[i], encoding_size, encode_params[i]) for i in 1:N]

	v̂s = ntuple(i -> value_functions[i][1], N)
	form_value_kwargs = ntuple(i -> value_functions[i][2], N)

	form_policy_kwargs = ntuple(i -> () -> (feature_vector = deepcopy(feature_vectors[i]), policy = zeros(T, length(game.agent_actions[i])), policy_args = form_policy_args(policy_params[i]), encoding_vector = zeros(T, encoding_size), combined_vector = zeros(T, length(feature_vectors[i]) + encoding_size), encoder_activations = FCANN.form_activations(encode_params[i].weights[1])), N)

	πs = ntuple(N) do i
		combined_size = length(feature_vectors[i]) + encoding_size
		function π(s::S; feature_vector::V = deepcopy(feature_vectors[i]), policy::Vector{T} = zeros(T, length(game.agent_actions[i])), policy_parameters::Θ = policy_params[i], policy_args = form_policy_args(policy_parameters), encoding_vector::Vector{T} = zeros(T, encoding_size), combined_vector::Vector{T} = zeros(T, combined_size), encode_parameters = encode_params[i], encoder_activations = FCANN.form_activations(encode_parameters.weights[1]), kwargs...) 
			update_feature_vectors![i](feature_vector, s)
			update_encoding!(encoding_vector, encode_parameters, feature_vector; activations = encoder_activations)
			update_combined_features!(combined_vector, feature_vector, encoding_vector)
			π!(policy, combined_vector, policy_parameters, policy_args...)
		end
	end

	π_samples = ntuple(N) do i
		function π_sample(s::S; kwargs...) 
			policy = πs[i](s; kwargs...)
			sample_action(policy)
		end
	end

	policies_and_values = ntuple(N) do i
		combined_size = length(feature_vectors[i]) + encoding_size
		function policy_and_value(s::S; feature_vector::V = deepcopy(feature_vectors[i]), policy::Vector{T} = zeros(T, length(game.agent_actions[i])), policy_parameters::Θ = policy_params[i], value_parameters::W = value_params[i], policy_args = form_policy_args(policy_parameters), encoding_vector::Vector{T} = zeros(T, encoding_size), combined_vector::Vector{T} = zeros(T, combined_size), encode_parameters = encode_params[i], encoder_activations = FCANN.form_activations(encode_parameters.weights[1]), kwargs...)
			update_feature_vectors![i](feature_vector, s)
			update_encoding!(encoding_vector, encode_parameters, feature_vector; activations = encoder_activations)
			update_combined_features!(combined_vector, feature_vector, encoding_vector)
			ReinforcementLearning.update_policy_dist!(policy, combined_vector, policy_parameters, policy_args...)
			v = v̂s[i](combined_vector, value_parameters; kwargs...)
			return (value = v, policy_dist = policy)
		end
	end

	form_policy_and_value_kwargs = ntuple(i -> () -> (;form_value_kwargs[i]()..., form_policy_kwargs[i]()...), N)

	return (policy_functions = πs, form_policy_kwargs = form_policy_kwargs, value_functions = v̂s, form_value_kwargs = form_value_kwargs, policy_sample_actions = π_samples, policies_and_values = policies_and_values, form_policy_and_value_kwargs = form_policy_and_value_kwargs)
end
  ╠═╡ =#

# ╔═╡ efee2d6e-7d4f-4e73-907b-cd3126600bee
md"""
### Algorithm
"""

# ╔═╡ fee7e48a-146c-429b-bace-e98a5731d59c
#=╠═╡
function synchronous_independent_actor_critic!(policy_params::NTuple{N, PP}, value_params::NTuple{N, VP}, encoding_size::Integer, game::StateStochasticGame{T, S, A, N, PTF, F1, F2}, γ::T, max_steps::Integer, num_env::Integer, feature_vectors::NTuple{N, V}, update_feature_vectors!::NTuple{N, Function}, value_args::Tuple, value_gradient_args::Tuple, policy_args::Tuple, policy_gradient_args::Tuple, encoding_setup::NTuple{N, NamedTuple}; α_w::T = one(T)/10, α_θ::T = one(T)/10, α_ψ::T = one(T)/10, nstep::Integer = 0, ∇v̂::NTuple{N, VP} = deepcopy(value_params), ∇lnπ::NTuple{N, PP} = deepcopy(policy_params), return_loss::Bool = false) where {T<:Real, S, A, N, PTF, F1, F2, VP, PP, V}

	∇ψ = ntuple(i -> deepcopy(encoding_setup[i].full_params), N)
	combined_vectors = ntuple(i -> vcat(feature_vectors[i], zeros(T, encoding_size)), N)
	
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
	encoding_vector = zeros(T, encoding_size)
	current_feature_vectors = ntuple(i -> ReinforcementLearning.initialize_synchronous_features(feature_vectors[i], num_env), N) #should store the feature vectors of the current time state for that environment
	update_feature_vectors = ntuple(i -> ReinforcementLearning.initialize_synchronous_features(feature_vectors[i], num_env), N) #should store the feature vectors of the state being updated
	current_encoding_vectors = ntuple(i -> zeros(T, encoding_size, num_env), N) #should store the encoding vectors of the current time state for that environment
	update_encoding_vectors = ntuple(i -> zeros(T, encoding_size, num_env), N) #should store the encoding vectors of the state being updated
	current_combined_vectors = ntuple(i -> ReinforcementLearning.initialize_synchronous_features(combined_vectors[i], num_env), N) #should store the combined vectors of the current time state for that environment
	update_combined_vectors = ntuple(i -> ReinforcementLearning.initialize_synchronous_features(combined_vectors[i], num_env), N) #should store the combined vectors of the state being updated
	policy_matrix = ntuple(i -> zeros(T, num_env, length(game.agent_actions[i])), N)
	batch_actions = ntuple(_ -> ones(Int64, num_env), N)
	batch_state_values = ntuple(_ -> zeros(T, num_env), N)
	batch_targets = ntuple(_ -> zeros(T, num_env), N)
	δs = ntuple(_ -> zeros(T, num_env), N)
	row_sums = zeros(T, num_env)
	row_mins = zeros(T, num_env)
	row_maxes = zeros(T, num_env)

	encoding_vectors = ntuple(i -> zeros(T, encoding_size), N-1)

	batch_nstep_rewards = ntuple(_ -> [CircularBuffer{T}(N+1) for _ in 1:num_env], N)
	batch_nstep_states = ntuple(_ -> [CircularBuffer{S}(N+1) for _ in 1:num_env], N)
	batch_nstep_actions = ntuple(_ -> [CircularBuffer{Int64}(N+1) for _ in 1:num_env], N)
	batch_bootstrap_discount = ones(T, N, num_env)
	batch_ready = fill(false, num_env) #tracks for each environment if it is ready for a batch update.  initially this will not be true for any before not enough n-step data has been accumulated yet
	batch_terminal_check = fill(false, num_env) #tracks for each environment if the current episode has terminated or not
	update_actions = ntuple(_ -> fill(0, num_env), N)

	encoding_errors = ntuple(i -> ntuple(j -> Vector{T}(), N-1), N)

	for (i, s) in enumerate(batch_states)
		for k in 1:N
			update_feature_vectors![k](feature_vectors[k], s)
			ReinforcementLearning.update_feature_matrix!(current_feature_vectors[k], feature_vectors[k], i)
		end
	end

	for k in 1:N
		encoding_setup[k].update_encoding!(current_encoding_vectors[k], encoding_setup[k].encode_params, current_feature_vectors[k])
		update_combined_features!(current_combined_vectors[k], current_feature_vectors[k], current_encoding_vectors[k])
	end

	num_updates = 0
	batch_steps = fill(0, num_env)
	
	while num_updates < max_steps
		# @info "Current batch states: $batch_states"
		# @info "Using a policy matrix of $policy_matrix sampled the following actions: $batch_actions"

		#for each environment update the policy distribution on a per row basis and then sample an action from each environment
		if !all(batch_ready) && !all(batch_terminal_check) #only envs that are NOT ready perform a step update so if all are ready we can just proceed straight to gradient updates
			for i in 1:N
				ReinforcementLearning.update_batch_policy_dist!(policy_matrix[i], current_combined_vectors[i], policy_params[i], row_sums, row_mins, row_maxes, policy_args[i]...)
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
							# ReinforcementLearning.update_feature_matrix!(current_feature_vectors[i], feature_vectors[i], k)
							update_encoding!(encoding_vector, encoding_setup[i].encode_params, feature_vectors[i]; activations = encoding_setup[i].vector_activations)
							update_combined_features!(current_combined_vectors[i], feature_vectors[i], encoding_vector, k)
							# ReinforcementLearning.update_feature_matrix!(current_encoding_vectors[i], encoding_vector, k)
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
						# ReinforcementLearning.update_feature_matrix!(current_feature_vectors[i], feature_vectors[i], k)
						update_encoding!(encoding_vector, encoding_setup[i].encode_params, feature_vectors[i]; activations = encoding_setup[i].vector_activations)
						update_combined_features!(current_combined_vectors[i], feature_vectors[i], encoding_vector, k)
						# ReinforcementLearning.update_feature_matrix!(current_encoding_vectors[i], encoding_vector, k)
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
				other_agent_inds = get_other_agent_inds(N, i)
				
				# encoding_setup[i].update_encoding!(current_encoding_vectors[i], encoding_setup[i].encode_params, current_feature_vectors[i])
				# update_combined_features!(current_combined_vectors[i], current_feature_vectors[i], current_encoding_vectors[i])

				#calculate state values for current states
				ReinforcementLearning.update_batch_state_values!(batch_state_values[i], current_combined_vectors[i], value_params[i], value_args[i]...)
		
				#zero out prediction values for terminal states and add discounted value to reward
				batch_targets[i] .+= batch_bootstrap_discount[i] .* batch_state_values[i]

				#compute gradient of encoder function for other agents policies, also updates the encodings for that agent
				for (j, other_agent) in enumerate(other_agent_inds)
					loss = encoding_setup[i].update_gradients![j](∇ψ[i][j], update_encoding_vectors[i], encoding_setup[i].full_params[j], update_feature_vectors[i], update_actions[other_agent]; return_loss = return_loss)
					if return_loss
						push!(encoding_errors[i][j], loss)
					end
				end
				update_combined_features!(update_combined_vectors[i], update_feature_vectors[i], update_encoding_vectors[i])
		
				#updates value gradient with the loss function and updates δs with the states values minus the target values for use later in the policy gradient calculation
				ReinforcementLearning.update_batch_value_gradient!(∇v̂[i], δs[i], value_params[i], batch_targets[i], update_combined_vectors[i], value_gradient_args[i]...)	
		
				#updates batch advantage values to use in policy gradient by multiplying by γ^n where n is the number of steps since the episode started
				δs[i] .*= cs
				
				#update value parameters using the value gradient
				ReinforcementLearning.update_params_with_gradient!(value_params[i], α_w, ∇v̂[i])
		
				# @info "Updating policy_params with the following information: δs = $δs, policy_matrix = $policy_matrix"
				#update policy parameters using the policy distribution, batch actions, and advantage values
				ReinforcementLearning.update_batch_policy_gradient!(∇lnπ[i], policy_params[i], δs[i], policy_matrix[i], update_actions[i], update_combined_vectors[i], policy_gradient_args[i]...)
				
				ReinforcementLearning.update_params_with_gradient!(policy_params[i], α_θ, ∇lnπ[i])

				#update encoder-decoder parameters
				for j in 1:N-1
					ReinforcementLearning.update_params_with_gradient!(encoding_setup[i].full_params[j], α_ψ, ∇ψ[i][j])
				end
			end
	
			batch_ready .= false #once a gradient update has occured, inform all environments they need to perform a new step
			cs .*= γ
			num_updates += 1
		end

		
	end

	encode_params = ntuple(i -> deepcopy(encoding_setup[i].encode_params), N)
	decode_params = ntuple(i -> deepcopy(encoding_setup[i].decode_params), N)
	policy_and_value_components = form_policy_and_value_functions(game, feature_vectors, update_feature_vectors!, policy_params, value_params, encoding_size, encode_params)

	#note that this step is a noop unless the gradients are gpu objects in which case they get deallocated
	for i in 1:N
		ReinforcementLearning.cleanup_gradient!(∇v̂[i])
		ReinforcementLearning.cleanup_gradient!(∇lnπ[i])
		for j in 1:N-1
			ReinforcementLearning.cleanup_gradient!(∇ψ[i][j])
		end
	end

	return (;avg_step_rewards = avg_step_rewards, batch_episodes = batch_episodes, batch_episode_steps = batch_episode_steps, batch_episode_rewards = batch_episode_rewards, policy_parameters = policy_params, value_parameters = value_params, encode_params = encode_params, decode_params = decode_params, encoding_errors = encoding_errors, policy_and_value_components...)
end
  ╠═╡ =#

# ╔═╡ 471678aa-b5e1-4c94-a5b5-127e247f3c3b
md"""
### Non-linear Approximation
"""

# ╔═╡ fb2c7284-8f4f-48c6-a544-982bd3639259
#=╠═╡
#pass encoder and decoder params into the setup function as keyword arguments so I can reuse them in future runs
function setup_fcann_policy_encoders(game::StateStochasticGame{T, S, A, N, PTF, F1, F2}, agent_index::Integer, feature_vector::Vector{T}, encode_hidden_layers::Vector{Int64}, code_size::Integer, decode_hidden_layers::Vector{Int64}, reslayers::Integer, use_μP::Bool, num_env::Integer; 
	l2::T = zero(T), 
	dropout::T = zero(T),
	#there should only be one network of encoder parameters
	encode_params::FCANNParams{T} = initialize_fcann_params(feature_vector, encode_hidden_layers, code_size, reslayers, use_μP),
	other_agent_inds = get_other_agent_inds(N, agent_index),
	num_actions::Vector{Int64} = [length(game.agent_actions[i]) for i in other_agent_inds],
	#there is a separate decoder for each other agent
	decode_params::NTuple{M, FCANNParams{T}} = ntuple(i -> initialize_fcann_params(code_size, decode_hidden_layers, num_actions[i], reslayers, use_μP), N-1)) where {T<:Real, S, A, N, M, PTF, F1, F2}
	
	input_size = length(feature_vector)
	hidden_layers = vcat(encode_hidden_layers, code_size, decode_hidden_layers)

	#concatenate the encoder with each respective decoder to make linked parameters for a full network that shares the encoder half of the parameters
	full_params = ntuple(i -> (weights = (vcat(encode_params.weights[1], decode_params[i].weights[1]), vcat(encode_params.weights[2], decode_params[i].weights[2])), reslayers = reslayers), N-1)

	activation_list = vcat(fill(true, length(encode_hidden_layers)), false, fill(true, length(decode_hidden_layers)))

	encoding_index = length(encode_hidden_layers) + 1

	encoder_activations = FCANN.form_activations(encode_params.weights[1], num_env)
	encoder_vector_activations = FCANN.form_activations(encode_params.weights[1])

	activations = ntuple(i -> FCANN.form_activations(full_params[i].weights[1], num_env), N-1)
	tanh_grad_z = deepcopy(activations)
	deltas = deepcopy(activations)
	onesvec = ones(T, num_env)

	scales = fill(-one(T), length(full_params[1].weights[1]))
	if use_μP
		for i in eachindex(hidden_layers)
			i′ = i + 1
			scales[i′] /= size(full_params[1].weights[1][i′], 2)
		end
	end
	
	function update_encoding!(encoding_matrix::Matrix{T}, params::FCANNParams{T}, feature_matrix::Matrix{T})
		FCANN.forwardNOGRAD_base!(encoder_activations, params.weights..., feature_matrix, reslayers; input_orientation = 'T')
		
		#note that the activation output has examples per row.  We want the encoding matrix to match the transposed style of input matrices where each example is in a column, so we need to transpose the output before transfering it to the matrix
		for j in 1:size(last(encoder_activations), 2)
			@inbounds @simd for i in 1:size(last(encoder_activations), 1)
				encoding_matrix[j, i] = last(encoder_activations)[i, j]
			end
		end
		encoding_matrix .-= mean(encoding_matrix; dims = 1)
		if size(encoding_matrix, 1) > 1
			encoding_matrix ./= (std(encoding_matrix; dims = 1) .+ one(T)/1_000_000)
		end
		return encoding_matrix
	end

	#the gradient computation should also update the encoding matrix
	update_gradients! = ntuple(N-1) do other_agent_index
		function update_gradient!(∇ψ::FCANNParams{T}, encoding_matrix::Matrix{T}, params::FCANNParams{T}, input_matrix, output_actions::Vector{Int64}; return_loss::Bool = false)
			# debug_print = rand() < 0.00001
			debug_print = false
			FCANN.nnCostFunction(params.weights..., hidden_layers, input_matrix, output_actions, l2, ∇ψ.weights..., tanh_grad_z[other_agent_index], activations[other_agent_index], deltas[other_agent_index], onesvec, dropout; resLayers = reslayers, activation_list = activation_list, loss_type = CrossEntropyLoss(), input_orientation = 'T')
			ReinforcementLearning.scale_fcann_params!(∇ψ, scales)
			if other_agent_index == 1
				for j in 1:size(activations[other_agent_index][encoding_index], 2)
					@inbounds @simd for i in 1:size(activations[other_agent_index][encoding_index], 1)
						encoding_matrix[j, i] = activations[other_agent_index][encoding_index][i, j]
					end
				end
				encoding_matrix .-= mean(encoding_matrix; dims = 1)
				if size(encoding_matrix, 1) > 1
					encoding_matrix ./= (std(encoding_matrix; dims = 1) .+ one(T)/1_000_000)
				end
				debug_print && @info "Updated encoding matrix for agent $other_agent_index: $(encoding_matrix)"
			end

			# debug_print && @info "Gradients for agent $other_agent_index: $(∇ψ.weights[1][1])"
			# debug_print && @info "output actions: $output_actions"

			if return_loss
				output = activations[other_agent_index][end]
				softmax_output = output 
				softmax_sums = sum(softmax_output, dims=2)
				softmax_output ./= softmax_sums
				debug_print && @info "Softmax output for agent $other_agent_index: $(softmax_output[1, :])"
				cross_entropy_loss = sum(i -> -log(softmax_output[i, output_actions[i]]), 1:length(output_actions))
				return cross_entropy_loss / num_env
			else
				return zero(T)
			end
		end
	end

	return (full_params = full_params, encode_params = encode_params, decode_params = decode_params, update_encoding! = update_encoding!, update_gradients! = update_gradients!, vector_activations = encoder_vector_activations)
end
  ╠═╡ =#

# ╔═╡ 17f76bf9-19dc-4cde-823c-8ea9c1100d52
#=╠═╡
begin
	function synchronous_independent_actor_critic_fcann(game::StateStochasticGame{T, S, A, N, PTF, F1, F2}, γ::T, max_steps::Integer, feature_vectors::NTuple{N, V}, update_feature_vectors!::NTuple{N, Function}, hidden_layers::Vector{Int64}, encode_hidden_layers::Vector{Int64}, encoding_size::Integer, decode_hidden_layers::Vector{Int64}; 
		reslayers::Integer = 0, 
		encoding_reslayers::Integer = 0,
		code_reslayers::Integer = 0, 
		use_μP::Bool = true, 
		combined_vectors::NTuple{N, V} = ntuple(i -> vcat(feature_vectors[i], zeros(T, encoding_size)), N), 
		policy_params::NTuple{N, FCANNParams{T}} = ntuple(i -> initialize_fcann_params(combined_vectors[i], hidden_layers, length(game.agent_actions[i]), reslayers, use_μP), N), 
		value_params::NTuple{N, FCANNParams{T}} = ntuple(i -> initialize_fcann_value_params(policy_params[i], use_μP), N), 
		#there should only be one network of encoder parameters
		encode_params::NTuple{N, FCANNParams{T}} = ntuple(i -> initialize_fcann_params(feature_vectors[i], encode_hidden_layers, encoding_size, reslayers, use_μP), N),
		other_agent_inds = ntuple(agent_index -> get_other_agent_inds(N, agent_index), N),
		num_actions::NTuple{N, Vector{Int64}} = ntuple(agent_index -> [length(game.agent_actions[i]) for i in other_agent_inds[agent_index]], N),
		#there is a separate decoder for each other agent
		decode_params::NTuple{N, NTuple{Nm1, FCANNParams{T}}} = ntuple(agent_index -> ntuple(i -> initialize_fcann_params(encoding_size, decode_hidden_layers, num_actions[agent_index][i], reslayers, use_μP), N-1), N), 
		l2::T = zero(T), 
		dropout::T = zero(T), 
		num_env::Integer = 8, 
		activation_list::Vector{Bool} = fill(true, length(hidden_layers)), 
		use_gpu::Bool = false, 
		kwargs...) where {T<:Real, S, A, N, PTF, F1, F2, V<:Vector{T}, Nm1}
														
		#add setup functions
		policy_setups = ntuple(i -> ReinforcementLearning.setup_fcann_batch_policy_arguments(policy_params[i], num_env, l2, dropout, use_μP, activation_list; use_gpu = use_gpu), N)
		value_setups = ntuple(i -> ReinforcementLearning.setup_fcann_batch_value_arguments(policy_setups[i], value_params[i], num_env, l2, dropout, use_μP, activation_list; use_gpu = use_gpu), N)

		encoding_setups = ntuple(i -> setup_fcann_policy_encoders(game, i, feature_vectors[i], encode_hidden_layers, encoding_size, decode_hidden_layers, encoding_reslayers, use_μP, num_env; l2 = l2, dropout = dropout, encode_params = encode_params[i], decode_params = decode_params[i]), N)
	
		!use_gpu && return synchronous_independent_actor_critic!(policy_params, value_params, encoding_size, game, γ, max_steps, num_env, feature_vectors, update_feature_vectors!, ntuple(i -> value_setups[i].value_args, N), ntuple(i -> value_setups[i].value_gradient_args, N), ntuple(i -> policy_setups[i].policy_args, N), ntuple(i -> policy_setups[i].policy_gradient_args, N), encoding_setups; kwargs...)

		# isempty(value_setups[1].gpu_args) && error("GPU backend is not available")
		# isempty(policy_setups[1].gpu_args) && error("GPU backend is not available")
	
		# output = synchronous_independent_actor_critic!(ntuple(i -> policy_setups[i].gpu_args.params, N), ntuple(i -> value_setups[i].gpu_args.params, N), game, γ, max_steps, num_env, feature_vector, update_feature_vector!, ntuple(i -> value_setups[i].gpu_args..value_args, N), ntuple(i -> value_setups[i].gpu_args.value_gradient_args, N), ntuple(i -> policy_setups[i].gpu_args.policy_args, N), ntuple(i -> policy_setups[i].gpu_args.policy_gradient_args, N); kwargs...)
	
		# for i in 1:N
		# 	FCANN.GPU2Host(value_params[i].weights, value_setups[i].gpu_args.params.weights)
		# 	FCANN.GPU2Host(policy_params[i].weights, policy_setups[i].gpu_args.params.weights)
	
		# 	value_setups[i].gpu_args.cleanup_vars()
		# 	policy_setups[i].gpu_args.cleanup_vars()
		# end
	
		# return (;output..., policy_parameters = deepcopy(policy_params), value_parameters = deepcopy(value_params))
	end

	synchronous_independent_actor_critic_fcann(game::StateStochasticGame{T, S, A, N, PTF, F1, F2}, γ::T, max_steps::Integer, feature_vector::V, update_feature_vector!::Function, hidden_layers::Vector{Int64}, encode_hidden_layers::Vector{Int64}, encoding_size::Integer, decode_hidden_layers::Vector{Int64}; kwargs...) where {T<:Real, S, A, N, PTF, F1, F2, V} = synchronous_independent_actor_critic_fcann(game, γ, max_steps, convert_to_ntuple(feature_vector, N), convert_to_ntuple(update_feature_vector!, N), hidden_layers, encode_hidden_layers, encoding_size, decode_hidden_layers; kwargs...)
end
  ╠═╡ =#

# ╔═╡ a60d22c5-fbf8-4f66-ae14-21542177eb7c
#idea to see how accurate the encoding models are for the behavior of the other agents over time.  Can get the cross entropy loss output itself I think

# ╔═╡ a57030aa-a68c-44ac-9722-de0d97a43a0c
#need to add function that forms the final value functions which requires getting the encodings as well and passing that through to the input

# ╔═╡ acb37f7c-dea3-4066-9d50-21a2ccc8f852
md"""
### Test
"""

# ╔═╡ aafd9f0f-e8e2-44e2-ad4e-994a19f72553
md"""
# Test Environment
"""

# ╔═╡ ff934cf6-96cf-4fa5-b1a7-6d876023657a
# ╠═╡ skip_as_script = true
#=╠═╡
const lbf_2244 = LevelBasedForaging.make_environment(;num_agents = 2, num_items = 2, width = 4, height = 4, force_cooperation = true, reset_chance = 0.01f0)
  ╠═╡ =#

# ╔═╡ cd390cd1-713e-40b9-8d64-cd65f7532019
# ╠═╡ skip_as_script = true
#=╠═╡
function make_lbf_relative_feature(width::Integer, height::Integer, agent_levels::NTuple{N, Int64}, item_levels::NTuple{M, Int64}, max_agent_level::Integer, max_item_level::Integer) where {N, M}
	#first calculate the number of features and the number of active features
	num_features = (N-1)*(width - 1 +height - 1) + M*(width-1+height-1+4) #the number of indicators span the length, width, and max level for all agents and items.  The other agents and items will have a level indicator relative to the level of the agent in question represented as a real number from 0 to 1 where 0 means the agent level exceeds the agent or item and 1 means the agent or item is a higher level with the largest possible difference

	#note that relative positions for x and y mean that the maximum distance away anything can be is width-1 or height-1

	#initialize feature vector 
	feature_vector = zeros(Float32, num_features)

	max_level = max(max_agent_level, max_item_level)

	#gets the feature vector for the X and Y position of all of the other agents relative to the agent index
	function update_relative_agent_positions!(feature_vector::Vector{T}, s::LevelBasedForaging.ForagingState{N, M, X, Y}, agent_index::Integer) where {X, Y, T<:Real}
		(x0, y0) = s.agent_positions[agent_index]
		for (i, other_agent_index) in enumerate(vcat(1:agent_index - 1, agent_index + 1:N))
			base_ind = (i - 1)*(X-1+Y-1)
			(x, y) = s.agent_positions[other_agent_index]
			d_x = x - x0
			ind = abs(d_x)
			if !iszero(ind)
				feature_vector[base_ind + ind] = sign(d_x)
			end
			d_y = y - y0
			ind = abs(d_y)
			if !iszero(ind)
				feature_vector[base_ind + X-1 + ind] = sign(d_y)
			end
		end
	end

	function update_relative_item_positions!(feature_vector::Vector{T}, s::LevelBasedForaging.ForagingState{N, M, X, Y}, agent_index::Integer) where {X, Y, T<:Real}
		agent_position_ind = (N-1)*(X-1+Y-1)
		(x0, y0) = s.agent_positions[agent_index]
		for item_index in 1:M
			if !s.item_collect[item_index]
				base_ind = (item_index - 1) * (X-1 + Y-1)
				(x, y) = s.item_positions[item_index]
				d_x = x - x0
				ind = abs(d_x)
				if !iszero(ind)
					feature_vector[agent_position_ind + base_ind + ind] = sign(d_x)
				end
				d_y = y - y0
				ind = abs(d_y)
				if !iszero(ind)
					feature_vector[agent_position_ind + base_ind + X-1 + ind] = sign(d_y)
				end
			end
		end
	end

	function update_relative_agent_levels!(feature_vector::Vector{T}, s::LevelBasedForaging.ForagingState{N, M, X, Y}, agent_index::Integer) where {X, Y, T<:Real}
		base_position_ind = (N-1)*(X-1 + Y-1) + M*(X-1 + Y-1) 
		l0 = s.agent_levels[agent_index]
		for (i, other_agent_index) in enumerate(vcat(1:agent_index - 1, agent_index + 1:N))
			l = s.agent_levels[other_agent_index]
			normalized_level = max(zero(T), (l - l0) / (max_level - 1)) 
			feature_vector[base_position_ind + i] = normalized_level
		end
	end

	function update_relative_item_levels!(feature_vector::Vector{T}, s::LevelBasedForaging.ForagingState{N, M, X, Y}, agent_index::Integer) where {X, Y, T<:Real}
		base_ind = (N-1)*(X-1 + Y-1) + M*(X-1 + Y-1) + N-1 
		l0 = s.agent_levels[agent_index]
		for item_index in 1:M
			l = s.item_levels[item_index]
			normalized_level = max(zero(T), (l - l0) / (max_level - 1)) 
			feature_vector[base_ind + item_index] = normalized_level
		end
	end

	function update_feature_vector!(x::Vector{T}, s::LevelBasedForaging.ForagingState{N, M, X, Y}, agent_index::Integer) where {T<:Real, X, Y}
		x .= zero(T)
		update_relative_agent_positions!(x, s, agent_index)
		update_relative_item_positions!(x, s, agent_index)
		update_relative_agent_levels!(x, s, agent_index)
		update_relative_item_levels!(x, s, agent_index)
		return x
	end

	feature_vectors = ntuple(i -> copy(feature_vector), N)
	update_feature_vectors! = ntuple(i -> (x, s) -> update_feature_vector!(x, s, i), N)

	return (feature_vectors, update_feature_vectors!)
end
  ╠═╡ =#

# ╔═╡ e26b747e-7c5d-44c6-93ee-867c30745449
# ╠═╡ skip_as_script = true
#=╠═╡
function make_lbf_relative_feature(s::LevelBasedForaging.ForagingState{N, M, X, Y}, max_agent_level::Integer, max_item_level::Integer) where {N, M, X, Y}
	make_lbf_relative_feature(X, Y, s.agent_levels, s.item_levels, max_agent_level, max_item_level)
end
  ╠═╡ =#

# ╔═╡ 7fb1a159-119e-4f50-887e-e9421cc0a075
#=╠═╡
initialize_linear_jal_params(lbf_2244, make_lbf_relative_feature(lbf_2244.initialize_state(), 2, 4)[1][1], 1, 0f0) |> typeof
  ╠═╡ =#

# ╔═╡ 67417a8b-1338-4d14-b261-e76996e3dded
#=╠═╡
joint_action_learning_linear(lbf_2244, 0.99f0, typemax(Int64), 100_000, make_lbf_relative_feature(lbf_2244.initialize_state(), 2, 4)...)
  ╠═╡ =#

# ╔═╡ a3228244-c224-4cf3-9790-9a0667ff65ae
#=╠═╡
const test_output = synchronous_independent_actor_critic_fcann(lbf_2244, 0.99f0, 100_000, make_lbf_relative_feature(lbf_2244.initialize_state(), 2, 4)..., [64, 64], [64, 64], 64, [64, 64, 64]; α_θ = 3f-1, α_w = 3f-1, α_ψ = 3f-1, nstep = 10, l2 = 0.01f0)
  ╠═╡ =#

# ╔═╡ d6788776-209b-4a35-9c92-6898f2175662
#=╠═╡
plot(test_output.avg_step_rewards |> a -> a[1] .+ a[2] |> cumsum |> v -> v[1:1000:end] ./ (1:1000:length(v)))
  ╠═╡ =#

# ╔═╡ 27bc8217-96d3-4409-9638-f404cdab6e8c
#=╠═╡
const test_ep = runepisode(lbf_2244; πs = test_output.policy_sample_actions)
  ╠═╡ =#

# ╔═╡ 1da94f7b-ea96-4f02-9c05-a01395c18946
#=╠═╡
sum(a -> sum(a), test_ep[3])
  ╠═╡ =#

# ╔═╡ b202d9d7-6d43-4fb7-99e0-8a5ccf3e64b1
#=╠═╡
const test_policy_kwargs = ntuple(i -> test_output.form_policy_kwargs[i](), 2)
  ╠═╡ =#

# ╔═╡ 6c1af149-2170-4f2f-a6d0-04914405e15a
#=╠═╡
const test_policies = ntuple(i -> s -> test_output.policy_sample_actions[i](s; test_policy_kwargs[i]...), 2)
  ╠═╡ =#

# ╔═╡ cc044d81-b1db-47a5-bd50-22e2e9120042
#=╠═╡
runepisode(lbf_2244; πs = test_policies, max_steps = 10_000)
  ╠═╡ =#

# ╔═╡ 14dbb722-252c-4c15-900e-d7f4a803e856
#=╠═╡
test_output.form_value_kwargs[1]()
  ╠═╡ =#

# ╔═╡ 2ebdf022-b16e-4ff8-a141-41d0b49e7264
#=╠═╡
test_output.form_policy_and_value_kwargs[1]()
  ╠═╡ =#

# ╔═╡ e9f36954-4f2d-4797-a6f6-641c17d03bf7
#=╠═╡
ntuple(i -> test_output.policies_and_values[i](test_ep[1][1]), 2)
  ╠═╡ =#

# ╔═╡ 7eb9f189-d783-4733-bedf-4debd5b1e746
#=╠═╡
synchronous_independent_actor_critic_fcann(lbf_2244, 0.99f0, 0, make_lbf_relative_feature(lbf_2244.initialize_state(), 2, 4)..., [2, 2], [2, 2], 2, [2, 2]; α_θ = 3f-2, α_w = 3f-2, α_ψ = 3f-2, nstep = 0)
  ╠═╡ =#

# ╔═╡ 71a7fc3c-3f46-11f1-ac55-c93979544e11
md"""
# Dependencies
"""

# ╔═╡ 5c0d6435-51da-4769-a7de-ea97b3b18053
# ╠═╡ skip_as_script = true
#=╠═╡
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
  ╠═╡ =#

# ╔═╡ 7cff61b0-4d86-11f1-9c48-3752f377ae54
#=╠═╡
function test_encoder_decoder_fixed_policies!(policy_params::NTuple{N, PP}, value_params::NTuple{N, VP}, encoding_size::Integer, game::StateStochasticGame{T, S, A, N, PTF, F1, F2}, γ::T, max_steps::Integer, num_env::Integer, feature_vectors::NTuple{N, V}, update_feature_vectors!::NTuple{N, Function}, value_args::Tuple, value_gradient_args::Tuple, policy_args::Tuple, policy_gradient_args::Tuple, encoding_setup::NTuple{N, NamedTuple}; α_ψ::T = one(T)/10, nstep::Integer = 0, ∇ψ::NTuple{N, NTuple{Nm1, VP}} = ntuple(i -> ntuple(j -> deepcopy(encoding_setup[i].full_params[j]), N-1), N), fixed_action::Integer = 1) where {T<:Real, S, A, N, PTF, F1, F2, VP, PP, V, Nm1}

	combined_vectors = ntuple(i -> vcat(feature_vectors[i], zeros(T, encoding_size)), N)
	
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
	encoding_vector = zeros(T, encoding_size)
	current_feature_vectors = ntuple(i -> ReinforcementLearning.initialize_synchronous_features(feature_vectors[i], num_env), N)
	update_feature_vectors = ntuple(i -> ReinforcementLearning.initialize_synchronous_features(feature_vectors[i], num_env), N)
	current_encoding_vectors = ntuple(i -> zeros(T, encoding_size, num_env), N) #should store the encoding vectors of the current time state for that environment
	update_encoding_vectors = ntuple(i -> zeros(T, encoding_size, num_env), N) #should store the encoding vectors of the state being updated
	policy_matrix = ntuple(i -> zeros(T, num_env, length(game.agent_actions[i])), N)
	batch_actions = ntuple(_ -> ones(Int64, num_env), N)
	batch_state_values = ntuple(_ -> zeros(T, num_env), N)
	batch_targets = ntuple(_ -> zeros(T, num_env), N)
	δs = ntuple(_ -> zeros(T, num_env), N)
	row_sums = zeros(T, num_env)
	row_mins = zeros(T, num_env)
	row_maxes = zeros(T, num_env)

	encoding_vectors = ntuple(i -> zeros(T, encoding_size), N-1)

	batch_nstep_rewards = ntuple(_ -> [CircularBuffer{T}(N+1) for _ in 1:num_env], N)
	batch_nstep_states = ntuple(_ -> [CircularBuffer{S}(N+1) for _ in 1:num_env], N)
	batch_nstep_actions = ntuple(_ -> [CircularBuffer{Int64}(N+1) for _ in 1:num_env], N)
	batch_bootstrap_discount = ones(T, N, num_env)
	batch_ready = fill(false, num_env)
	batch_terminal_check = fill(false, num_env)
	update_actions = ntuple(_ -> fill(0, num_env), N)

	encoding_errors = ntuple(i -> ntuple(j -> Vector{T}(), N-1), N)

	for (i, s) in enumerate(batch_states)
		for k in 1:N
			update_feature_vectors![k](feature_vectors[k], s)
			ReinforcementLearning.update_feature_matrix!(current_feature_vectors[k], feature_vectors[k], i)
		end
	end

	# for k in 1:N
	# 	encoding_setup[k].update_encoding!(current_encoding_vectors[k], encoding_setup[k].encode_params, current_feature_vectors[k])
	# 	update_combined_features!(current_combined_vectors[k], current_feature_vectors[k], current_encoding_vectors[k])
	# end

	num_updates = 0
	batch_steps = fill(0, num_env)
	
	while num_updates < max_steps
		# Fixed policy: all agents always choose action 1
		for i in 1:N
			batch_actions[i] .= fixed_action
		end

		#each environment has a tuple of joint actions for all of the agents in that environment
		env_joint_actions = ntuple(num_env) do k
			ntuple(i -> batch_actions[i][k], N)
		end
		
		r_avg = zeros(T, N) 
		#perform transitions for entire batch
		for k in 1:num_env
			if !batch_ready[k]
				if !batch_terminal_check[k]
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
						
					if (length(batch_nstep_rewards[1][k]) == N + 1) || terminal
						batch_ready[k] = true
					end
				elseif length(batch_nstep_rewards[1][k]) > 1
					for i in 1:N
						popfirst!(batch_nstep_rewards[i][k])
						popfirst!(batch_nstep_states[i][k])
						popfirst!(batch_nstep_actions[i][k])
						batch_bootstrap_discount[i, k] = zero(T)
					end
					batch_ready[k] = true
				else
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
					batch_targets[i][k] = sum(batch_nstep_rewards[i][k][t]*γ^(t-1) for t in eachindex(batch_nstep_rewards[i][k]); init = zero(T))
					r_avg[i] += first(batch_nstep_rewards[i][k])
				end
			end

			for i in 1:N
				push!(avg_step_rewards[i], r_avg[i] / num_env)
			end
			
			for i in 1:N
				other_agent_inds = get_other_agent_inds(N, i)
				
				# SKIP: encoding and combined feature computation since we don't update policies/values
				
				#calculate state values for current states (but don't update value parameters)
				# Note: We still need state values for bootstrapping, but we can compute them directly without encodings
				# For simplicity, we'll use zero values since we're not updating value params anyway
				batch_state_values[i] .= zero(T)
		
				#zero out prediction values for terminal states and add discounted value to reward
				batch_targets[i] .+= batch_bootstrap_discount[i] .* batch_state_values[i]

				#compute gradient of encoder function for other agents policies, also updates the encodings for that agent
				for (j, other_agent) in enumerate(other_agent_inds)
					loss = encoding_setup[i].update_gradients![j](∇ψ[i][j], update_encoding_vectors[i], encoding_setup[i].full_params[j], update_feature_vectors[i], update_actions[other_agent]; return_loss = true)
					push!(encoding_errors[i][j], loss)
				end
				# SKIP: combined feature updates since we don't use them
				
				# SKIP: value gradient computation and updates
				# SKIP: policy gradient computation and updates
		
				#update encoder-decoder parameters only
				for j in 1:N-1
					ReinforcementLearning.update_params_with_gradient!(encoding_setup[i].full_params[j], α_ψ, ∇ψ[i][j])
				end
			end
	
			batch_ready .= false
			cs .*= γ
			num_updates += 1
		end
	end

	encode_params = ntuple(i -> encoding_setup[i].encode_params, N)
	decode_params = ntuple(i -> encoding_setup[i].decode_params, N)
	policy_and_value_components = form_policy_and_value_functions(game, feature_vectors, update_feature_vectors!, policy_params, value_params, encoding_size, encode_params)

	#cleanup gradients
	for i in 1:N
		for j in 1:N-1
			ReinforcementLearning.cleanup_gradient!(∇ψ[i][j])
		end
	end

	return (;avg_step_rewards = avg_step_rewards, batch_episodes = batch_episodes, batch_episode_steps = batch_episode_steps, batch_episode_rewards = batch_episode_rewards, policy_parameters = policy_params, value_parameters = value_params, encode_params = encode_params, decode_params = decode_params, encoding_errors = encoding_errors, policy_and_value_components...)
end
  ╠═╡ =#

# ╔═╡ 7cff85aa-4d86-11f1-ba0c-99c1395b48ff
function test_encoder_decoder_fixed_policies(game::StateStochasticGame{T, S, A, N, PTF, F1, F2}, γ::T, max_steps::Integer, feature_vectors::NTuple{N, V}, update_feature_vectors!::NTuple{N, Function}, hidden_layers::Vector{Int64}, encode_hidden_layers::Vector{Int64}, encoding_size::Integer, decode_hidden_layers::Vector{Int64}; 
	reslayers::Integer = 0, 
	code_reslayers::Integer = 0, 
	use_μP::Bool = true, 
	combined_vectors::NTuple{N, V} = ntuple(i -> vcat(feature_vectors[i], zeros(T, encoding_size)), N), 
	policy_params::NTuple{N, FCANNParams{T}} = ntuple(i -> initialize_fcann_params(combined_vectors[i], hidden_layers, length(game.agent_actions[i]), reslayers, use_μP), N), 
	value_params::NTuple{N, FCANNParams{T}} = ntuple(i -> initialize_fcann_value_params(policy_params[i], use_μP), N), 
	encode_params::NTuple{N, FCANNParams{T}} = ntuple(i -> initialize_fcann_params(feature_vectors[i], encode_hidden_layers, encoding_size, reslayers, use_μP), N),
	other_agent_inds = ntuple(agent_index -> get_other_agent_inds(N, agent_index), N),
	num_actions::NTuple{N, Vector{Int64}} = ntuple(agent_index -> [length(game.agent_actions[i]) for i in other_agent_inds[agent_index]], N),
	decode_params::NTuple{N, NTuple{Nm1, FCANNParams{T}}} = ntuple(agent_index -> ntuple(i -> initialize_fcann_params(encoding_size, decode_hidden_layers, num_actions[agent_index][i], reslayers, use_μP), N-1), N), 
	l2::T = zero(T), 
	dropout::T = zero(T), 
	num_env::Integer = 8, 
	activation_list::Vector{Bool} = fill(true, length(hidden_layers)), 
	use_gpu::Bool = false, 
	α_ψ::T = one(T)/10,
	kwargs...) where {T<:Real, S, A, N, PTF, F1, F2, V, Nm1}
														
	#add setup functions
	policy_setups = ntuple(i -> ReinforcementLearning.setup_fcann_batch_policy_arguments(policy_params[i], num_env, l2, dropout, use_μP, activation_list; use_gpu = use_gpu), N)
	value_setups = ntuple(i -> ReinforcementLearning.setup_fcann_batch_value_arguments(policy_setups[i], value_params[i], num_env, l2, dropout, use_μP, activation_list; use_gpu = use_gpu), N)

	encoding_setups = ntuple(i -> setup_fcann_policy_encoders(game, i, feature_vectors[i], encode_hidden_layers, encoding_size, decode_hidden_layers, reslayers, use_μP, num_env; l2 = l2, dropout = dropout, encode_params = encode_params[i], decode_params = decode_params[i]), N)
	
	!use_gpu && return test_encoder_decoder_fixed_policies!(policy_params, value_params, encoding_size, game, γ, max_steps, num_env, feature_vectors, update_feature_vectors!, ntuple(i -> value_setups[i].value_args, N), ntuple(i -> value_setups[i].value_gradient_args, N), ntuple(i -> policy_setups[i].policy_args, N), ntuple(i -> policy_setups[i].policy_gradient_args, N), encoding_setups; α_ψ = α_ψ, kwargs...)

	# GPU version would go here if needed
	error("GPU version not implemented for test function")
end

test_encoder_decoder_fixed_policies(game::StateStochasticGame{T, S, A, N, PTF, F1, F2}, γ::T, max_steps::Integer, feature_vector::V, update_feature_vector!::Function, hidden_layers::Vector{Int64}, encode_hidden_layers::Vector{Int64}, encoding_size::Integer, decode_hidden_layers::Vector{Int64}; kwargs...) where {T<:Real, S, A, N, PTF, F1, F2, V} = test_encoder_decoder_fixed_policies(game, γ, max_steps, convert_to_ntuple(feature_vector, N), convert_to_ntuple(update_feature_vector!, N), hidden_layers, encode_hidden_layers, encoding_size, decode_hidden_layers; kwargs...)

# ╔═╡ 7cffbe44-4d86-11f1-af54-93f72bb546a9
#=╠═╡
begin
	# Test the encoder-decoder with fixed policies (always action 1)
	(feature_vectors, update_feature_vectors!) = make_lbf_relative_feature(lbf_2244.initialize_state(), 2, 4)
	
	# Run test for 1000 steps
	test_result = test_encoder_decoder_fixed_policies(
		lbf_2244, 
		0.99f0, 
		1000, 
		feature_vectors, 
		update_feature_vectors!, 
		[32, 32],  # hidden layers
		[32, 32],  # encode layers  
		32,        # encoding size
		[32, 32];  # decode layers
		α_ψ = 1f-2,
		num_env = 4
	)
	
	# Check that cross-entropy loss is decreasing
	println("Test completed. Encoding errors shape: ", size.(test_result.encoding_errors))
	println("Final losses for agent 1 predicting agent 2: ", test_result.encoding_errors[1][1][end-10:end])
end
  ╠═╡ =#

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
BenchmarkTools = "~1.8.0"
CSV = "~0.10.16"
DataFrames = "~1.8.2"
DataStructures = "~0.19.4"
HiGHS = "~1.23.0"
HypertextLiteral = "~0.9.5"
JuMP = "~1.30.0"
LaTeXStrings = "~1.4.0"
PlutoDevMacros = "~0.9.2"
PlutoPlotly = "~0.6.5"
PlutoProfile = "~0.4.0"
PlutoUI = "~0.7.80"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.12.6"
manifest_format = "2.0"
project_hash = "71d6103283c8197e699614aaf4eb4eafda6e2322"

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
deps = ["Compat", "JSON", "Logging", "PrecompileTools", "Printf", "Profile", "Statistics", "UUIDs"]
git-tree-sha1 = "9670d3febc2b6da60a0ae57846ba74670290653f"
uuid = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
version = "1.8.0"

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
deps = ["InteractiveUtils", "REPL", "UUIDs"]
git-tree-sha1 = "cfb7a2e89e245a9d5016b70323db412b3a7438d5"
uuid = "da1fd8a2-8d9e-5ec2-8556-3022fb5608a2"
version = "3.0.2"

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
git-tree-sha1 = "5fab31e2e01e70ad66e3e24c968c264d1cf166d6"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.8.2"

[[deps.DataStructures]]
deps = ["OrderedCollections"]
git-tree-sha1 = "e86f4a2805f7f19bec5129bc9150c38208e5dc23"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.19.4"

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
git-tree-sha1 = "cddeab6487248a39dae1a960fff0ac17b2a28888"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "1.3.3"

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
deps = ["HiGHS_jll", "LinearAlgebra", "MathOptIIS", "MathOptInterface", "OpenBLAS32_jll", "PrecompileTools", "SparseArrays"]
git-tree-sha1 = "bf5e946f72ebd1b4620249a6be7ff34832ba9ca0"
uuid = "87dc4568-4c63-4d18-b0c0-bb2238e4078b"
version = "1.23.0"

[[deps.HiGHS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Zlib_jll", "libblastrampoline_jll"]
git-tree-sha1 = "50ed12dc8c37ebb8d2b759f21755259d8512f2bd"
uuid = "8fd58aa0-07eb-5a78-9b36-339c94fd15ea"
version = "1.14.0+0"

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
git-tree-sha1 = "4091a1338a0e32766b11b9bd3fac247d34200c77"
uuid = "4076af6c-e467-56ae-b986-b466b2749572"
version = "1.30.0"

    [deps.JuMP.extensions]
    JuMPDimensionalDataExt = "DimensionalData"

    [deps.JuMP.weakdeps]
    DimensionalData = "0703355e-b756-11e9-17c0-8b28908087d0"

[[deps.JuliaInterpreter]]
deps = ["CodeTracking", "InteractiveUtils", "Random", "UUIDs"]
git-tree-sha1 = "58927c485919bf17ea308d9d82156de1adf4b006"
uuid = "aa1ae85d-cabe-5617-a682-6adf51b2e16a"
version = "0.10.12"

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

[[deps.MathOptIIS]]
deps = ["MathOptInterface"]
git-tree-sha1 = "3b3d69130d8ab8c39d5fa4d30e20a8e6428c9d37"
uuid = "8c4f8055-bd93-4160-a86b-a0c04941dbff"
version = "0.2.0"

[[deps.MathOptInterface]]
deps = ["BenchmarkTools", "CodecBzip2", "CodecZlib", "ForwardDiff", "JSON", "LinearAlgebra", "MutableArithmetics", "NaNMath", "OrderedCollections", "PrecompileTools", "Printf", "SparseArrays", "SpecialFunctions", "Test"]
git-tree-sha1 = "ce739e3d8a21313ea418772edfc3b7b15a1dfc16"
uuid = "b8f27783-ece8-5eb3-8dc8-9495eed66fee"
version = "1.50.1"

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
git-tree-sha1 = "7c25249fc13a070f5ba433c50e21e22bb33c6fb0"
uuid = "d8a4904e-b15c-11e9-3269-09a3773c0cb0"
version = "1.7.1"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "9b8215b1ee9e78a293f99797cd31375471b2bcae"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.1.3"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.3.0"

[[deps.OpenBLAS32_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "46cce8b42186882811da4ce1f4c7208b02deb716"
uuid = "656ef2d0-ae68-5445-9ca0-591084a874a2"
version = "0.3.30+0"

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
git-tree-sha1 = "fbc875044d82c113a9dee6fc14e16cf01fd48872"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.80"

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
git-tree-sha1 = "624de6279ab7d94fc9f672f0068107eb6619732c"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "3.3.2"

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
git-tree-sha1 = "2700b235561b0335d5bef7097a111dc513b8655e"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.7.2"

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
# ╟─54fd8229-4611-4cda-a36f-9b8df8ddb2da
# ╟─1e978fa9-e664-4a84-9451-ade2d09216ad
# ╠═672157a7-a7d4-4476-a34d-a192420daac0
# ╠═ea2647b7-4141-464b-8a30-0390831e0975
# ╟─1f416406-bcc6-4cee-bc2e-4def87e7caec
# ╠═bb19a7d2-70d7-4412-9fff-df0e2becc6ad
# ╟─559d8ec0-61ce-4837-bd3e-2e19acfbcfbf
# ╠═5a29909f-2268-412e-ade5-bb8ca1a6aba3
# ╠═8075e662-04a8-46ae-9d49-82ee2a465303
# ╠═2ff33994-2142-4c87-87ff-d585be2a5a0f
# ╠═7c167873-55b0-4a61-a1f0-a0c9d5573967
# ╠═7fb1a159-119e-4f50-887e-e9421cc0a075
# ╠═67417a8b-1338-4d14-b261-e76996e3dded
# ╠═413d6cd9-2b0d-404e-849e-d0ad18cde7f8
# ╠═01f46865-4d1c-491f-aa59-812bf0434b48
# ╠═92a76486-76fb-4ef6-a5f4-16c7add8ede7
# ╠═0482822e-8275-49c7-87a6-db08e6b46895
# ╠═651d1dd4-7fe1-4e2c-98ae-744cf346932f
# ╟─a9c977ae-8fef-4e6d-b4fe-220422f589a3
# ╟─c43e0fa7-3fc6-4af8-94b3-4be49faeb0a9
# ╠═318c9efc-fe93-49ed-93c2-9c98b0a91a04
# ╠═a997aa3f-b11c-42eb-96fc-daa518384f39
# ╠═199a483d-1eaf-4c90-b931-2359b0bee30c
# ╠═5cc2328f-e346-400c-b7e8-297a6b090baf
# ╠═70dd59f1-5dca-4598-8620-8042728a7561
# ╠═3784e659-5963-4c09-9035-f61745ce2512
# ╠═cb4eca54-3d51-4b4c-b2a9-b70ef1d90e1d
# ╟─efee2d6e-7d4f-4e73-907b-cd3126600bee
# ╠═fee7e48a-146c-429b-bace-e98a5731d59c
# ╟─471678aa-b5e1-4c94-a5b5-127e247f3c3b
# ╠═fb2c7284-8f4f-48c6-a544-982bd3639259
# ╠═17f76bf9-19dc-4cde-823c-8ea9c1100d52
# ╠═a60d22c5-fbf8-4f66-ae14-21542177eb7c
# ╠═a57030aa-a68c-44ac-9722-de0d97a43a0c
# ╟─acb37f7c-dea3-4066-9d50-21a2ccc8f852
# ╠═a3228244-c224-4cf3-9790-9a0667ff65ae
# ╠═d6788776-209b-4a35-9c92-6898f2175662
# ╠═7eb9f189-d783-4733-bedf-4debd5b1e746
# ╠═27bc8217-96d3-4409-9638-f404cdab6e8c
# ╠═1da94f7b-ea96-4f02-9c05-a01395c18946
# ╠═b202d9d7-6d43-4fb7-99e0-8a5ccf3e64b1
# ╠═6c1af149-2170-4f2f-a6d0-04914405e15a
# ╠═cc044d81-b1db-47a5-bd50-22e2e9120042
# ╠═14dbb722-252c-4c15-900e-d7f4a803e856
# ╠═2ebdf022-b16e-4ff8-a141-41d0b49e7264
# ╠═e9f36954-4f2d-4797-a6f6-641c17d03bf7
# ╠═aafd9f0f-e8e2-44e2-ad4e-994a19f72553
# ╠═ff934cf6-96cf-4fa5-b1a7-6d876023657a
# ╠═cd390cd1-713e-40b9-8d64-cd65f7532019
# ╠═e26b747e-7c5d-44c6-93ee-867c30745449
# ╠═71a7fc3c-3f46-11f1-ac55-c93979544e11
# ╠═bb0f0b2d-575c-43dd-ad5e-11902144b3e5
# ╠═c87289df-243d-4ee9-a1e6-d6a677730f99
# ╠═b08b4238-b047-485a-81fe-8670399b2342
# ╠═4ba61675-1b58-46fd-a5d5-ec260a061bf7
# ╠═5c0d6435-51da-4769-a7de-ea97b3b18053
# ╠═7cff61b0-4d86-11f1-9c48-3752f377ae54
# ╠═7cff85aa-4d86-11f1-ba0c-99c1395b48ff
# ╠═7cffbe44-4d86-11f1-af54-93f72bb546a9
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
