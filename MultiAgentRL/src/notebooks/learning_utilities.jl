### A Pluto.jl notebook ###
# v0.20.24

using Markdown
using InteractiveUtils

# ╔═╡ c098aeb7-c1f8-4c0a-94a3-28df60646895
using PlutoDevMacros

# ╔═╡ 2b9161b1-c68d-4182-8331-f6d3e89941de
using DataFrames, CSV, JuMP, HiGHS, DataStructures, Dates

# ╔═╡ 40210d93-8476-4980-bd3d-31dcd0267ca9
# ╠═╡ skip_as_script = true
#=╠═╡
begin
	using PlutoUI, PlutoPlotly, PlutoProfile, BenchmarkTools, LaTeXStrings, HypertextLiteral
	TableOfContents(;depth = 4)
end
  ╠═╡ =#

# ╔═╡ 3fa2f499-bd4c-4f75-92c1-6523759a8b27
@only_in_nb begin
	@frompackage @raw_str(joinpath(@__DIR__, "..", "RL_Module")) import *
	include(joinpath(@__DIR__, "multi_agent_types.jl"))
	include(joinpath(@__DIR__, "joint_action_learning.jl"))
	include(joinpath(@__DIR__, "independent_learning.jl"))
	include(joinpath(@__DIR__, "agent_modeling.jl"))
end

# ╔═╡ cbd924a8-0889-477d-afe0-f86107bc6f4f
md"""
# Test Environment
"""

# ╔═╡ 709be4b1-42e1-4ed4-a5a7-61f646eff4c7
# ╠═╡ skip_as_script = true
#=╠═╡
const game_1_tabular = LevelBasedForaging.make_small_environment()
  ╠═╡ =#

# ╔═╡ 1530f486-3424-47ff-bcc4-daf2184bfa52
#=╠═╡
const tabular_mdp_1 = TabularMDP(game_1_tabular, sum)
  ╠═╡ =#

# ╔═╡ ad7a5093-c8d4-4a2a-ae3e-b5da409cf1da
#=╠═╡
const game_1_exact = value_iteration_v(tabular_mdp_1, 0.99f0)
  ╠═╡ =#

# ╔═╡ 7aa80d07-02e2-4b21-a712-27302d3f974f
#=╠═╡
runepisode(tabular_mdp_1; π = game_1_exact.optimal_policy)
  ╠═╡ =#

# ╔═╡ 8049e06d-286f-45c2-8f4d-090261317168
md"""
Game 1 has an optimal solution that takes 4 steps to complete an episode
"""

# ╔═╡ ef90c554-058e-4c30-8e1c-3971bcb4ec83
#=╠═╡
const game_1_state = StateStochasticGame(game_1_tabular)
  ╠═╡ =#

# ╔═╡ 6b87627e-c9ce-4a25-9a99-953e6220ec49
# ╠═╡ disabled = true
#=╠═╡
const game_1_feature_sparse = StateAggregationFeatureVector(length(game_1_tabular.states))
  ╠═╡ =#

# ╔═╡ 094391df-77aa-4dc1-b01f-ba2729db25b1
#=╠═╡
const game_1_feature_dense = zeros(Float32, length(game_1_tabular.states))
  ╠═╡ =#

# ╔═╡ f3c6108e-33e5-48a7-a27d-4d765cb60b1a
# ╠═╡ disabled = true
#=╠═╡
const game_1_feature_binary = let
	x = BinaryFeatureVector(length(game_1_tabular.states))
	push!(x.active_features, 1)
	x.num_features = 1
	x
end
  ╠═╡ =#

# ╔═╡ 170747b0-a370-48cf-925b-27cd3638f04e
#=╠═╡
begin
	function update_game_1_feature!(x::StateAggregationFeatureVector, s)
		i_s = game_1_tabular.state_index[s]
		x.group_index = i_s
		return x
	end

	function update_game_1_feature!(x::BinaryFeatureVector, s)
		i_s = game_1_tabular.state_index[s]
		x.active_features[1] = i_s
		return x
	end

	function update_game_1_feature!(x::Vector{T}, s) where T<:Real
		x .= zero(T)
		i_s = game_1_tabular.state_index[s]
		x[i_s] = one(T)
	end
end
  ╠═╡ =#

# ╔═╡ 159c9d65-897c-4bb3-83ed-78b48d901593
md"""
# Training Utilities
"""

# ╔═╡ fa657909-32eb-41ab-a9f2-321b24e9c7c3
md"""
## Exhaustive Training
"""

# ╔═╡ b3df9dba-1c3f-482a-bacb-4591b9cd08ea
md"""
### Performance Evaluation

For general sum games, agents each receive an independent reward per step.  In order to measure progress with such a reward structure, one must reduce the multi-agent rewards into a single performance metric.  This type of reduction usually only is meaningful for cooperative games where one can, for example, sum over the agent rewards to arrive at a single value that represents the collective success of all agents in the environment.  Alternatively, if the environment is a common reward game then the reward value is already reduced to a single number.  For completeness, the evaluation functions take an arbitrary reduction function in order to reduce the rewards to a single value.  By default this reduction function is `+`.
"""

# ╔═╡ f531a3c9-78a0-446f-a3a1-1170f400a949
function reward_value(rewards::T, reducer::Function) where {T<:Real}
	return rewards
end

function reward_value(rewards, reducer::Function)
	return reduce(reducer, rewards)
end

function evaluate_policy_performance(game::StateStochasticGame{T, S, A, N, P, F1, F2}, πs::NTuple{N, Function}, eval_steps::Integer; use_steps::Bool = false, min_reward::T = typemin(T), reducer::Function = +) where {T<:Real, S, A, N, P, F1, F2}
	(states, actions, rewards, sterm, nsteps) = runepisode(game; πs = πs, max_steps = eval_steps)
	!game.isterm(sterm) && return min_reward
	reward_sum = sum(reward_value(r, reducer) for r in rewards)
	episode_count = 1
	step_count = nsteps
	remaining_steps = eval_steps - nsteps
	while remaining_steps > 0
		(states, actions, rewards, sterm, nsteps) = runepisode(game; πs = πs, max_steps = remaining_steps)
		if game.isterm(sterm) 
			reward_sum += sum(reward_value(r, reducer) for r in rewards)
			episode_count += 1
			step_count += nsteps
		end
		remaining_steps -= nsteps
	end
	!use_steps && return reward_sum / episode_count
	return reward_sum / step_count
end

# ╔═╡ ee78a3c4-a941-47dd-bd0e-6f4886b8a65a
begin
	#if the episode rewards are a single vector then just return that vector
	make_combined_rewards(episode_rewards::Vector{T}, reducer::Function) where T<:Real = episode_rewards

	#if the episode rewards are a vector for each agent, then combine them with the reducer function
	function make_combined_rewards(episode_rewards::NTuple{N, V}, reducer::Function) where {T<:Real, V<:Vector{T}, N}
		isempty(episode_rewards[1]) && return V()
		l = length(episode_rewards[1])
		[reduce(reducer, (rewards[i] for rewards in episode_rewards)) for i in 1:l]
	end
end

# ╔═╡ 1ba4cd28-f2fd-4e50-b9a6-22a3e48c81c9
function check_reward_progress(output::NamedTuple; use_steps::Bool = false, reducer::Function = +, kwargs...)
	combined_rewards = make_combined_rewards(output.episode_rewards, reducer)
	(!use_steps || isempty(output.episode_steps)) && return ReinforcementLearning.check_reward_progress(combined_rewards; kwargs...)

	l = length(output.episode_steps)
	l == 1 && return first(combined_rewards) / output.episode_steps[1]
	episode_check = ceil(Int64, l/2)
	sum(view(combined_rewards, episode_check:l)) / (output.episode_steps[l] - output.episode_steps[max(1, episode_check - 1)])
end

# ╔═╡ 52afd5fb-f426-4cf5-801e-e7c3d23a66e7
function check_batch_reward_progress(output::NamedTuple; use_steps::Bool = false, reducer::Function = +, kwargs...)
	if use_steps
		combined_rewards = make_combined_rewards(output.avg_step_rewards, reducer)
		l = length(combined_rewards)
		l == 1 && return first(combined_rewards)
		step_check = ceil(Int64, l/2)
		sum(view(combined_rewards, step_check:l)) / (l - step_check+1)
	else
		batch_combined_rewards = [make_combined_rewards(ntuple(i -> output.batch_episode_rewards[i][j], length(output.batch_episode_rewards)), reducer) for j in 1:length(output.batch_episode_rewards[1])]
		ReinforcementLearning.check_reward_progress(batch_combined_rewards; kwargs...)
	end
end

# ╔═╡ d996a1de-1dc6-43ac-bb6c-9948e0ac4e5d
md"""
### Value Training
"""

# ╔═╡ e4a1c364-64ad-40df-a10b-9939fee69a21
function Base.copy!(dest::NTuple{N, Array{T, M}}, src::NTuple{N, Array{T, M}}) where {T<:Real, N, M}
	for i in 1:N
		dest[i] .= src[i]
	end
	return dest
end

# ╔═╡ 62a2af5e-11c4-4e06-a4b7-043a9daef999
function Base.copy!(dest::NTuple{N, FCANNParams{T}}, src::NTuple{N, FCANNParams{T}}) where {T<:Real, N}
	for k in 1:N
		copy!(dest[k], src[k])
	end
end

# ╔═╡ fae0d63f-1cba-4dd7-a413-dc7d8b5c8432
function Base.copy!(dest::NTuple{N, NTuple{M, FCANNParams{T}}}, src::NTuple{N, NTuple{M, FCANNParams{T}}}) where {T<:Real, N, M}
	for k in 1:N
		for l in 1:M
			copy!(dest[k][l], src[k][l])
		end
	end
end

# ╔═╡ 880e73ca-976a-409d-ac65-380c4f56fe58
ReinforcementLearning.check_bad_params(params::Tuple) = any(ReinforcementLearning.check_bad_params, params)

# ╔═╡ fb823f3f-adbc-4496-aefe-34a6d5f3bf1a
function setup_MARL_value_training(game::StateStochasticGame{T, S, A, N, P, F1, F2}, feature_vectors::NTuple{N, V}, update_feature_vectors!::NTuple{N, Function}; linear_value_params::NTuple{N, Matrix{T}} = ntuple(i -> initialize_linear_parameters(feature_vectors[i], length(game.agent_actions[i]), zero(T)), N), fcann_value_params::Dict = Dict{NamedTuple, NTuple{N, FCANNParams{T}}}(), linear_utility_params::NTuple{N, Matrix{T}} = ntuple(i -> initialize_linear_parameters(feature_vectors[i], length(game.agent_actions[i]), zero(T)), N), fcann_utility_params::Dict = Dict{NamedTuple, NTuple{N, FCANNParams{T}}}(), min_reward::T = typemin(T), reward_reducer::Function = +) where {T<:Real, S, A, N, P, F1<:Function, F2<:Function, V}
	function reset_params!(params::NTuple{N, Matrix{T}})
		for i in 1:N
			params[i] .= initialize_linear_parameters(feature_vectors[i], length(game.agent_actions[i]), zero(T))
		end
	end

	function reset_params!(params::NTuple{N, FCANNParams{T}}; use_μP::Bool = true)
		for i in 1:N
			input_size, hidden_layers, num_layers = get_network_dimensions(params[i])
			newparams = initialize_fcann_params(input_size, hidden_layers, length(game.agent_actions[i]), params[i].reslayers, use_μP)
			copy!(params[i], newparams)
		end
		return params
	end

	function initialize_value_params(hidden_layers::Vector{Int64}, reslayers::Integer; reset_params::Bool = false, use_μP::Bool = true)
		key = (hidden_layers = hidden_layers, reslayers = reslayers)
		if !haskey(fcann_value_params, key)
			@info "Initializing new value parameters with hidden layers = $hidden_layers and reslayers = $reslayers"
			value_params = ntuple(i -> initialize_fcann_params(feature_vectors[i], hidden_layers, length(game.agent_actions[i]), reslayers, use_μP), N)
			fcann_value_params[key] = value_params
		elseif reset_params
			@info "Resetting value parameters with hidden layers = $hidden_layers and reslayers = $reslayers to new initial values"
			reset_params!(fcann_value_params[key]; use_μP = use_μP)
		else
			fcann_value_params[key]
		end
	end

	function initialize_utility_params(hidden_layers::Vector{Int64}, reslayers::Integer; reset_params::Bool = false, use_μP::Bool = true)
		key = (hidden_layers = hidden_layers, reslayers = reslayers)
		if !haskey(fcann_utility_params, key)
			@info "Initializing new utility parameters with hidden layers = $hidden_layers and reslayers = $reslayers"
			utility_params = ntuple(i -> initialize_fcann_params(feature_vectors[i], hidden_layers, length(game.agent_actions[i]), reslayers, use_μP), N)
			fcann_utility_params[key] = utility_params
		elseif reset_params
			@info "Resetting utility parameters with hidden layers = $hidden_layers and reslayers = $reslayers to new initial values"
			reset_params!(fcann_utility_params[key]; use_μP = use_μP)
		else
			fcann_utility_params[key]
		end
	end

	function idqn_train(γ::T, α::T, nstep::Integer, max_steps::Integer; max_episodes::Integer = typemax(Int64), new_params::Bool = true, kwargs...)
		new_params && reset_params!(linear_value_params)
		independent_dqn_linear(game, γ, max_episodes, max_steps, deepcopy(feature_vectors), update_feature_vectors!; value_params = linear_value_params, α = α, nstep = nstep, kwargs...)
	end

	function idqn_train(hidden_layers::Vector{Int64}, reslayers::Integer, γ::T, α::T, nstep::Integer, max_steps::Integer; max_episodes::Integer = typemax(Int64), new_params::Bool = true, use_μP::Bool = true, kwargs...)
		value_params = initialize_value_params(hidden_layers, reslayers; reset_params = new_params, use_μP = use_μP)
		independent_dqn_fcann(game, γ, max_episodes, max_steps, deepcopy(feature_vectors), update_feature_vectors!, hidden_layers; reslayers = reslayers, value_params = value_params, α = α, nstep = nstep, kwargs...)
	end

	function vdn_train(γ::T, α::T, nstep::Integer, max_steps::Integer; max_episodes::Integer = typemax(Int64), new_params::Bool = true, kwargs...)
		new_params && reset_params!(linear_utility_params)
		independent_vdn_linear(game, γ, max_episodes, max_steps, deepcopy(feature_vectors), update_feature_vectors!; utility_params = linear_utility_params, α = α, nstep = nstep, kwargs...)
	end

	function vdn_train(hidden_layers::Vector{Int64}, reslayers::Integer, γ::T, α::T, nstep::Integer, max_steps::Integer; max_episodes::Integer = typemax(Int64), new_params::Bool = true, use_μP::Bool = true, kwargs...)
		utility_params = initialize_utility_params(hidden_layers, reslayers; reset_params = new_params, use_μP = use_μP)
		independent_vdn_fcann(game, γ, max_episodes, max_steps, deepcopy(feature_vectors), update_feature_vectors!, hidden_layers; reslayers = reslayers, utility_params = utility_params, α = α, nstep = nstep, kwargs...)
	end

	function train_exhaustive(method::AbstractString, γ::T, α::T, nstep::Integer, trial_steps::Integer; kwargs...)
		f, params = if method == "idqn"
			idqn_train, linear_value_params
		elseif method == "vdn"
			vdn_train, linear_utility_params
		else
			error("Unknown value training method of $method.  Allowed methods are: \"idqn\" and \"vdn\"")
		end

		function train(α, nstep, trial_steps, new_params; kwargs...)
			f(γ, α, nstep, trial_steps; new_params = new_params, kwargs...)
		end

		@info "Starting exhaustive $method linear training with γ = $γ, α = $α with $nstep step returns and $trial_steps steps per trial"
		train_exhaustive(train, params, γ, α, nstep, trial_steps; kwargs...)
	end

	function train_exhaustive(method::AbstractString, hidden_layers::Vector{Int64}, reslayers::Integer, γ::T, α::T, nstep::Integer, trial_steps::Integer; use_μP::Bool = true, kwargs...)
		f, params = if method == "idqn"
			idqn_train, initialize_value_params(hidden_layers, reslayers; use_μP = use_μP)
		elseif method == "vdn"
			vdn_train, initialize_utility_params(hidden_layers, reslayers; use_μP = use_μP)
		else
			error("Unknown value training method of $method.  Allowed methods are: \"idqn\" and \"vdn\"")
		end

		function train(α, nstep, trial_steps, new_params; kwargs...)
			f(hidden_layers, reslayers, γ, α, nstep, trial_steps; new_params = new_params, use_μP = use_μP, kwargs...)
		end

		@info "Starting exhaustive $method non-linear training with hidden layers: $hidden_layers, reslayers: $reslayers, γ = $γ, α = $α with $nstep step returns and $trial_steps steps per trial"
		train_exhaustive(train, params, γ, α, nstep, trial_steps; kwargs...)
	end
	
	function train_exhaustive(train::Function, params, γ::T, α::T, nstep::Integer, trial_steps::Integer; new_params::Bool = false, ϵ::T = one(T)/10, use_steps::Bool = false, kwargs...)
		ReinforcementLearning.check_bad_params(params) && error("Current parameter values are bad")
		
		output1 = train(zero(T), 0, 0, new_params; kwargs...)
		π_kwargs = ntuple(i -> output1.form_kwargs[i](), N)
		πs = extract_joint_policies(game, output1.value_functions; ϵ=ϵ)
		baseline_reward = evaluate_policy_performance(game, ntuple(i -> s -> πs[i](s; π_kwargs[i]...), N), trial_steps; use_steps = use_steps, min_reward = min_reward, reducer = reward_reducer)
		reward1 = baseline_reward
		trial = 0

		@info "Baseline episode reward is $reward1, beginning first trial"
		backup_params = deepcopy(params)
		output2 = train(α, nstep, trial_steps, false; ϵ = ϵ, kwargs...)
		reward2 = check_reward_progress(output2; use_steps = use_steps, min_reward = min_reward, reducer = reward_reducer)

		if ReinforcementLearning.check_bad_params(params)
			@info "First trial resulted in bad parameter values"
			copy!(params, backup_params)
			return (;output1..., performance = reward1)
		elseif reward2 ≤ reward1
			@info "First trial performance of $reward2 failed to improve reward"
			copy!(params, backup_params)
			return (;output1..., performance = reward1)
		end

		episode_rewards = output2.episode_rewards
		while (reward2 > reward1) && !ReinforcementLearning.check_bad_params(params)
			trial += 1
			@info "On trial $trial, reward improved from $reward1 to $reward2"
			output1 = output2
			reward1 = reward2
			copy!(backup_params, params)
			episode_rewards = if isa(episode_rewards, Tuple)
				ntuple(i -> vcat(episode_rewards[i], output1.episode_rewards[i]), N)
			else
				vcat(episode_rewards, output1.episode_rewards)
			end

			output2 = train(α, nstep, trial_steps, false; ϵ = ϵ, kwargs...)
			reward2 = check_reward_progress(output2; use_steps = use_steps, min_reward = min_reward, reducer = reward_reducer)
		end

		if ReinforcementLearning.check_bad_params(params)
			@info "Final trial resulted in bad parameter values"
		else
			@info "Final trial performance of $reward2 failed to improve reward"
		end

		@info "Performance after $trial trials improved from $baseline_reward to $reward1"

		copy!(params, backup_params)
		return (;output1..., episode_rewards = episode_rewards, performance = reward1)
	end

	function train_rate_decay(train_args::Tuple, γ::T, α_init::T, nstep::Integer, trial_steps::Integer; kwargs...)
		output1 = train_exhaustive(train_args..., γ, α_init, nstep, trial_steps; kwargs...)
		episode_rewards = output1.episode_rewards

		α = α_init / 2
		@info "Reducing learning rate to $α for next set of trials"
		output2 = train_exhaustive(train_args..., γ, α, nstep, trial_steps; kwargs..., new_params = false)

		if output2.performance ≤ output1.performance
			@info "Second round performance of $(output2.performance) failed to improve reward"
			@info "Completed rate decay training after 1 round with performance of $(output1.performance)"
		end

		round = 2
		while output2.performance > output1.performance
			round += 1
			α /= 2
			output1 = output2
			episode_rewards = if isa(episode_rewards, Tuple)
				ntuple(i -> vcat(episode_rewards[i], output1.episode_rewards[i]), N)
			else
				vcat(episode_rewards, output1.episode_rewards)
			end
			@info "On round $round, reducing learning rate to $α"
			output2 = train_exhaustive(train_args..., γ, α, nstep, trial_steps; kwargs..., new_params = false)
		end
		@info "Completed rate decay training after $round rounds with performance $(output1.performance)."
		return (;output1..., episode_rewards = episode_rewards)
	end

	function train_ϵ_decay(train_args::Tuple, γ::T, ϵ_init::T, α_init::T, nstep::Integer, trial_steps::Integer; ϵ_min::T = one(T) / 20, kwargs...)
		output1 = train_rate_decay(train_args, γ, α_init, nstep, trial_steps; kwargs..., ϵ = ϵ_init)
		episode_rewards = output1.episode_rewards

		ϵ = ϵ_init / 2
		@info "Reducing exploration parameter to $ϵ for next set of trials"
		output2 = train_rate_decay(train_args, γ, α_init, nstep, trial_steps; kwargs..., ϵ = ϵ, new_params = false)

		if output2.performance ≤ output1.performance
			@info "Performance with ϵ = $ϵ of $(output2.performance) failed to improve over ϵ = $ϵ_init"
			@info "Completed ϵ decay training after 1 round with performance of $(output1.performance)"
		end

		round = 2
		while (output2.performance > output1.performance) && (ϵ > ϵ_min)
			round += 1
			ϵ /= 2
			output1 = output2
			episode_rewards = if isa(episode_rewards, Tuple)
				ntuple(i -> vcat(episode_rewards[i], output1.episode_rewards[i]), N)
			else
				vcat(episode_rewards, output1.episode_rewards)
			end
			@info "On round $round, reducing exploration parameter to $ϵ"
			output2 = train_rate_decay(train_args, γ, α_init, nstep, trial_steps; kwargs..., ϵ = ϵ, new_params = false)
		end

		if output2.performance > output1.performance
			output1 = output2
			episode_rewards = if isa(episode_rewards, Tuple)
				ntuple(i -> vcat(episode_rewards[i], output1.episode_rewards[i]), N)
			else
				vcat(episode_rewards, output1.episode_rewards)
			end
		end
		
		@info "Completed ϵ decay training after $round rounds with performance $(output1.performance)."
		return (;output1..., episode_rewards = episode_rewards)
	end	

	function train_rate_decay(method::AbstractString, γ::T, α_init::T, nstep::Integer, trial_steps::Integer; kwargs...)
		@info "Starting rate decay $method linear training with an initial learning rate of $α_init"
		train_rate_decay((method,), γ, α_init, nstep, trial_steps; kwargs...)
	end

	function train_rate_decay(method::AbstractString, hidden_layers::Vector{Int64}, reslayers::Integer, γ::T, α_init::T, nstep::Integer, trial_steps::Integer; kwargs...)
		@info "Starting rate decay $method nonlinear training with an initial learning rate of $α_init"
		train_rate_decay((method, hidden_layers, reslayers), γ, α_init, nstep, trial_steps; kwargs...)
	end

	function train_ϵ_decay(method::AbstractString, γ::T, α_init::T, nstep::Integer, trial_steps::Integer; ϵ_init::T = one(T) / 2, kwargs...)
		@info "Starting ϵ decay $method linear training with an initial exploration parameter of $ϵ_init"
		train_ϵ_decay((method,), γ, ϵ_init, α_init, nstep, trial_steps; kwargs...)
	end

	function train_ϵ_decay(method::AbstractString, hidden_layers::Vector{Int64}, reslayers::Integer, γ::T, α_init::T, nstep::Integer, trial_steps::Integer; ϵ_init::T = one(T) / 2, kwargs...)
		@info "Starting ϵ decay $method nonlinear training with an initial exploration parameter of $ϵ_init"
		train_ϵ_decay((method, hidden_layers, reslayers), γ, ϵ_init, α_init, nstep, trial_steps; kwargs...)
	end

	return (train_idqn = idqn_train, train_vdn = vdn_train, train_exhaustive = train_exhaustive, train_rate_decay = train_rate_decay, train_ϵ_decay = train_ϵ_decay, linear_value_params = linear_value_params, linear_utility_params = linear_utility_params, fcann_value_params = fcann_value_params, fcann_utility_params = fcann_utility_params)
end		

# ╔═╡ dc7910ff-4906-4384-a691-e8ce5150750b
setup_MARL_value_training(game::StateStochasticGame{T, S, A, N, P, F1, F2}, feature_vector::V, update_feature_vector!::Function; kwargs...) where {T<:Real, S, A, N, P, F1<:Function, F2<:Function, V} = setup_MARL_value_training(game, convert_to_ntuple(feature_vector, N), convert_to_ntuple(update_feature_vector!, N); kwargs...)

# ╔═╡ 329cf3dd-a981-4f28-949e-2f72a1b37548
md"""
### Value Tests
"""

# ╔═╡ 1aff84c6-a4ff-439f-940d-fbdc0b0276d3
# ╠═╡ disabled = true
#=╠═╡
const game_1_sparse_value_test = setup_MARL_value_training(game_1_state, game_1_feature_sparse, update_game_1_feature!; min_reward = 0f0)
  ╠═╡ =#

# ╔═╡ 948c4b2c-32be-43ff-8584-dc4312c6e127
# ╠═╡ disabled = true
#=╠═╡
const game_1_dense_value_test = setup_MARL_value_training(game_1_state, game_1_feature_dense, update_game_1_feature!; min_reward = 0f0)
  ╠═╡ =#

# ╔═╡ c7f4f8a6-19e1-420e-9dd5-fe348837b67d
# ╠═╡ disabled = true
#=╠═╡
const game_1_binary_value_test = setup_MARL_value_training(game_1_state, game_1_feature_binary, update_game_1_feature!; min_reward = 0f0)
  ╠═╡ =#

# ╔═╡ 7ba601ae-fd52-4063-9fa3-6e8f43f59c9c
#=╠═╡
game_1_sparse_value_test.train_idqn(0.99f0, 0.1f0, 10, 1_000; batch_size = 16)
  ╠═╡ =#

# ╔═╡ fe69bcdb-9ece-423a-b9b2-a31ee259c0be
#=╠═╡
game_1_sparse_value_test.train_idqn([4, 4], 1, 0.99f0, 0.1f0, 10, 1_000; batch_size = 16, new_params = false)
  ╠═╡ =#

# ╔═╡ a16a7f91-8c47-4600-9b77-a6bb0e49aff3
#=╠═╡
game_1_sparse_value_test.train_vdn(0.99f0, 0.1f0, 10, 1_000; batch_size = 16)
  ╠═╡ =#

# ╔═╡ 40b5763d-936c-41ad-9de6-62dc3f786ca2
#=╠═╡
game_1_sparse_value_test.train_vdn([4, 4], 1, 0.99f0, 0.1f0, 10, 1_000; batch_size = 16, new_params = false)
  ╠═╡ =#

# ╔═╡ 6b5a56b0-3252-4f7a-a829-dfdbc3f0147b
md"""
#### State Aggregation Test
"""

# ╔═╡ df7dd868-a3d5-4d2b-afb9-18e871b684a3
#=╠═╡
const game_1_sparse_linear_idqn_value_output = game_1_sparse_value_test.train_ϵ_decay("idqn", 0.99f0, 0.1f0, 10, 10_000; batch_size = 16, use_steps = true, new_params = true)
  ╠═╡ =#

# ╔═╡ 38b4bf3b-0835-4425-9b26-6ee3b70cf2b0
md"""
After training, agent policies can complete episodes in 7 steps which is ideal.  Exhaustive IDQN linear training is successful with the `+` reducer on rewards.
"""

# ╔═╡ 1bbe0183-c6b6-45c2-8c4d-f2034391d8db
#=╠═╡
runepisode(game_1_state; πs = extract_joint_policies(game_1_state, game_1_sparse_linear_idqn_value_output.value_functions; ϵ = 0.0f0), max_steps = 1_000)
  ╠═╡ =#

# ╔═╡ 490a29fc-c1bc-4935-a8f9-780002ee3563
#=╠═╡
const game_1_sparse_nonlinear_idqn_value_output = game_1_sparse_value_test.train_ϵ_decay("idqn", [2, 2], 1, 0.99f0, 0.1f0, 10, 10_000; batch_size = 16, use_steps = true)
  ╠═╡ =#

# ╔═╡ 6513bba4-6bea-4a5c-b372-97b01f147849
#=╠═╡
runepisode(game_1_state; πs = extract_joint_policies(game_1_state, game_1_sparse_nonlinear_idqn_value_output.value_functions; ϵ = 0.0f0), max_steps = 1_000)
  ╠═╡ =#

# ╔═╡ 22d9860b-cf49-43a3-b935-65583952f56c
#=╠═╡
const game_1_sparse_linear_vdn_value_output = game_1_sparse_value_test.train_ϵ_decay("vdn", 0.99f0, 0.01f0, 10, 10_000; batch_size = 16, use_steps = true, new_params = true)
  ╠═╡ =#

# ╔═╡ 512614d5-d07e-446e-9717-72a299311586
#=╠═╡
runepisode(game_1_state; πs = extract_joint_policies(game_1_state, game_1_sparse_linear_vdn_value_output.value_functions; ϵ = 0.0f0), max_steps = 1_000)
  ╠═╡ =#

# ╔═╡ fdf3d679-d767-44c0-b841-20872cc50c92
md"""
After training, agent policies can complete episodes in 7 steps which is ideal.  Exhaustive VDN linear training is successful with the `+` reducer on rewards.
"""

# ╔═╡ 85110098-63e8-48b5-98c6-2758651b8a24
#=╠═╡
const game_1_sparse_nonlinear_vdn_value_output = game_1_sparse_value_test.train_ϵ_decay("vdn", [2, 2], 1, 0.99f0, 0.1f0, 10, 10_000; batch_size = 16, use_steps = true)
  ╠═╡ =#

# ╔═╡ 7925d5be-4eba-4e80-81a3-c0cfc2fde561
#=╠═╡
runepisode(game_1_state; πs = extract_joint_policies(game_1_state, game_1_sparse_nonlinear_vdn_value_output.value_functions; ϵ = 0.0f0), max_steps = 1_000)
  ╠═╡ =#

# ╔═╡ 9a93c45f-8fff-4517-a1e6-6f0d47963e75
md"""
#### Dense Vector Test
"""

# ╔═╡ 4af81e90-7a8d-4fd7-9fdc-3ae5cf52a581
#=╠═╡
const game_1_dense_linear_vdn_value_output = game_1_dense_value_test.train_ϵ_decay("vdn", 0.99f0, 0.1f0, 10, 10_000; batch_size = 16, use_steps = true, new_params = true)
  ╠═╡ =#

# ╔═╡ a5169ac3-e089-41c9-ad61-b75292fd1b86
#=╠═╡
runepisode(game_1_state; πs = extract_joint_policies(game_1_state, game_1_dense_linear_vdn_value_output.value_functions; ϵ = 0.0f0), max_steps = 1_000)
  ╠═╡ =#

# ╔═╡ 88cae643-9a2d-4980-8934-db0db6e6cd55
#=╠═╡
const game_1_dense_nonlinear_vdn_value_output = game_1_dense_value_test.train_ϵ_decay("vdn", [2, 2], 1, 0.99f0, 0.1f0, 10, 10_000; batch_size = 16, use_steps = true)
  ╠═╡ =#

# ╔═╡ ad124b03-31de-4ab6-9f00-b422e716b76c
#=╠═╡
runepisode(game_1_state; πs = extract_joint_policies(game_1_state, game_1_dense_nonlinear_vdn_value_output.value_functions; ϵ = 0.0f0), max_steps = 1_000)
  ╠═╡ =#

# ╔═╡ 8dab0aef-c550-4a2d-babb-934082488c2f
md"""
#### Binary Feature Test
"""

# ╔═╡ 8985bdc3-4bc9-4ff6-ad20-671fff1ff1a2
#=╠═╡
const game_1_binary_linear_idqn_value_output = game_1_binary_value_test.train_ϵ_decay("idqn", 0.99f0, 0.1f0, 10, 10_000; batch_size = 16, use_steps = true, new_params = true)
  ╠═╡ =#

# ╔═╡ 8727f317-a2bc-44e1-bece-f3171cc7878c
#=╠═╡
runepisode(game_1_state; πs = extract_joint_policies(game_1_state, game_1_binary_linear_idqn_value_output.value_functions; ϵ = 0.0f0), max_steps = 1_000)
  ╠═╡ =#

# ╔═╡ 53154ff1-5296-4b62-b4ba-76631f739440
#=╠═╡
const game_1_binary_nonlinear_idqn_value_output = game_1_binary_value_test.train_ϵ_decay("idqn", [2, 2], 1, 0.99f0, 0.5f0, 10, 10_000; batch_size = 16, use_steps = true)
  ╠═╡ =#

# ╔═╡ 481e9182-9c2c-45a5-bfeb-48943792ed02
#=╠═╡
runepisode(game_1_state; πs = extract_joint_policies(game_1_state, game_1_binary_nonlinear_idqn_value_output.value_functions; ϵ = 0.0f0), max_steps = 1_000)
  ╠═╡ =#

# ╔═╡ 023cfc09-a0be-4e2d-ba90-79b30314bffa
#=╠═╡
const game_1_binary_linear_vdn_value_output = game_1_binary_value_test.train_ϵ_decay("vdn", 0.99f0, 0.1f0, 10, 10_000; batch_size = 16, use_steps = true, new_params = true)
  ╠═╡ =#

# ╔═╡ fdcf63d5-17d2-4a29-b174-43a20b06897a
#=╠═╡
runepisode(game_1_state; πs = extract_joint_policies(game_1_state, game_1_binary_linear_vdn_value_output.value_functions; ϵ = 0.0f0), max_steps = 1_000)
  ╠═╡ =#

# ╔═╡ ca5b24fa-d401-40a8-be9f-0eb0282832b9
#=╠═╡
const game_1_binary_nonlinear_vdn_value_output = game_1_binary_value_test.train_ϵ_decay("vdn", [2, 2], 1, 0.99f0, 0.5f0, 10, 10_000; batch_size = 16, use_steps = true)
  ╠═╡ =#

# ╔═╡ 58bc7bd1-c44a-401b-b69a-cb509a0edf8b
#=╠═╡
runepisode(game_1_state; πs = extract_joint_policies(game_1_state, game_1_binary_nonlinear_vdn_value_output.value_functions; ϵ = 0.0f0), max_steps = 1_000)
  ╠═╡ =#

# ╔═╡ 88bed632-6696-4675-a10f-770687f14b14
md"""
### Policy Training
"""

# ╔═╡ a6287074-c72a-47f6-bff2-d17104bb77a6
function setup_MARL_policy_training(game::StateStochasticGame{T, S, A, N, P, F1, F2}, feature_vectors::NTuple{N, V}, update_feature_vectors!::NTuple{N, Function};
	linear_policy_params::NTuple{N, Matrix{T}} = ntuple(i -> initialize_linear_parameters(feature_vectors[i], length(game.agent_actions[i]), zero(T)), N),
	linear_value_params::NTuple{N, Vector{T}} = ntuple(i -> initialize_linear_parameters(feature_vectors[i], zero(T)), N), 
	shared_linear_policy_params::NTuple{N, Matrix{T}} = let 
		params = initialize_linear_parameters(feature_vectors[1], length(game.agent_actions[1]), zero(T))
		ntuple(i -> params, N)
	end,
	shared_linear_value_params::NTuple{N, Vector{T}} = let 
		params = initialize_linear_parameters(feature_vectors[1], zero(T))
		ntuple(i -> params, N)
	end,
	fcann_value_params::Dict = Dict{NamedTuple, NTuple{N, FCANNParams{T}}}(),
	fcann_value_encode_params::Dict = Dict{NamedTuple, NTuple{N, FCANNParams{T}}}(), 
	fcann_policy_params::Dict = Dict{NamedTuple, NTuple{N, FCANNParams{T}}}(), 
	fcann_policy_encode_params::Dict = Dict{NamedTuple, NTuple{N, FCANNParams{T}}}(), 
	fcann_encoder_params::Dict = Dict{NamedTuple, NTuple{N, FCANNParams{T}}}(), 
	fcann_decoder_params::Dict = Dict{NamedTuple, NTuple{N, NTuple{N-1, FCANNParams{T}}}}(), 
	min_reward::T = typemin(T), 
	reward_reducer::Function = +) where {T<:Real, S, A, N, P, F1<:Function, F2<:Function, V}

	function reset_params!(params::NTuple{N, Matrix{T}})
		for i in 1:N
			params[i] .= initialize_linear_parameters(feature_vectors[i], length(game.agent_actions[i]), zero(T))
		end
	end

	function reset_params!(params::NTuple{N, Vector{T}})
		for i in 1:N
			params[i] .= initialize_linear_parameters(feature_vectors[i], zero(T))
		end
	end

	function initialize_params(hidden_layers::Vector{Int64}, reslayers::Integer; reset_params::Bool = false, use_μP::Bool = true, share_value_params::Bool = false, share_policy_params::Bool = false)
		key = (hidden_layers = hidden_layers, reslayers = reslayers, share_value_params = share_value_params, share_policy_params = share_policy_params)
		if !haskey(fcann_value_params, key) || reset_params
			@info "Initializing new parameters with hidden layers = $hidden_layers and reslayers = $reslayers"
			policy_params = if share_policy_params
				let 
					params = initialize_fcann_params(feature_vectors[1], hidden_layers, length(game.agent_actions[1]), reslayers, use_μP)
					ntuple(i -> params, N)
				end	
			else
				ntuple(i -> initialize_fcann_params(feature_vectors[i], hidden_layers, length(game.agent_actions[i]), reslayers, use_μP), N)
			end
			value_params = if share_value_params
				let 
					params = initialize_fcann_value_params(policy_params[1], use_μP)
					ntuple(i -> params, N)
				end	
			else
				ntuple(i -> initialize_fcann_value_params(policy_params[i], use_μP), N)
			end
			fcann_value_params[key] = value_params
			fcann_policy_params[key] = policy_params
		end
		(fcann_policy_params[key], fcann_value_params[key])
	end

	#for each agent, stores a tuple of the number of actions for the other agents in the game
	other_agent_actions = ntuple(N) do agent_index
		other_agent_inds = get_other_agent_inds(N, agent_index)
		ntuple(i -> length(game.agent_actions[other_agent_inds[i]]), N-1)
	end

	#initialize parameters for a version of actor critic learning that attempts to learn an encoded representation of other agent's policies to use as input for the actor and critic functions of each agent.  The input size is the sum of the length of the feature vector with the size of the encoding representation.
	function initialize_params(hidden_layers::Vector{Int64}, encode_layers::Vector{Int64}, code_size::Integer, decode_layers::Vector{Int64}, reslayers::Integer, encoding_reslayers::Integer; reset_params::Bool = false, use_μP::Bool = true)
		key = (hidden_layers = hidden_layers, encode_layers = encode_layers, code_size = code_size, decode_layers = decode_layers, reslayers = reslayers, encoding_reslayers = encoding_reslayers)
		# encode_key = (hidden_layers = encode_layers, code_size = code_size, reslayers = reslayers)
		# decode_key = (hidden_layers = decode_layers, code_size = code_size, reslayers = reslayers)
		if !haskey(fcann_value_encode_params, key) || reset_params
			@info "Initializing new parameters with encoding input of size $code_size, hidden layers = $hidden_layers and reslayers = $reslayers"
			# @info "Initializing policy and value parameters"
			policy_params = ntuple(i -> initialize_fcann_params(length(feature_vectors[i])+code_size, hidden_layers, length(game.agent_actions[i]), reslayers, use_μP), N)
			value_params = ntuple(i -> initialize_fcann_value_params(policy_params[i], use_μP), N)
			fcann_value_encode_params[key] = value_params
			fcann_policy_encode_params[key] = policy_params
			# @info "Initializing encoder and decoder parameters"
			encode_params = ntuple(i -> initialize_fcann_params(feature_vectors[i], encode_layers, code_size, encoding_reslayers, use_μP), N)
			decode_params = ntuple(i -> ntuple(j -> initialize_fcann_params(code_size, decode_layers, other_agent_actions[i][j], encoding_reslayers, use_μP), N-1), N)
			fcann_encoder_params[key] = encode_params
			fcann_decoder_params[key] = decode_params
		end
		(fcann_policy_encode_params[key], fcann_value_encode_params[key], fcann_encoder_params[key], fcann_decoder_params[key])
	end

	function ac_train(γ::T, α_θ::T, α_w::T, nstep::Integer, max_steps::Integer; new_params::Bool = true, share_policy_params::Bool = false, share_value_params::Bool = false, kwargs...)
		value_params = share_value_params ? shared_linear_value_params : linear_value_params
		policy_params = share_policy_params ? shared_linear_policy_params : linear_policy_params
		new_params && reset_params!(value_params)
		new_params && reset_params!(policy_params)
		synchronous_independent_actor_critic_linear(game, γ, max_steps, deepcopy(feature_vectors), update_feature_vectors!; value_params = value_params, policy_params = policy_params, α_θ = α_θ, α_w = α_w, share_policy_params = share_policy_params, share_value_params = share_value_params, nstep = nstep, kwargs...)
	end

	function ac_train(hidden_layers::Vector{Int64}, reslayers::Integer, γ::T, α_θ::T, α_w::T, nstep::Integer, max_steps::Integer; new_params::Bool = true, use_μP::Bool = true, share_policy_params::Bool = false, share_value_params::Bool = false, kwargs...)
		(policy_params, value_params) = initialize_params(hidden_layers, reslayers; reset_params = new_params, use_μP = use_μP, share_policy_params = share_policy_params, share_value_params = share_value_params)
		synchronous_independent_actor_critic_fcann(game, γ, max_steps, deepcopy(feature_vectors), update_feature_vectors!, hidden_layers; reslayers = reslayers, policy_params = policy_params, value_params = value_params, α_θ = α_θ, α_w = α_w, nstep = nstep, share_policy_params = share_policy_params, share_value_params = share_value_params, kwargs...)
	end

	function ac_train(hidden_layers::Vector{Int64}, encoder_layers::Vector{Int64}, code_size::Integer, decoder_layers::Vector{Int64}, reslayers::Integer, γ::T, α_θ::T, α_w::T, α_ψ, nstep::Integer, max_steps::Integer; new_params::Bool = true, use_μP::Bool = true, encoding_reslayers::Integer = 0, kwargs...)
		(policy_params, value_params, encoder_params, decoder_params) = initialize_params(hidden_layers, encoder_layers, code_size, decoder_layers, reslayers, encoding_reslayers; reset_params = new_params, use_μP = use_μP)
		synchronous_independent_actor_critic_fcann(game, γ, max_steps, deepcopy(feature_vectors), update_feature_vectors!, hidden_layers, encoder_layers, code_size, decoder_layers; reslayers = reslayers, policy_params = policy_params, value_params = value_params, encode_params = encoder_params, decode_params = decoder_params, α_θ = α_θ, α_w = α_w, α_ψ = α_ψ, nstep = nstep, kwargs...)
	end

	function train_exhaustive(method::AbstractString, γ::T, α_θ::T, α_w::T, nstep::Integer, trial_steps::Integer; kwargs...)
		f, params = if method == "ac"
			ac_train, (linear_policy_params, linear_value_params)
		else
			error("Unknown policy training method of $method.  Allowed methods are: \"ac\"")
		end

		function train(αs, nstep, trial_steps, new_params; kwargs...)
			f(γ, αs..., nstep, trial_steps; new_params = new_params, kwargs...)
		end

		@info "Starting exhaustive $method linear training with γ = $γ, α_θ = $α_θ, α_w = $α_w with $nstep step returns and $trial_steps steps per trial"
		train_exhaustive(train, params, γ, (α_θ, α_w), nstep, trial_steps; kwargs...)
	end

	function train_exhaustive(method::AbstractString, hidden_layers::Vector{Int64}, reslayers::Integer, γ::T, α_θ::T, α_w::T, nstep::Integer, trial_steps::Integer; use_μP::Bool = true, kwargs...)
		f, params = if method == "ac"
			policy_params, value_params = initialize_params(hidden_layers, reslayers; use_μP = use_μP)
			ac_train, (policy_params, value_params)
		else
			error("Unknown value training method of $method.  Allowed methods are: \"ac\"")
		end

		function train(αs, nstep, trial_steps, new_params; kwargs...)
			f(hidden_layers, reslayers, γ, αs..., nstep, trial_steps; new_params = new_params, use_μP = use_μP, kwargs...)
		end

		@info "Starting exhaustive $method non-linear training with hidden layers: $hidden_layers, reslayers: $reslayers, γ = $γ, α_θ = $α_θ, α_w = $α_w with $nstep step returns and $trial_steps steps per trial"
		train_exhaustive(train, params, γ, (α_θ, α_w), nstep, trial_steps; kwargs...)
	end

	function train_exhaustive(method::AbstractString, hidden_layers::Vector{Int64}, encode_layers::Vector{Int64}, code_size::Integer, decode_layers::Vector{Int64}, reslayers::Integer, γ::T, α_θ::T, α_w::T, α_ψ::T, nstep::Integer, trial_steps::Integer; use_μP::Bool = true, encoding_reslayers::Integer = 0, kwargs...)
		f, params = if method == "ac"
			policy_params, value_params, encoder_params, decoder_params = initialize_params(hidden_layers, encode_layers, code_size, decode_layers, reslayers, encoding_reslayers; use_μP = use_μP)
			ac_train, (policy_params, value_params, encoder_params, decoder_params)
		else
			error("Unknown value training method of $method.  Allowed methods are: \"ac\"")
		end

		function train(αs, nstep, trial_steps, new_params; kwargs...)
			f(hidden_layers, encode_layers, code_size, decode_layers, reslayers, γ, αs..., nstep, trial_steps; new_params = new_params, use_μP = use_μP, kwargs...)
		end

		@info "Starting exhaustive $method non-linear training with hidden layers: $hidden_layers, encoding size: $code_size, reslayers: $reslayers, γ = $γ, α_θ = $α_θ, α_w = $α_w with $nstep step returns and $trial_steps steps per trial"
		train_exhaustive(train, params, γ, (α_θ, α_w, α_ψ), nstep, trial_steps; kwargs...)
	end
	
	function train_exhaustive(train::Function, params::Tuple, γ::T, αs::Tuple, nstep::Integer, trial_steps::Integer; new_params::Bool = false, use_steps::Bool = false, kwargs...)
		ReinforcementLearning.check_bad_params(params) && error("Current parameter values are bad")

		# @info "Extracting current policy functions"
		output1 = train(αs, 0, 0, new_params; kwargs...)
		# @info "Forming reusable kwargs for policy function to run more efficiently"
		π_kwargs = ntuple(i -> output1.form_policy_kwargs[i](), N)
		# @info "Computing baseline reward value by executing policies in environment"
		baseline_reward = evaluate_policy_performance(game, ntuple(i -> s -> output1.policy_sample_actions[i](s; π_kwargs[i]...), N), trial_steps; use_steps = use_steps, min_reward = min_reward, reducer = reward_reducer)
		reward1 = baseline_reward
		trial = 0

		@info "Baseline episode reward is $reward1, beginning first trial"
		backup_params = deepcopy(params)
		output2 = train(αs, nstep, trial_steps, false; kwargs...)
		reward2 = check_batch_reward_progress(output2; use_steps = use_steps, min_reward = min_reward, reducer = reward_reducer)

		if ReinforcementLearning.check_bad_params(params)
			@info "First trial resulted in bad parameter values"
			for i in eachindex(params)
				copy!(params[i], backup_params[i])
			end
			return (;output1..., performance = reward1)
		elseif reward2 ≤ reward1
			@info "First trial performance of $reward2 failed to improve reward"
			for i in eachindex(params)
				copy!(params[i], backup_params[i])
			end
			return (;output1..., performance = reward1)
		end

		avg_step_rewards = output2.avg_step_rewards
		while (reward2 > reward1) && !ReinforcementLearning.check_bad_params(params)
			trial += 1
			@info "On trial $trial, reward improved from $reward1 to $reward2"
			output1 = output2
			reward1 = reward2
			# @info "Backing up parameter values"
			for i in eachindex(params)
				copy!(backup_params[i], params[i])
			end
			avg_step_rewards = ntuple(i -> vcat(avg_step_rewards[i], output1.avg_step_rewards[i]), N)

			output2 = train(αs, nstep, trial_steps, false; kwargs...)
			reward2 = check_batch_reward_progress(output2; use_steps = use_steps, min_reward = min_reward, reducer = reward_reducer)
		end

		if ReinforcementLearning.check_bad_params(params)
			@info "Final trial resulted in bad parameter values"
		else
			@info "Final trial performance of $reward2 failed to improve reward"
		end

		@info "Performance after $trial trials improved from $baseline_reward to $reward1"

		
		for i in eachindex(params)
			copy!(params[i], backup_params[i])
		end
		return (;output1..., avg_step_rewards = avg_step_rewards, performance = reward1)
	end

	function train_rate_decay(train_args::Tuple, γ::T, α_inits::Tuple, nstep::Integer, trial_steps::Integer; kwargs...)
		output1 = train_exhaustive(train_args..., γ, α_inits..., nstep, trial_steps; kwargs...)
		avg_step_rewards = output1.avg_step_rewards

		αs = ntuple(i -> α_inits[i]/2, length(α_inits))
		
		@info "Reducing learning rates to: $αs for next set of trials"
		output2 = train_exhaustive(train_args..., γ, αs..., nstep, trial_steps; kwargs..., new_params = false)

		if output2.performance ≤ output1.performance
			@info "Second round performance of $(output2.performance) failed to improve reward"
			@info "Completed rate decay training after 1 round with performance of $(output1.performance)"
		end

		round = 2
		while output2.performance > output1.performance
			round += 1
			αs = ntuple(i -> αs[i]/2, length(α_inits))
			output1 = output2
			avg_step_rewards = ntuple(i -> vcat(avg_step_rewards[i], output1.avg_step_rewards[i]), N)
			@info "On round $round, reducing learning rates to: $αs for next set of trials"
			output2 = train_exhaustive(train_args..., γ, αs..., nstep, trial_steps; kwargs..., new_params = false)
		end
		@info "Completed rate decay training after $round rounds with performance $(output1.performance)."
		return (;output1..., avg_step_rewards = avg_step_rewards)
	end

	function train_rate_decay(method::AbstractString, γ::T, α_θ_init::T, α_w_init::T, nstep::Integer, trial_steps::Integer; kwargs...)
		@info "Starting rate decay $method linear training with initial learning rates of α_θ: $α_θ_init and α_w: $α_w_init"
		train_rate_decay((method,), γ, (α_θ_init, α_w_init), nstep, trial_steps; kwargs...)
	end

	function train_rate_decay(method::AbstractString, hidden_layers::Vector{Int64}, reslayers::Integer, γ::T, α_θ_init::T, α_w_init::T, nstep::Integer, trial_steps::Integer; kwargs...)
		@info "Starting rate decay $method nonlinear training with initial learning rates of α_θ: $α_θ_init and α_w: $α_w_init"
		train_rate_decay((method, hidden_layers, reslayers), γ, (α_θ_init, α_w_init), nstep, trial_steps; kwargs...)
	end

	function train_rate_decay(method::AbstractString, hidden_layers::Vector{Int64}, encode_layers::Vector{Int64}, code_size::Integer, decode_layers::Vector{Int64}, reslayers::Integer, γ::T, α_θ_init::T, α_w_init::T, α_ψ_init::T, nstep::Integer, trial_steps::Integer; kwargs...)
		@info "Starting rate decay $method nonlinear training with policy encoding and initial learning rates of α_θ: $α_θ_init, α_w: $α_w_init, and α_ψ: $α_ψ_init"
		train_rate_decay((method, hidden_layers, encode_layers, code_size, decode_layers, reslayers), γ, (α_θ_init, α_w_init, α_ψ_init), nstep, trial_steps; kwargs...)
	end

	return (train_ac = ac_train, train_exhaustive = train_exhaustive, train_rate_decay = train_rate_decay, linear_value_params = linear_value_params, linear_policy_params = linear_policy_params, fcann_value_params = fcann_value_params, fcann_policy_params = fcann_policy_params)
end	

# ╔═╡ fdc7362f-d8d1-412f-934a-ebfecb06035e
setup_MARL_policy_training(game::StateStochasticGame{T, S, A, N, P, F1, F2}, feature_vector::V, update_feature_vector!::Function; kwargs...) where {T<:Real, S, A, N, P, F1<:Function, F2<:Function, V} = setup_MARL_policy_training(game, convert_to_ntuple(feature_vector, N), convert_to_ntuple(update_feature_vector!, N); kwargs...)

# ╔═╡ ecc2d3c6-2d68-4082-90b8-a5ee11105dc9
md"""
### Policy Tests
"""

# ╔═╡ 7787dcaa-6270-45eb-8827-ab0b8f9261ef
# ╠═╡ disabled = true
#=╠═╡
const game_1_sparse_policy_test = setup_MARL_policy_training(game_1_state, game_1_feature_sparse, update_game_1_feature!; min_reward = 0f0)
  ╠═╡ =#

# ╔═╡ d9931e52-c651-41d2-a8b5-9dcdf454fa08
#=╠═╡
const game_1_dense_policy_test = setup_MARL_policy_training(game_1_state, game_1_feature_dense, update_game_1_feature!; min_reward = 0f0)
  ╠═╡ =#

# ╔═╡ b74f7df0-c7cf-4ea1-bc6e-a0d961e952f1
# ╠═╡ disabled = true
#=╠═╡
const game_1_binary_policy_test = setup_MARL_policy_training(game_1_state, game_1_feature_binary, update_game_1_feature!; min_reward = 0f0)
  ╠═╡ =#

# ╔═╡ d8f5a14a-fb1f-4e60-a428-afdf584e08c8
md"""
#### State Aggregation Test
"""

# ╔═╡ 1162fc5b-bd63-4913-851f-9a8e9dbd983e
#=╠═╡
const game_1_sparse_linear_ac_output = game_1_sparse_policy_test.train_rate_decay("ac", 0.99f0, 0.5f0, 0.1f0, 10, 10_000; num_env = 4, use_steps = true, new_params = true)
  ╠═╡ =#

# ╔═╡ f5f13a65-4154-43f0-a42c-23bd5c9d7877
#=╠═╡
runepisode(game_1_state; πs = game_1_sparse_linear_ac_output.policy_sample_actions, max_steps = 1_000)
  ╠═╡ =#

# ╔═╡ d7c96172-f5e5-44f4-ab15-1e4689708301
#=╠═╡
const game_1_sparse_nonlinear_ac_output = game_1_sparse_policy_test.train_rate_decay("ac", [2, 2], 1, 0.99f0, 0.1f0, 0.1f0, 10, 10_000; num_env = 4, use_steps = true)
  ╠═╡ =#

# ╔═╡ 88b6cb51-289a-4630-aaed-5c4da290fdd0
#=╠═╡
runepisode(game_1_state; πs = game_1_sparse_nonlinear_ac_output.policy_sample_actions, max_steps = 1_000)
  ╠═╡ =#

# ╔═╡ ac78e8a0-b9bb-426c-8bb7-6ff77b2a39a0
md"""
#### Dense Vector Test
"""

# ╔═╡ 04a57228-9d48-401e-84bb-a88214ad91b9
#=╠═╡
const game_1_dense_linear_ac_output = game_1_dense_policy_test.train_rate_decay("ac", 0.99f0, 0.5f0, 0.5f0, 10, 10_000; num_env = 4, use_steps = true)
  ╠═╡ =#

# ╔═╡ c48e7f9b-4b53-4d69-9f77-fccfab64ca53
#=╠═╡
runepisode(game_1_state; πs = game_1_dense_linear_ac_output.policy_sample_actions, max_steps = 1_000)
  ╠═╡ =#

# ╔═╡ 2066ff6c-2db0-4a84-b34a-72f351a373f4
#=╠═╡
const game_1_dense_nonlinear_ac_output = game_1_dense_policy_test.train_rate_decay("ac", [4, 4], 1, 0.99f0, 0.01f0, 0.01f0, 10, 100_000; num_env = 8, use_steps = true)
  ╠═╡ =#

# ╔═╡ d6cf5f0f-27ce-4762-8562-ba77bfa1e389
#=╠═╡
runepisode(game_1_state; πs = game_1_dense_nonlinear_ac_output.policy_sample_actions, max_steps = 1_000)
  ╠═╡ =#

# ╔═╡ 26d3c66f-76ce-4b62-9074-f3cec5a05754
#=╠═╡
game_1_dense_policy_test.train_rate_decay("ac", [16, 16], [4, 4], 4, [4, 4, 4], 1, 0.99f0, 0.0001f0, 0.0001f0, 0.0001f0, 10, 10_000_000; num_env = 4, use_steps = true, new_params = true)
  ╠═╡ =#

# ╔═╡ f6294409-274f-4972-9111-7f4b1a366393
#=╠═╡
mean(rand(10, 100), dims = 1)
  ╠═╡ =#

# ╔═╡ 1cfd36ca-fc05-48cd-afa1-4bb12e0cc400
md"""
#### Binary Feature Test
"""

# ╔═╡ 59138d49-34f2-41a5-8733-afc78296505d
#=╠═╡
const game_1_binary_linear_ac_output = game_1_binary_policy_test.train_rate_decay("ac", 0.99f0, 0.5f0, 0.1f0, 10, 10_000; num_env = 4, use_steps = true, new_params = true)
  ╠═╡ =#

# ╔═╡ 5788a91e-0056-4e24-93c0-fcfcf2202d32
#=╠═╡
runepisode(game_1_state; πs = game_1_binary_linear_ac_output.policy_sample_actions, max_steps = 1_000)
  ╠═╡ =#

# ╔═╡ 6035c588-8a93-4e4d-8758-db124a6a61b8
#=╠═╡
const game_1_binary_nonlinear_ac_output = game_1_binary_policy_test.train_rate_decay("ac", [2, 2], 1, 0.99f0, 0.5f0, 0.2f0, 10, 10_000; num_env = 4, use_steps = true)
  ╠═╡ =#

# ╔═╡ 69c2e85c-fdc8-483d-83c3-9ad05508c748
#=╠═╡
runepisode(game_1_state; πs = game_1_binary_nonlinear_ac_output.policy_sample_actions, max_steps = 1_000)
  ╠═╡ =#

# ╔═╡ 82585df6-3468-11f1-a729-2128b8837878
md"""
# Dependencies
"""

# ╔═╡ 6b97d51a-f74a-4e87-ae52-06609e556929
# ╠═╡ skip_as_script = true
#=╠═╡
BLAS.set_num_threads(4)
  ╠═╡ =#

# ╔═╡ 9a0105c5-5dca-4727-bb46-9accaee954c8
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
BenchmarkTools = "~1.7.0"
CSV = "~0.10.16"
DataFrames = "~1.8.1"
DataStructures = "~0.19.4"
HiGHS = "~1.18.2"
HypertextLiteral = "~0.9.5"
JuMP = "~1.29.4"
LaTeXStrings = "~1.4.0"
PlutoDevMacros = "~0.9.2"
PlutoPlotly = "~0.6.5"
PlutoProfile = "~0.3.2"
PlutoUI = "~0.7.80"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.12.6"
manifest_format = "2.0"
project_hash = "3e6b3f7c67b2070506c883a029d991a7513458fc"

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
git-tree-sha1 = "6876e30dc02dc69f0613cb6ece242144f2ca9e56"
uuid = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
version = "1.7.0"

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
git-tree-sha1 = "d8928e9169ff76c6281f39a659f9bca3a573f24c"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.8.1"

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
deps = ["Dates", "Logging", "Parsers", "PrecompileTools", "StructUtils", "UUIDs", "Unicode"]
git-tree-sha1 = "67c6f1f085cb2671c93fe34244c9cccde30f7a26"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "1.5.0"

    [deps.JSON.extensions]
    JSONArrowExt = ["ArrowTypes"]

    [deps.JSON.weakdeps]
    ArrowTypes = "31f734f8-188a-4ce0-8406-c8a06bd891cd"

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
git-tree-sha1 = "85dca20e9f02d05e0642ee04b7e374157ee2003a"
uuid = "ee419aa8-929d-45cd-acf6-76bd043cd7ba"
version = "0.3.2"

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
deps = ["Base64", "JSON", "Pkg", "Profile", "REPL"]
git-tree-sha1 = "990016fb1508b0726a70039f39569720d054c78d"
uuid = "efd6af41-a80b-495e-886c-e51b0c7d77a3"
version = "0.1.7"

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

[[deps.StructUtils]]
deps = ["Dates", "UUIDs"]
git-tree-sha1 = "fa95b3b097bcef5845c142ea2e085f1b2591e92c"
uuid = "ec057cc2-7a8d-4b58-b3b3-92acb9f63b42"
version = "2.7.1"

    [deps.StructUtils.extensions]
    StructUtilsMeasurementsExt = ["Measurements"]
    StructUtilsStaticArraysCoreExt = ["StaticArraysCore"]
    StructUtilsTablesExt = ["Tables"]

    [deps.StructUtils.weakdeps]
    Measurements = "eff96d63-e80a-5855-80a2-b1b0885c5ab7"
    StaticArraysCore = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
    Tables = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"

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
# ╠═cbd924a8-0889-477d-afe0-f86107bc6f4f
# ╠═709be4b1-42e1-4ed4-a5a7-61f646eff4c7
# ╠═1530f486-3424-47ff-bcc4-daf2184bfa52
# ╠═ad7a5093-c8d4-4a2a-ae3e-b5da409cf1da
# ╠═7aa80d07-02e2-4b21-a712-27302d3f974f
# ╟─8049e06d-286f-45c2-8f4d-090261317168
# ╠═ef90c554-058e-4c30-8e1c-3971bcb4ec83
# ╠═6b87627e-c9ce-4a25-9a99-953e6220ec49
# ╠═094391df-77aa-4dc1-b01f-ba2729db25b1
# ╠═f3c6108e-33e5-48a7-a27d-4d765cb60b1a
# ╠═170747b0-a370-48cf-925b-27cd3638f04e
# ╟─159c9d65-897c-4bb3-83ed-78b48d901593
# ╟─fa657909-32eb-41ab-a9f2-321b24e9c7c3
# ╟─b3df9dba-1c3f-482a-bacb-4591b9cd08ea
# ╠═f531a3c9-78a0-446f-a3a1-1170f400a949
# ╠═ee78a3c4-a941-47dd-bd0e-6f4886b8a65a
# ╠═1ba4cd28-f2fd-4e50-b9a6-22a3e48c81c9
# ╠═52afd5fb-f426-4cf5-801e-e7c3d23a66e7
# ╟─d996a1de-1dc6-43ac-bb6c-9948e0ac4e5d
# ╠═e4a1c364-64ad-40df-a10b-9939fee69a21
# ╠═62a2af5e-11c4-4e06-a4b7-043a9daef999
# ╠═fae0d63f-1cba-4dd7-a413-dc7d8b5c8432
# ╠═880e73ca-976a-409d-ac65-380c4f56fe58
# ╠═fb823f3f-adbc-4496-aefe-34a6d5f3bf1a
# ╠═dc7910ff-4906-4384-a691-e8ce5150750b
# ╟─329cf3dd-a981-4f28-949e-2f72a1b37548
# ╠═1aff84c6-a4ff-439f-940d-fbdc0b0276d3
# ╠═948c4b2c-32be-43ff-8584-dc4312c6e127
# ╠═c7f4f8a6-19e1-420e-9dd5-fe348837b67d
# ╠═7ba601ae-fd52-4063-9fa3-6e8f43f59c9c
# ╠═fe69bcdb-9ece-423a-b9b2-a31ee259c0be
# ╠═a16a7f91-8c47-4600-9b77-a6bb0e49aff3
# ╠═40b5763d-936c-41ad-9de6-62dc3f786ca2
# ╟─6b5a56b0-3252-4f7a-a829-dfdbc3f0147b
# ╠═df7dd868-a3d5-4d2b-afb9-18e871b684a3
# ╟─38b4bf3b-0835-4425-9b26-6ee3b70cf2b0
# ╠═1bbe0183-c6b6-45c2-8c4d-f2034391d8db
# ╠═490a29fc-c1bc-4935-a8f9-780002ee3563
# ╠═6513bba4-6bea-4a5c-b372-97b01f147849
# ╠═22d9860b-cf49-43a3-b935-65583952f56c
# ╠═512614d5-d07e-446e-9717-72a299311586
# ╟─fdf3d679-d767-44c0-b841-20872cc50c92
# ╠═85110098-63e8-48b5-98c6-2758651b8a24
# ╠═7925d5be-4eba-4e80-81a3-c0cfc2fde561
# ╟─9a93c45f-8fff-4517-a1e6-6f0d47963e75
# ╠═4af81e90-7a8d-4fd7-9fdc-3ae5cf52a581
# ╟─a5169ac3-e089-41c9-ad61-b75292fd1b86
# ╠═88cae643-9a2d-4980-8934-db0db6e6cd55
# ╠═ad124b03-31de-4ab6-9f00-b422e716b76c
# ╟─8dab0aef-c550-4a2d-babb-934082488c2f
# ╠═8985bdc3-4bc9-4ff6-ad20-671fff1ff1a2
# ╠═8727f317-a2bc-44e1-bece-f3171cc7878c
# ╠═53154ff1-5296-4b62-b4ba-76631f739440
# ╠═481e9182-9c2c-45a5-bfeb-48943792ed02
# ╠═023cfc09-a0be-4e2d-ba90-79b30314bffa
# ╠═fdcf63d5-17d2-4a29-b174-43a20b06897a
# ╠═ca5b24fa-d401-40a8-be9f-0eb0282832b9
# ╠═58bc7bd1-c44a-401b-b69a-cb509a0edf8b
# ╟─88bed632-6696-4675-a10f-770687f14b14
# ╠═a6287074-c72a-47f6-bff2-d17104bb77a6
# ╠═fdc7362f-d8d1-412f-934a-ebfecb06035e
# ╟─ecc2d3c6-2d68-4082-90b8-a5ee11105dc9
# ╠═7787dcaa-6270-45eb-8827-ab0b8f9261ef
# ╠═d9931e52-c651-41d2-a8b5-9dcdf454fa08
# ╠═b74f7df0-c7cf-4ea1-bc6e-a0d961e952f1
# ╟─d8f5a14a-fb1f-4e60-a428-afdf584e08c8
# ╠═1162fc5b-bd63-4913-851f-9a8e9dbd983e
# ╠═f5f13a65-4154-43f0-a42c-23bd5c9d7877
# ╠═d7c96172-f5e5-44f4-ab15-1e4689708301
# ╠═88b6cb51-289a-4630-aaed-5c4da290fdd0
# ╟─ac78e8a0-b9bb-426c-8bb7-6ff77b2a39a0
# ╠═04a57228-9d48-401e-84bb-a88214ad91b9
# ╠═c48e7f9b-4b53-4d69-9f77-fccfab64ca53
# ╠═2066ff6c-2db0-4a84-b34a-72f351a373f4
# ╠═d6cf5f0f-27ce-4762-8562-ba77bfa1e389
# ╠═26d3c66f-76ce-4b62-9074-f3cec5a05754
# ╠═f6294409-274f-4972-9111-7f4b1a366393
# ╟─1cfd36ca-fc05-48cd-afa1-4bb12e0cc400
# ╠═59138d49-34f2-41a5-8733-afc78296505d
# ╠═5788a91e-0056-4e24-93c0-fcfcf2202d32
# ╠═6035c588-8a93-4e4d-8758-db124a6a61b8
# ╠═69c2e85c-fdc8-483d-83c3-9ad05508c748
# ╟─82585df6-3468-11f1-a729-2128b8837878
# ╠═c098aeb7-c1f8-4c0a-94a3-28df60646895
# ╠═2b9161b1-c68d-4182-8331-f6d3e89941de
# ╠═3fa2f499-bd4c-4f75-92c1-6523759a8b27
# ╠═6b97d51a-f74a-4e87-ae52-06609e556929
# ╠═40210d93-8476-4980-bd3d-31dcd0267ca9
# ╠═9a0105c5-5dca-4727-bb46-9accaee954c8
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
