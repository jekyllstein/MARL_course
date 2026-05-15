### A Pluto.jl notebook ###
# v0.20.24

using Markdown
using InteractiveUtils

# ╔═╡ 6f7fbd5f-f907-491e-8257-22aee4374529
using PlutoDevMacros

# ╔═╡ dbcef8d5-c6b3-489f-9776-8d1b356b8543
using DataFrames, CSV, JuMP, HiGHS, DataStructures, Dates

# ╔═╡ ea36b694-2293-4588-b51f-47d37d36d0a5
begin
	using PlutoUI, PlutoPlotly, PlutoProfile, BenchmarkTools, LaTeXStrings, HypertextLiteral
	TableOfContents(depth = 4)
end

# ╔═╡ b8ecf71c-a2da-4c48-ba52-f18de34f67c0
@only_in_nb begin
	@frompackage @raw_str(joinpath(@__DIR__, "..", "RL_Module")) import *
	include(joinpath(@__DIR__, "multi_agent_types.jl"))
	include(joinpath(@__DIR__, "joint_action_learning.jl"))
	include(joinpath(@__DIR__, "independent_learning.jl"))
	include(joinpath(@__DIR__, "agent_modeling.jl"))
	include(joinpath(@__DIR__, "learning_utilities.jl"))
end

# ╔═╡ f5429487-d444-46a2-8d93-8e58afb337d3
md"""
# Environment Module

The "Landmark" game consists of a gridworld with 1 or more landmarks positioned randomly at the start of an episode and matching number of agents.  In order for the episode to end successfully, exactly one agent must move to the position of each landmark.  In other words, the agents must distribute themselves to cover all landmarks.  The step function also has a small probability to terminate the episode on every step.  If that occurs on a step that does not result in all landmarks being "covered", then the episode will end unsuccessfully with no agent receiving any non-zero reward.  A successful episode will award a reward of 1 to each agent, making this a common reward game.

Agents are free to move to any position in the grid without colliding with other agents.  They can choose to move in one of the four cardinal directions as well as to stay in place.  

By adjusting how much information is available to all of the agents involved, we can use this environment to test success and failure cases of experience sharing and homogenous agents.
"""

# ╔═╡ e8e0a91a-a316-4868-b562-8aebd8998e9d
module LandmarkGame
	import ..RL_Module
	import ..StateGameTransitionDeterministic
	import ..StateStochasticGame
	import ..makelookup

	abstract type Action end
	struct Up <: Action end
	struct Down <: Action end
	struct Left <: Action end
	struct Right <: Action end
	struct Noop <: Action end

	const Position = Tuple{Int64, Int64}

	const action_list = [Up(), Down(), Left(), Right(), Noop()]
	const action_tuple = (Up(), Down(), Left(), Right(), Noop())
	const action_index = makelookup(action_list)

	#a state stores the positions of agents and landmarks with an equal number of each.  It also stores whether the state is terminal for cases when the reset is triggered or all landmarks have been covered.  the type parameters are N for the number of agents/landmarks, X/Y for the size of the grid in the X and Y position
	struct State{N, X, Y}
		agent_positions::NTuple{N, Position}
		landmark_positions::NTuple{N, Position}
		terminated::Bool
	end

	State(agent_positions::NTuple{N, Position}, landmark_positions::NTuple{N, Position}, width::Integer, height::Integer) where N = State{N, width, height}(agent_positions, landmark_positions, false)

	#check if a landmark is covered
	is_covered(landmark::Position, agent_positions::NTuple{N, Position}) where N = any(p -> isequal(p, landmark), agent_positions)

	#check if all landmarks are covered
	all_covered(landmark_positions::NTuple{N, Position}, agent_positions::NTuple{N, Position}) where {N} = all(l -> is_covered(l, agent_positions), landmark_positions)

	#rules to update an agent's position in the grid
	move(p::Position, ::Noop) = p
	move(p::Position, ::Up) = (p[1], p[2]+1)
	move(p::Position, ::Right) = (p[1]+1, p[2])
	move(p::Position, ::Down) = (p[1], p[2]-1)
	move(p::Position, ::Left) = (p[1]-1, p[2])

	function initialize_state(width::Integer, height::Integer, N::Integer)
		grid_positions = Set((x, y) for x in 1:width for y in 1:height)
		p1 = rand(grid_positions)
		landmark_positions = [p1]
		allowed_positions = setdiff(grid_positions, p1)
		for i in 2:N
			isempty(allowed_positions) && error("Cannot place landmark $i of $N")
			p = rand(allowed_positions)
			push!(landmark_positions, p)
			setdiff!(allowed_positions, [p])
		end

		isempty(allowed_positions) && error("Cannot form state with no available positions for agents")

		agent_positions = [rand(allowed_positions)]
		for i in 2:N
			setdiff!(allowed_positions, agent_positions[[i-1]])
			isempty(allowed_positions) && error("Cannot place agent $i of $N")
			push!(agent_positions, rand(allowed_positions))
		end

		State(NTuple{N, Position}(agent_positions), NTuple{N, Position}(landmark_positions), width, height)
	end

	isterm(s::State{N, X, Y}) where {N, X, Y} = s.terminated

	function step(s::State{N, X, Y}, a::NTuple{N, I}, reset_chance::T) where {N, X, Y, T<:Real, I<:Integer}
		#check if state is already terminal
		isterm(s) && return (ntuple(i -> zero(T), N), s)

		#move agents
		new_positions = ntuple(N) do i
			p′ = move(s.agent_positions[i], action_tuple[a[i]])
			x = p′[1]
			y = p′[2]
			if (x > X) || (y > Y) || (x < 1) || (y < 1)
				s.agent_positions[i]
			else
				p′
			end
		end

		success = all_covered(s.landmark_positions, new_positions)
		rewards = ntuple(i -> T(success), N)

		success && return (rewards, State{N, X, Y}(new_positions, s.landmark_positions, success))
		term = (rand() < reset_chance)
		return (rewards, State{N, X, Y}(new_positions, s.landmark_positions, term))
	end

	function make_environment(;width::Integer = 8, height::Integer = 8, N::Integer = 3, reset_chance::T = 0.01f0) where T<:Real
		init_state() = initialize_state(width, height, N)
		ptf = StateGameTransitionDeterministic((s, a) -> step(s, a, reset_chance), init_state(), N)
		StateStochasticGame(ntuple(Returns(action_list), N), ptf, init_state, isterm)
	end
		
	
end

# ╔═╡ e1a9c4a1-7131-4ad0-8e7b-b2fa50f7a5ba
# ╠═╡ skip_as_script = true
#=╠═╡
const landmark_883 = LandmarkGame.make_environment()
  ╠═╡ =#

# ╔═╡ 6bc13bad-aac4-4dc5-bb52-d10edbc67fd1
const landmark_883_plot = LandmarkGame.make_environment(;reset_chance = 0f0)

# ╔═╡ 37654c57-b40b-4041-92e0-1f4dda839640
#=╠═╡
landmark_883.initialize_state()
  ╠═╡ =#

# ╔═╡ 0d1de58c-0e81-485c-b1a2-15ad82f8e92e
#=╠═╡
test_episode = runepisode(landmark_883)
  ╠═╡ =#

# ╔═╡ dabd9ce8-1e2a-4e07-9d47-242d22c743cb
#=╠═╡
@bind test_episode_step Slider(1:test_episode[5]; show_value = true)
  ╠═╡ =#

# ╔═╡ 0540b6f3-f0bf-45b2-b1f5-56f1b8dd2874
md"""
# Feature Vector Construction
"""

# ╔═╡ 59fd75a5-827a-4d11-92b9-508847969e17
md"""
## Absolute Position Information
"""

# ╔═╡ 7dfae4b3-f333-4038-ac16-97181142c22a
md"""
## Relative Position Information
"""

# ╔═╡ 81021156-c74d-470a-b398-308ceab92435
function make_landmark_relative_feature(width::Integer, height::Integer, n::Integer; use_agent_positions::Bool = true)
	num_features = n * (width - 1 + height - 1) #relative positions features to every landmark
	if use_agent_positions
		num_features += (n-1) * (width - 1 + height - 1) #relative position features to every other agent
	end

	#note that relative positions for x and y mean that the maximum distance away anything can be is width-1 or height-1

	#initialize feature vector 
	feature_vector = zeros(Float32, num_features)

	function update_relative_landmark_positions!(feature_vector::Vector{T}, s::LandmarkGame.State{N, X, Y}, agent_index::Integer) where {X, Y, T<:Real, N}
		(x0, y0) = s.agent_positions[agent_index]
		for landmark_index in 1:N
			base_ind = (landmark_index - 1) * (X-1 + Y-1)
			(x, y) = s.landmark_positions[landmark_index]
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
		return feature_vector
	end

	#gets the feature vector for the X and Y position of all of the other agents relative to the agent index
	function update_relative_agent_positions!(feature_vector::Vector{T}, s::LandmarkGame.State{N, X, Y}, agent_index::Integer) where {X, Y, T<:Real, N}
		!use_agent_positions && return feature_vector
		(x0, y0) = s.agent_positions[agent_index]
		other_agent_inds = get_other_agent_inds(N, agent_index)
		landmark_base_ind = N*(width - 1 + height - 1)
		for (i, other_agent_index) in enumerate(other_agent_inds)
			base_ind = (i - 1)*(X-1+Y-1)
			(x, y) = s.agent_positions[other_agent_index]
			d_x = x - x0
			ind = abs(d_x)
			if !iszero(ind)
				feature_vector[landmark_base_ind + base_ind + ind] = sign(d_x)
			end
			d_y = y - y0
			ind = abs(d_y)
			if !iszero(ind)
				feature_vector[landmark_base_ind + base_ind + X-1 + ind] = sign(d_y)
			end
		end
		return feature_vector
	end

	function update_feature_vector!(x::Vector{T}, s::LandmarkGame.State{N, X, Y}, agent_index::Integer) where {T<:Real, X, Y, N}
		x .= zero(T)
		update_relative_landmark_positions!(x, s, agent_index)
		update_relative_agent_positions!(x, s, agent_index)
	end

	feature_vectors = ntuple(i -> copy(feature_vector), n)
	update_feature_vectors! = ntuple(i -> (x, s) -> update_feature_vector!(x, s, i), n)

	return (feature_vectors, update_feature_vectors!)
end

# ╔═╡ 1bec5557-ce3d-49b4-b58f-459934b406b2
md"""
# Performance Testing
"""

# ╔═╡ 27569545-046d-40b5-9a63-11520a0e792d
function setup_landmark_features(s::LandmarkGame.State{N, X, Y}, use_relative::Bool, use_agent_positions::Bool) where {N, X, Y}
	f = use_relative ? make_landmark_relative_feature : make_landmark_absolute_feature
	f(X, Y, N; use_agent_positions = use_agent_positions)
end

# ╔═╡ a5b02994-1718-4d0d-8e6a-66b82523b43b
function setup_landmark_training(game::StateStochasticGame, use_relative::Bool, use_agent_positions::Bool)
	s0 = game.initialize_state()
	(feature_vectors, update_feature_vectors!) = setup_landmark_features(s0, use_relative, use_agent_positions)
	policy_setup = setup_MARL_policy_training(game, feature_vectors, update_feature_vectors!; min_reward = 0f0)
	value_setup = setup_MARL_value_training(game, feature_vectors, update_feature_vectors!; min_reward = 0f0)
	(policy_setup = policy_setup, value_setup = value_setup)
end

# ╔═╡ 8f88a834-76b6-4bcf-8ac6-d9b50bfd6839
function evaluate_policy_performance(game::StateStochasticGame; num_trials = 1_000, kwargs...)
	f(i) = runepisode(game; kwargs...)[5]
	1:num_trials |> Map(i -> f(i)) |> tcollect |> summarystats
end

# ╔═╡ 36c6e60a-b613-40f6-9430-8a7b25871a1f
function extract_threadsafe_joint_policies(game::StateStochasticGame{T, S, A, N, P, F1, F2}, output::NamedTuple; ϵ::T = zero(T)) where {T<:Real, S, A, N, P, F1, F2}
	value_functions = output.value_functions
	kws = output.form_kwargs
	if iszero(ϵ)
		ntuple(N) do i
			f(s; kwargs...) = value_functions[i](s; kwargs..., kws[i]()...).maximizing_action
		end
	else
		ntuple(N) do i
			f(s; kwargs...) = rand() ≤ ϵ ? rand(eachindex(game.agent_actions[i])) : value_functions[i](s; kwargs..., kws[i]()...).maximizing_action
		end
	end
end

# ╔═╡ 47958e9f-e2dc-423a-81be-9b01821e5408
evaluate_policy_performance(landmark_883_plot; max_steps = 1_000_000)

# ╔═╡ 6f4d6869-a771-4061-877d-d37174801139
#=╠═╡
evaluate_policy_performance(landmark_883)
  ╠═╡ =#

# ╔═╡ ace6eac9-2cd9-45f2-a846-a89706d4186a
md"""
## Blind to Other Agents
"""

# ╔═╡ 4d2a941b-0a6e-4142-9f03-f96bc673b462
#=╠═╡
const landmark_883_setup_relative_noagentpos = setup_landmark_training(landmark_883, true, false)
  ╠═╡ =#

# ╔═╡ 31b401e2-2048-4ccd-9d2d-81aaa54e5ba1
md"""
### Policy Training
"""

# ╔═╡ ef0b2bb9-5c02-4fb5-a54a-d455ac510239
#=╠═╡
const landmark_883_relative_noagentpos_policy_output = landmark_883_setup_relative_noagentpos.policy_setup.train_rate_decay("ac", [64, 64], 1, 0.99f0, 5f-1, 1f-1, 10, 1_000_000, new_params = true)
  ╠═╡ =#

# ╔═╡ 14fca46c-ad50-4d9c-85e1-48218715b54e
md"""
The agents must develop distinct behavior to avoid inefficiently going for the same landmark.  However since each episode randomly positions the agents, exchanging their policies will not change the expected outcome of the game.  This property is due to the inability for agents to see the position of their companions.  If agents could see all of the available positions, then there would be no need to develop distinct behavior because agents could coordinate by only pursuing landmarks they are closest to (they could also develop some tiebreaking criteria for equal distance based on some XY position indicator).  Below is an example of the same game but where agents can see all positional information which should allow for higher scoring performance.
"""

# ╔═╡ 3a704735-a233-40b3-805e-9496150624f1
#=╠═╡
const landmark_883_relative_noagentpos_policy_encode_output = landmark_883_setup_relative_noagentpos.policy_setup.train_rate_decay("ac", [64, 64], [64, 64], 64, [64, 64, 64], 1, 0.99f0, 5f-1, 1f-1, 1f-1, 10, 100_000; new_params = true)
  ╠═╡ =#

# ╔═╡ e427fdb5-7888-4083-ba37-5f8419b79382
#=╠═╡
const landmark_883_relative_noagentpos_policy_shared_output = landmark_883_setup_relative_noagentpos.policy_setup.train_rate_decay("ac", [64, 64], 1, 0.99f0, 5f-1, 1f-1, 10, 1_000_000, new_params = true, share_policy_params = true, share_value_params = true)
  ╠═╡ =#

# ╔═╡ 47b17860-93b4-464c-b47d-9d72a25106a4
md"""
#### Learning Curves
"""

# ╔═╡ 55c87e94-490d-4514-af3e-9834f43338bc
md"""
#### Average Performance
"""

# ╔═╡ 978d5716-4fc2-49b6-ba0d-d3e4da4bd6da
#=╠═╡
NamedTuple(Symbol(name) => evaluate_policy_performance(landmark_883_plot; πs = output.policy_sample_actions, max_steps = 10_000) for (name, output) in [("Vanilla Actor Critic", landmark_883_relative_noagentpos_policy_output), ("AC with Policy Representations", landmark_883_relative_noagentpos_policy_encode_output), ("AC with Homogeneous Agents", landmark_883_relative_noagentpos_policy_shared_output)])
  ╠═╡ =#

# ╔═╡ 2993ddf1-2116-49f6-a8f3-b95c9b5e2ce5
md"""
#### Episode Visualization
"""

# ╔═╡ 549acb9e-6cb5-4264-8fe0-52617849e650
#=╠═╡
landmark_883_ac_comp_ep = let
	s0 = landmark_883.initialize_state()
	ep1 = runepisode(landmark_883_plot; πs = landmark_883_relative_noagentpos_policy_output.policy_sample_actions, max_steps = 1_000, s0 = s0)
	ep2 = runepisode(landmark_883_plot; πs = landmark_883_relative_noagentpos_policy_encode_output.policy_sample_actions, max_steps = 1_000, s0 = s0)
	ep3 = runepisode(landmark_883_plot; πs = landmark_883_relative_noagentpos_policy_shared_output.policy_sample_actions, max_steps = 1_000, s0 = s0)
	(ep1, ep2, ep3)
end
  ╠═╡ =#

# ╔═╡ 710641b3-d0e0-4e63-9cc7-76b7a78f1ed5
#=╠═╡
@bind landmark_883_ac_comp_ep_step Slider(1:maximum(ep[5] for ep in landmark_883_ac_comp_ep)+1; show_value=true)
  ╠═╡ =#

# ╔═╡ b9eafcc7-eae2-4591-981f-c47e72a701ee
md"""
### Value Training
"""

# ╔═╡ 641ccd85-2a03-4ef5-a2a8-4521d8896ce9
#=╠═╡
const landmark_883_relative_noagentpos_idqn_output = landmark_883_setup_relative_noagentpos.value_setup.train_ϵ_decay("idqn", [64, 64], 1, 0.99f0, 1f-2, 10, 1_000_000, new_params = true, batch_size = 64, target_update_interval = 200, use_double_q = true)
  ╠═╡ =#

# ╔═╡ 1dbb3b3f-3f32-4704-915e-78ebb3fb44cc
#=╠═╡
const landmark_883_relative_noagentpos_idqn_output2 = landmark_883_setup_relative_noagentpos.value_setup.train_ϵ_decay("idqn", [64, 64], 1, 0.99f0, 1f-2, 10, 1_000_000, new_params = true, batch_size = 64, target_update_interval = 200, use_double_q = false)
  ╠═╡ =#

# ╔═╡ 9f47fdab-dfd8-4347-8dc4-d34e8870c834
#=╠═╡
const landmark_883_relative_noagentpos_vdn_output = landmark_883_setup_relative_noagentpos.value_setup.train_ϵ_decay("vdn", [64, 64], 1, 0.99f0, 1f-2, 10, 1_000_000, new_params = true, batch_size = 64, target_update_interval = 200)
  ╠═╡ =#

# ╔═╡ fd36dc48-5530-414f-bcc3-9b7d13614281
md"""
#### Learning Curves
"""

# ╔═╡ 4a6d1a6b-e24a-4746-8bb7-6c1ec6c36d3e
md"""
#### Average Performance
"""

# ╔═╡ 29c48bfe-7882-4183-8043-47932dedfd29
#=╠═╡
evaluate_policy_performance(landmark_883_plot; πs = extract_threadsafe_joint_policies(landmark_883_plot, landmark_883_relative_noagentpos_idqn_output; ϵ = 0.01f0), max_steps = 10_000)
  ╠═╡ =#

# ╔═╡ eb51758e-cc2d-407e-b329-b13d6fec0a09
#=╠═╡
evaluate_policy_performance(landmark_883_plot; πs = extract_threadsafe_joint_policies(landmark_883_plot, landmark_883_relative_noagentpos_vdn_output; ϵ = 0.01f0), max_steps = 10_000)
  ╠═╡ =#

# ╔═╡ 4a4d37eb-08dc-4b20-b7bd-b9dea75b3eb2
md"""
## Full Position Information
"""

# ╔═╡ 3a6190af-80c2-4afd-9749-1831c633aafe
#=╠═╡
const landmark_883_setup_relative = setup_landmark_training(landmark_883, true, true)
  ╠═╡ =#

# ╔═╡ d0cfdc57-ce0b-4dc3-8b11-4647b6a551fb
md"""
### Policy Training
"""

# ╔═╡ 02fdf5a0-388f-4c71-b700-481013795c7b
#=╠═╡
const landmark_883_relative_policy_output = landmark_883_setup_relative.policy_setup.train_rate_decay("ac", [64, 64], 1, 0.99f0, 5f-1, 1f-1, 10, 1_000_000, new_params = true)
  ╠═╡ =#

# ╔═╡ 0896966a-333a-4269-8ce0-7c0cf6ba8caa
#=╠═╡
const landmark_883_relative_policy_encode_output = landmark_883_setup_relative.policy_setup.train_rate_decay("ac", [64, 64], [64, 64], 64, [64, 64, 64], 1, 0.99f0, 5f-2, 1f-2, 1f-2, 10, 1_000_000, new_params = true)
  ╠═╡ =#

# ╔═╡ 9fa906e2-c4e6-42ad-bae5-1303a364b8d0
#=╠═╡
const landmark_883_relative_policy_shared_output = landmark_883_setup_relative.policy_setup.train_rate_decay("ac", [64, 64], 1, 0.99f0, 5f-2, 1f-2, 10, 1_000_000, new_params = true, share_policy_params = true, share_value_params = true)
  ╠═╡ =#

# ╔═╡ 41247f8c-d265-44b2-b1a1-41801eef780c
md"""
#### Learning Curves
"""

# ╔═╡ fc77c3ee-1ec3-4d91-8968-3edb22045118
md"""
#### Average Performance
"""

# ╔═╡ dc4b6026-93e5-4a6e-9961-a4ab7208ed76
#=╠═╡
NamedTuple(Symbol(name) => evaluate_policy_performance(landmark_883_plot; πs = output.policy_sample_actions, max_steps = 10_000) for (name, output) in [("Vanilla Actor Critic", landmark_883_relative_policy_output), ("AC with Policy Representations", landmark_883_relative_policy_encode_output), ("AC with Homogeneous Agents", landmark_883_relative_policy_shared_output)])
  ╠═╡ =#

# ╔═╡ b81e52f7-de3a-40d1-843c-ddebd6fbeb9d
md"""
#### Compare Behavior in Episode
"""

# ╔═╡ 691357d5-ef22-40ed-a516-a8bc5b1394e3
#=╠═╡
landmark_883_ac_comp_ep2 = let
	s0 = landmark_883.initialize_state()
	ep1 = runepisode(landmark_883_plot; πs = landmark_883_relative_policy_output.policy_sample_actions, max_steps = 1_000, s0 = s0)
	ep2 = runepisode(landmark_883_plot; πs = landmark_883_relative_policy_encode_output.policy_sample_actions, max_steps = 1_000, s0 = s0)
	ep3 = runepisode(landmark_883_plot; πs = landmark_883_relative_policy_shared_output.policy_sample_actions, max_steps = 1_000, s0 = s0)
	(ep1, ep2, ep3)
end
  ╠═╡ =#

# ╔═╡ e6a3ce3e-fcd9-474b-9588-e143d2066519
#=╠═╡
@bind landmark_883_ac_comp_ep2_step Slider(1:maximum(ep[5] for ep in landmark_883_ac_comp_ep2)+1; show_value=true)
  ╠═╡ =#

# ╔═╡ e63a5ee4-172f-493a-990c-0c0c9d901671
md"""
### Value Training
"""

# ╔═╡ 8272ef65-5739-4f17-aba0-e12a8a1f5e3b
#=╠═╡
const landmark_883_relative_idqn_output = landmark_883_setup_relative.value_setup.train_ϵ_decay("idqn", [64, 64], 1, 0.99f0, 1f-2, 10, 100_000, new_params = true, batch_size = 64, target_update_interval = 200, use_double_q = true)
  ╠═╡ =#

# ╔═╡ c3f2ff70-478c-4855-a2e0-c39b4688b4c0
#=╠═╡
const landmark_883_relative_vdn_output = landmark_883_setup_relative.value_setup.train_ϵ_decay("vdn", [64, 64], 1, 0.99f0, 1f-2, 10, 100_000, new_params = true, batch_size = 64, target_update_interval = 200)
  ╠═╡ =#

# ╔═╡ 6aad109a-71b8-4fe6-b811-a61d9c22dd39
md"""
#### Learning Curves
"""

# ╔═╡ 360b8520-7a85-4fec-8117-aaf1bef56803
md"""
# Visualization Tools
"""

# ╔═╡ a6826b4f-236b-4e35-8c9a-95a8af8c1de8
function plot_landmark_state(s::LandmarkGame.State{N, X, Y}) where {N, X, Y}
	bottom_border = scatter(x = [0, X+1], y = [0, 0], mode = "lines", line_color = "black", showlegend = false, name = "Edge", line_width = 3)
	top_border = scatter(x = [0, X+1], y = [Y+1, Y+1], mode = "lines", line_color = "black", showlegend = false, name = "Edge", line_width = 3)
	left_border = scatter(x = [0, 0], y = [0, Y+1], mode = "lines", line_color = "black", showlegend = false, name = "Edge", line_width = 3)
	right_border = scatter(x = [X+1, X+1], y = [0, Y+1], mode = "lines", line_color = "black", showlegend = false, name = "Edge", line_width = 3)
	agent_xs = [s.agent_positions[i][1] for i in 1:N]
	agent_ys = [s.agent_positions[i][2] for i in 1:N]
	agent_trace = scatter(x = agent_xs, y = agent_ys, name = "Agents", mode = "markers")

	landmark_xs = [s.landmark_positions[i][1] for i in 1:N]
	landmark_ys = [s.landmark_positions[i][2] for i in 1:N]
	landmark_trace = scatter(x = landmark_xs, y = landmark_ys, name = "Landmarks", mode = "markers", marker_size = 20)

	plot([bottom_border, top_border, left_border, right_border, landmark_trace, agent_trace], Layout(xaxis_title = "x", yaxis_title = "y", xaxis_range = [0, X+1], yaxis_range = [0, Y+1], yaxis_scaleanchor="x", width = 450, height = 400, xaxis_dtick = 1, yaxis_dtick = 1, xaxis_tickvals = 1:X, xaxis_ticktext = string.(1:X), yaxis_tickvals = 1:Y, yaxis_ticktext = string.(1:Y), annotations = vcat([attr(x = s.agent_positions[i][1], y = s.agent_positions[i][2], text = "Agent $i", showarrow = true,  font = attr(color = "blue", size = 14, weight = 1000, showdow = "auto")) for i in 1:N])))
end

# ╔═╡ a122ef1a-4e23-401b-b4cf-338eca6e4cfa
#=╠═╡
plot_landmark_state(landmark_883.initialize_state())
  ╠═╡ =#

# ╔═╡ 84153fbc-81ac-4416-9d44-50d25d5d4af7
#=╠═╡
plot_landmark_state(test_episode[1][test_episode_step])
  ╠═╡ =#

# ╔═╡ 8b9bf677-87f1-4585-92e5-e5d38f942c59
#=╠═╡
@htl("""
	 <div style = "display: flex;">
	 <div style = "width: 33%; transform: scale(0.8);">
	 <div>Vanilla Actor Critic: $(landmark_883_ac_comp_ep[1][5]) Steps</div>
	$(plot_landmark_state(vcat(landmark_883_ac_comp_ep[1][1], landmark_883_ac_comp_ep[1][4])[min(landmark_883_ac_comp_ep_step, landmark_883_ac_comp_ep[1][5]+1)]))
	 </div>
	 <div style = "width: 33%; transform: scale(0.8);">
	 <div>Actor Critic with Policy Representations: $(landmark_883_ac_comp_ep[2][5]) Steps</div>
	 $(plot_landmark_state(vcat(landmark_883_ac_comp_ep[2][1], landmark_883_ac_comp_ep[2][4])[min(landmark_883_ac_comp_ep_step, landmark_883_ac_comp_ep[2][5]+1)]))
	 </div>
	 <div style = "width: 33%; transform: scale(0.8);">
	 <div>Actor Critic with Parameter Sharing: $(landmark_883_ac_comp_ep[3][5]) Steps</div>
	 $(plot_landmark_state(vcat(landmark_883_ac_comp_ep[3][1], landmark_883_ac_comp_ep[3][4])[min(landmark_883_ac_comp_ep_step, landmark_883_ac_comp_ep[3][5]+1)]))
	 </div>
	 </div>
	 """)
  ╠═╡ =#

# ╔═╡ d3f763d6-64ab-4915-a752-accdd9eacdb4
#=╠═╡
@htl("""
	 <div style = "display: flex;">
	 <div style = "width: 33%; transform: scale(0.8);">
	 <div>Vanilla Actor Critic: $(landmark_883_ac_comp_ep2[1][5]) Steps</div>
	$(plot_landmark_state(vcat(landmark_883_ac_comp_ep2[1][1], landmark_883_ac_comp_ep2[1][4])[min(landmark_883_ac_comp_ep2_step, landmark_883_ac_comp_ep2[1][5]+1)]))
	 </div>
	 <div style = "width: 33%; transform: scale(0.8);">
	 <div>Actor Critic with Policy Representations: $(landmark_883_ac_comp_ep2[2][5]) Steps</div>
	 $(plot_landmark_state(vcat(landmark_883_ac_comp_ep2[2][1], landmark_883_ac_comp_ep2[2][4])[min(landmark_883_ac_comp_ep2_step, landmark_883_ac_comp_ep2[2][5]+1)]))
	 </div>
	 <div style = "width: 33%; transform: scale(0.8);">
	 <div>Actor Critic with Parameter Sharing: $(landmark_883_ac_comp_ep2[3][5]) Steps</div>
	 $(plot_landmark_state(vcat(landmark_883_ac_comp_ep2[3][1], landmark_883_ac_comp_ep2[3][4])[min(landmark_883_ac_comp_ep2_step, landmark_883_ac_comp_ep2[3][5]+1)]))
	 </div>
	 </div>
	 """)
  ╠═╡ =#

# ╔═╡ 966c349a-05ee-4842-a9bf-87f812c642d5
function plot_policy_performance(policy_output::NamedTuple; num_points::Integer = 1_000, smoothing_factor::Integer = 10_000)
	agent_rewards = policy_output.avg_step_rewards
	l = length(first(agent_rewards))
	rewards = first(agent_rewards)

	avg_points = l < num_points ? 1 : ceil(Int64, l / num_points)
	smoothed_performance = [mean(view(rewards, i-smoothing_factor:i)) for i in smoothing_factor+1:avg_points:l]
	plot(scatter(x = smoothing_factor+1:avg_points:l, y = smoothed_performance), Layout(xaxis_title = "Step", yaxis_title = "Average Reward per Step"))
end

# ╔═╡ 8119df54-db01-488e-8425-1420feea5afd
function plot_policy_performance(policy_output::NamedTuple, name::AbstractString; num_points::Integer = 1_000, smoothing_factor::Integer = 10_000)
	agent_rewards = policy_output.avg_step_rewards
	l = length(first(agent_rewards))
	rewards = first(agent_rewards)

	
	avg_points = l < num_points ? 1 : ceil(Int64, l / num_points)
	smoothed_performance = [mean(view(rewards, i-smoothing_factor:i)) for i in smoothing_factor+1:avg_points:l]
	scatter(x = smoothing_factor+1:avg_points:l, y = smoothed_performance, name = name)
end

# ╔═╡ 69a756ce-176c-4d01-860e-5b8c7e5923f2
plot_policy_performance(v::AbstractVector{T}) where T<:GenericTrace = plot(v, Layout(xaxis_title = "Step", yaxis_title = "Average Reward per Step"))

# ╔═╡ cdbb4602-a89a-4695-98e6-47531de46784
#=╠═╡
plot_policy_performance(landmark_883_relative_noagentpos_policy_output)
  ╠═╡ =#

# ╔═╡ 0c012a01-696c-42a8-81c0-7a2dc7883b2b
#=╠═╡
plot_policy_performance([plot_policy_performance(landmark_883_relative_noagentpos_policy_output, "Vanilla Actor Critic"), plot_policy_performance(landmark_883_relative_noagentpos_policy_encode_output, "AC with Policy Representations"), plot_policy_performance(landmark_883_relative_noagentpos_policy_shared_output, "AC with Parameter Sharing")])
  ╠═╡ =#

# ╔═╡ 03f06c81-1e7f-4a1e-badb-c344aad36927
#=╠═╡
plot_policy_performance([plot_policy_performance(landmark_883_relative_policy_output, "Vanilla Actor Critic"), plot_policy_performance(landmark_883_relative_policy_encode_output, "AC with Policy Representations"), plot_policy_performance(landmark_883_relative_policy_shared_output, "AC with Parameter Sharing")])
  ╠═╡ =#

# ╔═╡ 47c2b366-0d83-4570-807e-338d43940b4f
function create_episode_policy_visualization(game::StateStochasticGame, output::NamedTuple)
	ep = runepisode(game; πs = output.policy_sample_actions)

	states = vcat(ep[1], ep[4])

	step_select = Slider(1:ep[5]+1; show_value=true)
	make_plot(i::Integer) = plot_landmark_state(states[i])
	(step_select, make_plot)
end

# ╔═╡ f8de6fb3-4275-48f1-b753-3b654d647005
#=╠═╡
landmark_883_relative_noagentpos_policy_vis = create_episode_policy_visualization(landmark_883, landmark_883_relative_noagentpos_policy_output)
  ╠═╡ =#

# ╔═╡ 1e929357-1ffd-4c51-8db2-21b0e64e2566
#=╠═╡
@bind landmark_883_relative_noagentpos_policy_vis_step landmark_883_relative_noagentpos_policy_vis[1]
  ╠═╡ =#

# ╔═╡ b9c112ad-adfd-4394-b9fb-82d2bbd6afd7
#=╠═╡
landmark_883_relative_noagentpos_policy_vis[2](landmark_883_relative_noagentpos_policy_vis_step)
  ╠═╡ =#

# ╔═╡ d9f0f5e9-1c64-4eba-82dd-d6120e84fc20
function plot_vdn_performance(vdn_output::NamedTuple; num_points::Integer = 1_000, smoothing_factor::Integer = 1_000)
	rewards = vdn_output.episode_rewards
	l = length(rewards)
	num_agents = length(vdn_output.value_functions)
	avg_points = l < num_points ? 1 : ceil(Int64, l / num_points)
	smoothed_performance = [mean(view(rewards, i-smoothing_factor:i)) / num_agents for i in smoothing_factor+1:avg_points:l]
	plot(scatter(x = smoothing_factor+1:avg_points:l, y = smoothed_performance), Layout(xaxis_title = "Episode", yaxis_title = "Agent Reward Per Episode"))
end

# ╔═╡ ad4d0c82-f42c-4868-a8b9-ec07ef00de4f
function plot_vdn_performance(vdn_output::NamedTuple, name::AbstractString; num_points::Integer = 1_000, smoothing_factor::Integer = 1_000)
	rewards = vdn_output.episode_rewards
	l = length(rewards)
	num_agents = length(vdn_output.value_functions)
	avg_points = l < num_points ? 1 : ceil(Int64, l / num_points)
	smoothed_performance = [mean(view(rewards, i-smoothing_factor:i)) / num_agents for i in smoothing_factor+1:avg_points:l]
	scatter(x = smoothing_factor+1:avg_points:l, y = smoothed_performance, name = name)
end

# ╔═╡ 821eb847-7acf-49f7-94bd-f783b0a07986
function plot_idqn_performance(idqn_output::NamedTuple; num_points::Integer = 1_000, smoothing_factor::Integer = 1_000)
	rewards = idqn_output.episode_rewards[1]
	l = length(rewards)
	avg_points = l < num_points ? 1 : ceil(Int64, l / num_points)
	smoothed_performance = [mean(view(rewards, i-smoothing_factor:i)) for i in smoothing_factor+1:avg_points:l]
	plot(scatter(x = smoothing_factor+1:avg_points:l, y = smoothed_performance), Layout(xaxis_title = "Episode", yaxis_title = "Agent Reward Per Episode"))
end

# ╔═╡ 8a3cf1c2-c223-42c4-95e0-d5ca555e5d3a
function plot_idqn_performance(idqn_output::NamedTuple, name::AbstractString; num_points::Integer = 1_000, smoothing_factor::Integer = 1_000)
	rewards = idqn_output.episode_rewards[1]
	l = length(rewards)
	avg_points = l < num_points ? 1 : ceil(Int64, l / num_points)
	smoothed_performance = [mean(view(rewards, i-smoothing_factor:i)) for i in smoothing_factor+1:avg_points:l]
	scatter(x = smoothing_factor+1:avg_points:l, y = smoothed_performance, name = name)
end

# ╔═╡ deb889be-7d0d-4b8b-858e-1bba3784cbcc
plot_value_performance(v::AbstractVector{T}) where T<:GenericTrace = plot(v, Layout(xaxis_title = "Episode", yaxis_title = "Agent Reward per Episode"))

# ╔═╡ d5dab0a0-94e1-4c0a-b809-70e2ecdb59f3
#=╠═╡
plot_value_performance([plot_idqn_performance(landmark_883_relative_noagentpos_idqn_output, "IDQN Double Q"), plot_idqn_performance(landmark_883_relative_noagentpos_idqn_output2, "IDQN"), plot_vdn_performance(landmark_883_relative_noagentpos_vdn_output, "VDN")])
  ╠═╡ =#

# ╔═╡ 8fa92721-57c5-43eb-a757-260393e67ab3
#=╠═╡
plot_value_performance([plot_idqn_performance(landmark_883_relative_idqn_output, "IDQN"), plot_vdn_performance(landmark_883_relative_vdn_output, "VDN")])
  ╠═╡ =#

# ╔═╡ a99b2b36-4a6f-11f1-8d2d-33857cdc162b
md"""
# Dependencies
"""

# ╔═╡ 3f1123fc-ef67-430a-8b3d-ec5f533beb5d
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
BenchmarkTools = "~1.8.0"
CSV = "~0.10.16"
DataFrames = "~1.8.2"
DataStructures = "~0.19.4"
HiGHS = "~1.23.0"
HypertextLiteral = "~0.9.5"
JuMP = "~1.30.1"
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
project_hash = "55c67acb127cba623e9e6f21ae9945061ed82bab"

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
git-tree-sha1 = "6941586d9cf3c0af718bc6e6250dcf24057d412e"
uuid = "4076af6c-e467-56ae-b986-b466b2749572"
version = "1.30.1"

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
deps = ["CodecBzip2", "CodecZlib", "ForwardDiff", "JSON", "LinearAlgebra", "MutableArithmetics", "NaNMath", "OrderedCollections", "PrecompileTools", "Printf", "SparseArrays", "SpecialFunctions", "Test"]
git-tree-sha1 = "73939c06e863f8d68322106fdc2464f3443b5e1a"
uuid = "b8f27783-ece8-5eb3-8dc8-9495eed66fee"
version = "1.51.0"

    [deps.MathOptInterface.extensions]
    MathOptInterfaceBenchmarkToolsExt = "BenchmarkTools"
    MathOptInterfaceCliqueTreesExt = "CliqueTrees"

    [deps.MathOptInterface.weakdeps]
    BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
    CliqueTrees = "60701a23-6482-424a-84db-faee86b9b1f8"

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
git-tree-sha1 = "dd0d9979377e43918a80486a562ddedcc6d9bdf3"
uuid = "656ef2d0-ae68-5445-9ca0-591084a874a2"
version = "0.3.33+0"

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
git-tree-sha1 = "5d5e0a78e971354b1c7bff0655d11fdc1b0e12c8"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.4"

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
git-tree-sha1 = "0716e01c3b40413de5dedbc9c5c69f27cddfddfc"
uuid = "ea10d353-3f73-51f8-a26c-33c1cb351aa5"
version = "1.4.3"

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
# ╠═f5429487-d444-46a2-8d93-8e58afb337d3
# ╠═e8e0a91a-a316-4868-b562-8aebd8998e9d
# ╠═e1a9c4a1-7131-4ad0-8e7b-b2fa50f7a5ba
# ╠═6bc13bad-aac4-4dc5-bb52-d10edbc67fd1
# ╠═a122ef1a-4e23-401b-b4cf-338eca6e4cfa
# ╠═37654c57-b40b-4041-92e0-1f4dda839640
# ╠═0d1de58c-0e81-485c-b1a2-15ad82f8e92e
# ╟─dabd9ce8-1e2a-4e07-9d47-242d22c743cb
# ╠═84153fbc-81ac-4416-9d44-50d25d5d4af7
# ╟─0540b6f3-f0bf-45b2-b1f5-56f1b8dd2874
# ╠═59fd75a5-827a-4d11-92b9-508847969e17
# ╟─7dfae4b3-f333-4038-ac16-97181142c22a
# ╠═81021156-c74d-470a-b398-308ceab92435
# ╟─1bec5557-ce3d-49b4-b58f-459934b406b2
# ╠═27569545-046d-40b5-9a63-11520a0e792d
# ╠═a5b02994-1718-4d0d-8e6a-66b82523b43b
# ╠═8f88a834-76b6-4bcf-8ac6-d9b50bfd6839
# ╠═36c6e60a-b613-40f6-9430-8a7b25871a1f
# ╠═47958e9f-e2dc-423a-81be-9b01821e5408
# ╠═6f4d6869-a771-4061-877d-d37174801139
# ╟─ace6eac9-2cd9-45f2-a846-a89706d4186a
# ╠═4d2a941b-0a6e-4142-9f03-f96bc673b462
# ╟─31b401e2-2048-4ccd-9d2d-81aaa54e5ba1
# ╠═ef0b2bb9-5c02-4fb5-a54a-d455ac510239
# ╠═cdbb4602-a89a-4695-98e6-47531de46784
# ╠═f8de6fb3-4275-48f1-b753-3b654d647005
# ╟─1e929357-1ffd-4c51-8db2-21b0e64e2566
# ╠═b9c112ad-adfd-4394-b9fb-82d2bbd6afd7
# ╟─14fca46c-ad50-4d9c-85e1-48218715b54e
# ╠═3a704735-a233-40b3-805e-9496150624f1
# ╠═e427fdb5-7888-4083-ba37-5f8419b79382
# ╟─47b17860-93b4-464c-b47d-9d72a25106a4
# ╟─0c012a01-696c-42a8-81c0-7a2dc7883b2b
# ╟─55c87e94-490d-4514-af3e-9834f43338bc
# ╠═978d5716-4fc2-49b6-ba0d-d3e4da4bd6da
# ╟─2993ddf1-2116-49f6-a8f3-b95c9b5e2ce5
# ╠═549acb9e-6cb5-4264-8fe0-52617849e650
# ╟─710641b3-d0e0-4e63-9cc7-76b7a78f1ed5
# ╟─8b9bf677-87f1-4585-92e5-e5d38f942c59
# ╟─b9eafcc7-eae2-4591-981f-c47e72a701ee
# ╠═641ccd85-2a03-4ef5-a2a8-4521d8896ce9
# ╠═1dbb3b3f-3f32-4704-915e-78ebb3fb44cc
# ╠═9f47fdab-dfd8-4347-8dc4-d34e8870c834
# ╟─fd36dc48-5530-414f-bcc3-9b7d13614281
# ╠═d5dab0a0-94e1-4c0a-b809-70e2ecdb59f3
# ╟─4a6d1a6b-e24a-4746-8bb7-6c1ec6c36d3e
# ╠═29c48bfe-7882-4183-8043-47932dedfd29
# ╠═eb51758e-cc2d-407e-b329-b13d6fec0a09
# ╟─4a4d37eb-08dc-4b20-b7bd-b9dea75b3eb2
# ╠═3a6190af-80c2-4afd-9749-1831c633aafe
# ╟─d0cfdc57-ce0b-4dc3-8b11-4647b6a551fb
# ╠═02fdf5a0-388f-4c71-b700-481013795c7b
# ╠═0896966a-333a-4269-8ce0-7c0cf6ba8caa
# ╠═9fa906e2-c4e6-42ad-bae5-1303a364b8d0
# ╟─41247f8c-d265-44b2-b1a1-41801eef780c
# ╠═03f06c81-1e7f-4a1e-badb-c344aad36927
# ╟─fc77c3ee-1ec3-4d91-8968-3edb22045118
# ╠═dc4b6026-93e5-4a6e-9961-a4ab7208ed76
# ╟─b81e52f7-de3a-40d1-843c-ddebd6fbeb9d
# ╠═691357d5-ef22-40ed-a516-a8bc5b1394e3
# ╟─e6a3ce3e-fcd9-474b-9588-e143d2066519
# ╟─d3f763d6-64ab-4915-a752-accdd9eacdb4
# ╟─e63a5ee4-172f-493a-990c-0c0c9d901671
# ╠═8272ef65-5739-4f17-aba0-e12a8a1f5e3b
# ╠═c3f2ff70-478c-4855-a2e0-c39b4688b4c0
# ╟─6aad109a-71b8-4fe6-b811-a61d9c22dd39
# ╠═8fa92721-57c5-43eb-a757-260393e67ab3
# ╟─360b8520-7a85-4fec-8117-aaf1bef56803
# ╠═a6826b4f-236b-4e35-8c9a-95a8af8c1de8
# ╠═966c349a-05ee-4842-a9bf-87f812c642d5
# ╠═8119df54-db01-488e-8425-1420feea5afd
# ╠═69a756ce-176c-4d01-860e-5b8c7e5923f2
# ╠═47c2b366-0d83-4570-807e-338d43940b4f
# ╠═d9f0f5e9-1c64-4eba-82dd-d6120e84fc20
# ╠═ad4d0c82-f42c-4868-a8b9-ec07ef00de4f
# ╠═821eb847-7acf-49f7-94bd-f783b0a07986
# ╠═8a3cf1c2-c223-42c4-95e0-d5ca555e5d3a
# ╠═deb889be-7d0d-4b8b-858e-1bba3784cbcc
# ╠═a99b2b36-4a6f-11f1-8d2d-33857cdc162b
# ╠═6f7fbd5f-f907-491e-8257-22aee4374529
# ╠═dbcef8d5-c6b3-489f-9776-8d1b356b8543
# ╠═b8ecf71c-a2da-4c48-ba52-f18de34f67c0
# ╠═ea36b694-2293-4588-b51f-47d37d36d0a5
# ╠═3f1123fc-ef67-430a-8b3d-ec5f533beb5d
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
