### A Pluto.jl notebook ###
# v0.20.21

using Markdown
using InteractiveUtils

# ╔═╡ 4e509812-7bdf-4928-bee6-55f2d142be67
using PlutoDevMacros

# ╔═╡ 9260398e-6842-4b11-9845-98900884b94a
begin
	PlutoDevMacros.@frompackage @raw_str(joinpath(@__DIR__, "..", "RL_Module")) begin 
		using RL_Module
		using >.Random, >.Statistics, >.LinearAlgebra, >.Transducers, >.StaticArrays, >.DataStructures, >.SparseArrays
	end
end

# ╔═╡ 9f1aa27d-2214-4b7c-9110-1566603887d2
# ╠═╡ skip_as_script = true
#=╠═╡
begin
	using PlutoUI, PlutoPlotly, PlutoProfile, BenchmarkTools, LaTeXStrings, HypertextLiteral, DataFrames, Dates, CSV
	
	TableOfContents()
end
  ╠═╡ =#

# ╔═╡ 73ddfc82-cf10-11f0-a45e-3da27f44d6bc
md"""
# Multi-Agent RL Types
"""

# ╔═╡ 04715feb-f04a-4a4a-b7c5-76180e0508df
md"""
## Tabular State Transition Functions
For the multi-agent problem, transitions depend on the action selections of all agents involved.  In the general case, a reward value must also be assigned to each agent individually.  Since the number of agents is fundamental to the structure of the transition, it is helpful to include a type parameter `N` which specifies how many agents are included in the dynamics.  The dimensionality of the state transition map will have one additional dimension for the state space.  In a deterministic problem, only one transition state results for each state/join-action pair so the transition map can simply store each index.
"""

# ╔═╡ f6de9a64-7dee-4291-815b-7a891c52a146
begin
	abstract type AbstractGameTransition{T<:Real, N} end
	abstract type AbstractTabularGameTransition{T<:Real, N} <: AbstractGameTransition{T, N} end
	
	struct TabularGameTransition{T<:Real, N, Np1, ST<:Union{Int64, SparseVector{T, Int64}}, RT<:Union{T, Vector{T}}} <: AbstractTabularGameTransition{T, N}
		state_transition_map::Array{ST, Np1}
		reward_transition_map::Array{NTuple{N, RT}, Np1}

		function TabularGameTransition(state_transition_map::Array{ST, Np1}, reward_transition_map::Array{NTuple{N, RT}, Np1}) where {T<:Real, N, Np1, ST<:Union{Int64, SparseVector{T, Int64}}, RT <: Union{T, Vector{T}}}
			@assert N == Np1 - 1
			new{T, N, Np1, ST, RT}(state_transition_map, reward_transition_map)
		end
	end
	
	const TabularGameDeterministicTransition{T<:Real, N, Np1} = TabularGameTransition{T, N, Np1, Int64, T}
	const TabularGameStochasticTransition{T<:Real, N, Np1} = TabularGameTransition{T, N, Np1, SparseVector{T, Int64}, Vector{T}}
	
	TabularGameDeterministicTransition(m1, m2) = TabularGameTransition(m1, m2)
	
	TabularMultAgentStochasticTransition(m1, m2) = TabularGameTransition(m1, m2)

	function (ptf::TabularGameDeterministicTransition{T, N, Np1})(i_s::Integer, a::NTuple{N, I}) where {T<:Real, N, Np1, I<:Integer}
		i_s′ = ptf.state_transition_map[a..., i_s]
		r = ptf.reward_transition_map[a..., i_s]
		(r, i_s′)
	end
end

# ╔═╡ fe796b4e-fb05-4071-ad4b-7e3ce58092c4
#convert a multi-agent transition into an MDP transition with the action space being each joint action and the reward transformed into a single scalar value
function TabularRL.TabularDeterministicTransition(ptf::TabularGameDeterministicTransition{T, N, Np1}, reward_function::Function) where {T<:Real, N, Np1}
	dims = size(ptf.state_transition_map)
	num_states = dims[Np1]
	num_actions = dims[1:N]
	total_actions = prod(num_actions)
	state_transition_map = reshape(ptf.state_transition_map, total_actions, num_states)
	reward_intermediate = reshape(ptf.reward_transition_map, total_actions, num_states)
	reward_transition_map = Matrix{T}(undef, total_actions, num_states)
	for i in eachindex(reward_transition_map)
		reward_transition_map[i] = reward_function(reward_intermediate[i])
	end
	TabularDeterministicTransition(state_transition_map, reward_transition_map)
end

# ╔═╡ 85251e1c-6704-4dc5-85dd-0c8d9706358e
#note that for a zero-sum and common-reward game only one reward value is needed for all of the agent rewards

# ╔═╡ c42b7d58-922d-4b17-a03c-62433c9adeb9
begin
	struct TabularZeroSumGameTransition{T<:Real, ST<:Union{Int64, SparseVector{T, Int64}}, RT<:Union{T, Vector{T}}} <: AbstractTabularGameTransition{T, 2}
		state_transition_map::Array{ST, 3}
		reward_transition_map::Array{RT, 3}
		function TabularZeroSumGameTransition(state_transition_map::Array{ST, 3}, reward_transition_map::Array{RT, 3}) where {T<:Real, ST<:Union{Int64, SparseVector{T, Int64}}, RT <: Union{T, Vector{T}}}
			new{T, ST, RT}(state_transition_map, reward_transition_map)
		end
	end

	const TabularZeroSumGameDeterministicTransition{T<:Real} = TabularZeroSumGameTransition{T, Int64, T}
	const TabularZeroSumGameStochasticTransition{T<:Real} = TabularZeroSumGameTransition{T, SparseVector{T, Int64}, Vector{T}}

	function (ptf::TabularZeroSumGameDeterministicTransition{T})(i_s::Integer, a::NTuple{2, I}) where {T<:Real, I<:Integer}
		i_s′ = ptf.state_transition_map[a[1], a[2], i_s]
		r = ptf.reward_transition_map[a[1], a[2], i_s]
		(r, i_s′)
	end
end

# ╔═╡ 6f98c460-6d48-457d-a40c-6b8588fb01ef
begin
	struct TabularCommonRewardGameTransition{T<:Real, N, Np1, ST<:Union{Int64, SparseVector{T, Int64}}, RT<:Union{T, Vector{T}}} <: AbstractTabularGameTransition{T, N}
		state_transition_map::Array{ST, Np1}
		reward_transition_map::Array{RT, Np1}
		function TabularCommonRewardGameTransition(state_transition_map::Array{ST, Np1}, reward_transition_map::Array{RT, Np1}) where {T<:Real, Np1, ST<:Union{Int64, SparseVector{T, Int64}}, RT <: Union{T, Vector{T}}}
			new{T, Np1-1, N, ST, RT}(state_transition_map, reward_transition_map)
		end
	end

	const TabularCommonRewardGameDeterministicTransition{T<:Real, N, Np1} = TabularCommonRewardGameTransition{T, N, Np1, Int64, T}
	const TabularCommonRewardGameStochasticTransition{T<:Real, N, Np1} = TabularCommonRewardGameTransition{T, N, Np1, SparseVector{T, Int64}, Vector{T}}

	function (ptf::TabularCommonRewardGameDeterministicTransition{T, N, Np1})(i_s::Integer, a::NTuple{N, I}) where {T<:Real, N, Np1, I<:Integer}
		i_s′ = ptf.state_transition_map[a..., i_s]
		r = ptf.reward_transition_map[a..., i_s]
		(r, i_s′)
	end
end

# ╔═╡ ed41ec94-38be-40de-bf61-c927c242cef7
#convert a multi-agent transition into an MDP transition with the action space being each joint action and the common reward used as the reward
function TabularRL.TabularDeterministicTransition(ptf::TabularCommonRewardGameDeterministicTransition{T, N, Np1}) where {T<:Real, N, Np1}
	dims = size(ptf.state_transition_map)
	num_states = dims[Np1]
	num_actions = dims[1:N]
	total_actions = prod(num_actions)
	state_transition_map = reshape(ptf.state_transition_map, total_actions, num_states)
	reward_transition_map = reshape(ptf.reward_transition_map, total_actions, num_states)
	TabularDeterministicTransition(state_transition_map, reward_transition_map)
end

# ╔═╡ b815429b-fbe3-443c-9a76-d4b194eca26d
#example transitions for two agents with 6 actions each
ex_reward_transitions = rand(6, 6, 100)

# ╔═╡ 23a5f4c3-5f6a-473a-9ee5-0c82cf8c7f8e
#joint action space for single agent RL is 36 representing all combinations of agent actions
reshape(ex_reward_transitions, 36, 100)

# ╔═╡ 75726eb9-eb0e-42a3-a648-dbfce92cbd6e
abstract type AbstractStochasticGame{T<:Real, S, A, N, P<:AbstractGameTransition{T, N}, F<:Function} end

# ╔═╡ ed152a1a-0fc9-40f3-91b4-e40246725847
md"""
## Tabular Multi-Agent Stochastic Games

To form a complete stochastic game, the transition function must be accompanied by the enumerated states, agent actions, state initialization function, and terminal states indicator.  For convenience, lookup tables are stored to easy find the index of any state and agent action.
"""

# ╔═╡ 79888a7b-c604-4db2-932a-ee9bfd378d9e
begin
	struct TabularStochasticGame{T<:Real, S, A, N, P<:AbstractTabularGameTransition{T, N}, F<:Function} <: AbstractStochasticGame{T, S, A, N, P, F}
		states::Vector{S}
		agent_actions::NTuple{N, Vector{A}}
		ptf::P
		initialize_state_index::F
		terminal_states::BitVector
		state_index::Dict{S, Int64}
		action_index::NTuple{N, Dict{A, Int64}}
	end

	TabularStochasticGame(states::Vector{S}, agent_actions::NTuple{N, Vector{A}}, ptf::P, initialize_state_index::F, terminal_states::BitVector; state_index::Dict{S, Int64} = makelookup(states), action_index::NTuple{N, Dict{A, Int64}} = Tuple(makelookup(a) for a in agent_actions)) where {S, A, N, P<:AbstractTabularGameTransition, F<:Function} = TabularStochasticGame(states, agent_actions, ptf, initialize_state_index, terminal_states, state_index, action_index)

	TabularStochasticGame(states::Vector{S}, agent_actions::NTuple{N, Vector{A}}, ptf::P, terminal_states::BitVector; state_index::Dict{S, Int64} = makelookup(states), action_index::NTuple{N, Dict{A, Int64}} = Tuple(makelookup(a) for a in agent_actions)) where {S, A, N, P<:AbstractTabularGameTransition} = TabularStochasticGame(states, agent_actions, ptf, Returns(1:length(states)), terminal_states, state_index, action_index)

	TabularStochasticGame(states::Vector{S}, agent_actions::NTuple{N, Vector{A}}, ptf::P, initialize_state_index::F; state_index::Dict{S, Int64} = makelookup(states), action_index::NTuple{N, Dict{A, Int64}} = Tuple(makelookup(a) for a in agent_actions)) where {S, A, N, P<:AbstractTabularGameTransition, F<:Function} = TabularStochasticGame(states, agent_aactions, ptf, initialize_state_index, BitMatrix(undef, length(states)), state_index, action_index)

	TabularStochasticGame(states::Vector{S}, agent_actions::NTuple{N, Vector{A}}, ptf::P; state_index::Dict{S, Int64} = makelookup(states), action_index::NTuple{N, Dict{A, Int64}} = Tuple(makelookup(a) for a in agent_actions)) where {S, A, N, P<:AbstractTabularGameTransition} = TabularStochasticGame(states, agent_actions, ptf, Returns(1:length(states)), BitMatrix(undef, length(states)), state_index, action_index)
end

# ╔═╡ faac6925-b526-4ee0-a84c-b660e6819313
#convert a tabular stochastic game into a tabular mdp using a scalar reward function
function TabularRL.TabularMDP(stochastic_game::TabularStochasticGame{T, S, A, N, P, F}, reward_function::Function) where {T<:Real, S, A, N, P<:TabularGameDeterministicTransition, F<:Function}
	agent_actions = stochastic_game.agent_actions
	num_actions = Tuple(length(a) for a in agent_actions)
	joint_action_matrix = Array{A, N}(undef, num_actions...)
	inds = CartesianIndices(joint_action_matrix)
	joint_action_list = [Tuple(agent_actions[i][inds[n][i]] for i in 1:N) for n in 1:length(joint_action_matrix)]

	ptf = TabularDeterministicTransition(stochastic_game.ptf, reward_function)
	
	TabularMDP(stochastic_game.states, joint_action_list, ptf, stochastic_game.initialize_state_index, stochastic_game.terminal_states; state_index = stochastic_game.state_index)
end

# ╔═╡ 776a9ef9-8477-449f-b195-2cc5d7c93749
# ╠═╡ skip_as_script = true
#=╠═╡
const test_agent_actions = [Tuple(1:6) for i in 1:3]
  ╠═╡ =#

# ╔═╡ 45668f60-2dbc-48d4-9903-111b80665845
#=╠═╡
const test_inds = collect(CartesianIndices(Array{Int64, length(test_agent_actions)}(undef, Tuple(length(a) for a in test_agent_actions)...)))[:]
  ╠═╡ =#

# ╔═╡ 04b35d98-e60f-4e3c-b89d-891f7139b5ec
md"""
## Non-tabular State Transition Functions

The step function takes a state and joint action as input.  The joint action is represented by a tuple of integers of length equal to the number of agents.  Each index in the tuple refers to the index of the selected action for that agent from its list of available actions.
"""

# ╔═╡ 738eadf8-7bbf-4942-a90d-e8accbb52bea
begin
	abstract type AbstractStateGameTransition{T<:Real, S, F<:Function, N} <: AbstractGameTransition{T, N} end

	struct StateGameTransitionDeterministic{T<:Real, S, F<:Function, N} <: AbstractStateGameTransition{T, S, F, N}
		step::F
		function StateGameTransitionDeterministic(step::F, s::S, num_agents::Integer) where {F<:Function, S}
			(r, s′) = step(s, ntuple(Returns(1), num_agents))
			@assert promote_type(S, typeof(s′)) != Any "There is no common type between the provided state $s and the transition state $s′"
			@assert length(r) == num_agents "The reward output length of $(length(r)) does not match the expected number of agents: $num_agents"
			new{eltype(r), promote_type(S, typeof(s′)), F, num_agents}(step)
		end
	end

	(ptf::StateGameTransitionDeterministic{T, S, F, N})(s::S, a::NTuple{N, I}) where {T<:Real, S, F<:Function, N, I<:Integer} = ptf.step(s, a)
end

# ╔═╡ ae213605-c98f-4939-8e29-48eea2ff5ed8
md"""
## Non-tabular Multi-Agent Stochastic Games

For the non-tabular case, the list of states is omitted and now the initialization function returns a state rather than a state index.  Similarly, the list of terminal states is replaced by a function that can evaluate whether or not a state is terminal.
"""

# ╔═╡ 736aacc0-5592-4439-b8c3-cf76525983a5
begin
	struct StateStochasticGame{T<:Real, S, A, N, P<:AbstractStateGameTransition{T, S, F, N} where F<:Function, StateInit<:Function, IsTerm<:Function} <: AbstractStochasticGame{T, S, A, N, P, StateInit}
		agent_actions::NTuple{N, Vector{A}}
		ptf::P
		initialize_state::StateInit
		isterm::IsTerm
		agent_action_index::NTuple{N, Dict{A, Int64}} #lookup table mapping actions to their index for each agent
	end

	#automatically generate the action lookup when constructing MDP
	function StateStochasticGame(agent_actions::NTuple{N, A}, ptf::AbstractStateGameTransition{T, S, F, N}, initialize_state::StateInit, isterm::IsTerm; agent_action_index = Tuple(makelookup(a) for a in agent_actions)) where {T<:Real, S, F<:Function, N, A<:AbstractVector, StateInit<:Function, IsTerm<:Function}
		s0 = initialize_state()
		isterm(s0)
		@assert typeof(s0) <: S
		StateStochasticGame(Tuple(Vector(a) for a in agent_actions), ptf, initialize_state, isterm, agent_action_index)
	end

	#if terminal check is not provided assume there are no terminal states
	StateStochasticGame(agent_actions::NTuple{N, A}, ptf::AbstractStateGameTransition{T, S, F, N}, initialize_state::StateInit; kwargs...) where {T<:Real, S, F<:Function, N, A<:AbstractVector, StateInit<:Function} = StateStochasticGame(agent_actions, ptf, initialize_state, Returns(false); kwargs...)
end

# ╔═╡ 39182e4e-6ee9-4c14-8b5f-3164e38d5b5a
begin
	convert_multiagent_transition(mdp::StateStochasticGame{T, S, A, N, P, F1, F2}, joint_step::Function) where {T<:Real, S, A, N, P<:StateGameTransitionDeterministic, F1<:Function, F2<:Function} = StateMDPTransitionDeterministic(joint_step, mdp.initialize_state())

	#add other transition types later
end

# ╔═╡ 0bb34d75-0422-4336-9e62-d6431ce2b1c3
#convert a multi-agent state mdp into an mdp using a scalar reward function
function TabularRL.StateMDP(mdp::StateStochasticGame{T, S, A, N, P, F1, F2}, reward_function::Function) where {T<:Real, S, A, N, P<:AbstractStateGameTransition, F1<:Function, F2<:Function}
	agent_actions = mdp.agent_actions
	num_actions = Tuple(length(a) for a in agent_actions)
	joint_action_matrix = Array{A, N}(undef, num_actions...)
	inds = CartesianIndices(joint_action_matrix)
	joint_action_list = [Tuple(agent_actions[i][inds[n][i]] for i in 1:N) for n in 1:length(joint_action_matrix)]

	function joint_step(s::S, i_a::Integer)
		a = CartesianIndices(joint_action_matrix)[i_a] |> Tuple
		(rewards, s′) = mdp.ptf.step(s, a)
		r = reward_function(rewards)
		return (r, s′)
	end

	ptf = convert_multiagent_transition(mdp, joint_step)
	
	StateMDP(joint_action_list, ptf, mdp.initialize_state, mdp.isterm)
end

# ╔═╡ ae85884a-a23a-48f3-947e-19fc4a21d39f
md"""
# Test Environments
"""

# ╔═╡ 660d62d3-b967-41ba-8dab-8d1003dcd1fc
md"""
## Level-Based Foraging

Multiple agents move within a gridworld and have the option to collect items.  For randomly generated environments, items must be placed so none occupy the same grid point and an agent can never be adjacent to two items at once (otherwise the collect action is ambiguous).  Each item is associated with a set of 4 or fewer locations from which it is possible to collect (adjacent locations).  These collection locations can be saved in the state to speed up evaluation of the step function.
"""

# ╔═╡ 79b114c9-8999-4e78-86b7-8bc0bd81dcad
module LevelBasedForaging
	#import general RL and MARL types and functions
	import RL_Module
	import TabularRL.makelookup

	import ..StateGameTransitionDeterministic
	import ..StateStochasticGame
	import ..TabularGameTransition
	import ..TabularStochasticGame

	#simple empty types to specify the six LBF moves
	abstract type ForagingMove end
	struct Up <: ForagingMove end
	struct Down <: ForagingMove end
	struct Left <: ForagingMove end
	struct Right <: ForagingMove end
	struct Collect <: ForagingMove end
	struct Noop <: ForagingMove end

	#position x/y coordinate in gridworld
	const Position = Tuple{Int64, Int64}

	#six actions available to each agent
	const action_list = [Up(), Down(), Left(), Right(), Collect(), Noop()]
	const action_index = makelookup(action_list)

	#a state in the LBF environment must track the location of each agent and item.  Associated with every entity is also a level.  To aid the environment step function, the state also stores a Set of available positions to agents and the valid positions to collect each item.
	struct ForagingState{N, M}
		agent_positions::Vector{Position}
		item_positions::Vector{Position}
		agent_levels::Vector{Int64}
		item_levels::Vector{Int64}
		available_positions::Set{Position} #keeps track of which positions an agent could move into by excluding squares occupied by items
		item_collection_positions::Vector{Set{Position}} #keeps a list of the positions that are allowed for collecting each item.  These are just the square adjacent to it
		item_level_sum::Integer

		ForagingState(agent_positions, item_positions, agent_levels, item_levels, available_positions, item_collection_positions, item_level_sum) = new{length(agent_positions), length(item_positions)}(agent_positions, item_positions, agent_levels, item_levels, available_positions, item_collection_positions, item_level_sum)
	end

	#hash and isequal definition for ForagingState so that dictionary lookup is not affected by the non-essential state information
	Base.hash(s::ForagingState) = hash((s.agent_positions, s.item_positions, s.agent_levels, s.item_levels, s.item_level_sum))
	Base.isequal(s1::ForagingState, s2::ForagingState) = isequal(s1.agent_positions, s2.agent_positions) && isequal(s1.item_positions, s2.item_positions) && isequal(s1.agent_levels, s2.agent_levels) && isequal(s1.item_levels, s2.item_levels) && isequal(s1.item_level_sum, s2.item_level_sum)

	#generate a ForagingState with only the essential information
	function ForagingState(agent_positions::Vector{Position}, item_positions::Vector{Position}, agent_levels::Vector{Int64}, item_levels::Vector{Int64}, item_level_sum::Integer, x_max::Integer, y_max::Integer)
		position_set = Set((x, y) for x in 1:x_max for y in 1:y_max)
		available_positions = setdiff(position_set, item_positions)
		item_collection_positions = [get_adjacent(p) for p in item_positions]
		ForagingState(agent_positions, item_positions, agent_levels, item_levels, available_positions, item_collection_positions, item_level_sum)
	end

	#LBF episodes end after all items have been collected
	isterm(s::ForagingState) = isempty(s.item_positions)

	function check_adjacent(p1::Position, p2::Position) 
		((p1[1] == p2[1]) && (abs(p1[2] - p2[2]) == 1)) ||
		((p1[2] == p2[2]) && (abs(p1[1] - p2[1]) == 1))
	end

	get_adjacent(p::Position) = Set(move(p, m) for m in (Up(), Down(), Right(), Left()))

	#For all actions except `collect` there is no resulting item collected
	attempt_collect(s::ForagingState, a::ForagingMove, agent_index::Integer) = (0, 0)

	#For the `collect` action, adds the desired item and agent level to the tally for that item
	function attempt_collect(s::ForagingState{N, M}, a::Collect, agent_index::Integer) where {N, M}
		agent_position = s.agent_positions[agent_index]
		agent_level = s.agent_levels[agent_index]
		i = 1
		item_number = 0
		level = 0
		for i in 1:M
			flag = in(agent_position, s.item_collection_positions[i])
			item_number += flag*i
			level += flag*agent_level
		end
		return (item_number, level)
	end

	attempt_collect(s::ForagingState, i_a::Integer, agent_index::Integer) = attempt_collect(s, action_list[i_a], agent_index)

	#for a joint action, checks to see which items if any are collected based on if the level checks pass for a particular item.  Then rewards are allotted to the agents involved in collection
	function attempt_collect(s::ForagingState{N, M}, a::NTuple{N, I}) where {N, M, I<:Integer}
		item_checks = zeros(Int64, M)
		agent_checks = [Set{Int64}() for _ in 1:M]
		rewards = zeros(Float32, N)
	
		for i in 1:N
			(item_number, level) = attempt_collect(s, a[i], i)
			if level > 0
				item_checks[item_number] += level
				push!(agent_checks[item_number], i)
			end
		end

		for i in 1:M
			item_level = s.item_levels[i]
			success = (item_checks[i] >= item_level)
			if success
				denom = s.item_level_sum * sum(s.agent_levels[j] for j in agent_checks[i])
				for j in agent_checks[i]
					rewards[j] = s.agent_levels[j]*item_level / denom
				end
			end
			item_checks[i] = success
		end

		return rewards, item_checks
	end		

	#rules to update an agent's position in the grid
	move(p::Position, ::Noop) = p
	move(p::Position, ::Up) = (p[1], p[2]+1)
	move(p::Position, ::Right) = (p[1]+1, p[2])
	move(p::Position, ::Down) = (p[1], p[2]-1)
	move(p::Position, ::Left) = (p[1]-1, p[2])
	move(p::Position, ::Collect) = p

	move(p::Position, i_a::Integer) = move(p, action_list[i_a])

	#initialize a state given the grid size, number of agents, number of items and the level distribution for both.  Randomly assigns levels using a uniform distribution from 1 to maxlevel
	function initialize_state(grid_positions, position_set, num_agents, num_items, max_agent_level, max_item_level)
		p1 = rand(grid_positions)

		excluded_positions = Set((p1, move(p1, Up()), move(p1, Down()), move(p1, Right()), move(p1, Left())))
		item_positions = [p1]
		allowed_positions = setdiff(grid_positions, excluded_positions)
		
		for i in 2:num_items
			p = rand(allowed_positions)
			push!(item_positions, p)
			setdiff!(allowed_positions, p)
			for m in (Up(), Down(), Right(), Left())
				setdiff!(allowed_positions, move(p, m))
			end
		end

		agent_positions = [rand(allowed_positions)]
		for i in 2:num_agents
			setdiff!(allowed_positions, agent_positions[i-1])
			push!(agent_positions, rand(allowed_positions))
		end

		item_levels = [rand(1:max_item_level) for _ in 1:num_items]
		agent_levels = [rand(1:max_agent_level) for _ in 1:num_agents]
		
		ForagingState(agent_positions, item_positions, agent_levels, item_levels, setdiff(position_set, item_positions), [get_adjacent(p) for p in item_positions], sum(item_levels))
	end

	#make an environment for LBF as a Multi Agent MDP
	function make_environment(;width = 8, height = 8, num_agents = 3, num_items = 5, max_agent_level = 2, max_item_level = 4)
		grid_positions = [(x, y) for x in 1:width for y in 1:height]
		position_set = Set(grid_positions)
		position_inds = collect(eachindex(grid_positions))
		position_lookup = makelookup(grid_positions)
		num_positions = length(grid_positions)

		init_state() = initialize_state(grid_positions, position_set, num_agents, num_items, max_agent_level, max_item_level)

		clamp_positions(p::Position) = (clamp(p[1], 1, width), clamp(p[2], 1, height))

		function step(s::ForagingState{N, M}, a::NTuple{N, I}) where {N, M, I<:Integer}
			s′ = deepcopy(s)
			for i in 1:N
				s′.agent_positions[i] = move(s.agent_positions[i], a[i])
			end
			
			for i in 1:N
				p′ = s′.agent_positions[i]
				if !in(p′, s.available_positions) || in(p′, view(s′.agent_positions, 1:i-1)) || in(p′, view(s′.agent_positions, i+1:N))
					s′.agent_positions[i] = s.agent_positions[i]
				end
			end

			(rewards, item_checks) = attempt_collect(s, a)

			#modify state for collected item
			for i in 1:M
				if isone(item_checks[i])
					deleteat!(s′.item_positions, i)
					deleteat!(s′.item_levels, i)
					deleteat!(s′.item_collection_positions, i)
					push!(s′.available_positions, s.item_positions[i])
				end
			end

			return rewards, ForagingState(s′.agent_positions, s′.item_positions, s′.agent_levels, s′.item_levels, s′.available_positions, s′.item_collection_positions, s′.item_level_sum)
		end

		

		ptf = StateGameTransitionDeterministic(step, init_state(), num_agents)
			
		StateStochasticGame(ntuple(Returns(action_list), num_agents), ptf, init_state, isterm)
	end

	#make an environment for LBF based on example 5.3 which always uses the same initial state with 2 agents and 2 items in an 11x11 grid.  The tabular state space is constructed by iterating through all agent positions with all possible item combinations
	function make_5_3_environment()
		width = 11
		height = 11
		num_agents = 2
		num_items = 2
		max_agent_level = 1
		max_item_level = 2
		
		grid_positions = [(x, y) for x in 1:width for y in 1:height]
		position_set = Set(grid_positions)
		position_inds = collect(eachindex(grid_positions))
		position_lookup = makelookup(grid_positions)
		num_positions = length(grid_positions)

		function initialize_state()
			p1 = (5, 5)
			p2 = (7, 7)
			item_positions = [p1, p2]

			agent_positions = [(1, 1), (11, 1)]
	
			item_levels = [1, 2]
			agent_levels = [1, 1]
			
			ForagingState(agent_positions, item_positions, agent_levels, item_levels, setdiff(position_set, item_positions), [get_adjacent(p) for p in item_positions], 3)
		end

		clamp_positions(p::Position) = (clamp(p[1], 1, width), clamp(p[2], 1, height))

		function step(s::ForagingState{2, M}, a::NTuple{2, I}) where {M, I<:Integer}
			isterm(s) && return ([0f0, 0f0], s)
			s′ = deepcopy(s)
			for i in 1:2
				s′.agent_positions[i] = move(s.agent_positions[i], a[i])
			end
			
			for i in 1:2
				p′ = s′.agent_positions[i]
				if !in(p′, s.available_positions) || in(p′, view(s′.agent_positions, 1:i-1)) || in(p′, view(s′.agent_positions, i+1:2))
					s′.agent_positions[i] = s.agent_positions[i]
				end
			end

			(rewards, item_checks) = attempt_collect(s, a)

			#modify state for collected item
			for i in 1:M
				if isone(item_checks[i])
					deleteat!(s′.item_positions, i)
					deleteat!(s′.item_levels, i)
					deleteat!(s′.item_collection_positions, i)
					push!(s′.available_positions, s.item_positions[i])
				end
			end

			return rewards, ForagingState(s′.agent_positions, s′.item_positions, s′.agent_levels, s′.item_levels, s′.available_positions, s′.item_collection_positions, 3)
		end

		#initialize state with two items and iterate through all agent positions
		s0 = initialize_state()
		states = Vector{ForagingState}()
		for p1 in s0.available_positions
			for p2 in setdiff(s0.available_positions, p1)
				push!(states, ForagingState([p1, p2], s0.item_positions, s0.agent_levels, s0.item_levels, s0.available_positions, s0.item_collection_positions, 3))
			end
		end

		#only keep item 1 and iterate through all agent positions
		s1 = ForagingState(s0.agent_positions, s0.item_positions[[1]], s0.agent_levels, s0.item_levels[[1]], 3, 11, 11)
		for p1 in s1.available_positions
			for p2 in setdiff(s1.available_positions, p1)
				push!(states, ForagingState([p1, p2], s1.item_positions, s1.agent_levels, s1.item_levels, s1.available_positions, s1.item_collection_positions, 3))
			end
		end

		#only keep item 2 and iterate through all agent positions
		s2 = ForagingState(s0.agent_positions, s0.item_positions[[2]], s0.agent_levels, s0.item_levels[[2]], 3, 11, 11)
		for p1 in s2.available_positions
			for p2 in setdiff(s2.available_positions, p1)
				push!(states, ForagingState([p1, p2], s2.item_positions, s2.agent_levels, s2.item_levels, s2.available_positions, s2.item_collection_positions, 3))
			end
		end

		#remove both items and iterate through all agent positions
		s3 = ForagingState(s0.agent_positions, Vector{Position}(), s0.agent_levels, Vector{Int64}(), 3, 11, 11)
		for p1 in s3.available_positions
			for p2 in setdiff(s3.available_positions, p1)
				push!(states, ForagingState([p1, p2], s3.item_positions, s3.agent_levels, s3.item_levels, s3.available_positions, s3.item_collection_positions, 3))
			end
		end

		state_index = Dict{ForagingState, Int64}(makelookup(states))
		
		actions = ntuple(Returns(action_list), 2)
		
		#transition maps have two dimensions of length 6 for the 6 actions available to each agent and then the last dimension is the state space
		state_transition_map = Array{Int64, 3}(undef, 6, 6, length(states))
		reward_transition_map = Array{NTuple{2, Float32}, 3}(undef, 6, 6, length(states))
		terminal_states = BitVector(undef, length(states))

		for i_s in eachindex(states)
			s = states[i_s]
			terminal_states[i_s] = isempty(s.item_positions)
		end

		for i_s in eachindex(states)
			for i_a1 in 1:6
				for i_a2 in 1:6
					s = states[i_s]
					(rewards, s′) = step(s, (i_a1, i_a2))
					i_s′ = state_index[s′]
					state_transition_map[i_a1, i_a2, i_s] = i_s′
					reward_transition_map[i_a1, i_a2, i_s] = Tuple(rewards)
				end
			end
		end

		ptf = TabularGameTransition(state_transition_map, reward_transition_map)

		function initialize_state_index()
			s0 = initialize_state()
			state_index[s0]
		end
			
		TabularStochasticGame(states, actions, ptf, initialize_state_index, terminal_states; state_index = state_index)
	end
end

# ╔═╡ 0eec6fce-eefb-48cb-8b1e-0d77bfcb1129
const lbf_test = LevelBasedForaging.make_environment(;num_agents = 3)

# ╔═╡ b149a025-3b7b-4ee5-a4a5-a6bcd035ad8b
const lbf_single_agent_reduction = StateMDP(lbf_test, sum)

# ╔═╡ c1cc33d2-ad34-4ad8-b47a-6be3aa44c8ed
md"""
### Tabular Environment
"""

# ╔═╡ ac0a9ef8-6fbc-42f2-9433-727b235224e5
const ex_5_3 = LevelBasedForaging.make_5_3_environment()

# ╔═╡ 63062544-0375-4de4-86e3-385347033e5d
md"""
### Central Learning Reductions
"""

# ╔═╡ a91367b5-2bbf-4f17-9e04-5c939223c590
md"""
Using the sum of agent rewards as the scalar transformation, we can convert this multi-agent stochastic game into a Tabular MDP
"""

# ╔═╡ dac84639-2388-4ee8-9372-aae518a37f17
const tab_5_3 = TabularMDP(ex_5_3, sum)

# ╔═╡ c46aa7de-2b71-4abe-b420-17802f027ab4
md"""
#### Value Iteration Solution

With a tabular MDP, we can apply an exactly solution technique such as value iteration.
"""

# ╔═╡ 83583ca7-bd96-4589-bbe0-e967849c3bd3
const value_iter = value_iteration_v(tab_5_3, 0.99f0)

# ╔═╡ aa1faa8e-1e85-4ce1-91c2-adacac30e80c
md"""
#### Ideal Episode Example 5.3 LBF
"""

# ╔═╡ 3f61b8ad-5f70-48f1-b967-33da2169e0fb
md"""
Sweep episode timesteps
"""

# ╔═╡ ac51b4e7-3097-420d-a2dc-0d939dea5456
#=╠═╡
md"""
Play Movie: $(@bind timestep_select CheckBox())
"""
  ╠═╡ =#

# ╔═╡ 1fac0df2-9ce4-40d1-9296-a067906c2b01
md"""
#### Sarsa Solution

We can also apply estimation techniques from reinforcement learning that rely on environment sampling.  Below an example of using sarsa and its variations.
"""

# ╔═╡ e5ef0212-ff93-4e2b-a7c1-d7ff53cbb8aa
const lbf_sarsa = q_learning(tab_5_3, 0.99f0; max_steps = 1_000_000, α = 0.2f0, ϵ = 0.01f0, save_history = true)

# ╔═╡ b6df5773-94cd-488b-bb14-4713314a4575
function get_lbf_sarsa_statistics(algo; nruns = Base.Threads.nthreads(), kwargs...)
	1:nruns |> Map() do i
		output = algo(tab_5_3, 0.99f0; kwargs..., save_history = true)
		output.reward_history
	end |> foldxt((a, b) -> a .+ b) |> v -> v ./ nruns
end

# ╔═╡ 1d609b68-52a1-4459-a605-cae08785f306
const lbf_sarsa_avg = get_lbf_sarsa_statistics(sarsa; max_steps = 1_000_000, α = 0.2f0, ϵ = 0.01f0, save_history = true)

# ╔═╡ 7201de00-04d0-4eea-b814-08fe26fbabb5
const lbf_expected_sarsa_avg = get_lbf_sarsa_statistics(expected_sarsa; max_steps = 1_000_000, α = 0.2f0, ϵ = 0.1f0, save_history = true)

# ╔═╡ 2d15d9d3-4fec-498f-a1c0-9ae3c10f2d2f
const lbf_q_learning_avg = get_lbf_sarsa_statistics(q_learning; max_steps = 1_000_000, α = 0.2f0, ϵ = 0.01f0, save_history = true)

# ╔═╡ 09474fdf-df26-4e33-8f29-fc6ad9b85368
const lbf_double_q_learning_avg = get_lbf_sarsa_statistics(double_q_learning; max_steps = 1_000_000, α = 0.2f0, ϵ = 0.01f0, save_history = true)

# ╔═╡ 5a291aa8-5aeb-4427-aff2-c30a154c435d
#=╠═╡
function plot_episode_rewards(v::Vector{T}, n::Integer) where T<:Real
	inds = [i for i in n+1:n:length(v)]
	r_avg = inv.([mean(v[i-n:i]) for i in n+1:n:length(v)])
	tr = scatter(x = inds, y = r_avg, mode = "lines+markers")
	plot(tr, Layout(xaxis_title = "Environment time steps", yaxis_title = "Average Steps Over Last $n Episodes", yaxis_type = "log"))
end
  ╠═╡ =#

# ╔═╡ c9694e12-001f-4f14-867e-7ca1659d36a8
#=╠═╡
function plot_episode_rewards(vs::Vector{Vector{T}}, names::Vector{String}, n::Integer) where T<:Real
	traces = [begin
		inds = [i for i in n+1:n:length(v)]
		r_avg = inv.([mean(v[i-n:i]) for i in n+1:n:length(v)])
		tr = scatter(x = inds, y = r_avg, mode = "lines+markers", name = names[j])
	end
	for (j, v) in enumerate(vs)]
	plot(traces, Layout(xaxis_title = "Environment time steps", yaxis_title = "Average Steps Over Last $n Episodes", yaxis_type = "log"))
end
  ╠═╡ =#

# ╔═╡ 791a661b-77fd-4ddd-8b82-56c64dcc5c22
const value_iter_episode = runepisode(tab_5_3; π = value_iter.optimal_policy)

# ╔═╡ a113a20e-a599-4dbb-87a0-f943245d75ce
#=╠═╡
@bind timestep Slider(1:length(value_iter_episode[1])+1)
  ╠═╡ =#

# ╔═╡ 57d02e2c-4387-4345-bed3-504fafcd2a26
#=╠═╡
@bind movie_timestep Clock(;max_value = length(value_iter_episode[1])+1, repeat = true, interval = 0.4)
  ╠═╡ =#

# ╔═╡ cc259b71-cbe0-4990-ba6f-b495f6b0a2b7
md"""
# Independing Learning Algorithms
"""

# ╔═╡ ff335a51-e2ed-4313-af7b-1983c0a488a7
function independent_q_learning!((value_estimates, policies)::Tuple{NTuple{N, Matrix{T}}, NTuple{N, Matrix{T}}}, mdp::TabularStochasticGame{T, S, A, N, P, F}, γ::T, α::T, max_episodes::Integer, max_steps::Integer, policy_update!::Function; i_s0 = mdp.initialize_state_index(), save_history = false) where {T<:Real,S, A, P, F<:Function, N}
	ep = 1
	step = 0
	i_s = i_s0
	#there might be two policies in the case of off policy learning with a target and behavior policy.   the convention is that if there is a behavior policy that should be used to sample actions, it will be last
	a = Tuple(sample_action(policies[i], i_s) for i in 1:N)

	if save_history
		reward_history = Vector{NTuple{N, T}}()
		episode_steps = Vector{Int64}()
	end
	
	while (ep < max_episodes) && (step < max_steps)
		a = Tuple(sample_action(policies[i], i_s) for i in 1:N)
		rewards, i_s′ = mdp.ptf(i_s, a)
		step += 1
		
		for n in 1:N
			qs = value_estimates[n]
			r = rewards[n]
			i_a = a[n]
			qs[i_a, i_s] += α*(r + γ*maximum(view(qs, :, i_s′)) - qs[i_a, i_s])
			policy_update!(policies[n], qs, i_s)
		end
		

		if save_history
			push!(reward_history, rewards)
		end
		#if a terminal state is reached, need to reset episode
		if mdp.terminal_states[i_s′]
			save_history && push!(episode_steps, step)
			ep += 1
			i_s = mdp.initialize_state_index()
			a = Tuple(sample_action(policies[i], i_s) for i in 1:N)
		else
			i_s = i_s′
		end
	end
	basereturn = (value_estimates = value_estimates, policies = policies)
	!save_history && return basereturn
	(;basereturn..., reward_history = reward_history, episode_steps = episode_steps)
end

# ╔═╡ 3ab5ea43-f8d5-46ac-a407-fd7c9af50540
independent_q_learning(mdp::TabularStochasticGame{T, S, A, N, P, F}, γ::T; α::T = one(T)/10, ϵ::T = one(T)/10, max_steps = 100_000, max_episodes = typemax(Int64), init_value = zero(T), qs::NTuple{N, Matrix{T}} = Tuple(ones(T, length(mdp.agent_actions[i]), length(mdp.states)) .* init_value for i in 1:N), πs::NTuple{N, Matrix{T}} = Tuple(ones(T, length(mdp.agent_actions[i]), length(mdp.states)) ./ length(mdp.agent_actions[i]) for i in 1:N), kwargs...) where {T<:Real, S, A, N, P, F<:Function} = independent_q_learning!((qs, πs), mdp, γ, α, max_episodes, max_steps, (π, q, i_s) -> make_ϵ_greedy_policy!(π, i_s, q; ϵ = ϵ); kwargs...) 

# ╔═╡ d4661036-1e7a-48ba-a05a-51537b7b4910
const ex_5_3_iql = independent_q_learning(ex_5_3, 0.99f0; α = 0.01f0, max_steps = 1_000_000, save_history = true)

# ╔═╡ b27cbf6a-2202-40d8-8946-7d9fc6def3e9
function get_lbf_iql_statistics(;nruns = Base.Threads.nthreads(), kwargs...)
	1:nruns |> Map() do i
		output = independent_q_learning(ex_5_3, 0.99f0; kwargs..., save_history = true)
		[sum(a) for a in output.reward_history]
	end |> foldxt((a, b) -> a .+ b) |> v -> v ./ nruns
end

# ╔═╡ 13f1c44a-ca9b-42f2-9e6c-4d7dd1658af7
const lbf_iql_avg = get_lbf_iql_statistics(;max_steps = 1_000_000, α = 0.2f0, ϵ = 0.01f0, save_history = true)

# ╔═╡ 664429bb-d0fc-409f-9760-a14827c3159a
#=╠═╡
plot_episode_rewards([lbf_sarsa_avg, lbf_expected_sarsa_avg, lbf_q_learning_avg, lbf_double_q_learning_avg, lbf_iql_avg], ["Sarsa", "Expected Sarsa", "Q-learning", "Double Q-learning", "Independent Q-learning"], 30_000)
  ╠═╡ =#

# ╔═╡ 75a6b525-a507-4cf0-9ba7-ec1e76e86465
md"""
# Visualization Tools

"""

# ╔═╡ 9a0a63a1-0292-4937-9d7f-8c18bc3eb28b
md"""
## Level-Based Foraging
"""

# ╔═╡ a7acda55-6a77-4b38-91cc-5b19a18c5934
#=╠═╡
function plot_foraging_state(s::LevelBasedForaging.ForagingState{N, M}, xmax::Integer, ymax::Integer) where {N, M}
	bottom_border = scatter(x = [0, xmax+1], y = [0, 0], mode = "lines", line_color = "black", showlegend = false, name = "Edge", line_width = 3)
	top_border = scatter(x = [0, xmax+1], y = [ymax+1, ymax+1], mode = "lines", line_color = "black", showlegend = false, name = "Edge", line_width = 3)
	left_border = scatter(x = [0, 0], y = [0, ymax+1], mode = "lines", line_color = "black", showlegend = false, name = "Edge", line_width = 3)
	right_border = scatter(x = [xmax+1, xmax+1], y = [0, ymax+1], mode = "lines", line_color = "black", showlegend = false, name = "Edge", line_width = 3)
	agent_xs = [s.agent_positions[i][1] for i in 1:N]
	agent_ys = [s.agent_positions[i][2] for i in 1:N]
	agent_trace = scatter(x = agent_xs, y = agent_ys, name = "Agents", mode = "markers")

	item_xs = [s.item_positions[i][1] for i in 1:M]
	item_ys = [s.item_positions[i][2] for i in 1:M]
	item_trace = scatter(x = item_xs, y = item_ys, name = "Items", mode = "markers")

	plot([agent_trace, item_trace, bottom_border, top_border, left_border, right_border], Layout(xaxis_title = "x", yaxis_title = "y", xaxis_range = [0, xmax+1], yaxis_range = [0, ymax+1], yaxis_scaleanchor="x", width = 450, xaxis_dtick = 1, yaxis_dtick = 1, xaxis_tickvals = 1:xmax, xaxis_ticktext = string.(1:xmax), yaxis_tickvals = 1:ymax, yaxis_ticktext = string.(1:ymax), annotations = vcat([attr(x = s.agent_positions[i][1], y = s.agent_positions[i][2], text = s.agent_levels[i], showarrow = true,  font = attr(color = "blue", size = 14, weight = 1000, showdow = "auto")) for i in 1:N], [attr(x = s.item_positions[i][1], y = s.item_positions[i][2], text = s.item_levels[i], showarrow = true,  font = attr(color = "orange", size = 14, weight = 1000, showdow = "auto")) for i in 1:M])))
end
  ╠═╡ =#

# ╔═╡ 1f7c4a86-d530-4d17-9640-6c1c7315f1bc
#=╠═╡
function plot_lbf_episode(timestep)
	@htl("""
		 <div style = "display: flex; height: 420px;">
		 <div>
		 State Value Function: $(round(value_iter.final_value[vcat(value_iter_episode[1], value_iter_episode[4])[timestep]]; sigdigits = 3)), Numerical Annotations = Level
		 $(plot_foraging_state(tab_5_3.states[vcat(value_iter_episode[1], value_iter_episode[4])[timestep]], 11, 11))
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

# ╔═╡ fdcfd9d8-5c2e-4e6f-a408-7c530b1bc5ff
#=╠═╡
if timestep_select
	plot_lbf_episode(movie_timestep)
else
	plot_lbf_episode(timestep)
end
  ╠═╡ =#

# ╔═╡ 924d646f-8fd7-4b0b-a5c5-0446abb6434e
#=╠═╡
plot_foraging_state(ex_5_3.states[6188], 11, 11)
  ╠═╡ =#

# ╔═╡ 1a488045-380d-4441-a6ec-186d3c53420d
md"""
# Dependencies
"""

# ╔═╡ 5904f5e9-93b4-479d-9fc5-58efb00f8e01
html"""
<style>
	main {
		margin: 0 auto;
		max-width: min(1600px, 90%);
		padding-left: max(50px, 10%);
		padding-right: max(10px, 5%);
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
Dates = "ade2ca70-3891-5945-98fb-dc099432e06a"
HypertextLiteral = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
PlutoDevMacros = "a0499f29-c39b-4c5c-807c-88074221b949"
PlutoPlotly = "8e989ff0-3d88-8e9f-f020-2b208a939ff0"
PlutoProfile = "ee419aa8-929d-45cd-acf6-76bd043cd7ba"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"

[compat]
BenchmarkTools = "~1.6.3"
CSV = "~0.10.15"
DataFrames = "~1.8.1"
HypertextLiteral = "~0.9.5"
LaTeXStrings = "~1.4.0"
PlutoDevMacros = "~0.9.2"
PlutoPlotly = "~0.6.5"
PlutoProfile = "~0.4.0"
PlutoUI = "~0.7.75"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.12.2"
manifest_format = "2.0"
project_hash = "170a5c24fc6462387ae568ecfb37939d5fab920d"

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

[[deps.CSV]]
deps = ["CodecZlib", "Dates", "FilePathsBase", "InlineStrings", "Mmap", "Parsers", "PooledArrays", "PrecompileTools", "SentinelArrays", "Tables", "Unicode", "WeakRefStrings", "WorkerUtilities"]
git-tree-sha1 = "deddd8725e5e1cc49ee205a1964256043720a6c3"
uuid = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
version = "0.10.15"

[[deps.CodeTracking]]
deps = ["InteractiveUtils", "UUIDs"]
git-tree-sha1 = "b7231a755812695b8046e8471ddc34c8268cbad5"
uuid = "da1fd8a2-8d9e-5ec2-8556-3022fb5608a2"
version = "3.0.0"

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

    [deps.ColorVectorSpace.extensions]
    SpecialFunctionsExt = "SpecialFunctions"

    [deps.ColorVectorSpace.weakdeps]
    SpecialFunctions = "276daf66-3868-5448-9aa4-cd146d93841b"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "362a287c3aa50601b0bc359053d5c2468f0e7ce0"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.11"

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
git-tree-sha1 = "d60eb76f37d7e5a40cc2e7c36974d864b82dc802"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.17.1"

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

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"
version = "1.11.0"

[[deps.HashArrayMappedTries]]
git-tree-sha1 = "2eaa69a7cab70a52b9687c8bf950a5a93ec895ae"
uuid = "076d061b-32b6-4027-95e0-9a2c6f6d7e74"
version = "0.2.0"

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

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.JuliaInterpreter]]
deps = ["CodeTracking", "InteractiveUtils", "Random", "UUIDs"]
git-tree-sha1 = "c98fa9e3a241e92895be025614ec33eca0ffa5f7"
uuid = "aa1ae85d-cabe-5617-a682-6adf51b2e16a"
version = "0.10.8"

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
version = "2025.5.20"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.3.0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.29+0"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.5.4+0"

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
version = "1.12.0"
weakdeps = ["REPL"]

    [deps.Pkg.extensions]
    REPLExt = "REPL"

[[deps.PlotlyBase]]
deps = ["ColorSchemes", "Colors", "Dates", "DelimitedFiles", "DocStringExtensions", "JSON", "LaTeXStrings", "Logging", "Parameters", "Pkg", "REPL", "Requires", "Statistics", "UUIDs"]
git-tree-sha1 = "49c457ee4c9c6f5bdf2f6f1a69e66976aaecfcdb"
uuid = "a03496cd-edff-5a9b-9e67-9cda94a718b5"
version = "0.8.22"

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
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Downloads", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "db8a06ef983af758d285665a0398703eb5bc1d66"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.75"

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
git-tree-sha1 = "0f27480397253da18fe2c12a4ba4eb9eb208bf3d"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.5.0"

[[deps.PrettyTables]]
deps = ["Crayons", "LaTeXStrings", "Markdown", "PrecompileTools", "Printf", "REPL", "Reexport", "StringManipulation", "Tables"]
git-tree-sha1 = "c5a07210bd060d6a8491b0ccdee2fa0235fc00bf"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "3.1.2"

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
git-tree-sha1 = "c3b2323466378a2ba15bea4b2f73b081e022f473"
uuid = "7e506255-f358-4e82-b7e4-beb19740aa63"
version = "1.5.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "9b81b8393e50b7d4e6d0a9f14e192294d3b7c109"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.3.0"

[[deps.SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "712fb0231ee6f9120e005ccd56297abbc053e7e0"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.4.8"

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

[[deps.Statistics]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "ae3bb1eb3bba077cd276bc5cfc337cc65c3075c0"
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.11.1"

    [deps.Statistics.extensions]
    SparseArraysExt = ["SparseArrays"]

    [deps.Statistics.weakdeps]
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.StringManipulation]]
deps = ["PrecompileTools"]
git-tree-sha1 = "a3c1536470bf8c5e02096ad4853606d7c8f62721"
uuid = "892a3eda-7b42-436c-8928-eab12a02cf0e"
version = "0.4.2"

[[deps.StyledStrings]]
uuid = "f489334b-da3d-4c2e-b8f0-e476e12c162b"
version = "1.11.0"

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
# ╟─73ddfc82-cf10-11f0-a45e-3da27f44d6bc
# ╟─04715feb-f04a-4a4a-b7c5-76180e0508df
# ╠═f6de9a64-7dee-4291-815b-7a891c52a146
# ╠═fe796b4e-fb05-4071-ad4b-7e3ce58092c4
# ╠═85251e1c-6704-4dc5-85dd-0c8d9706358e
# ╠═c42b7d58-922d-4b17-a03c-62433c9adeb9
# ╠═6f98c460-6d48-457d-a40c-6b8588fb01ef
# ╠═ed41ec94-38be-40de-bf61-c927c242cef7
# ╠═b815429b-fbe3-443c-9a76-d4b194eca26d
# ╠═23a5f4c3-5f6a-473a-9ee5-0c82cf8c7f8e
# ╠═75726eb9-eb0e-42a3-a648-dbfce92cbd6e
# ╟─ed152a1a-0fc9-40f3-91b4-e40246725847
# ╠═79888a7b-c604-4db2-932a-ee9bfd378d9e
# ╠═faac6925-b526-4ee0-a84c-b660e6819313
# ╠═776a9ef9-8477-449f-b195-2cc5d7c93749
# ╠═45668f60-2dbc-48d4-9903-111b80665845
# ╟─04b35d98-e60f-4e3c-b89d-891f7139b5ec
# ╠═738eadf8-7bbf-4942-a90d-e8accbb52bea
# ╟─ae213605-c98f-4939-8e29-48eea2ff5ed8
# ╠═736aacc0-5592-4439-b8c3-cf76525983a5
# ╠═39182e4e-6ee9-4c14-8b5f-3164e38d5b5a
# ╠═0bb34d75-0422-4336-9e62-d6431ce2b1c3
# ╟─ae85884a-a23a-48f3-947e-19fc4a21d39f
# ╟─660d62d3-b967-41ba-8dab-8d1003dcd1fc
# ╠═79b114c9-8999-4e78-86b7-8bc0bd81dcad
# ╠═0eec6fce-eefb-48cb-8b1e-0d77bfcb1129
# ╠═b149a025-3b7b-4ee5-a4a5-a6bcd035ad8b
# ╟─c1cc33d2-ad34-4ad8-b47a-6be3aa44c8ed
# ╠═ac0a9ef8-6fbc-42f2-9433-727b235224e5
# ╟─63062544-0375-4de4-86e3-385347033e5d
# ╟─a91367b5-2bbf-4f17-9e04-5c939223c590
# ╠═dac84639-2388-4ee8-9372-aae518a37f17
# ╟─c46aa7de-2b71-4abe-b420-17802f027ab4
# ╠═83583ca7-bd96-4589-bbe0-e967849c3bd3
# ╟─aa1faa8e-1e85-4ce1-91c2-adacac30e80c
# ╟─3f61b8ad-5f70-48f1-b967-33da2169e0fb
# ╟─a113a20e-a599-4dbb-87a0-f943245d75ce
# ╟─57d02e2c-4387-4345-bed3-504fafcd2a26
# ╟─ac51b4e7-3097-420d-a2dc-0d939dea5456
# ╠═fdcfd9d8-5c2e-4e6f-a408-7c530b1bc5ff
# ╟─1fac0df2-9ce4-40d1-9296-a067906c2b01
# ╠═e5ef0212-ff93-4e2b-a7c1-d7ff53cbb8aa
# ╠═1d609b68-52a1-4459-a605-cae08785f306
# ╠═7201de00-04d0-4eea-b814-08fe26fbabb5
# ╠═2d15d9d3-4fec-498f-a1c0-9ae3c10f2d2f
# ╠═09474fdf-df26-4e33-8f29-fc6ad9b85368
# ╠═b6df5773-94cd-488b-bb14-4713314a4575
# ╠═5a291aa8-5aeb-4427-aff2-c30a154c435d
# ╠═c9694e12-001f-4f14-867e-7ca1659d36a8
# ╠═791a661b-77fd-4ddd-8b82-56c64dcc5c22
# ╠═1f7c4a86-d530-4d17-9640-6c1c7315f1bc
# ╟─cc259b71-cbe0-4990-ba6f-b495f6b0a2b7
# ╠═ff335a51-e2ed-4313-af7b-1983c0a488a7
# ╠═3ab5ea43-f8d5-46ac-a407-fd7c9af50540
# ╠═d4661036-1e7a-48ba-a05a-51537b7b4910
# ╠═b27cbf6a-2202-40d8-8946-7d9fc6def3e9
# ╠═13f1c44a-ca9b-42f2-9e6c-4d7dd1658af7
# ╠═664429bb-d0fc-409f-9760-a14827c3159a
# ╟─75a6b525-a507-4cf0-9ba7-ec1e76e86465
# ╟─9a0a63a1-0292-4937-9d7f-8c18bc3eb28b
# ╠═924d646f-8fd7-4b0b-a5c5-0446abb6434e
# ╠═a7acda55-6a77-4b38-91cc-5b19a18c5934
# ╟─1a488045-380d-4441-a6ec-186d3c53420d
# ╠═4e509812-7bdf-4928-bee6-55f2d142be67
# ╠═9260398e-6842-4b11-9845-98900884b94a
# ╠═9f1aa27d-2214-4b7c-9110-1566603887d2
# ╠═5904f5e9-93b4-479d-9fc5-58efb00f8e01
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
