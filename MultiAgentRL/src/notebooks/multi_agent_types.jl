### A Pluto.jl notebook ###
# v0.20.25

using Markdown
using InteractiveUtils

# ╔═╡ 4e509812-7bdf-4928-bee6-55f2d142be67
using PlutoDevMacros

# ╔═╡ b9ee4c0e-e556-4f67-9cce-533b1f35ed96
using Dates, CSV

# ╔═╡ 9f1aa27d-2214-4b7c-9110-1566603887d2
# ╠═╡ skip_as_script = true
#=╠═╡
begin
	using PlutoUI, PlutoPlotly, PlutoProfile, BenchmarkTools, LaTeXStrings, HypertextLiteral, DataFrames
	
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

# ╔═╡ b815429b-fbe3-443c-9a76-d4b194eca26d
# ╠═╡ skip_as_script = true
#=╠═╡
#example transitions for two agents with 6 actions each
ex_reward_transitions = rand(6, 6, 100)
  ╠═╡ =#

# ╔═╡ 23a5f4c3-5f6a-473a-9ee5-0c82cf8c7f8e
#=╠═╡
#joint action space for single agent RL is 36 representing all combinations of agent actions
reshape(ex_reward_transitions, 36, 100)
  ╠═╡ =#

# ╔═╡ ed152a1a-0fc9-40f3-91b4-e40246725847
md"""
## Tabular Multi-Agent Stochastic Games

To form a complete stochastic game, the transition function must be accompanied by the enumerated states, agent actions, state initialization function, and terminal states indicator.  For convenience, lookup tables are stored to easy find the index of any state and agent action.
"""

# ╔═╡ 04b35d98-e60f-4e3c-b89d-891f7139b5ec
md"""
## Non-tabular State Transition Functions

The step function takes a state and joint action as input.  The joint action is represented by a tuple of integers of length equal to the number of agents.  Each index in the tuple refers to the index of the selected action for that agent from its list of available actions.
"""

# ╔═╡ ae213605-c98f-4939-8e29-48eea2ff5ed8
md"""
## Non-tabular Multi-Agent Stochastic Games

For the non-tabular case, the list of states is omitted and now the initialization function returns a state rather than a state index.  Similarly, the list of terminal states is replaced by a function that can evaluate whether or not a state is terminal.
"""

# ╔═╡ ae85884a-a23a-48f3-947e-19fc4a21d39f
md"""
# Test Environments
"""

# ╔═╡ 660d62d3-b967-41ba-8dab-8d1003dcd1fc
md"""
## Level-Based Foraging

Multiple agents move within a gridworld and have the option to collect items.  For randomly generated environments, items must be placed so none occupy the same grid point and an agent can never be adjacent to two items at once (otherwise the collect action is ambiguous).  Each item is associated with a set of 4 or fewer locations from which it is possible to collect (adjacent locations).  These collection locations can be saved in the state to speed up evaluation of the step function.
"""

# ╔═╡ c1cc33d2-ad34-4ad8-b47a-6be3aa44c8ed
md"""
### Tabular Environment
"""

# ╔═╡ 63062544-0375-4de4-86e3-385347033e5d
md"""
### Central Learning Reductions
"""

# ╔═╡ a91367b5-2bbf-4f17-9e04-5c939223c590
md"""
Using the sum of agent rewards as the scalar transformation, we can convert this multi-agent stochastic game into a Tabular MDP
"""

# ╔═╡ c46aa7de-2b71-4abe-b420-17802f027ab4
md"""
#### Value Iteration Solution

With a tabular MDP, we can apply an exactly solution technique such as value iteration.
"""

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

# ╔═╡ 775fd220-532d-49ba-968f-20781c5d6708
md"""
## Two-Player Soccer

Each episode starts with the ball randomly assigned to one agent and the two agents on their respective side of the "field" offset vertically by one square.  Each agent can move in the four cardinal grid directions or stand still.  If the agent with the ball ateempts to move into the location of the other agent, it loses the ball to the other agent.  An agent scores a reward of +1 if it moves the ball into the opponent goal (and the opponent gets a reward of -1), after which a new episode starts.  Selected actions are executed in a random order.
"""

# ╔═╡ 91c667a3-293d-44d7-9437-901274f27af9
md"""
## Non-Repeated Zero-Sum Games
"""

# ╔═╡ b53e1cfd-4f81-4e1e-bff0-3435d3fcd221
md"""
## Rock-Paper-Scissors
"""

# ╔═╡ c2bec8e1-9df1-4f5f-8e92-0f6f5f9955a4
md"""
## Non-Repeated General-Sum Games
"""

# ╔═╡ cc259b71-cbe0-4990-ba6f-b495f6b0a2b7
md"""
# Independing Learning Algorithms
"""

# ╔═╡ 50422bb8-ff98-4803-8a3b-9a42454d793d
function get_max_q(q::Matrix{T}, i_s::Integer) where T<:Real
	maxq = typemin(T)
	@inbounds @simd for i_a in 1:size(q, 1)
		maxq = max(maxq, q[i_a, i_s])
	end
	return maxq
end

# ╔═╡ eec99b81-8150-4a65-90a0-283cd5da877a
md"""
## Level-Based Foraging Test (General Sum Game)
"""

# ╔═╡ 8c6f4ab7-9172-4723-bc2b-e61fe0315e1c
md"""
## Soccer Game Test (Zero Sum Game)
"""

# ╔═╡ 75a6b525-a507-4cf0-9ba7-ec1e76e86465
md"""
# Visualization Tools

"""

# ╔═╡ 9a0a63a1-0292-4937-9d7f-8c18bc3eb28b
md"""
## Level-Based Foraging
"""

# ╔═╡ 1a488045-380d-4441-a6ec-186d3c53420d
md"""
# Dependencies
"""

# ╔═╡ 9260398e-6842-4b11-9845-98900884b94a
@only_in_nb @frompackage "../RL_Module" import *

# ╔═╡ f6de9a64-7dee-4291-815b-7a891c52a146
begin
	"""
		AbstractGameTransition{T, N}

	Abstract type for game transition functions in multi-agent stochastic games with N agents.

	Subtypes define how joint actions and states map to next states and per-agent rewards.

	# Subtypes
	- [`AbstractTabularGameTransition`](@ref) — tabular (state-indexed) transitions
	- [`AbstractStateGameTransition`](@ref) — non-tabular (state-based) transitions
	"""
	abstract type AbstractGameTransition{T<:Real, N} end

	"""
		AbstractTabularGameTransition{T, N} <: AbstractGameTransition{T, N}

	Abstract type for tabular game transition functions where states are indexed by integers.

	# Subtypes
	- [`TabularGameTransition`](@ref) — lookup-table based transitions
	- [`TabularGameTransitionSampler`](@ref) — sampling-based transitions
	- [`AbstractTabularZeroSumGameTransition`](@ref) — two-agent zero-sum games
	- [`AbstractTabularCommonRewardGameTransition`](@ref) — common reward games
	"""
	abstract type AbstractTabularGameTransition{T<:Real, N} <: AbstractGameTransition{T, N} end
	
	"""
		TabularGameTransition{T, N, Np1, ST, RT} <: AbstractTabularGameTransition{T, N}

	Tabular game transition function for N agents, storing state and reward
	transition maps as N+1 dimensional arrays.

	The first N dimensions index each agent's actions, and the last dimension
	indexes the current state.

	# Fields
	- `state_transition_map::Array{ST, Np1}`: Joint action → next state mapping
	- `reward_transition_map::Array{NTuple{N, RT}, Np1}`: Joint action → per-agent rewards
	"""
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

# ╔═╡ 28920a55-4e8c-4dfc-84a0-03eeb13b9975
begin
	"""
		TabularGameTransitionSampler{T, N, F} <: AbstractTabularGameTransition{T, N}

	Tabular game transition function for N agents that samples next states and
	rewards from a provided step function that returns the rewards and index of the next state.

	# Constructor
		TabularGameTransitionSampler(step::Function, joint_action::NTuple{N, Integer})

	where `step` has the signature:

		step(i_s::Integer, joint_action::NTuple{N, Integer}) -> rewards::NTuple{N, Real}, i_s′::Integer

	# Fields
	- `step::F`: Function `(i_s, joint_action) -> (rewards, i_s′)`
	"""
	struct TabularGameTransitionSampler{T<:Real, N, F<:Function} <: AbstractTabularGameTransition{T, N}
		step::F
		function TabularGameTransitionSampler(step::F, a::NTuple{N, I}) where {F<:Function, N, I<:Integer}
			(rewards, i_s′) = step(1, a)
			@assert isa(i_s′, Integer)
			@assert length(rewards) == N
			new{eltype(rewards), N, F}(step)
		end
	end

	#when used as a functor just apply the step function to the state action pair indices
	(ptf::TabularGameTransitionSampler{T, N, F})(i_s::Integer, a::NTuple{N, I}) where {T<:Real, F<:Function, N, I<:Integer} = ptf.step(i_s, a)
end

# ╔═╡ 0b20d83b-eef0-4006-891e-9d15282ea57b
begin
	"""
		AbstractTabularCommonRewardGameTransition{T, N} <: AbstractTabularGameTransition{T, N}

	Abstract type for tabular game transitions where all agents share the same reward signal.

	# Subtypes
	- [`TabularCommonRewardGameTransition`](@ref) — lookup-table based
	- [`TabularCommonRewardGameTransitionSampler`](@ref) — sampling-based
	"""
	abstract type AbstractTabularCommonRewardGameTransition{T<:Real, N} <: AbstractTabularGameTransition{T, N} end 

	"""
		AbstractTabularZeroSumGameTransition{T} <: AbstractTabularGameTransition{T, 2}

	Abstract type for tabular zero-sum game transitions with two agents,
	where one agent's reward is the negative of the other's.

	# Subtypes
	- [`TabularZeroSumGameTransition`](@ref) — lookup-table based
	- [`TabularZeroSumGameTransitionSampler`](@ref) — sampling-based
	"""
	abstract type AbstractTabularZeroSumGameTransition{T<:Real} <: AbstractTabularGameTransition{T, 2} end 
end

# ╔═╡ 71d96c8a-d036-45a9-95e4-bbf4ee079607
begin
	"""
		TabularZeroSumGameTransitionSampler{T, F} <: AbstractTabularZeroSumGameTransition{T}

	Tabular zero-sum game transition for two agents that samples next states and
	rewards from a user-provided step function.

	The inner constructor validates the step function by calling it with state
	index 1 and the joint action `(1, 1)`, asserting that the returned `i_s′` is an
	integer.  The reward type `T` is inferred from this test call.

	# Constructor
		TabularZeroSumGameTransitionSampler(step)

	where `step` has the signature:

		step(i_s::Integer, joint_action::NTuple{2, I}) -> reward::T, i_s′::Integer

	where `joint_action` is a tuple of two action indices.

	# Fields
	- `step::F`: The stored step function
	"""
	struct TabularZeroSumGameTransitionSampler{T<:Real, F<:Function} <: AbstractTabularZeroSumGameTransition{T}
		step::F
		function TabularZeroSumGameTransitionSampler(step::F) where {F<:Function}
			(r, i_s′) = step(1, (1, 1))
			@assert isa(i_s′, Integer)
			new{typeof(r), F}(step)
		end
	end

	#when used as a functor just apply the step function to the state action pair indices
	(ptf::TabularZeroSumGameTransitionSampler{T, F})(i_s::Integer, a::NTuple{2, I}) where {T<:Real, F<:Function, I<:Integer} = ptf.step(i_s, a)
end

# ╔═╡ c9b36b87-4a88-4398-95ce-4feb4f9839f7
begin
	"""
		TabularCommonRewardGameTransitionSampler{T, N, F} <: AbstractTabularCommonRewardGameTransition{T, N}

	Tabular common-reward game transition for N agents that samples next states and
	rewards from a user-provided step function.

	The inner constructor validates the step function by calling it with state
	index 1 and an example joint action, asserting that the returned `i_s′` is an
	integer.  The reward type `T` and agent count `N` are inferred from this test
	call.

	# Constructor
		TabularCommonRewardGameTransitionSampler(step, example_joint_action)

	where `step` has the signature:

		step(i_s::Integer, joint_action::NTuple{N, I}) -> reward::T, i_s′::Integer

	where `example_joint_action` is an example joint action used to infer `N`.

	# Fields
	- `step::F`: The stored step function
	"""
	struct TabularCommonRewardGameTransitionSampler{T<:Real, N, F<:Function} <: AbstractTabularCommonRewardGameTransition{T, N}
		step::F
		function TabularCommonRewardGameTransitionSampler(step::F, a::NTuple{N, I}) where {F<:Function, I<:Integer, N}
			(r, i_s′) = step(1, a)
			@assert isa(i_s′, Integer)
			new{typeof(r), N, F}(step)
		end
	end

	#when used as a functor just apply the step function to the state action pair indices
	(ptf::TabularCommonRewardGameTransitionSampler{T, N, F})(i_s::Integer, a::NTuple{N, I}) where {T<:Real, F<:Function, I<:Integer, N} = ptf.step(i_s, a)
end

# ╔═╡ 9b75d562-ac45-406a-867b-b6d2af5822ee

begin
	"""
		get_game_reward(ptf, rewards, agent_index)

	Extract the reward for a given agent from the transition function's reward output,
	handling the different reward representations for each game type.
	
	- **General-sum tabular games**: `rewards` is an `NTuple{N, T}` — returns `rewards[agent_index]`.
	- **Zero-sum tabular games**: `rewards` is a scalar `T` — returns `r` for agent 1 and `-r` for agent 2.
	- **Common-reward tabular games**: `rewards` is a scalar `T` — returns `r` regardless of agent index.
	"""
	get_game_reward(ptf, rewards::NTuple{N, T}, n::Integer) where {N, T<:Real} = rewards[n]
	get_game_reward(::AbstractTabularZeroSumGameTransition{T}, r::T, ::Val{1}) where T<:Real = r
	get_game_reward(::AbstractTabularZeroSumGameTransition{T}, r::T, ::Val{2}) where T<:Real = -r
	get_game_reward(ptf::AbstractTabularZeroSumGameTransition{T}, r::T, n::Integer) where T<:Real = get_game_reward(ptf, r, Val(n))
	get_game_reward(::AbstractTabularCommonRewardGameTransition{T}, r::T, ::Integer) where T<:Real = r
end

# ╔═╡ 75726eb9-eb0e-42a3-a648-dbfce92cbd6e
"""
	AbstractStochasticGame{T, S, A, N, P, F}

Abstract type for a complete stochastic game including state space, agent actions,
transition function, and initialization.

# Type Parameters
- `T`: Numeric type for rewards
- `S`: State type
- `A`: Action type
- `N`: Number of agents
- `P`: Transition function type
- `F`: Initialization function type

# Subtypes
- [`TabularStochasticGame`](@ref) — tabular (enumerated states)
- [`StateStochasticGame`](@ref) — state-based (non-tabular)
"""
abstract type AbstractStochasticGame{T<:Real, S, A, N, P<:AbstractGameTransition{T, N}, F<:Function} end

# ╔═╡ 9908b24b-77c4-4ece-99b5-337dc6725a13
"""
	get_other_inds(ptf, agent_index)

Return a tuple of the indices of other agents and the state index dimension
in the transition array.

# Returns
- `(other_inds, state_index)` where `other_inds` contains the dimensions
  corresponding to agents other than `agent_index`, and `state_index` is the
  dimension index of the state (always `N+1`).
"""
get_other_inds(ptf::AbstractTabularGameTransition{T, N}, agent_index::Integer) where {T<:Real, N} = (setdiff(1:N, agent_index), N+1)

# ╔═╡ 738eadf8-7bbf-4942-a90d-e8accbb52bea
begin
	"""
		AbstractStateGameTransition{T, S, F, N} <: AbstractGameTransition{T, N}

	Abstract type for state-based (non-tabular) game transition functions,
	where states are arbitrary objects rather than integer indices.

	# Subtypes
	- [`StateGameTransitionDeterministic`](@ref) — general-sum deterministic
	- [`StateZeroSumGameTransitionDeterministic`](@ref) — two-agent zero-sum
	- [`StateCommonRewardGameTransitionDeterministic`](@ref) — common reward
	"""
	abstract type AbstractStateGameTransition{T<:Real, S, F<:Function, N} <: AbstractGameTransition{T, N} end

	"""
		StateGameTransitionDeterministic{T, S, F, N} <: AbstractStateGameTransition{T, S, F, N}

	State-based deterministic game transition for N agents with per-agent
	rewards returned as a tuple.

	The inner constructor validates the step function by calling it with an
	example state and the joint action `ntuple(Returns(1), num_agents)` (every
	agent selects action index 1).  It asserts that `s′` shares a common type
	with `s` and that the reward tuple contains exactly `num_agents` elements.
	The type parameters `T`, `S`, and `N` are inferred from this test call.

	# Constructor
		StateGameTransitionDeterministic(step, s, num_agents; test_joint_action = (1, 1, ...))

	Option to pass a joint action to the constructor other than the first agent action for each agent

	where `step` has the signature:

		step(s::S, joint_action::NTuple{N, I}) -> rewards::NTuple{N, T}, s′::S

	where `joint_action` is a tuple of action indices, one per agent.

	# Fields
	- `step::F`: The stored step function
	"""
	struct StateGameTransitionDeterministic{T<:Real, S, F<:Function, N} <: AbstractStateGameTransition{T, S, F, N}
		step::F
		function StateGameTransitionDeterministic(step::F, s::S, num_agents::Integer; test_joint_action::NTuple{N, Integer} = ntuple(Returns(1), num_agents)) where {N, F<:Function, S}
			(r, s′) = step(s, test_joint_action)
			@assert promote_type(S, typeof(s′)) != Any "There is no common type between the provided state $s and the transition state $s′"
			@assert length(r) == num_agents "The reward output length of $(length(r)) does not match the expected number of agents: $num_agents"
			new{eltype(r), promote_type(S, typeof(s′)), F, num_agents}(step)
		end
	end

	(ptf::StateGameTransitionDeterministic{T, S, F, N})(s::S, a::NTuple{N, I}) where {T<:Real, S, F<:Function, N, I<:Integer} = ptf.step(s, a)

	"""
		StateZeroSumGameTransitionDeterministic{T, S, F} <: AbstractStateGameTransition{T, S, F, 2}

	State-based deterministic zero-sum game transition for two agents,
	where a single scalar reward is returned (positive for agent 1,
	negative for agent 2).

	The inner constructor validates the step function by calling it with an
	example state and the joint action `(1, 1)`.  It asserts that the returned
	reward is a scalar real value and that `s′` shares a common type with `s`.
	The type parameters `T` and `S` are inferred from this test call.

	# Constructor
		StateZeroSumGameTransitionDeterministic(step, s; test_joint_action = (1, 1))

	Option to pass a joint action to the constructor other than the first agent action for each agent

	where `step` has the signature:

		step(s::S, joint_action::NTuple{2, I}) -> reward::T, s′::S

	where `joint_action` is a tuple of two action indices.

	# Fields
	- `step::F`: The stored step function
	"""
	struct StateZeroSumGameTransitionDeterministic{T<:Real, S, F<:Function} <: AbstractStateGameTransition{T, S, F, 2}
		step::F
		function StateZeroSumGameTransitionDeterministic(step::F, s::S; test_joint_action::Tuple{Integer, Integer} = (1, 1)) where {F<:Function, S}
			(r, s′) = step(s, test_joint_action)
			@assert r isa Real "The reward output must be a scalar real value representing agent 1's reward"
			@assert promote_type(S, typeof(s′)) != Any "There is no common type between the provided state $s and the transition state $s′"
			new{typeof(r), promote_type(S, typeof(s′)), F}(step)
		end
	end

	(ptf::StateZeroSumGameTransitionDeterministic{T, S, F})(s::S, a::NTuple{2, I}) where {T<:Real, S, F<:Function, I<:Integer} = ptf.step(s, a)

	"""
		StateCommonRewardGameTransitionDeterministic{T, S, F, N} <: AbstractStateGameTransition{T, S, F, N}

	State-based deterministic common-reward game transition for N agents,
	where all agents receive the same scalar reward.

	The inner constructor validates the step function by calling it with an
	example state and the joint action `ntuple(Returns(1), num_agents)` (every
	agent selects action index 1).  It asserts that the returned reward is a
	scalar real value and that `s′` shares a common type with `s`.  The type
	parameters `T`, `S`, and `N` are inferred from this test call.

	# Constructor
		StateCommonRewardGameTransitionDeterministic(step, s, num_agents; test_joint_action = (1, 1,...))

	Option to pass a joint action to the constructor other than the first agent action for each agent

	where `step` has the signature:

		step(s::S, joint_action::NTuple{N, I}) -> reward::T, s′::S

	where `joint_action` is a tuple of action indices, one per agent.

	# Fields
	- `step::F`: The stored step function
	"""
	struct StateCommonRewardGameTransitionDeterministic{T<:Real, S, F<:Function, N} <: AbstractStateGameTransition{T, S, F, N}
		step::F
		function StateCommonRewardGameTransitionDeterministic(step::F, s::S, num_agents::Integer; test_joint_action::NTuple{N, Integer} = ntuple(Returns(1), num_agents)) where {F<:Function, S, N}
			(r, s′) = step(s, test_joint_action)
			@assert r isa Real "The reward output must be a scalar real value shared by all agents"
			@assert promote_type(S, typeof(s′)) != Any "There is no common type between the provided state $s and the transition state $s′"
			new{typeof(r), promote_type(S, typeof(s′)), F, num_agents}(step)
		end
	end

	(ptf::StateCommonRewardGameTransitionDeterministic{T, S, F, N})(s::S, a::NTuple{N, I}) where {T<:Real, S, F<:Function, N, I<:Integer} = ptf.step(s, a)
end

# ╔═╡ c755f0b1-2299-42b4-a28d-d53980260e79
begin
	"""
		initialize_reward_history(ptf)
	
	Create an empty reward history vector appropriate for the given transition type.
	
	- For general-sum games (N agents): returns `Vector{NTuple{N, T}}()` to store per-agent reward tuples.
	- For zero-sum games: returns `Vector{T}()` to store scalar rewards.
	- For common-reward games: returns `Vector{T}()` to store shared scalar rewards.
	"""
	initialize_reward_history(::AbstractTabularGameTransition{T, N}) where {N, T<:Real} = Vector{NTuple{N, T}}()
	initialize_reward_history(::AbstractTabularZeroSumGameTransition{T}) where T<:Real = Vector{T}()
	initialize_reward_history(::AbstractTabularCommonRewardGameTransition{T}) where T<:Real = Vector{T}()
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

# ╔═╡ 11b17cb5-30fa-4a66-bf71-52b4dc9f2f02
"""
	sample_joint_action(πs, i_s)

Sample a joint action from per-agent policies at state index `i_s`.

Each element of the returned tuple is the action index sampled from the
corresponding agent's policy matrix.

# Arguments
- `πs::NTuple{N, AbstractMatrix{T}}`: Per-agent policy matrices, where
  `πs[n][i_a, i_s]` is the probability agent `n` selects action index `i_a` in state index `i_s`.
- `i_s::Integer`: The current state index.

# Returns
- `joint_action::NTuple{N, Int64}`: A tuple of action indices, one per agent.
"""
sample_joint_action(πs::NTuple{N, M}, i_s::Integer) where {N, T<:Real, M <:AbstractMatrix{T}} = NTuple{N, Int64}(sample_action(πs[n], i_s) for n in 1:N)

# ╔═╡ dd64a78d-a8f1-4b9b-aa8b-95ee48936268
function (ptf::AbstractGameTransition{T, N})(s, πs::NTuple{N, A}) where {T<:Real, A<:AbstractMatrix{T}, N}
	a = sample_joint_action(πs, s)
	(r, s′) = ptf(s, a)
	return (r, s′, a)
end

# ╔═╡ c42b7d58-922d-4b17-a03c-62433c9adeb9
begin
	"""
		TabularZeroSumGameTransition{T, ST, RT} <: AbstractTabularZeroSumGameTransition{T}

	Tabular two-player zero-sum game transition storing state and reward maps
	as 3-dimensional arrays indexed by (agent 1 action index, agent 2 action index, state index).

	The reward is a single scalar value: positive for agent 1, negative for agent 2.

	The following type aliases fix the `ST` and `RT` parameters for common cases:
	- `TabularZeroSumGameDeterministicTransition{T}` = `TabularZeroSumGameTransition{T, Int64, T}`
	  (deterministic with `ST = Int64` for single next state index, `RT = T` for scalar reward)
	- `TabularZeroSumGameStochasticTransition{T}` = `TabularZeroSumGameTransition{T, SparseVector{T, Int64}, Vector{T}}`
	  (stochastic with `ST = SparseVector` for transition probabilities, `RT = Vector{T}` for per-outcome rewards)

	# Fields
	- `state_transition_map::Array{ST, 3}`: Next state index indexed by (agent 1 action index, agent 2 action index, current state index)
	- `reward_transition_map::Array{RT, 3}`: Scalar reward indexed by (agent 1 action index, agent 2 action index, current state index)
	"""
	struct TabularZeroSumGameTransition{T<:Real, ST<:Union{Int64, SparseVector{T, Int64}}, RT<:Union{T, Vector{T}}} <: AbstractTabularZeroSumGameTransition{T}
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

	function (ptf::TabularZeroSumGameStochasticTransition{T})(i_s::Integer, a::NTuple{2, I}) where {T<:Real, I<:Integer}
		state_transition_probabilities = ptf.state_transition_map[a[1], a[2], i_s]
		i = sample_action(state_transition_probabilities.nzval)
		i_s′ = state_transition_probabilities.nzind[i]
		r = ptf.reward_transition_map[a[1], a[2], i_s][i]
		(r, i_s′)
	end
end

# ╔═╡ 6f98c460-6d48-457d-a40c-6b8588fb01ef
begin
	"""
		TabularCommonRewardGameTransition{T, N, Np1, ST, RT} <: AbstractTabularCommonRewardGameTransition{T, N}

	Tabular common-reward game transition for N agents storing state and reward
	maps as N+1 dimensional arrays indexed by
	(agent 1 action index, ..., agent N action index, state index).

	All agents share the same scalar reward value.

	The following type aliases fix the `ST` and `RT` parameters for common cases:
	- `TabularCommonRewardGameDeterministicTransition{T, N, Np1}` = `TabularCommonRewardGameTransition{T, N, Np1, Int64, T}`
	  (deterministic with `ST = Int64` for single next state index, `RT = T` for scalar reward)
	- `TabularCommonRewardGameStochasticTransition{T, N, Np1}` = `TabularCommonRewardGameTransition{T, N, Np1, SparseVector{T, Int64}, Vector{T}}`
	  (stochastic with `ST = SparseVector` for transition probabilities, `RT = Vector{T}` for per-outcome rewards)

	# Fields
	- `state_transition_map::Array{ST, Np1}`: Next state index indexed by (agent 1 action index, ..., agent N action index, current state index)
	- `reward_transition_map::Array{RT, Np1}`: Shared scalar reward indexed by (agent 1 action index, ..., agent N action index, current state index)
	"""
	struct TabularCommonRewardGameTransition{T<:Real, N, Np1, ST<:Union{Int64, SparseVector{T, Int64}}, RT<:Union{T, Vector{T}}} <: AbstractTabularCommonRewardGameTransition{T, N}
		state_transition_map::Array{ST, Np1}
		reward_transition_map::Array{RT, Np1}
		function TabularCommonRewardGameTransition(state_transition_map::Array{ST, Np1}, reward_transition_map::Array{RT, Np1}) where {T<:Real, Np1, ST<:Union{Int64, SparseVector{T, Int64}}, RT <: Union{T, Vector{T}}}
			new{T, Np1-1, Np1, ST, RT}(state_transition_map, reward_transition_map)
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

# ╔═╡ 727885f8-ad7a-44b4-845d-9a9d8aebdb87
begin
	function check_policies(ptf::Union{TabularGameTransition{T, N, Np1, ST, RT}, TabularCommonRewardGameTransition{T, N, Np1, ST, RT}}, π_others::NTuple{Nm1, M}) where {T<:Real, N, Np1, ST, RT, Nm1, T2<:Real, M <:AbstractMatrix{T2}}
		@assert Nm1 == (N-1) "There must be $(N-1) policies for the other agents"
	end

	check_policies(ptf::TabularZeroSumGameTransition{T, ST, RT}, π_others::NTuple{1, M}) where {T<:Real, ST, RT, T2<:Real, M <:AbstractMatrix{T2}} = nothing
		
	check_policies(ptf::TabularZeroSumGameTransition{T, ST, RT}, π_others::NTuple{Nm1, M}) where {T<:Real, ST, RT, T2<:Real, Nm1, M <:AbstractMatrix{T2}} =  error("There must be 1 policy for the other agent")
end	

# ╔═╡ b0b41d5e-4944-4fc8-b108-175c559e3d63
begin
	get_num_states(ptf::Union{TabularGameTransition{T, N, Np1, ST, RT}, TabularCommonRewardGameTransition{T, N, Np1, ST, RT}}) where {T<:Real, N, Np1, ST, RT} = size(ptf.state_transition_map, Np1)

	get_num_states(ptf::TabularZeroSumGameTransition{T, ST, RT}) where {T<:Real, ST, RT} = size(ptf.state_transition_map, 3)
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

# ╔═╡ 79888a7b-c604-4db2-932a-ee9bfd378d9e
begin
	"""
		TabularStochasticGame{T, S, A, N, P, F} <: AbstractStochasticGame{T, S, A, N, P, F}

	Complete stochastic game with enumerated states and tabular transition function.

	Stores the full list of states, per-agent action spaces, a tabular game transition
	function, initialization and termination logic, and lookup tables for converting
	states and actions to their integer indices.

	Multiple convenience constructors are provided:
	- `TabularStochasticGame(states, agent_actions, ptf, initialize_state_index, terminal_states)`
	  — full specification with all fields
	- `TabularStochasticGame(states, agent_actions, ptf, terminal_states)`
	  — initial state index sampled uniformly from `1:length(states)`
	- `TabularStochasticGame(states, agent_actions, ptf, initialize_state_index)`
	  — terminal states default is `BitVector(undef, length(states))` (no terminal states)
	- `TabularStochasticGame(states, agent_actions, ptf)`
	  — both initialization and terminal states use defaults

	Lookup tables `state_index` and `action_index` are automatically generated from
	the provided `states` and `agent_actions` unless explicitly overridden with keyword arguments.

	# Fields
	- `states::Vector{S}`: Enumerated state space
	- `agent_actions::NTuple{N, Vector{A}}`: Available actions for each agent
	- `ptf::P`: Tabular game transition function
	- `initialize_state_index::F`: Function returning the index of the initial state
	- `terminal_states::BitVector`: Boolean vector marking terminal states
	- `state_index::Dict{S, Int64}`: State → index lookup table
	- `action_index::NTuple{N, Dict{A, Int64}}`: Per-agent action → index lookup tables
	"""
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

	TabularStochasticGame(states::Vector{S}, agent_actions::NTuple{N, Vector{A}}, ptf::P, initialize_state_index::F; state_index::Dict{S, Int64} = makelookup(states), action_index::NTuple{N, Dict{A, Int64}} = Tuple(makelookup(a) for a in agent_actions)) where {S, A, N, P<:AbstractTabularGameTransition, F<:Function} = TabularStochasticGame(states, agent_actions, ptf, initialize_state_index, BitVector(undef, length(states)), state_index, action_index)

	TabularStochasticGame(states::Vector{S}, agent_actions::NTuple{N, Vector{A}}, ptf::P; state_index::Dict{S, Int64} = makelookup(states), action_index::NTuple{N, Dict{A, Int64}} = Tuple(makelookup(a) for a in agent_actions)) where {S, A, N, P<:AbstractTabularGameTransition} = TabularStochasticGame(states, agent_actions, ptf, Returns(1:length(states)), BitVector(undef, length(states)), state_index, action_index)
end

# ╔═╡ 2bfa6ee7-d580-4ea0-9551-28c34570dc1e
"""
	create_non_repeated_zero_sum_game(reward_matrix, agent_actions)

Create a single-state zero-sum stochastic game with terminal state for a
non-repeated (one-shot) matrix game.

The game has two states: `:play` (initial) and `:term` (terminal).
The transition always goes from `:play` to `:term` regardless of the joint action,
so the game lasts exactly one step.

# Arguments
- `reward_matrix::Matrix{T}`: The `(n_actions_1 × n_actions_2)` payoff matrix
  (positive values favor agent 1, negative favor agent 2).
- `agent_actions::NTuple{2, Vector{A}}`: Available actions for each agent.

# Returns
- `TabularStochasticGame{T, Symbol, A, 2, TabularZeroSumGameTransition, F}`:
  A zero-sum stochastic game with two states.
"""
function create_non_repeated_zero_sum_game(reward_matrix::Matrix{T}, agent_actions::NTuple{2, Vector{A}}) where {A, T<:Real}
	(na1, na2) = size(reward_matrix)

	state_transition_map = ones(Int64, na1, na2, 2) .* 2
	reward_transition_map = zeros(Float32, na1, na2, 2)
	for i_a1 in 1:na1
		for i_a2 in 1:na2
			reward_transition_map[i_a1, i_a2, 1] = reward_matrix[i_a1, i_a2]
		end
	end

	ptf = TabularZeroSumGameTransition(state_transition_map, reward_transition_map)

	initialize_state_index() = 1

	TabularStochasticGame([:play, :term], agent_actions, ptf, initialize_state_index, BitVector([false, true]))
end

# ╔═╡ 769e5dc3-684e-4d36-a63f-0edb90331c59
module RockPaperScissors
	const actions = [:rock, :paper, :scissors]
	const payout_matrix = [0f0 -1f0 1f0; 1f0 0f0 -1f0; -1f0 1f0 0f0]

	import ..create_non_repeated_zero_sum_game
	non_repeated_game = create_non_repeated_zero_sum_game(payout_matrix, ntuple(i -> actions, 2))
end

# ╔═╡ 5629f70c-b2cc-4a68-b5f0-9a11aef6dcbc
# ╠═╡ skip_as_script = true
#=╠═╡
RockPaperScissors.non_repeated_game.ptf(1, (3, 2))
  ╠═╡ =#

# ╔═╡ 9f05e777-fb94-42f9-9cd7-b461f10f199d
"""
	create_non_repeated_game(reward_matrices, agent_actions)

Create a single-state general-sum stochastic game with terminal state for a
non-repeated (one-shot) matrix game with N agents.

The game has two states: `:play` (initial) and `:term` (terminal).
The transition always goes from `:play` to `:term` regardless of the joint action,
so the game lasts exactly one step.

# Arguments
- `reward_matrices::NTuple{N, Matrix{T}}`: Per-agent payoff matrices, each of
  shape `(n_actions_1 × ⋯ × n_actions_N)`.
- `agent_actions::NTuple{N, Vector{A}}`: Available actions for each agent.

# Returns
- `TabularStochasticGame{T, Symbol, A, N, TabularGameTransition, F}`:
  A general-sum stochastic game with two states.
"""
function create_non_repeated_game(reward_matrices::NTuple{N, Matrix{T}}, agent_actions::NTuple{N, Vector{A}}) where {A, T<:Real, N}
	ns = size(first(reward_matrices))

	test_mat = ones(Int64, ns...)
	state_transition_map = ones(Int64, ns..., 2) .* 2
	reward_transition_map = Array{NTuple{N, T}, N+1}(undef, ns..., 2)
	
	for i in CartesianIndices(test_mat)
		reward_transition_map[i, 1] = NTuple{N, T}(reward_matrices[n][i] for n in 1:N)
		reward_transition_map[i, 2] = NTuple{N, T}(zero(T) for n in 1:N)
	end

	ptf = TabularGameTransition(state_transition_map, reward_transition_map)

	initialize_state_index() = 1

	TabularStochasticGame([:play, :term], agent_actions, ptf, initialize_state_index, BitVector([false, true]))
end

# ╔═╡ a2284047-0aaf-41b2-a6a0-de082cdeb5a2
# ╠═╡ skip_as_script = true
#=╠═╡
create_non_repeated_game(([1. -1.; -1. 1.], [-1. 1.; 1. -1.]), ([1, 2], [1, 2]))
  ╠═╡ =#

# ╔═╡ 3f44f098-5aef-4287-901e-67a6062f0f02
"""
	make_random_policies(game::TabularStochasticGame)

Create a tuple of uniformly random policy matrices for each agent.  In a tabular setting, a policy is represented as a matrix where each column corresponds to a state and each row corresponds to an action.  The entry at (i_a, i_s) gives the probability of selecting action i_a in state i_s.

Each policy matrix has shape `(n_actions[n], n_states)` where all entries are
`1 / n_actions[n]`, representing a uniform distribution over actions for every state.
"""
function make_random_policies(game::TabularStochasticGame{T, S, A, N, P, F}) where {T<:Real, S, A, P, F, N}
	n_actions = Tuple(length(a) for a in game.agent_actions)
	n_states = length(game.states)
	NTuple{N, Matrix{T}}(ones(T, n_actions[n], n_states) ./ n_actions[n] for n in 1:N)
end

# ╔═╡ d95d9945-944a-497b-b07b-7b4d4eb1f01d
"""
	initialize_agent_action_values(game::TabularStochasticGame, init_value)

Create a tuple of Q-value matrices initialized to a constant value for each agent.  In a tabular setting, an agent's Q-value function is represented as a matrix where each column corresponds to a state and each row corresponds to an action.  The entry at (i_a, i_s) gives the estimated value of taking action i_a in state i_s for that particular agent.

Each matrix has shape `(n_actions[n], n_states)` filled with `init_value`.
"""
function initialize_agent_action_values(game::TabularStochasticGame{T, S, A, N, P, F}, init_value::T) where {T<:Real, S, A, P, F<:Function, N}
	n_actions = NTuple{N, Int64}(length(a) for a in game.agent_actions)
	n_states = length(game.states)
	NTuple{N, Matrix{T}}(ones(T, n_actions[n], n_states) .* init_value for n in 1:N)
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

# ╔═╡ d2bfbf48-2ca0-483f-afab-9a292343beca
#convert a tabular stochastic game into a tabular mdp using a scalar reward function
function TabularRL.TabularMDP(stochastic_game::TabularStochasticGame{T, S, A, N, P, F}, reward_function::Function) where {T<:Real, S, A, N, P<:TabularGameTransitionSampler, F<:Function}
	agent_actions = stochastic_game.agent_actions
	num_actions = Tuple(length(a) for a in agent_actions)
	joint_action_matrix = Array{A, N}(undef, num_actions...)
	inds = CartesianIndices(joint_action_matrix)
	joint_action_list = [Tuple(agent_actions[i][inds[n][i]] for i in 1:N) for n in 1:length(joint_action_matrix)]

	function step(i_s::Integer, i_a::Integer)
		a = joint_action_list[i_a]
		(rewards, i_s′) = stochastic_game.ptf.step(i_s, a)
		r = reward_function(rewards)
		return (r, i_s′)
	end
	
	ptf = TabularMDPTransitionSampler(step)
	
	TabularMDP(stochastic_game.states, joint_action_list, ptf, stochastic_game.initialize_state_index, stochastic_game.terminal_states; state_index = stochastic_game.state_index)
end

# ╔═╡ e838d71d-4120-4e4c-8727-82caf6a1431c
function TabularRL.TabularMDP(stochastic_game::TabularStochasticGame{T, S, A, N, P, F}) where {T<:Real, S, A, N, P<:TabularCommonRewardGameTransitionSampler, F<:Function}
	agent_actions = stochastic_game.agent_actions
	num_actions = Tuple(length(a) for a in agent_actions)
	joint_action_matrix = Array{A, N}(undef, num_actions...)
	inds = CartesianIndices(joint_action_matrix)
	joint_action_list = [Tuple(agent_actions[i][inds[n][i]] for i in 1:N) for n in 1:length(joint_action_matrix)]

	function step(i_s::Integer, i_a::Integer)
		a = joint_action_list[i_a]
		stochastic_game.ptf.step(i_s, a)
	end
	
	ptf = TabularMDPTransitionSampler(step)
	
	TabularMDP(stochastic_game.states, joint_action_list, ptf, stochastic_game.initialize_state_index, stochastic_game.terminal_states; state_index = stochastic_game.state_index)
end

# ╔═╡ 53c087e9-ad7d-4a83-a373-0e0d3e406724
#convert a tabular stochastic game into a tabular mdp for agent i using fixed policies for the other agents
function TabularRL.TabularMDP(stochastic_game::TabularStochasticGame{T, S, A, N, P, F}, agent_index::Integer, πs_other::NTuple{Nm1, M}) where {T<:Real, S, A, N, Nm1, P<:TabularGameTransitionSampler{T, N}, F<:Function, T2<:Real, M <: AbstractMatrix{T2}}
	@assert Nm1 == (N - 1) "There must be $(N-1) policies for the other agents"
	agent_actions = stochastic_game.agent_actions
	function step(i_s::Integer, i_a::Integer)
		a_other = sample_joint_action(πs_other, i_s)
		a = (Tuple(a_other[i] for i in 1:agent_index-1)..., i_a, Tuple(a_other[i] for i in agent_index+1:N))
		(rewards, i_s′) = stochastic_game.ptf.step(i_s, a)
		r = rewards[agent_index]
		return (r, i_s′)
	end

	ptf = TabularMDPTransitionSampler(step)
	
	TabularMDP(stochastic_game.states, agent_actions[agent_index], ptf, stochastic_game.initialize_state_index, stochastic_game.terminal_states; state_index = stochastic_game.state_index)
end

# ╔═╡ 77719b6d-893e-4011-b676-cbd322d3a835
TabularRL.TabularMDP(stochastic_game::TabularStochasticGame{T, S, A, 2, P, F}, player1::Bool, π_other::M) where {T<:Real, S, A, P<:TabularGameTransitionSampler{T, 2}, F<:Function, T2<:Real, M <: AbstractMatrix{T2}} = TabularMDP(stochastic_game, player1*1 + !player1*2, (π_other,))

# ╔═╡ eb6189fd-68e0-435f-b736-dab754522ea9
begin
	function update_transitions!(state_transition_map::Matrix{SparseVector{T, Int64}}, reward_transitions::Matrix{Dict{Int64, T}}, ptf::Union{TabularGameStochasticTransition{T, N, Np1}, TabularZeroSumGameStochasticTransition{T}, TabularCommonRewardGameStochasticTransition{T, N}}, cart_ind::CartesianIndex, i_a::Integer, i_s::Integer, p::T, agent_index::Integer) where {T<:Real, N, Np1}
		probs = ptf.state_transition_map[cart_ind]
		agent_rewards = ptf.reward_transition_map[cart_ind]

		for i in eachindex(probs.nzind)
			i_s′ = probs.nzind[i]
			r = get_game_reward(ptf, agent_rewards[i], agent_index)
			p2 = probs.nzval[i]
			state_transition_map[i_a, i_s][i_s′] += p*p2
			
			if !haskey(reward_transitions[i_a, i_s], i_s′)
				reward_transitions[i_a, i_s][i_s′] = p*p2*r
			else
				reward_transitions[i_a, i_s][i_s′] += p*p2*r
			end
		end
	end

	function update_transitions!(state_transition_map::Matrix{SparseVector{T, Int64}}, reward_transitions::Matrix{Dict{Int64, T}}, ptf::Union{TabularGameDeterministicTransition{T, N, Np1}, TabularZeroSumGameDeterministicTransition{T}, TabularCommonRewardGameDeterministicTransition{T, N}}, cart_ind::CartesianIndex, i_a::Integer, i_s::Integer, p::T, agent_index::Integer) where {T<:Real, N, Np1}
		i_s′ = ptf.state_transition_map[cart_ind]
		r = get_game_reward(ptf, ptf.reward_transition_map[cart_ind], agent_index)
		state_transition_map[i_a, i_s][i_s′] += p
		if !haskey(reward_transitions[i_a, i_s], i_s′)
			reward_transitions[i_a, i_s][i_s′] = p*r
		else
			reward_transitions[i_a, i_s][i_s′] += p*r
		end
	end
end

# ╔═╡ 475e855e-ca4b-4466-ba8b-ca0edefb4f82
#convert a tabular stochastic game into a tabular mdp for agent i using fixed policies for the other agents
function TabularRL.TabularStochasticTransition(ptf::Union{TabularGameTransition{T, N, Np1, ST, RT}, TabularZeroSumGameTransition{T, ST, RT}, TabularCommonRewardGameTransition{T, N, Np1, ST, RT}}, agent_index::Integer, π_others::NTuple{Nm1, M}) where {T<:Real, N, Np1, Nm1, T2<:Real, M <: AbstractMatrix{T2}, ST <: Union{Int64, SparseVector{T, Int64}}, RT <: Union{T, Vector{T}}}
	check_policies(ptf, π_others)
	
	num_actions = size(ptf.state_transition_map, agent_index)
	num_states = get_num_states(ptf)

	(other_inds, state_index) = get_other_inds(ptf, agent_index)
	
	state_transition_map = Matrix{SparseVector{T, Int64}}(undef, num_actions, num_states)
	reward_transitions = Matrix{Dict{Int64, T}}(undef, num_actions, num_states)
	reward_transition_map = Matrix{Vector{T}}(undef, num_actions, num_states)
	for i in eachindex(state_transition_map)
		state_transition_map[i] = SparseVector{T, Int64}(zeros(T, num_states))
		reward_transitions[i] = Dict{Int64, T}()
		reward_transition_map[i] = Vector{T}()
	end

	cart_inds = CartesianIndices(ptf.state_transition_map)
	
	for cart_ind in cart_inds
		i_s = cart_ind[state_index]
		i_a = cart_ind[agent_index]

		p = one(T)
		for (i, n) in enumerate(other_inds)
			i_a_other = cart_ind[n]
			p *= π_others[i][i_a_other, i_s]
		end

		update_transitions!(state_transition_map, reward_transitions, ptf, cart_ind, i_a, i_s, p, agent_index)
	end

	for i_s in 1:num_states
		for i_a in 1:num_actions
			for i_s′ in state_transition_map[i_a, i_s].nzind
				ptot = state_transition_map[i_a, i_s][i_s′]
				r_avg = reward_transitions[i_a, i_s][i_s′] / ptot
				push!(reward_transition_map[i_a, i_s], r_avg)
			end
		end
	end

	TabularTransitionDistribution(state_transition_map, reward_transition_map)
end

# ╔═╡ 89b1cb3b-bf58-4ca3-a08d-736ee2af2b0f
function TabularRL.TabularMDP(stochastic_game::TabularStochasticGame{T, S, A, N, P, F}, agent_index::Integer, π_others::NTuple{Nm1, M}) where {T<:Real, S, A, N, P<:Union{TabularGameTransition, TabularZeroSumGameTransition, TabularCommonRewardGameTransition}, F<:Function, Nm1, T2<:Real, M <: AbstractMatrix{T2}}
	@assert Nm1 == (N - 1) "There must be $(N-1) policies for the other agents"
	agent_actions = stochastic_game.agent_actions

	ptf = TabularStochasticTransition(stochastic_game.ptf, agent_index, π_others)
	TabularMDP(stochastic_game.states, agent_actions[agent_index], ptf, stochastic_game.initialize_state_index, stochastic_game.terminal_states; state_index = stochastic_game.state_index)
end

# ╔═╡ fab57783-bcf7-4fd8-a5d1-5824db9226e4
TabularRL.TabularMDP(stochastic_game::TabularStochasticGame{T, S, A, 2, P, F}, player1::Bool, π_other::M) where {T<:Real, S, A, P<:Union{TabularGameTransition, TabularZeroSumGameTransition, TabularCommonRewardGameTransition}, F<:Function, T2<:Real, M <: AbstractMatrix{T2}} = TabularMDP(stochastic_game, player1*1 + !player1*2, (π_other,))

# ╔═╡ 3f9d927e-4158-4531-b31f-56aae3ac0c7b
#convert a tabular stochastic game into a tabular mdp for agent i using fixed policies for the other agents
function TabularRL.TabularMDP(stochastic_game::TabularStochasticGame{T, S, A, 2, P, F}, player1::Bool, π_other::M) where {T<:Real, S, A, P<:TabularZeroSumGameTransitionSampler{T}, F<:Function, T2<:Real, M <: AbstractMatrix{T2}}
	agent_actions = stochastic_game.agent_actions
	inds = player1 ? (1, 2) : (2, 1)
	form_joint_action(a::Tuple{Int64, Int64}) = (a[inds[1]], a[inds[2]])  
	c = player1 ? one(T) : -1*one(T)
	agent_index = player1 ? 1 : 2
	function step(i_s::Integer, i_a::Integer)
		i_a_other = sample_action(π_other, i_s)
		a = form_joint_action((i_a, i_a_other))
		(r, i_s′) = stochastic_game.ptf.step(i_s, a)
		return (c*r, i_s′)
	end

	ptf = TabularMDPTransitionSampler(step)
	
	TabularMDP(stochastic_game.states, agent_actions[agent_index], ptf, stochastic_game.initialize_state_index, stochastic_game.terminal_states; state_index = stochastic_game.state_index)
end

# ╔═╡ 736aacc0-5592-4439-b8c3-cf76525983a5
begin
	"""
		StateStochasticGame{T, S, A, N, P, StateInit, IsTerm} <: AbstractStochasticGame{T, S, A, N, P, StateInit}

	Complete stochastic game with state-based (non-tabular) transition function,
	where states are arbitrary objects rather than enumerated state indices.

	Unlike [`TabularStochasticGame`](@ref), the state space is not explicitly
	enumerated.  Instead, `initialize_state` returns a concrete state object,
	and `isterm` is a predicate function that determines whether a state is
	terminal.

	Per-agent validity predicates `is_valid_action` can be provided to restrict
	which actions are available in a given state.

	# Fields
	- `agent_actions::NTuple{N, Vector{A}}`: Available actions for each agent
	- `ptf::P`: State-based game transition function
	- `initialize_state::StateInit`: Function returning the initial state
	- `isterm::IsTerm`: Predicate function `isterm(s) -> Bool` for terminal state detection
	- `agent_action_index::NTuple{N, Dict{A, Int64}}`: Per-agent action → index lookup tables
	- `is_valid_action::NTuple{N, Function}`: Per-agent action validity predicates
	"""
	struct StateStochasticGame{T<:Real, S, A, N, P<:AbstractStateGameTransition{T, S, F, N} where F<:Function, StateInit<:Function, IsTerm<:Function} <: AbstractStochasticGame{T, S, A, N, P, StateInit}
		agent_actions::NTuple{N, Vector{A}}
		ptf::P
		initialize_state::StateInit
		isterm::IsTerm
		agent_action_index::NTuple{N, Dict{A, Int64}} #lookup table mapping actions to their index for each agent
		is_valid_action::NTuple{N, Function}
	end

	#automatically generate the action lookup when constructing MDP
	"""
		StateStochasticGame(agent_actions, ptf, initialize_state, isterm; kwargs...)

	Construct a state-based stochastic game with explicit terminal state predicate.

	The inner constructor validates the initial state type against the transition
	function's state type parameter `S`.

	# Constructor
		StateStochasticGame(agent_actions, ptf, initialize_state, isterm;
		                    agent_action_index = Tuple(makelookup(a) for a in agent_actions),
		                    is_valid_action = ntuple(i -> (s, i_a) -> true, N))

	# Arguments
	- `agent_actions::NTuple{N, AbstractVector}`: Available actions per agent
	- `ptf::AbstractStateGameTransition`: State-based transition function
	- `initialize_state::Function`: Function `() -> s::S` returning the initial state
	- `isterm::Function`: Function `isterm(s::S) -> Bool`
	"""
	function StateStochasticGame(agent_actions::NTuple{N, A}, ptf::AbstractStateGameTransition{T, S, F, N}, initialize_state::StateInit, isterm::IsTerm; agent_action_index = Tuple(makelookup(a) for a in agent_actions), is_valid_action = ntuple(i -> (s, i_a) -> true, N)) where {T<:Real, S, F<:Function, N, A<:AbstractVector, StateInit<:Function, IsTerm<:Function}
		s0 = initialize_state()
		isterm(s0)
		@assert typeof(s0) <: S
		StateStochasticGame(Tuple(Vector(a) for a in agent_actions), ptf, initialize_state, isterm, agent_action_index, is_valid_action)
	end

	#if terminal check is not provided assume there are no terminal states
	StateStochasticGame(agent_actions::NTuple{N, A}, ptf::AbstractStateGameTransition{T, S, F, N}, initialize_state::StateInit; kwargs...) where {T<:Real, S, F<:Function, N, A<:AbstractVector, StateInit<:Function} = StateStochasticGame(agent_actions, ptf, initialize_state, Returns(false); kwargs...)

	"""
		StateStochasticGame(game::TabularStochasticGame)

	Convert a [`TabularStochasticGame`](@ref) with a tabular deterministic transition
	into an equivalent [`StateStochasticGame`](@ref).

	A step function is constructed that looks up the state index, applies the tabular
	transition map, and returns the actual state object.  The conversion preserves
	action indices from the original game's lookup tables.
	"""
	function StateStochasticGame(game::TabularStochasticGame{T, S, A, N, P, F}) where {T<:Real, S, A, N, P<:TabularGameDeterministicTransition, F<:Function}
		agent_actions = game.agent_actions
		initialize_state() = game.states[game.initialize_state_index()]
		isterm(s::S) = game.terminal_states[game.state_index[s]]
	
		function step(s::S, a::NTuple{N, Int64})
			i_s = game.state_index[s]
			i_s′ = game.ptf.state_transition_map[a..., i_s]
			rewards = game.ptf.reward_transition_map[a..., i_s]
			return rewards, game.states[i_s′]
		end
		
		ptf = StateGameTransitionDeterministic(step, initialize_state(), N)
		StateStochasticGame(agent_actions, ptf, initialize_state, isterm; agent_action_index = game.action_index)
	end

	"""
		StateStochasticGame(game::TabularStochasticGame)

	Convert a tabular zero-sum deterministic stochastic game into a state-based
	[`StateStochasticGame`](@ref) using [`StateZeroSumGameTransitionDeterministic`](@ref).
	"""
	function StateStochasticGame(game::TabularStochasticGame{T, S, A, 2, P, F}) where {T<:Real, S, A, P<:TabularZeroSumGameDeterministicTransition, F<:Function}
		agent_actions = game.agent_actions
		initialize_state() = game.states[game.initialize_state_index()]
		isterm(s::S) = game.terminal_states[game.state_index[s]]
	
		function step(s::S, a::NTuple{2, Int64})
			i_s = game.state_index[s]
			i_s′ = game.ptf.state_transition_map[a..., i_s]
			r = game.ptf.reward_transition_map[a..., i_s]
			return r, game.states[i_s′]
		end
		
		ptf = StateZeroSumGameTransitionDeterministic(step, initialize_state())
		StateStochasticGame(agent_actions, ptf, initialize_state, isterm; agent_action_index = game.action_index)
	end

	"""
		StateStochasticGame(game::TabularStochasticGame)

	Convert a tabular common-reward deterministic stochastic game into a state-based
	[`StateStochasticGame`](@ref) using [`StateCommonRewardGameTransitionDeterministic`](@ref).
	"""
	function StateStochasticGame(game::TabularStochasticGame{T, S, A, N, P, F}) where {T<:Real, S, A, N, P<:TabularCommonRewardGameDeterministicTransition, F<:Function}
		agent_actions = game.agent_actions
		initialize_state() = game.states[game.initialize_state_index()]
		isterm(s::S) = game.terminal_states[game.state_index[s]]
	
		function step(s::S, a::NTuple{N, Int64})
			i_s = game.state_index[s]
			i_s′ = game.ptf.state_transition_map[a..., i_s]
			r = game.ptf.reward_transition_map[a..., i_s]
			return r, game.states[i_s′]
		end
		
		ptf = StateCommonRewardGameTransitionDeterministic(step, initialize_state(), N)
		StateStochasticGame(agent_actions, ptf, initialize_state, isterm; agent_action_index = game.action_index)
	end
end

# ╔═╡ acdb7ffb-df0a-4609-8b07-87b4f3f9a145
StateStochasticGame(create_non_repeated_game(([1. -1.; -1. 1.], [-1. 1.; 1. -1.]), ([1, 2], [1, 2])))

# ╔═╡ 39182e4e-6ee9-4c14-8b5f-3164e38d5b5a
begin
	convert_multiagent_transition(game::StateStochasticGame{T, S, A, N, P, F1, F2}, joint_step::Function) where {T<:Real, S, A, N, P<:StateGameTransitionDeterministic, F1<:Function, F2<:Function} = StateMDPTransitionDeterministic(joint_step, game.initialize_state())

	convert_multiagent_transition(game::StateStochasticGame{T, S, A, N, P, F1, F2}, joint_step::Function) where {T<:Real, S, A, N, P<:StateCommonRewardGameTransitionDeterministic, F1<:Function, F2<:Function} = StateMDPTransitionDeterministic(joint_step, game.initialize_state())
end

# ╔═╡ 0bb34d75-0422-4336-9e62-d6431ce2b1c3
#convert a multi-agent state mdp into an mdp using a scalar reward function
begin
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
	function TabularRL.StateMDP(mdp::StateStochasticGame{T, S, A, N, P, F1, F2}) where {T<:Real, S, A, N, P<:StateCommonRewardGameTransitionDeterministic, F1<:Function, F2<:Function}
		agent_actions = mdp.agent_actions
		num_actions = Tuple(length(a) for a in agent_actions)
		joint_action_matrix = Array{A, N}(undef, num_actions...)
		inds = CartesianIndices(joint_action_matrix)
		joint_action_list = [Tuple(agent_actions[i][inds[n][i]] for i in 1:N) for n in 1:length(joint_action_matrix)]

		function joint_step(s::S, i_a::Integer)
			a = CartesianIndices(joint_action_matrix)[i_a] |> Tuple
			(r, s′) = mdp.ptf.step(s, a)
			return (r, s′)
		end

		ptf = convert_multiagent_transition(mdp, joint_step)
		
		StateMDP(joint_action_list, ptf, mdp.initialize_state, mdp.isterm)
	end
end

# ╔═╡ 79b114c9-8999-4e78-86b7-8bc0bd81dcad
module LevelBasedForaging
	#import general RL and MARL types and functions
	import ..RL_Module

	import ..makelookup

	import ..StateGameTransitionDeterministic
	import ..StateStochasticGame
	import ..TabularGameTransition
	import ..TabularStochasticGame

	"""
		ForagingMove

	Abstract type for the six movement/action primitives in Level-Based Foraging.

	# Subtypes
	- [`Up`](@ref), [`Down`](@ref), [`Left`](@ref), [`Right`](@ref) — cardinal movement
	- [`Collect`](@ref) — collect an adjacent item
	- [`Noop`](@ref) — stand still
	"""
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
	const action_tuple = (Up(), Down(), Left(), Right(), Collect(), Noop())
	const action_index = makelookup(action_list)

	"""
		ForagingState{N, M, X, Y}

	State representation for the Level-Based Foraging environment.

	Tracks N agent positions and M item positions on an X × Y grid world.
	Each agent and item has an associated integer level.  Items that have been
	collected are marked by `item_collect`.

	The type parameters `N`, `M`, `X`, `Y` are used for dispatch and to enable
	compile-time optimizations: `ForagingState{3, 5, 8, 8}` is a different type
	from `ForagingState{2, 2, 11, 11}`.

	# Fields
	- `agent_positions::NTuple{N, Position}`: Grid positions of each agent
	- `item_positions::NTuple{M, Position}`: Grid positions of each item
	- `item_collect::NTuple{M, Bool}`: Whether each item has been collected
	- `agent_levels::NTuple{N, Int64}`: Level of each agent (contribution to collection)
	- `item_levels::NTuple{M, Int64}`: Required level to collect each item
	"""
	struct ForagingState{N, M, X, Y}
		agent_positions::NTuple{N, Position}
		item_positions::NTuple{M, Position}
		item_collect::NTuple{M, Bool} #keeps track of whether or not an item has been collected, initial values will be all false
		agent_levels::NTuple{N, Int64}
		item_levels::NTuple{M, Int64}

		# ForagingState(agent_positions, item_positions, item_collect, agent_levels, item_levels, available_positions, item_collection_positions, item_level_sum) = new{length(agent_positions), length(item_positions)}(agent_positions, item_positions, agent_levels, item_levels, available_positions, item_collection_positions, item_level_sum)
	end

	#hash and isequal definition for ForagingState so that dictionary lookup is not affected by the non-essential state information
	Base.hash(s::ForagingState) = hash((s.agent_positions, s.item_positions, s.item_collect, s.agent_levels, s.item_levels))
	Base.isequal(s1::ForagingState{N1, M1, X1, Y1}, s2::ForagingState{N2, M2, X2, Y2}) where {N1, M1, X1, Y1, N2, M2, X2, Y2} = false
	Base.isequal(s1::ForagingState{N, M, X, Y}, s2::ForagingState{N, M, X, Y}) where {N, M, X, Y} = isequal(s1.agent_positions, s2.agent_positions) && isequal(s1.item_positions, s2.item_positions) && isequal(s1.agent_levels, s2.agent_levels) && isequal(s1.item_levels, s2.item_levels) && isequal(s1.item_collect, s2.item_collect)

	#generate a ForagingState with only the essential information
	function ForagingState(agent_positions::NTuple{N, Position}, item_positions::NTuple{M, Position}, agent_levels::NTuple{N, Int64}, item_levels::NTuple{M, Int64}, x_max::Integer, y_max::Integer) where {N, M}
		ForagingState{N, M, x_max, y_max}(agent_positions, item_positions, ntuple(i -> false, M), agent_levels, item_levels)
	end

	function ForagingState(agent_positions, item_positions, agent_levels, item_levels, x_max::Integer, y_max::Integer)
		N = length(agent_positions)
		M = length(item_positions)
		ForagingState(NTuple{N, Position}(agent_positions), NTuple{M, Position}(item_positions), NTuple{N, Int64}(agent_levels), NTuple{M, Int64}(item_levels), x_max, y_max)
	end

	function ForagingState(agent_positions::NTuple{N, Position}, item_positions::NTuple{M, Position}, item_collect::NTuple{M, Bool}, agent_levels::NTuple{N, Int64}, item_levels::NTuple{M, Int64}, x_max::Integer, y_max::Integer) where {N, M}
		ForagingState{N, M, x_max, y_max}(agent_positions, item_positions, item_collect, agent_levels, item_levels)
	end

	function ForagingState(agent_positions, item_positions, item_collect, agent_levels, item_levels, x_max::Integer, y_max::Integer)
		N = length(agent_positions)
		M = length(item_positions)
		ForagingState(NTuple{N, Position}(agent_positions), NTuple{M, Position}(item_positions), NTuple{M, Bool}(item_collect), NTuple{N, Int64}(agent_levels), NTuple{M, Int64}(item_levels), x_max, y_max)
	end

	#LBF episodes end after all items have been collected
	isterm(s::ForagingState) = all(s.item_collect)

	function check_adjacent(p1::Position, p2::Position) 
		((p1[1] == p2[1]) && (abs(p1[2] - p2[2]) == 1)) ||
		((p1[2] == p2[2]) && (abs(p1[1] - p2[1]) == 1))
	end

	get_adjacent(p::Position) = Set(move(p, m) for m in (Up(), Down(), Right(), Left()))

	#For all actions except `collect` there is no resulting item collected
	attempt_collect(s::ForagingState, a::ForagingMove, agent_index::Integer) = (0, 0)

	#For the `collect` action, adds the desired item and agent level to the tally for that item
	function attempt_collect(s::ForagingState{N, M, X, Y}, a::Collect, agent_index::Integer) where {N, M, X, Y}
		agent_position = s.agent_positions[agent_index]
		agent_level = s.agent_levels[agent_index]
		i = 1
		item_number = 0
		level = 0
		for i in 1:M
			if !s.item_collect[i] #can only collect an item that hasn't yet been collected
				flag = check_adjacent(agent_position, s.item_positions[i])
				item_number += flag*i
				level += flag*agent_level
			end
		end
		return (item_number, level)
	end

	attempt_collect(s::ForagingState, i_a::Integer, agent_index::Integer) = attempt_collect(s, action_tuple[i_a], agent_index)

	#for a joint action, checks to see which items if any are collected based on if the level checks pass for a particular item.  Then rewards are allotted to the agents involved in collection
	function attempt_collect(s::ForagingState{N, M, X, Y}, a::NTuple{N, I}) where {N, M, X, Y, I<:Integer}
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

		item_level_sum = sum(s.item_levels)

		for i in 1:M
			if !s.item_collect[i] #can only collect an item that hasn't already been collected
				item_level = s.item_levels[i]
				success = (item_checks[i] >= item_level)
				if success
					denom = item_level_sum * sum(s.agent_levels[j] for j in agent_checks[i])
					for j in agent_checks[i]
						rewards[j] = s.agent_levels[j]*item_level / denom
					end
				end
				item_checks[i] = success
			else #in this case the item has already been collected so just reflect that in the check without assigning any reward
				item_checks[i] = 1
			end
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

	#ensures that no item is in a collectable position with another item on the grid which creates ambiguity about the `collect` action
	function make_item_bubble(p::Position)
		excluded_positions = Set{Position}()
		push!(excluded_positions, p)
		moves = ((Up(), Down(), Left(), Right()))
		for m in moves
			push!(excluded_positions, move(p, m))
		end
		
		for m1 in moves
			for m2 in moves
				push!(excluded_positions, move(move(p, m1), m2))
			end
		end
		return excluded_positions
	end

	#initialize a state given the grid size, number of agents, number of items and the level distribution for both.  Randomly assigns levels using a uniform distribution from 1 to maxlevel
	function initialize_state(width::Integer, height::Integer, num_agents, num_items, min_agent_level, max_agent_level, min_item_level, max_item_level; force_cooperation::Bool = false)
		grid_positions = Set((x, y) for x in 1:width for y in 1:height)
		p1 = rand(grid_positions)

		excluded_positions = make_item_bubble(p1)
		item_positions = [p1]
		allowed_positions = setdiff(grid_positions, excluded_positions)
		
		for i in 2:num_items
			isempty(allowed_positions) && break
			p = rand(allowed_positions)
			push!(item_positions, p)
			setdiff!(allowed_positions, make_item_bubble(p))
		end

		#allowed positions for agents
		allowed_positions = setdiff(grid_positions, item_positions)

		isempty(allowed_positions) && error("Cannot form state with no available positions for agents")

		agent_positions = [rand(allowed_positions)]
		for i in 2:num_agents
			setdiff!(allowed_positions, agent_positions[[i-1]])
			push!(agent_positions, rand(allowed_positions))
			isempty(allowed_positions) && error("Cannot form state with no available positions for agents")
		end

		agent_levels = [rand(min_agent_level:max_agent_level) for _ in 1:num_agents]
		agent_level_sum = sum(agent_levels)

		num_items = length(item_positions)

		min_item_level > agent_level_sum && error("The minimum item level of $min_item_level is higher than the agents can collect of $agent_level_sum")

		min_item_level = force_cooperation ? maximum(agent_levels) + 1 : min_item_level

		max_item_level = min(agent_level_sum, max_item_level)
		min_item_level > max_item_level && error("No valid item levels between $min_item_level and $max_item_level")
		
		#restrict item levels to only be as high as the sum of all the agent levels.  this ensures that every item is probably collectable
		item_levels = [rand(min_item_level:max_item_level) for _ in 1:num_items]
		
		ForagingState(agent_positions, item_positions, agent_levels, item_levels, width, height)
	end

	function step(s::ForagingState{N, M, X, Y}, a::NTuple{N, I}, reset_chance::T) where {N, M, X, Y, I<:Integer, T<:Real}
		isterm(s) && return (ntuple(i -> 0f0, N), s)
		#check to see if any items have been collected and calculate the rewards
		(rewards, item_checks) = attempt_collect(s, a)

		item_collect′ = ntuple(M) do i
			isone(item_checks[i])
		end

		!all(item_collect′) && (rand() < reset_chance) && return NTuple{N, Float32}(rewards), ForagingState(s.agent_positions, s.item_positions, ntuple(i -> true, M), s.agent_levels, s.item_levels, X, Y)

		#make any agent moves that are attempted prior to checking to see if any intersect
		candidate_positions = Vector{Position}([begin
							   		p′ = move(s.agent_positions[i], action_tuple[a[i]])
									x = p′[1]
									y = p′[2]
									if (x > X) || (y > Y) || (x < 1) || (y < 1) || any(i -> !item_collect′[i] && (s.item_positions[i] == p′), 1:M)
										s.agent_positions[i]
									else
										p′
									end
							   end
							   for i in 1:N])

		agent_positions′ = ntuple(N) do i
			p′ = candidate_positions[i]
			#prevent an agent from moving if its move takes it into a space occupied by another agent
			if in(p′, view(candidate_positions, 1:i-1)) || in(p′, view(candidate_positions, i+1:N))
				s.agent_positions[i]
			else
				candidate_positions[i]
			end
		end

		#modify state for collected item
		# for i in 1:M
		# 	if isone(item_checks[i])
		# 		deleteat!(s′.item_positions, i)
		# 		deleteat!(s′.item_levels, i)
		# 		deleteat!(s′.item_collection_positions, i)
		# 		push!(s′.available_positions, s.item_positions[i])
		# 	end
		# end

		return NTuple{N, Float32}(rewards), ForagingState(agent_positions′, s.item_positions, item_collect′, s.agent_levels, s.item_levels, X, Y)
	end

	"""
		make_environment(; kwargs...)

	Create a Level-Based Foraging environment as a [`StateStochasticGame`](@ref)
	using a state-based deterministic transition function.

	Keyword arguments configure the grid size, number of agents, number of items,
	and level distributions.

	# Keywords
	- `width::Integer = 8`: Grid width
	- `height::Integer = 8`: Grid height
	- `num_agents::Integer = 3`: Number of agents
	- `num_items::Integer = 5`: Number of items
	- `min_agent_level::Integer = 1`: Minimum agent level
	- `max_agent_level::Integer = 2`: Maximum agent level
	- `min_item_level::Integer = 1`: Minimum item level
	- `max_item_level::Integer = 4`: Maximum item level
	- `reset_chance::Real = 0.0`: Probability of resetting all items at each step
	- `force_cooperation::Bool = false` (passed via kwargs): If true, forces items
	  to require cooperation by setting minimum item level above max agent level

	# Returns
	- `StateStochasticGame{...}`: A state-based stochastic game with LBF dynamics
	"""
	function make_environment(;width = 8, height = 8, num_agents = 3, num_items = 5, min_agent_level = 1, max_agent_level = 2, min_item_level = 1, max_item_level = 4, reset_chance = 0f0, kwargs...)
		init_state() = initialize_state(width, height, num_agents, num_items, min_agent_level, max_agent_level, min_item_level, max_item_level; kwargs...)

		ptf = StateGameTransitionDeterministic((s, a) -> step(s, a, reset_chance), init_state(), num_agents)
			
		StateStochasticGame(ntuple(Returns(action_list), num_agents), ptf, init_state, isterm)
	end

	#make an environment for LBF based on example 5.3 which always uses the same initial state with 2 agents and 2 items in an 11x11 grid.  The tabular state space is constructed by iterating through all agent positions with all possible item combinations
	"""
		make_5_3_environment(;width=11, height=11)

	Create a tabular Level-Based Foraging environment matching Example 5.3 from
	the textbook "Multi-Agent Reinforcement Learning" with:
	- 2 agents (levels 1 and 1)
	- 2 items (levels 1 and 2)
	- 11×11 grid world
	- Predefined initial positions

	The environment is converted to a [`TabularStochasticGame`](@ref) with 
	enumerated states for exact solution methods. States track agent positions
	and item collection status.
	"""
	function make_5_3_environment(;width = 11, height = 11)
		num_agents = 2
		num_items = 2
		max_agent_level = 1
		max_item_level = 2

		reset_positions = [false, false]
		
		grid_positions = [(x, y) for x in 1:width for y in 1:height]
		position_set = Set(grid_positions)
		position_inds = collect(eachindex(grid_positions))
		position_lookup = makelookup(grid_positions)
		num_positions = length(grid_positions)

		function initialize_state()
			x1 = floor(Int64, width / 2)
			x2 = x1 + 2
			p1 = (x1, x1)
			p2 = (x2, x2)
			item_positions = [p1, p2]

			agent_positions = [(1, 1), (width, 1)]
	
			item_levels = [1, 2]
			agent_levels = [1, 1]
			
			ForagingState(agent_positions, item_positions, agent_levels, item_levels, width, height)
		end

		#initialize state with two items and iterate through all agent positions
		s0 = initialize_state()
		available_positions = setdiff(grid_positions, s0.item_positions)
		
		states = Vector{ForagingState{2, 2, width, height}}()
		for p1 in available_positions
			for p2 in setdiff(available_positions, (p1,))
				push!(states, ForagingState((p1, p2), s0.item_positions, s0.agent_levels, s0.item_levels, width, height))
			end
		end

		#only keep item 1 and iterate through all agent positions
		available_positions′ = setdiff(grid_positions, s0.item_positions[2])
		item_collect′ = (false, true)
		for p1 in available_positions′
			for p2 in setdiff(available_positions′, (p1,))
				push!(states, ForagingState((p1, p2), s0.item_positions, item_collect′, s0.agent_levels, s0.item_levels, width, height))
			end
		end

		#only keep item 2 and iterate through all agent positions
		available_positions′ = setdiff(grid_positions, s0.item_positions[1])
		item_collect′ = (true, false)
		for p1 in available_positions′
			for p2 in setdiff(available_positions′, (p1,))
				push!(states, ForagingState((p1, p2), s0.item_positions, item_collect′, s0.agent_levels, s0.item_levels, width, height))
			end
		end

		#remove both items and iterate through all agent positions
		available_positions′ = grid_positions
		item_collect′ = (true, true)
		for p1 in available_positions′
			for p2 in setdiff(available_positions′, (p1,))
				push!(states, ForagingState((p1, p2), s0.item_positions, item_collect′, s0.agent_levels, s0.item_levels, width, height))
			end
		end

		state_index = Dict{ForagingState{2, 2, width, height}, Int64}(makelookup(states))
		
		actions = ntuple(Returns(action_list), 2)
		
		#transition maps have two dimensions of length 6 for the 6 actions available to each agent and then the last dimension is the state space
		state_transition_map = Array{Int64, 3}(undef, 6, 6, length(states))
		reward_transition_map = Array{NTuple{2, Float32}, 3}(undef, 6, 6, length(states))
		terminal_states = BitVector(undef, length(states))

		for i_s in eachindex(states)
			s = states[i_s]
			terminal_states[i_s] = isterm(s)
		end

		for i_s in eachindex(states)
			for i_a1 in 1:6
				for i_a2 in 1:6
					s = states[i_s]
					(rewards, s′) = step(s, (i_a1, i_a2), 0f0)
					i_s′ = state_index[s′]
					state_transition_map[i_a1, i_a2, i_s] = i_s′
					reward_transition_map[i_a1, i_a2, i_s] = rewards
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

	"""
		make_small_environment(;width=3, height=3)

	Create a minimal tabular Level-Based Foraging environment for quick testing:
	- 2 agents (both level 1)
	- 1 item (level 2) 
	- 3×3 grid world
	- Predefined initial positions

	The small state space makes this suitable for debugging and algorithm 
	validation. Converted to a [`TabularStochasticGame`](@ref) with enumerated
	states.
	"""
	function make_small_environment(;width = 3, height = 3)
		num_agents = 2
		num_items = 1
		agent_level = 1
		item_level = 2

		reset_positions = [false, false]
		
		grid_positions = [(x, y) for x in 1:width for y in 1:height]
		position_set = Set(grid_positions)
		position_inds = collect(eachindex(grid_positions))
		position_lookup = makelookup(grid_positions)
		num_positions = length(grid_positions)

		function initialize_state()
			item_positions = [(width, height)]

			agent_positions = [(1, 1), (width, 1)]
	
			item_levels = [2]
			agent_levels = [1, 1]
			
			ForagingState(agent_positions, item_positions, agent_levels, item_levels, width, height)
		end

		#initialize state with two items and iterate through all agent positions
		s0 = initialize_state()
		available_positions = setdiff(grid_positions, s0.item_positions)
		
		states = Vector{ForagingState{2, 1, width, height}}()
		for p1 in available_positions
			for p2 in setdiff(available_positions, (p1,))
				push!(states, ForagingState((p1, p2), s0.item_positions, s0.agent_levels, s0.item_levels, width, height))
			end
		end

		#remove item and iterate through all agent positions
		available_positions′ = grid_positions
		item_collect′ = (true,)
		for p1 in available_positions′
			for p2 in setdiff(available_positions′, (p1,))
				push!(states, ForagingState((p1, p2), s0.item_positions, item_collect′, s0.agent_levels, s0.item_levels, width, height))
			end
		end

		state_index = Dict{ForagingState{2, 1, width, height}, Int64}(makelookup(states))
		
		actions = ntuple(Returns(action_list), 2)
		
		#transition maps have two dimensions of length 6 for the 6 actions available to each agent and then the last dimension is the state space
		state_transition_map = Array{Int64, 3}(undef, 6, 6, length(states))
		reward_transition_map = Array{NTuple{2, Float32}, 3}(undef, 6, 6, length(states))
		terminal_states = BitVector(undef, length(states))

		for i_s in eachindex(states)
			s = states[i_s]
			terminal_states[i_s] = isterm(s)
		end

		for i_s in eachindex(states)
			for i_a1 in 1:6
				for i_a2 in 1:6
					s = states[i_s]
					(rewards, s′) = step(s, (i_a1, i_a2), 0f0)
					i_s′ = state_index[s′]
					state_transition_map[i_a1, i_a2, i_s] = i_s′
					reward_transition_map[i_a1, i_a2, i_s] = rewards
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
# ╠═╡ skip_as_script = true
#=╠═╡
const lbf_test = LevelBasedForaging.make_environment(;num_agents = 3)
  ╠═╡ =#

# ╔═╡ ac0a9ef8-6fbc-42f2-9433-727b235224e5
# ╠═╡ skip_as_script = true
#=╠═╡
const ex_5_3 = LevelBasedForaging.make_5_3_environment()
  ╠═╡ =#

# ╔═╡ a691fc4c-8ac6-484a-983f-ad2f32d2e5e1
const ex_small = LevelBasedForaging.make_small_environment()

# ╔═╡ a7acda55-6a77-4b38-91cc-5b19a18c5934
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

# ╔═╡ 68a327a1-886c-4b87-9a26-0ecc97d77311
#=╠═╡
plot_foraging_state(lbf_test.initialize_state())
  ╠═╡ =#

# ╔═╡ 924d646f-8fd7-4b0b-a5c5-0446abb6434e
#=╠═╡
plot_foraging_state(ex_5_3.states[6188])
  ╠═╡ =#

# ╔═╡ b149a025-3b7b-4ee5-a4a5-a6bcd035ad8b
#=╠═╡
const lbf_single_agent_reduction = StateMDP(lbf_test, sum)
  ╠═╡ =#

# ╔═╡ dac84639-2388-4ee8-9372-aae518a37f17
#=╠═╡
const tab_5_3 = TabularMDP(ex_5_3, sum)
  ╠═╡ =#

# ╔═╡ 83583ca7-bd96-4589-bbe0-e967849c3bd3
#=╠═╡
const value_iter = value_iteration_v(tab_5_3, 0.99f0)
  ╠═╡ =#

# ╔═╡ b0eb099f-d55e-4f2b-aa61-5b7dce374056
#=╠═╡
runepisode(tab_5_3; π = value_iter.optimal_policy)
  ╠═╡ =#

# ╔═╡ e5ef0212-ff93-4e2b-a7c1-d7ff53cbb8aa
#=╠═╡
const lbf_sarsa = q_learning(tab_5_3, 0.99f0; max_steps = 1_000_000, α = 0.2f0, ϵ = 0.01f0, save_history = true)
  ╠═╡ =#

# ╔═╡ b6df5773-94cd-488b-bb14-4713314a4575
#=╠═╡
function get_lbf_sarsa_statistics(algo; nruns = Base.Threads.nthreads(), kwargs...)
	1:nruns |> Map() do i
		output = algo(tab_5_3, 0.99f0; kwargs..., save_history = true)
		output.reward_history
	end |> foldxt((a, b) -> a .+ b) |> v -> v ./ nruns
end
  ╠═╡ =#

# ╔═╡ 1d609b68-52a1-4459-a605-cae08785f306
#=╠═╡
const lbf_sarsa_avg = get_lbf_sarsa_statistics(sarsa; max_steps = 1_000_000, α = 0.2f0, ϵ = 0.01f0, save_history = true)
  ╠═╡ =#

# ╔═╡ 7201de00-04d0-4eea-b814-08fe26fbabb5
#=╠═╡
const lbf_expected_sarsa_avg = get_lbf_sarsa_statistics(expected_sarsa; max_steps = 1_000_000, α = 0.2f0, ϵ = 0.1f0, save_history = true)
  ╠═╡ =#

# ╔═╡ 2d15d9d3-4fec-498f-a1c0-9ae3c10f2d2f
#=╠═╡
const lbf_q_learning_avg = get_lbf_sarsa_statistics(q_learning; max_steps = 1_000_000, α = 0.2f0, ϵ = 0.01f0, save_history = true)
  ╠═╡ =#

# ╔═╡ 09474fdf-df26-4e33-8f29-fc6ad9b85368
#=╠═╡
const lbf_double_q_learning_avg = get_lbf_sarsa_statistics(double_q_learning; max_steps = 1_000_000, α = 0.2f0, ϵ = 0.01f0, save_history = true)
  ╠═╡ =#

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
#=╠═╡
const value_iter_episode = runepisode(tab_5_3; π = value_iter.optimal_policy)
  ╠═╡ =#

# ╔═╡ a113a20e-a599-4dbb-87a0-f943245d75ce
#=╠═╡
@bind timestep Slider(1:length(value_iter_episode[1])+1)
  ╠═╡ =#

# ╔═╡ 57d02e2c-4387-4345-bed3-504fafcd2a26
#=╠═╡
@bind movie_timestep Clock(;max_value = length(value_iter_episode[1])+1, repeat = true, interval = 0.4)
  ╠═╡ =#

# ╔═╡ 1f7c4a86-d530-4d17-9640-6c1c7315f1bc
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

# ╔═╡ fdcfd9d8-5c2e-4e6f-a408-7c530b1bc5ff
#=╠═╡
if timestep_select
	plot_lbf_episode(movie_timestep)
else
	plot_lbf_episode(timestep)
end
  ╠═╡ =#

# ╔═╡ 604461b5-484e-468d-ada7-8f70d3b222c8
module TwoPlayerSoccer
	#import general RL and MARL types and functions
	import ..RL_Module
	import ..makelookup

	import ..TabularZeroSumGameTransitionSampler
	import ..TabularZeroSumGameStochasticTransition
	import ..TabularZeroSumGameTransition
	import ..TabularGameTransition
	import ..TabularStochasticGame
	import ..SparseVector
	
	"""
		Move

	Abstract type for the five movement primitives in Two-Player Soccer.

	# Subtypes
	- [`Up`](@ref), [`Down`](@ref), [`Left`](@ref), [`Right`](@ref) — cardinal movement
	- [`Noop`](@ref) — stand still
	"""
	abstract type Move end
	struct Up <: Move end
	struct Down <: Move end
	struct Left <: Move end
	struct Right <: Move end
	struct Noop <: Move end

	#position x/y coordinate in gridworld
	const Position = Tuple{Int64, Int64}

	const action_list = [Up(), Down(), Left(), Right(), Noop()]
	const action_index = makelookup(action_list)

	const action_list_tuple = (Up(), Down(), Left(), Right(), Noop())

	"""
		State{Width, Height, GoalHeight}

	State representation for the Two-Player Soccer environment.

	Tracks the grid positions of both agents and which agent currently has
	the ball.  The type parameters encode the grid dimensions and goal height,
	enabling compile-time dispatch for different field configurations.

	# Fields
	- `agent_positions::Tuple{Position, Position}`: (x, y) positions of agent 1 and agent 2
	- `agent1_ball::Bool`: Whether agent 1 currently possesses the ball
	"""
	struct State{Width, Height, GoalHeight}
		agent_positions::Tuple{Position, Position}
		agent1_ball::Bool
	end


	function State(agent_positions::Tuple{Position, Position}, agent1_ball::Bool, width::Integer, height::Integer)
		@assert isodd(width)
		@assert iseven(height)
		goal_y = Int64(height/2)
		State{width, height, goal_y}(agent_positions, agent1_ball)
	end

	#LBF episodes end after all items have been collected
	function isterm(s::State{W, H, GH}) where {W, H, GH} 
		s.agent1_ball && return (s.agent_positions[1][1] == 0)
		!s.agent1_ball && return (s.agent_positions[2][1] == W+1)
	end

	#rules to update an agent's position in the grid
	move(p::Position, ::Noop) = p
	move(p::Position, ::Up) = (p[1], p[2]+1)
	move(p::Position, ::Right) = (p[1]+1, p[2])
	move(p::Position, ::Down) = (p[1], p[2]-1)
	move(p::Position, ::Left) = (p[1]-1, p[2])

	move(p::Position, i_a::Integer) = move(p, action_list_tuple[i_a])

	
	function step(s::State{W, H, GH}, (a1, a2)::Tuple{M1, M2}, agent1_first::Bool) where {W, H, GH, M1<:Move, M2<:Move}
		isterm(s) && return (0f0, s)
		(i1, i2, i1ball, i1score_x, i2score_x, a_first, a_second) = if agent1_first
			(1, 2, s.agent1_ball, 0, W+1, a1, a2)
		else
			(2, 1, !s.agent1_ball, W+1, 0, a2, a1)
		end

		switch_ball = false
		p1′ = move(s.agent_positions[i1], a_first)
		move_into1 = (clamp(p1′[1], 1, W) == s.agent_positions[i2][1]) && (clamp(p1′[2], 1, H) == s.agent_positions[i2][2])
		if move_into1
			p1′ = s.agent_positions[i1]
		end

		p2′ = move(s.agent_positions[i2], a_second)
		move_into2 = (clamp(p2′[1], 1, W) == clamp(p1′[1], 1, W)) && (clamp(p2′[2], 1, H) == clamp(p1′[2], 1, H))
		
		switch_ball = (move_into1 && i1ball) || (move_into2 && !i1ball)

		if move_into2
			p2′ = s.agent_positions[i2]
		end

		i1ball′ = if switch_ball
			!i1ball
		else
			i1ball
		end

		i1score = (i1ball′ && ((p1′[2] == GH) || (p1′[2] == GH+1)) && (p1′[1] == i1score_x))
		i2score = (!i1ball′ && ((p2′[2] == GH) || (p2′[2] == GH+1)) && (p2′[1] == i2score_x))

		p1′ = (i1score ? i1score_x : clamp(p1′[1], 1, W), clamp(p1′[2], 1, H))
		p2′ = (i2score ? i2score_x : clamp(p2′[1], 1, W), clamp(p2′[2], 1, H))

		agent1_ball′ = if switch_ball
			!s.agent1_ball
		else
			s.agent1_ball
		end
		
		if (i1 == 1)
			r = Float32(i1score - i2score)
			s′ = State{W, H, GH}((p1′, p2′), agent1_ball′)
		else
			r = Float32(i2score - i1score)
			s′ = State{W, H, GH}((p2′, p1′), agent1_ball′)
		end

		return (r, s′)
	end
	
	# step(s::State, a) = step(s, a, rand(Bool))
			
	function make_sample_environment(;width = 5, height = 4)
		goal_y1 = Int64(height / 2)
		goal_y2 = goal_y1 + 1
		x_mid = floor(Int64, width / 2)
		function initialize_state()
			p1 = (x_mid + 2, goal_y2)
			p2 = (x_mid, goal_y1)
			agent1_ball = rand((true, false))
			State((p1, p2), agent1_ball, width, height)
		end

		agent_actions = ntuple(i -> action_list, 2)

		positions = Set((x, y) for x in 1:width for y in 1:height)
		states = Vector{State{width, height, goal_y1}}()
		for p1 in positions
			for p2 in setdiff(positions, (p1,))
				push!(states, State((p1, p2), true, width, height))
				push!(states, State((p1, p2), false, width, height))
			end
		end

		for p in positions
			push!(states, State((p, (width+1, goal_y1)), false, width, height))
			push!(states, State((p, (width+1, goal_y2)), false, width, height))
			push!(states, State(((0, goal_y1), p), true, width, height))
			push!(states, State(((0, goal_y2), p), true, width, height))
		end

		state_index = Dict{State{width, height, goal_y1}, Int64}(makelookup(states))

		function tabular_step(i_s::Integer, (i_a1, i_a2)::Tuple{Int64, Int64}, agent1_first::Bool = rand(Bool)) 
			(r, s′) = step(states[i_s], (action_list_tuple[i_a1], action_list_tuple[i_a2]), agent1_first)
			i_s′ = state_index[s′]
			(r, i_s′)
		end

		ptf = TabularZeroSumGameTransitionSampler(tabular_step)

		terminal_states = BitVector(isterm(s) for s in states)

		function initialize_state_index()
			s0 = initialize_state()
			state_index[s0]
		end
		
		#add sampler type transition function for tabular
		TabularStochasticGame(states, agent_actions, ptf, initialize_state_index, terminal_states; state_index = state_index)
	end	

	function make_distribution_environment(;width = 5, height = 4)

		game_sampler = make_sample_environment(;width = 5, height = 4)

		states = game_sampler.states
		agent_actions = game_sampler.agent_actions
		ptf_sampler = game_sampler.ptf
		terminal_states = game_sampler.terminal_states
		state_index = game_sampler.state_index

		reward_transition_map = Array{Vector{Float32}, 3}(undef, length(agent_actions[1]), length(agent_actions[2]), length(states))
		state_transition_map = Array{SparseVector{Float32, Int64}, 3}(undef, length(agent_actions[1]), length(agent_actions[2]), length(states))

		for (i, inds) in enumerate(CartesianIndices(reward_transition_map))
			i_a1 = inds[1]
			i_a2 = inds[2]
			i_s = inds[3]
			probs = SparseVector{Float32, Int64}(undef, length(states))
			rewards = Vector{Float32}()
			
			(r1, i_s′1) = ptf_sampler.step(i_s, (i_a1, i_a2), true)
			probs[i_s′1] += 0.5f0
			push!(rewards, r1)
			
			(r2, i_s′2) = ptf_sampler.step(i_s, (i_a1, i_a2), false)
			probs[i_s′2] += 0.5f0
			if i_s′2 == i_s′1
				@assert (r2 == r1)
			else 
				if i_s′2 > i_s′1
					push!(rewards, r2)
				else
					pushfirst!(rewards, r2)
				end
			end

			reward_transition_map[i] = rewards
			state_transition_map[i] = probs
		end

		ptf = TabularZeroSumGameTransition(state_transition_map, reward_transition_map)

		TabularStochasticGame(states, agent_actions, ptf, game_sampler.initialize_state_index, terminal_states; state_index = state_index)
	end
end

# ╔═╡ d00e3993-1189-49ed-803a-a8506e3b4b80
# ╠═╡ skip_as_script = true
#=╠═╡
const soccer_env = TwoPlayerSoccer.make_sample_environment()
  ╠═╡ =#

# ╔═╡ 07f82b02-a479-4b1d-824d-7d18ced98ad3
#=╠═╡
findall(s -> (s.agent_positions[1] == s.agent_positions[2]), soccer_env.states)
  ╠═╡ =#

# ╔═╡ db0fa811-6a74-4bf0-8806-58a4b513dd5e
#=╠═╡
soccer_env.ptf(soccer_env.initialize_state_index(), (3, 5))
  ╠═╡ =#

# ╔═╡ 74be33d5-a7ae-4b8c-8db9-0c37e7974549
# ╠═╡ skip_as_script = true
#=╠═╡
const soccer_env_dist = TwoPlayerSoccer.make_distribution_environment()
  ╠═╡ =#

# ╔═╡ 90680746-e84b-4a63-8809-568f91cd0475
#=╠═╡
soccer_env_dist.ptf(soccer_env.initialize_state_index(), (3, 5))
  ╠═╡ =#

# ╔═╡ ff335a51-e2ed-4313-af7b-1983c0a488a7
function independent_q_learning!((value_estimates, policies)::Tuple{NTuple{N, Matrix{T}}, NTuple{N, Matrix{T}}}, game::TabularStochasticGame{T, S, A, N, P, F}, γ::T, max_episodes::Integer, max_steps::Integer; i_s0 = game.initialize_state_index(), save_history = false, train_policies::AbstractVector{I} = 1:N, α::T = one(T) / 10, α_decay::T = one(T), ϵ::T = one(T) / 10) where {T<:Real,S, A, P, F<:Function, N, I <: Integer}
	ep = 1
	step = 0
	i_s = i_s0
	
	a = zeros(Int64, N)

	α_step = α

	reward_history = initialize_reward_history(game.ptf)
	episode_steps = Vector{Int64}()
	
	while (ep < max_episodes) && (step < max_steps)
		@inbounds @simd for n in 1:N
			a[n] = sample_action(policies[n], i_s)
		end
		
		rewards, i_s′ = game.ptf(i_s, NTuple{N, Int64}(a))
		step += 1
		
		for n in train_policies
			qs = value_estimates[n]
			r = get_game_reward(game.ptf, rewards, n)
			i_a = a[n]
			max_q′ = get_max_q(qs, i_s′)
			qs[i_a, i_s] += α_step*(r + γ*max_q′ - qs[i_a, i_s])
			make_ϵ_greedy_policy!(policies[n], i_s, qs; ϵ = ϵ)
		end
		

		if save_history
			push!(reward_history, rewards)
		end
		
		#if a terminal state is reached, need to reset episode
		if game.terminal_states[i_s′]
			save_history && push!(episode_steps, step)
			ep += 1
			i_s = game.initialize_state_index()
		else
			i_s = i_s′
		end

		α_step *= α_decay
	end

	for n in train_policies
		for i_s in eachindex(game.states)
			make_greedy_policy!(policies[n], i_s, value_estimates[n])
		end
	end
	
	(value_estimates = value_estimates, policies = policies, reward_history = reward_history, episode_steps = episode_steps)
end

# ╔═╡ 3ab5ea43-f8d5-46ac-a407-fd7c9af50540
independent_q_learning(game::TabularStochasticGame{T, S, A, N, P, F}, γ::T; max_steps = 100_000, max_episodes = typemax(Int64), init_value = zero(T), qs::NTuple{N, Matrix{T}} = initialize_agent_action_values(game, init_value), πs::NTuple{N, Matrix{T}} = make_random_policies(game), kwargs...) where {T<:Real, S, A, N, P, F<:Function} = independent_q_learning!((qs, πs), game, γ, max_episodes, max_steps; kwargs...) 

# ╔═╡ d4661036-1e7a-48ba-a05a-51537b7b4910
# ╠═╡ skip_as_script = true
#=╠═╡
const ex_5_3_iql = independent_q_learning(ex_5_3, 0.99f0; α = 0.01f0, max_steps = 1_000_000, save_history = true)
  ╠═╡ =#

# ╔═╡ bf52bae8-97d4-40b3-80c9-4afa2f77bd99
#=╠═╡
const soccer_sample_iql = independent_q_learning(soccer_env, 0.9f0; α = 0.1f0, max_steps = 1_000_000, save_history = true)
  ╠═╡ =#

# ╔═╡ 4e008a70-2536-440a-afcb-c8b9772f4c8c
#=╠═╡
const soccer_dist_iql = independent_q_learning(soccer_env_dist, 0.9f0; α = 0.1f0, max_steps = 1_000_000, save_history = true)
  ╠═╡ =#

# ╔═╡ b27cbf6a-2202-40d8-8946-7d9fc6def3e9
# ╠═╡ skip_as_script = true
#=╠═╡
function get_lbf_iql_statistics(;nruns = Base.Threads.nthreads(), kwargs...)
	1:nruns |> Map() do i
		output = independent_q_learning(ex_5_3, 0.99f0; kwargs..., save_history = true)
		[sum(a) for a in output.reward_history]
	end |> foldxt((a, b) -> a .+ b) |> v -> v ./ nruns
end
  ╠═╡ =#

# ╔═╡ 13f1c44a-ca9b-42f2-9e6c-4d7dd1658af7
#=╠═╡
const lbf_iql_avg = get_lbf_iql_statistics(;max_steps = 1_000_000, α = 0.2f0, ϵ = 0.01f0, save_history = true)
  ╠═╡ =#

# ╔═╡ 664429bb-d0fc-409f-9760-a14827c3159a
#=╠═╡
plot_episode_rewards([lbf_sarsa_avg, lbf_expected_sarsa_avg, lbf_q_learning_avg, lbf_double_q_learning_avg, lbf_iql_avg], ["Sarsa", "Expected Sarsa", "Q-learning", "Double Q-learning", "Independent Q-learning"], 30_000)
  ╠═╡ =#

# ╔═╡ d9102b45-90da-4c86-8ffe-5fc35847efbf
#=╠═╡
TabularMDP(soccer_env, true, soccer_sample_iql.policies[1])
  ╠═╡ =#

# ╔═╡ d24f3f83-ecad-414d-9fed-6d2c59969ea7
#=╠═╡
TabularMDP(soccer_env, false, soccer_sample_iql.policies[1])
  ╠═╡ =#

# ╔═╡ a5530b30-a169-408f-a297-9c96e39c7800
#=╠═╡
TabularMDP(soccer_env_dist, true, make_random_policies(soccer_env)[2])
  ╠═╡ =#

# ╔═╡ 75934603-1fd2-4a65-87f5-21fc14367c04
#=╠═╡
lbf_conv_test = TabularMDP(ex_5_3, true, make_random_policies(ex_5_3)[2])
  ╠═╡ =#

# ╔═╡ 5904f5e9-93b4-479d-9fc5-58efb00f8e01
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
Dates = "ade2ca70-3891-5945-98fb-dc099432e06a"
HypertextLiteral = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
PlutoDevMacros = "a0499f29-c39b-4c5c-807c-88074221b949"
PlutoPlotly = "8e989ff0-3d88-8e9f-f020-2b208a939ff0"
PlutoProfile = "ee419aa8-929d-45cd-acf6-76bd043cd7ba"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"

[compat]
BenchmarkTools = "~1.7.0"
CSV = "~0.10.16"
DataFrames = "~1.8.1"
HypertextLiteral = "~0.9.5"
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
project_hash = "b754673dde3a5e79a95d244265098982d65ce822"

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
version = "2025.11.4"

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
git-tree-sha1 = "d05693d339e37d6ab134c5ab53c29fce5ee5d7d5"
uuid = "892a3eda-7b42-436c-8928-eab12a02cf0e"
version = "0.4.4"

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
# ╠═dd64a78d-a8f1-4b9b-aa8b-95ee48936268
# ╠═28920a55-4e8c-4dfc-84a0-03eeb13b9975
# ╠═11b17cb5-30fa-4a66-bf71-52b4dc9f2f02
# ╠═0b20d83b-eef0-4006-891e-9d15282ea57b
# ╠═c42b7d58-922d-4b17-a03c-62433c9adeb9
# ╠═71d96c8a-d036-45a9-95e4-bbf4ee079607
# ╠═6f98c460-6d48-457d-a40c-6b8588fb01ef
# ╠═c9b36b87-4a88-4398-95ce-4feb4f9839f7
# ╠═ed41ec94-38be-40de-bf61-c927c242cef7
# ╠═b815429b-fbe3-443c-9a76-d4b194eca26d
# ╠═23a5f4c3-5f6a-473a-9ee5-0c82cf8c7f8e
# ╠═75726eb9-eb0e-42a3-a648-dbfce92cbd6e
# ╟─ed152a1a-0fc9-40f3-91b4-e40246725847
# ╠═79888a7b-c604-4db2-932a-ee9bfd378d9e
# ╠═faac6925-b526-4ee0-a84c-b660e6819313
# ╠═d2bfbf48-2ca0-483f-afab-9a292343beca
# ╠═e838d71d-4120-4e4c-8727-82caf6a1431c
# ╠═53c087e9-ad7d-4a83-a373-0e0d3e406724
# ╠═77719b6d-893e-4011-b676-cbd322d3a835
# ╠═eb6189fd-68e0-435f-b736-dab754522ea9
# ╠═727885f8-ad7a-44b4-845d-9a9d8aebdb87
# ╠═b0b41d5e-4944-4fc8-b108-175c559e3d63
# ╠═9908b24b-77c4-4ece-99b5-337dc6725a13
# ╠═475e855e-ca4b-4466-ba8b-ca0edefb4f82
# ╠═89b1cb3b-bf58-4ca3-a08d-736ee2af2b0f
# ╠═fab57783-bcf7-4fd8-a5d1-5824db9226e4
# ╠═3f9d927e-4158-4531-b31f-56aae3ac0c7b
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
# ╠═68a327a1-886c-4b87-9a26-0ecc97d77311
# ╟─c1cc33d2-ad34-4ad8-b47a-6be3aa44c8ed
# ╠═ac0a9ef8-6fbc-42f2-9433-727b235224e5
# ╠═a691fc4c-8ac6-484a-983f-ad2f32d2e5e1
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
# ╠═b0eb099f-d55e-4f2b-aa61-5b7dce374056
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
# ╟─775fd220-532d-49ba-968f-20781c5d6708
# ╠═604461b5-484e-468d-ada7-8f70d3b222c8
# ╠═d00e3993-1189-49ed-803a-a8506e3b4b80
# ╠═74be33d5-a7ae-4b8c-8db9-0c37e7974549
# ╠═07f82b02-a479-4b1d-824d-7d18ced98ad3
# ╠═db0fa811-6a74-4bf0-8806-58a4b513dd5e
# ╠═90680746-e84b-4a63-8809-568f91cd0475
# ╟─91c667a3-293d-44d7-9437-901274f27af9
# ╠═2bfa6ee7-d580-4ea0-9551-28c34570dc1e
# ╟─b53e1cfd-4f81-4e1e-bff0-3435d3fcd221
# ╠═769e5dc3-684e-4d36-a63f-0edb90331c59
# ╠═5629f70c-b2cc-4a68-b5f0-9a11aef6dcbc
# ╟─c2bec8e1-9df1-4f5f-8e92-0f6f5f9955a4
# ╠═9f05e777-fb94-42f9-9cd7-b461f10f199d
# ╠═a2284047-0aaf-41b2-a6a0-de082cdeb5a2
# ╠═acdb7ffb-df0a-4609-8b07-87b4f3f9a145
# ╟─cc259b71-cbe0-4990-ba6f-b495f6b0a2b7
# ╠═9b75d562-ac45-406a-867b-b6d2af5822ee
# ╠═c755f0b1-2299-42b4-a28d-d53980260e79
# ╠═50422bb8-ff98-4803-8a3b-9a42454d793d
# ╠═ff335a51-e2ed-4313-af7b-1983c0a488a7
# ╠═3f44f098-5aef-4287-901e-67a6062f0f02
# ╠═d95d9945-944a-497b-b07b-7b4d4eb1f01d
# ╠═3ab5ea43-f8d5-46ac-a407-fd7c9af50540
# ╟─eec99b81-8150-4a65-90a0-283cd5da877a
# ╠═d4661036-1e7a-48ba-a05a-51537b7b4910
# ╠═b27cbf6a-2202-40d8-8946-7d9fc6def3e9
# ╠═13f1c44a-ca9b-42f2-9e6c-4d7dd1658af7
# ╠═664429bb-d0fc-409f-9760-a14827c3159a
# ╟─8c6f4ab7-9172-4723-bc2b-e61fe0315e1c
# ╠═bf52bae8-97d4-40b3-80c9-4afa2f77bd99
# ╠═4e008a70-2536-440a-afcb-c8b9772f4c8c
# ╠═d9102b45-90da-4c86-8ffe-5fc35847efbf
# ╠═d24f3f83-ecad-414d-9fed-6d2c59969ea7
# ╠═a5530b30-a169-408f-a297-9c96e39c7800
# ╠═75934603-1fd2-4a65-87f5-21fc14367c04
# ╟─75a6b525-a507-4cf0-9ba7-ec1e76e86465
# ╟─9a0a63a1-0292-4937-9d7f-8c18bc3eb28b
# ╠═924d646f-8fd7-4b0b-a5c5-0446abb6434e
# ╠═a7acda55-6a77-4b38-91cc-5b19a18c5934
# ╟─1a488045-380d-4441-a6ec-186d3c53420d
# ╠═4e509812-7bdf-4928-bee6-55f2d142be67
# ╠═9260398e-6842-4b11-9845-98900884b94a
# ╠═b9ee4c0e-e556-4f67-9cce-533b1f35ed96
# ╠═9f1aa27d-2214-4b7c-9110-1566603887d2
# ╠═5904f5e9-93b4-479d-9fc5-58efb00f8e01
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
