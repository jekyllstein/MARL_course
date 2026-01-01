### A Pluto.jl notebook ###
# v0.20.21

using Markdown
using InteractiveUtils

# ╔═╡ 20a8c921-816f-4c5c-9341-b7749644d249
using PlutoDevMacros

# ╔═╡ c3e732e2-2a10-4de3-800f-0378e0acd0dc
begin
	PlutoDevMacros.@frompackage @raw_str(joinpath(@__DIR__, "..", "RL_Module")) begin 
		using RL_Module
		using >.Random, >.Statistics, >.LinearAlgebra, >.Transducers, >.StaticArrays, >.DataStructures, >.SparseArrays
	end
end

# ╔═╡ e5a78e58-0208-4c6a-9f7f-4779e384855b
using JuMP

# ╔═╡ db2d2393-e984-4828-9ec7-d9dfe076bc79
# ╠═╡ skip_as_script = true
#=╠═╡
begin
	using PlutoUI, PlutoPlotly, PlutoProfile, BenchmarkTools, LaTeXStrings, HypertextLiteral, DataFrames, Dates, CSV
	
	TableOfContents()
end
  ╠═╡ =#

# ╔═╡ 00999193-757c-4bea-a24c-a5ed8633d826
begin
	@only_in_nb include(joinpath(@__DIR__, "multi_agent_types.jl"))
end

# ╔═╡ c62ca696-4fbe-42e2-ab00-4a916f9fac8d
md"""
# Tabular Stochastic Game Utilities
"""

# ╔═╡ 844c29f5-fe4d-49cc-9dab-4b0ed6256a85
begin
	function TabularRL.runepisode!((states, joint_actions, rewards)::Tuple{Vector{Int64}, Vector{NTuple{N, Int64}}, R}, game::TabularStochasticGame{T, S, A, N, P, F}; i_s0::Integer = game.initialize_state_index(), πs::NTuple{N, M} = make_random_policies(game), a0::NTuple{N, Int64} = sample_joint_action(πs, i_s0), max_steps = Inf) where {T<:Real, S, A, P, F, N, M <: AbstractMatrix{T2} where T2<:Real, R<:Union{Vector{NTuple{N, T}}, Vector{T}}}
		@assert any(game.terminal_states) #ensure that some terminal state exists since episodes are only defined for problems with terminal states
		i_s = i_s0
		l = length(states)
		@assert l == length(joint_actions) == length(rewards)
	
		function add_value!(v, x, i) 
			if i > l
				push!(v, x)
			else
				v[i] = x
			end
		end 
		add_value!(states, i_s, 1)
		a = a0
		(r, i_s′) = game.ptf(i_s, a0)
		add_value!(joint_actions, a, 1)
		add_value!(rewards, r, 1)
		step = 2
		i_sterm = i_s
		if game.terminal_states[i_s′]
			i_sterm = i_s′
		else
			i_sterm = i_s
		end
		i_s = i_s′
	
		#note that the terminal state will not be added to the state list
		while !game.terminal_states[i_s] && (step <= max_steps)
			add_value!(states, i_s, step)
			(r, i_s′, a) = game.ptf(i_s, πs)
			add_value!(joint_actions, a, step)
			add_value!(rewards, r, step)
			i_s = i_s′
			step += 1
			if game.terminal_states[i_s′]
				i_sterm = i_s′
			end
		end
		return states, joint_actions, rewards, i_sterm, step-1
	end

	initialize_episode_rewards(ptf::AbstractTabularGameTransition{T, N}) where {T<:Real, N} = Vector{NTuple{N, T}}()
	initialize_episode_rewards(ptf::AbstractTabularCommonRewardGameTransition{T, N}) where {T<:Real, N} = Vector{T}()
	initialize_episode_rewards(ptf::AbstractTabularZeroSumGameTransition{T}) where {T<:Real} = Vector{T}()
	
	function TabularRL.runepisode(game::TabularStochasticGame{T, S, A, N, P, F}; kwargs...) where {T<:Real, S, A, P<:AbstractTabularGameTransition, F, N}
		states = Vector{Int64}()
		actions = Vector{NTuple{N, Int64}}()
		rewards = initialize_episode_rewards(game.ptf)
		runepisode!((states, actions, rewards), game; kwargs...)
	end
end

# ╔═╡ 798386fc-f572-4cff-9222-00ab3452faae
md"""
## Soccer Game Example

Stochastic game which is mostly deterministic.  For a given action selection, each transition has two possible outcomes depending on which player goes first.  For transitions in which there is no possibility of collision, only one outcome is possible.  The visualization below shows an episode run using two random policies.
"""

# ╔═╡ 3120f952-d848-4f37-b20c-922b5e897587
const soccer_game = TwoPlayerSoccer.make_distribution_environment()

# ╔═╡ 69ed465f-7d0b-480b-90a2-186f51258b38
const soccer_random_policy = make_random_policies(soccer_game)[1]

# ╔═╡ 96b067f8-5e7e-459a-9e3f-5640aa18680b
md"""
### Soccer Visulization with Random Policies
"""

# ╔═╡ 164251a2-a42c-45d7-839f-bf06d082a971
md"""
### Soccer Game Statistics of Random Policies
"""

# ╔═╡ bd5e2748-4a79-4fab-8019-f6a733c9a653
md"""
With two random policies we'd expect the win percentage to be 50%, but we can see from the statistics that starting with the ball produces an advantage of about 10% and games last an average of 100 steps.
"""

# ╔═╡ 1ed1be16-d0d0-446f-b096-5cc5e48e2ab5
function display_soccer_statistics(results::NamedTuple, name1::AbstractString, name2::AbstractString)
f(x) = round(Float64(x); sigdigits = 3)

md"""
|$name1 vs $name2|Win %|Loss %|Draw %|Avg Steps|Avg Discounted Reward|
|---|---|---|---|---|---|
|Overall|$(f(results.overall_stats.win_pct))|$(f(results.overall_stats.loss_pct))|$(f(results.overall_stats.draw_pct))|$(f(results.overall_stats.avg_steps))|$(f(results.overall_stats.avg_reward))|
|Start with Ball|$(f(results.ball_start_stats.win_pct))|$(f(results.ball_start_stats.loss_pct))|$(f(results.ball_start_stats.draw_pct))|$(f(results.ball_start_stats.avg_steps))|$(f(results.ball_start_stats.avg_reward))|
|Start without Ball|$(f(results.noball_start_stats.win_pct))|$(f(results.noball_start_stats.loss_pct))|$(f(results.noball_start_stats.draw_pct))|$(f(results.noball_start_stats.avg_steps))|$(f(results.noball_start_stats.avg_reward))|
"""
end

# ╔═╡ e7cc25e9-b4fe-42a6-b07f-1940e7018068
function test_soccer_policies(π1, π2; ntrials = 100_000, max_ep_length = 1_000)
	data = 0:ntrials |> Map() do i
		out = runepisode(soccer_game; πs = (π1, π2), max_steps = max_ep_length)
		player1_ball = soccer_game.states[out[1][1]].agent1_ball
		l = length(out[3])
		(player1_ball, l, out[3][end], out[3][end] * 0.9f0^l)
	end |> tcollect

	player1_ball_inds = findall(a -> a[1], data)
	player1_not_ball_inds = setdiff(eachindex(data), player1_ball_inds)

	player1_balln = length(player1_ball_inds)
	player1_not_balln = ntrials - player1_balln

	function calc_stats(data)
		n = length(data)
		win_pct = sum(isone(a[3]) for a in data) / n
		loss_pct = sum(a[3] == -1 for a in data) / n
		draw_pct = sum(iszero(a[3]) for a in data) / n
		avg_l = sum(a[2] for a in data) / n
		avg_reward = sum(a[4] for a in data) / n
	
		(win_pct = win_pct, loss_pct = loss_pct, draw_pct = draw_pct, avg_steps = avg_l, avg_reward = avg_reward)
	end

	overall_stats = calc_stats(data)
	ball_start_stats = calc_stats(view(data, player1_ball_inds))
	noball_start_stats = calc_stats(view(data, player1_not_ball_inds))
	
	(overall_stats = overall_stats, ball_start_stats = ball_start_stats, noball_start_stats = noball_start_stats)
end

# ╔═╡ fe4bd7bf-a73b-401c-8682-dc734122f5e8
function display_soccer_statistics(π1, π2, name1::AbstractString, name2::AbstractString; kwargs...)
	results = test_soccer_policies(π1, π2; kwargs...)
	display_soccer_statistics(results, name1, name2)
end

# ╔═╡ 939f23a4-d089-49ab-807c-ac107c3664d4
display_soccer_statistics(soccer_random_policy, soccer_random_policy, "Random", "Random"; max_ep_length = 10_000)

# ╔═╡ af9ba099-95b3-40cb-b1fe-b6091cbe7cd2
md"""
# Solution Methods
"""

# ╔═╡ bd7ce958-162b-419f-a177-6d1b21905e1c
md"""
## Value Iteration for Stochastic Games
"""

# ╔═╡ a01f5f05-788d-491e-9e2f-a047a11fbc03
md"""
### Minimax Solution

For two-player zero-sum games, there is a unique minimax solution that depends on the reward values for each joint action.  The function calculates the value and solution from the perspective of either the first or second player.
"""

# ╔═╡ e527160e-6d64-48d2-ab4d-abc67afebb97
md"""
### Bellman Value Iteration for Zero-Sum Games

Bellman style recursive reward calculation for zero sum games.  These games only require a single reward value for both players since ``r_x \doteq -r_y`` for all circumstances.  The value is calculated from the perspective of the first player by convention.  Below are implementations that cover deterministic and stochastic zero-sum games.
"""

# ╔═╡ 54db08a2-b7c5-4311-8bca-fce64d665563
begin
	function bellman_game_reward_value(ptf::TabularZeroSumGameDeterministicTransition{T}, i_s::Integer, a::Tuple{Int64, Int64}, γ::T, v_est::Vector{T}) where T<:Real
		r = ptf.reward_transition_map[a..., i_s]
		i_s′ = ptf.state_transition_map[a..., i_s]
		return r + γ*v_est[i_s′]
	end

	function bellman_game_reward_value(ptf::TabularZeroSumGameStochasticTransition{T}, i_s::Integer, a::Tuple{Int64, Int64}, γ::T, v_est::Vector{T}) where T<:Real
		reward_transitions = ptf.reward_transition_map[a..., i_s]
		state_transitions = ptf.state_transition_map[a..., i_s]
		isempty(reward_transitions) && return typemin(T) #if there are no reward transitions then the action is invalid
		v_avg = zero(T)
		
		@inbounds @simd for i in eachindex(reward_transitions)
			r = reward_transitions[i]
			p = state_transitions.nzval[i]
			i_s′ = state_transitions.nzind[i]
			v′ = r + γ*v_est[i_s′]
			v_avg += p * v′
		end
		return v_avg
	end
end

# ╔═╡ fe077be2-467e-43a7-8729-3f3d48b8c91b
md"""
### Soccer Game Example
"""

# ╔═╡ 9a615e94-6be8-47f5-aa38-91088f3accc3
md"""
#### Minimax vs Minimax
"""

# ╔═╡ bfd2d1a0-7b3e-4fbf-9a29-22de88bbc08d
#compare solution states with different discount rates to see which states are affected by that
#show policy and value as a grid with one agent in place and the other squqares show the value/policy for the first agent when the second agent is in that square, this would be more useful for value to see what is desirable for the position of the second agent when you are the first
#add value iteration for policy to train against one of these.  Need to create stochastic distribution MDP based on the other agent's policies
#note that the non-game based algorithms cannot reproduce stochastic policies like what is needed for some of the game states

# ╔═╡ badfc16b-3b6e-471a-8e6e-d7ced4885770
#likelihood to win depends on who starts with the ball.  the player with the ball wins about 2/3 of the time with most discount rates

# ╔═╡ 09e01730-c983-48bf-9e7a-a3d2ccc0a631
md"""
#### Minimax vs Random
"""

# ╔═╡ f31c6793-2bfb-4970-9b36-fbc95460be30
md"""
## Joint-Action Learning

When we can only sample data from a game, we must rely on a method which uses sample averaging to calculate game values.  The solution concept is the same as that for value iteration but with updates occuring stochastially rather than with expected updates.
"""

# ╔═╡ 2d8f6a70-1f8e-4ad6-a616-e725b5c0a377
md"""
### Soccer Game Example
"""

# ╔═╡ 9ea31e89-6279-456d-be11-e915686e0be4
md"""
#### Minimax Q vs Minimax Q
"""

# ╔═╡ 3ec640b5-9c7a-4973-b128-ec438d45f5b7
md"""
The lower table on the right represents the estimated game values compared to the exact values shown above.  During an episode we can see states where the true minimax policy does not match the one learned by joint-action learning.
"""

# ╔═╡ efcc1361-774a-4245-b034-090dc29ae131
md"""
#### Minimax Q vs Random
"""

# ╔═╡ b2872d0f-f780-449a-b17d-72d1506f90c0
md"""
#### Random Performance vs True Minimax

Comparing the two policies we can see a higher reward value and higher winrate for the true solution compared to the estimated one.
"""

# ╔═╡ bbdece04-95cd-4333-8244-eaf6f0be4b18
md"""
#### Minimax Q vs Minimax
"""

# ╔═╡ dfd93d6c-da3a-4f02-b8a4-6ebec3dc8284
md"""
Player A follows the estimated minimax solution against Player B with the exact minimax solution
"""

# ╔═╡ 989c2b5b-1d33-4292-b372-6c9a1f13be20
md"""
## Independent Q-Learning

This method does not learn joint-action values and is an example of a single-agent RL reduction from the previous chapter.  Each agent attempts to maximize its own performance based on action values learned from the non-stationary environment that includes the other player.
"""

# ╔═╡ c4929e3a-c48f-4ba1-bf78-32f42fa7446d
const soccer_iql = independent_q_learning(soccer_game, 0.9f0; α = 1f0, max_steps = 10_000_000, α_decay = 0.99999954f0, ϵ = 0.2f0, save_history = true)

# ╔═╡ e6f529bf-c71e-431a-90ae-884d59d12132
const soccer_iql_stats = display_soccer_statistics(soccer_iql.policies..., "IQL", "IQL")

# ╔═╡ 448ed962-b29f-445a-bb62-29c617a0b12e
md"""
Notice that these players do not perform according to the equilibrium solution which requires stochastic policies.  Instead they get trapped in a draw only sceneario but the algorithm does not appear to converge looking at the average reward over time.  That is because any deterministic policy can be exploited by another one resulting in cycles of performance swapping.
"""

# ╔═╡ eda86d7a-87ca-4f07-9ca9-e9045906da92
const iql_vs_random_stats = display_soccer_statistics(soccer_iql.policies[1], soccer_random_policy, "IQL", "Random")

# ╔═╡ 33669b7d-4609-4a11-a6d2-7569a582353e
md"""
### Policy Performance vs Optimal Opponents

Now that we have used different methods to calculate soccer policies, we can train opponents specifically against them to see if the policies are exploitable.  These opponents represent worst case scenarios in which a policy can be thoroughly studied by an adversary to tailor a solution designed to perform well against it ignoring all other possible opponents.  Fixing one player's startegy is turns the problem into an MDP which has an optimal solution for the other player that can be written as a deterministic strategy.
"""

# ╔═╡ 7e0e10fe-2c5c-4c27-b21a-2568bebe8e6d
md"""
#### Random Policy

Previously trained strategies had high success against the random policy, but training specifically against a random opponent can yield different results that achieve even higher expected rewards.
"""

# ╔═╡ c3418786-fa47-4eb3-b239-3183ff920cc7
const soccer_random_vs_optimal = independent_q_learning(soccer_game, 0.9f0; α = 1f0, max_steps = 1_000_000, train_policies = 2:2, save_history = true, ϵ = 0.2f0, α_decay = 0.9999954f0)

# ╔═╡ 91b0b90b-a1d7-42ac-a493-257dcc29bc6a
const random_vs_optimal_stats = display_soccer_statistics(soccer_random_vs_optimal.policies..., "Random", "Optimal")

# ╔═╡ b84645f7-7ce2-498a-9db7-0da603e2904a
md"""
#### Minimax Solution

This strategy should be unexploitable meaning the best expected discounted reward one could hope to achieve against it is 0
"""

# ╔═╡ 63af1889-a4a6-4d0e-a21f-21107c0efed2
const soccer_iql_vs_optimal = independent_q_learning(soccer_game, 0.9f0; α = 1f0, max_steps = 1_000_000, πs = (soccer_iql.policies[1], copy(soccer_random_policy)), train_policies = 2:2, save_history = true, ϵ = 0.05f0, α_decay = 0.9999954f0)

# ╔═╡ bd291ecb-f529-46da-8fab-ae172535afa9
#=╠═╡
plot(soccer_iql_vs_optimal.reward_history |> cumsum |> v -> v ./ (1:length(v)))
  ╠═╡ =#

# ╔═╡ 7f486f74-6b68-4e03-9d09-858091d51784
const iql_vs_optimal_stats = display_soccer_statistics(soccer_iql_vs_optimal.policies..., "IQL", "Optimal")

# ╔═╡ b4380430-e223-44e9-96e7-e784de33020d
md"""
## Joint-Action Learning with Agent Modeling
"""

# ╔═╡ 0ada2900-e35b-42c4-8e35-f357b57a2579
md"""
### Zero-sum Games
"""

# ╔═╡ 422d5853-3a5e-4f84-96a4-8b05a752c64b
begin
	function calculate_AV(i_s::Integer, i_a::Integer, πs::NTuple{2, Matrix{T}}, q_est::Array{T, 3}, ::Val{true}) where {T<:Real}
		x = zero(T)
		π2 = πs[2]
		i_a2_inds = 1:size(π2, 1)
		@inbounds @simd for i_a2 in i_a2_inds
			x += q_est[i_a, i_a2, i_s] * π2[i_a2, i_s]
		end
		return x
	end

	function calculate_AV(i_s::Integer, i_a::Integer, πs::NTuple{2, Matrix{T}}, q_est::Array{T, 3}, ::Val{false}) where {T<:Real}
		x = zero(T)
		π1 = πs[1]
		i_a1_inds = 1:size(π1, 1)
		@inbounds @simd for i_a1 in i_a1_inds
			x -= q_est[i_a1, i_a, i_s] * π1[i_a1, i_s]
		end
		return x
	end

	calculate_AV(i_s::Integer, i_a::Integer, πs::NTuple{2, Matrix{T}}, q_est::Array{T, 3}, player1::Bool) where {T<:Real} = calculate_AV(i_s, i_a, πs, q_est, Val(player1))
end

# ╔═╡ 1c18c351-9afe-41b0-8e12-c4a7441d8499
md"""
### General-sum Games
"""

# ╔═╡ 9abf652e-7ab2-43a1-80ed-3a992bf720f5
function calculate_AV(i_s::Integer, i_a::Integer, πs::NTuple{N, Matrix{T}}, q_est::Array{T, Np1}, other_player_inds::Vector{Int64}, a_minus_inds::Vector{CartesianIndex{N}}) where {N, Np1, T<:Real}
	x = zero(T)
	@inbounds @simd for a in a_minus_inds
		x += q_est[a, i_s] * prod(πs[n][a[n], i_s] for n in other_player_inds) 
	end
	return x
end

# ╔═╡ 42e13eeb-b6e4-49da-81b4-e8c16ed74688
function get_max_av(i_s::Integer, πs::NTuple{2, Matrix{T}}, q_est::Array{T, 3}, player1) where T<:Real
	i_a_max = 0
	av_max = typemin(T)
	for i_a in 1:size(πs[1], 1)
		av = calculate_AV(i_s, i_a, πs, q_est, player1)
		newmax = (av > av_max)
		av_max = av_max*!newmax + newmax*av
		i_a_max = i_a_max*!newmax + newmax*i_a
	end
	return (i_a_max, av_max)
end

# ╔═╡ b199d65b-95ab-41ab-bce9-fd43bda79f35
function get_max_av(i_s::Integer, πs::NTuple{N, Matrix{T}}, q_est::Array{T, Np1}, player_index::Integer, other_player_inds::Vector{Int64}, a_minus_inds::Vector{Vector{CartesianIndex{N}}}) where {N, Np1, T<:Real}
	i_a_max = 0
	av_max = typemin(T)
	for i_a in 1:size(πs[player_index], 1)
		av = calculate_AV(i_s, i_a, πs, q_est, other_player_inds, a_minus_inds[i_a])
		newmax = (av > av_max)
		av_max = av_max*!newmax + newmax*av
		i_a_max = i_a_max*!newmax + newmax*i_a
	end
	return (i_a_max, av_max)
end

# ╔═╡ 9918d23e-a266-4c5c-b63b-f984c8db1bb5
#joint action learning with game theory
function jal_am!(q_est::Array{T, 3}, πs::NTuple{2, Matrix{T}}, game::TabularStochasticGame{T, S, A, 2, P, F}, γ::T, max_episodes::Integer, max_steps::Integer; α::T = one(T) / 10, ϵ::T = one(T) / 10, α_decay::T = one(T), save_history::Bool = false, save_policy_history::Bool = false) where {T<:Real, S, A, P<:AbstractTabularZeroSumGameTransition{T}, F<:Function}
	ep = 1
	step = 1
	i_s = game.initialize_state_index()
	a = [0, 0]

	reward_history = Vector{T}()
	policy_history = Vector{NTuple{2, Matrix{T}}}()

	num_actions = NTuple{2, Int64}(length(game.agent_actions[n]) for n in 1:2)

	action_counts = NTuple{2, Matrix{T}}(ones(T, size(πs[i])...) for i in 1:2)
	action_totals = NTuple{2, Vector{T}}(num_actions[i] .* ones(T, length(game.states)) for i in 1:2)

	α_step = α
	
	while (ep <= max_episodes) && (step <= max_steps)
		if rand() < ϵ
			a[1] = rand(eachindex(game.agent_actions[1]))
		else
			i_a_max, av_max = get_max_av(i_s, πs, q_est, true)
			a[1] = i_a_max
		end

		if rand() < ϵ
			a[2] = rand(eachindex(game.agent_actions[2]))
		else
			i_a_max, av_max = get_max_av(i_s, πs, q_est, false)
			a[2] = i_a_max
		end

		(r, i_s′) = game.ptf(i_s, NTuple{2, Int64}(a))

		save_history && push!(reward_history, r)

		#update agent models
		for n in 1:2
			π = πs[n]
			action_counts[n][a[n], i_s] += one(T)
			action_totals[n][i_s] += one(T)
			ntot = action_totals[n][i_s]
			for i_a in 1:num_actions[n]
				π[i_a, i_s] = action_counts[n][i_a, i_s] / ntot
			end
		end

		save_policy_history && push!(policy_history, deepcopy(πs))

		if game.terminal_states[i_s′]
			game_state_value = zero(T)
			i_s′ = game.initialize_state_index()
			ep += 1
		else
			i_a_max, av_max = get_max_av(i_s′, πs, q_est, true)
			game_state_value = av_max
		end

		
		target = r + γ*game_state_value
		δ = target - q_est[a[1], a[2], i_s]
		q_est[a[1], a[2], i_s] += α_step * δ

		i_s = i_s′
		step += 1
		α_step *= α_decay
	end
	
	return (joint_action_values = q_est, policies = πs, reward_history = reward_history, policy_history = policy_history)
end	

# ╔═╡ 865b6b13-531e-4960-8bfb-2e22906b077d
#joint action learning with game theory
function jal_am!(q_ests::NTuple{N, Array{T, Np1}}, πs::NTuple{N, Matrix{T}}, game::TabularStochasticGame{T, S, A, N, P, F}, γ::T, max_episodes::Integer, max_steps::Integer; α::T = one(T) / 10, ϵ::T = one(T) / 10, α_decay::T = one(T), ϵ_decay::T = one(T), save_history::Bool = false, use_linear_decay::Bool = false, α_min::T = zero(T), ϵ_min::T = zero(T), save_policy_history::Bool = false) where {T<:Real, S, A, N, Np1, P<:AbstractTabularGameTransition{T, N}, F<:Function}
	@assert Np1 == N + 1
	ep = 1
	step = 1
	i_s = game.initialize_state_index()
	a = zeros(Int64, N)

	reward_history = Vector{NTuple{N, T}}()
	episode_steps = Vector{Int64}()
	policy_history = Vector{NTuple{N, Matrix{T}}}()

	num_actions = NTuple{N, Int64}(length(game.agent_actions[n]) for n in 1:N)

	action_counts = NTuple{N, Matrix{T}}(ones(T, size(πs[i])...) for i in 1:N)
	action_totals = NTuple{N, Vector{T}}(num_actions[i] .* ones(T, length(game.states)) for i in 1:N)

	action_inds = CartesianIndices(rand(num_actions...))

	a_minus_inds = NTuple{N, Vector{Vector{CartesianIndex{N}}}}([filter(a -> a[n] == i, action_inds) for i in 1:num_actions[n]] for n in 1:N)

	other_player_inds = [vcat(1:n-1, n+1:N) for n in 1:N]

	game_state_values = zeros(T, N)

	α_step = α
	ϵ_step = ϵ
	
	while (ep <= max_episodes) && (step <= max_steps)
		for n in 1:N
			if rand() < ϵ_step
				a[n] = rand(eachindex(game.agent_actions[1]))
			else
				i_a_max, av_max = get_max_av(i_s, πs, q_ests[n], n, other_player_inds[n], a_minus_inds[n])
				a[n] = i_a_max
			end
		end

		joint_action = NTuple{N, Int64}(a)
		(rewards, i_s′) = game.ptf(i_s, joint_action)

		save_history && push!(reward_history, rewards)

		#update agent models
		for n in 1:N
			π = πs[n]
			action_counts[n][a[n], i_s] += one(T)
			action_totals[n][i_s] += one(T)
			ntot = action_totals[n][i_s]
			for i_a in 1:num_actions[n]
				π[i_a, i_s] = action_counts[n][i_a, i_s] / ntot
			end
		end

		save_policy_history && push!(policy_history, deepcopy(πs))

		if game.terminal_states[i_s′]
			game_state_values .= zero(T)
			i_s′ = game.initialize_state_index()
			save_history && push!(episode_steps, step)
			ep += 1
		else
			for n in 1:N
				i_a_max, av_max = get_max_av(i_s′, πs, q_ests[n], n, other_player_inds[n], a_minus_inds[n])
				game_state_values[n] = av_max
			end
		end

		for n in 1:N
			target = rewards[n] + γ*game_state_values[n]
			δ = target - q_ests[n][joint_action..., i_s]
			q_ests[n][joint_action..., i_s] += α_step * δ
		end

		i_s = i_s′
		step += 1

		if use_linear_decay && max_steps <= typemax(T)
			α_step -= (α - α_min) / max_steps
			ϵ_step -= (ϵ - ϵ_min) / max_steps
		else
			α_step *= α_decay
			ϵ_step *= ϵ_decay
		end
	end
	
	return (joint_action_values = q_ests, policies = πs, reward_history = reward_history, episode_steps = episode_steps, policy_history = policy_history)
end	

# ╔═╡ da51fa8e-02cc-43a8-be86-6d9e6514bd46
begin
	jal_am(game::TabularStochasticGame{T, S, A, 2, P, F}, γ::T, max_episodes::Integer, max_steps::Integer; init_value::T = zero(T), q_est::Array{T, 3} = ones(T, length(game.agent_actions[1]), length(game.agent_actions[2]), length(game.states)) .* init_value, πs::NTuple{2, Matrix{T}} = make_random_policies(game), kwargs...) where {T<:Real, S, A, P<:AbstractTabularZeroSumGameTransition{T}, F<:Function} = 
	jal_am!(q_est, πs, game, γ, max_episodes, max_steps; kwargs...)
end

# ╔═╡ 9cfe4831-18b1-4c16-aa79-ffcff88c3f4e
begin
	function initialize_joint_action_values(game::TabularStochasticGame{T, S, A, N, P, F}, init_value::T) where {T<:Real, S, A, N, P, F<:Function}
		num_actions = Tuple(length(game.agent_actions[n]) for n in 1:N)
		javs = ones(T, num_actions..., length(game.states))
		NTuple{N, Array{T, N+1}}(javs .* init_value for n in 1:N)
	end

	function initialize_joint_action_values(game::TabularStochasticGame{T, S, A, 2, P, F}, init_value::T) where {T<:Real, S, A, P<:AbstractTabularZeroSumGameTransition{T}, F<:Function}
		num_actions = Tuple(game.agent_actions[n] for n in 1:2)
		javs = ones(T, num_actions..., length(game.states))
		return javs .* init_value
	end
end

# ╔═╡ c5cd53c2-9057-492e-b2e2-dfed2dce814b
begin
	jal_am(game::TabularStochasticGame{T, S, A, N, P, F}, γ::T, max_episodes::Integer, max_steps::Integer; init_value::T = zero(T), q_ests::NTuple{N, Array{T, Np1}} = initialize_joint_action_values(game, init_value), πs::NTuple{N, Matrix{T}} = make_random_policies(game), kwargs...) where {T<:Real, S, A, N, Np1, P<:AbstractTabularGameTransition{T, N}, F<:Function} = 
	jal_am!(q_ests, πs, game, γ, max_episodes, max_steps; kwargs...)
end

# ╔═╡ 278ed71e-9872-472c-ab59-6541b16e13e8
md"""
### Soccer Game Example
"""

# ╔═╡ 731008b1-2d23-4a8e-ab19-ac61e1f6a01b
const soccer_jal_am = jal_am(soccer_game, 0.9f0, typemax(Int64), 10_000_000; α = 1f0, ϵ = 0.2f0, α_decay = 0.99999954f0, save_history = true)

# ╔═╡ 8087e0c6-c0e2-499d-baec-499e23b3a8e5
display_soccer_statistics(soccer_jal_am.policies..., "JAL-AM", "JAL-AM")

# ╔═╡ 026c1c3e-8ae2-4acb-8f2e-286187a4e408
display_soccer_statistics(soccer_jal_am.policies[1], soccer_random_policy, "JAL-AM", "Random")

# ╔═╡ 7429193c-9bee-4031-8c4b-ccb1e8f6ad84
const soccer_jal_am_vs_optimal = independent_q_learning(soccer_game, 0.9f0; α = 1f0, max_steps = 1_000_000, πs = (soccer_jal_am.policies[1], copy(soccer_random_policy)), train_policies = 2:2, save_history = true, ϵ = 0.2f0, α_decay = 0.9999954f0)

# ╔═╡ f1255c1c-6435-4eca-8d84-4b7f3a505ace
display_soccer_statistics(soccer_jal_am_vs_optimal.policies..., "JAL-AM", "Optimal")

# ╔═╡ f3a8c840-8c0c-4bcd-a541-2801cddd529f
const lbf_game = LevelBasedForaging.make_5_3_environment()

# ╔═╡ 1b671a28-ab2d-4a4f-ab4a-f6916daa34b6
const lbf_jal_am_prep = jal_am(lbf_game, 0.99f0, typemax(Int64), 1_000_000; α = .01f0, ϵ = 1f0, save_history = true, use_linear_decay = true, α_min = 0.01f0, ϵ_min = 0.05f0)

# ╔═╡ 8878e879-6e70-4d21-8eb9-c4e1898f0f48
const lbf_jal_am = jal_am(lbf_game, 0.99f0, typemax(Int64), 1_000_000; α = .01f0, ϵ = 0.05f0, ϵ_decay = 1f0, save_history = true, q_ests = lbf_jal_am_prep.joint_action_values)

# ╔═╡ 58cda3e3-9ba1-428b-bc80-b62951c4e193
runepisode(lbf_game; πs = lbf_jal_am.policies)

# ╔═╡ 8adec11d-e873-49c7-8a49-3e584cf76720
md"""
### Rock-Paper-Scissors Example
"""

# ╔═╡ 23db3972-76b7-4369-866b-d1f187ba7670
const jal_am_rps = jal_am(RockPaperScissors.non_repeated_game, 1f0, 500, typemax(Int64); α = 0.05f0, ϵ = 0.05f0, save_policy_history = true)

# ╔═╡ 549e81d0-9dbe-4f89-9849-333d7bdc4d4f
md"""
## Policy-Based Learning
"""

# ╔═╡ 3ab9580f-0b52-4b96-a386-91783e598798
md"""
### Win or learn fast with policy hill climbing
"""

# ╔═╡ abd8af73-c03d-468f-8bc3-5c5b5caaae44
begin
	get_agent_reward_value(ptf::AbstractTabularGameTransition{T, N}, rewards::NTuple{N, T}, n::Integer) where {N, T<:Real} = rewards[n]
	get_agent_reward_value(ptf::AbstractTabularZeroSumGameTransition{T}, rewards::T, ::Val{1}) where T<:Real = rewards
	get_agent_reward_value(ptf::AbstractTabularZeroSumGameTransition{T}, rewards::T, ::Val{2}) where T<:Real = -rewards
	get_agent_reward_value(ptf::AbstractTabularZeroSumGameTransition{T}, rewards::T, n::Integer) where T<:Real  = get_agent_reward_value(ptf, rewards, Val(n))
	get_agent_reward_value(ptf::AbstractTabularCommonRewardGameTransition{T, N}, rewards::T, ::Integer) where {N, T<:Real} = rewards
end

# ╔═╡ be5b8d2e-2d66-4291-93b4-056cdc68e3c4
function get_max_q_num!(max_action::BitVector, q::Matrix{T}, i_s::Integer) where T<:Real
	maxq = typemin(T)
	@inbounds @simd for i_a in 1:size(q, 1)
		maxq = max(maxq, q[i_a, i_s])
	end
	
	n = 0
	@inbounds @simd for i_a in 1:size(q, 1)
		ismax = isapprox(q[i_a, i_s], maxq)
		n += ismax
		max_action[i_a] = ismax
	end
	
	return maxq, n, max_action
end

# ╔═╡ 3efd8233-04da-4a6c-bbb7-ef7ed5c02057
#joint action learning with game theory
function wolf_phc!(q_ests::NTuple{N, Matrix{T}}, πs::NTuple{N, Matrix{T}}, π̄s::NTuple{N, Matrix{T}}, state_counts::Vector{T}, game::TabularStochasticGame{T, S, A, N, P, F}, γ::T, max_episodes::Integer, max_steps::Integer; α::T = one(T) / 10, l_l::T = one(T)/10, l_w::T = one(T) / 20, ϵ::T = one(T) / 10, α_decay::T = one(T), save_history::Bool = false) where {T<:Real, S, A, N, P<:AbstractTabularGameTransition{T, N}, F<:Function}
	@assert l_l > l_w "The learning rate for losing of $l_l must be greater than the learning rate for winning of $l_w"
	ep = 1
	step = 1
	i_s = game.initialize_state_index()
	a = zeros(Int64, N)

	reward_history = initialize_reward_history(game.ptf)
	episode_steps = Vector{Int64}()

	agent_policy_values = [zeros(T, length(game.agent_actions[n])) for n in 1:N]
	agent_max_action = [BitVector(zeros(T, length(game.agent_actions[n]))) for n in 1:N]

	α_step = α
	
	while (ep <= max_episodes) && (step <= max_steps)
		state_counts[i_s] += one(T)
		c = inv(state_counts[i_s])
		for n in 1:N
			if rand() < ϵ
				a[n] = rand(eachindex(game.agent_actions[n]))
			else
				a[n] = sample_action(πs[n], i_s)
			end
		end

		(rewards, i_s′) = game.ptf(i_s, NTuple{N, Int64}(a))

		terminated = game.terminal_states[i_s′]

		save_history && push!(reward_history, rewards)

		#update q values
		for n in 1:N
			num_actions = length(game.agent_actions[n])
			max_action = agent_max_action[n]
			#update q values
			π = πs[n]
			q_est = q_ests[n]
			r = get_agent_reward_value(game.ptf, rewards, n)
			target = r + (terminated ? zero(T) : γ*get_max_q(q_est, i_s′))
			δ = target - q_est[a[n], i_s]
			q_est[a[n], i_s] += α_step * δ

			q_max, num_max = get_max_q_num!(max_action, q_est, i_s)

			# @info "Maximum action value for agent $n is $q_max with $num_max values"

			#update average policy π̄
			π̄ = π̄s[n]
			@inbounds @simd for i_a in eachindex(game.agent_actions[n])
				δ = π[i_a, i_s] - π̄[i_a, i_s]
				π̄[i_a, i_s] += c * δ
			end

			# if num_max < num_actions
				#prepare to calculate Δ(s, a_i)
				x1 = zero(T)
				x2 = zero(T)
				@inbounds @simd for i_a in 1:num_actions
					x1 += π[i_a, i_s] * q_est[i_a, i_s]
					x2 += π̄[i_a, i_s] * q_est[i_a, i_s]
				end
				# x1 = sum(π[i_a, i_s] * q_est[i_a, i_s] for i_a in eachindex(game.agent_actions[n]))
				# x2 = sum(π̄[i_a, i_s] * q_est[i_a, i_s] for i_a in eachindex(game.agent_actions[n]))
				win_expect = (x1 > x2)
				δ = win_expect*l_w + !win_expect*l_l
				x3 = δ / (num_actions - num_max)

				# @info "Delta value is $x3"
				
				#update policy
				@inbounds @simd for i_a in 1:num_actions
					agent_policy_values[n][i_a] = π[i_a, i_s]
				end
				for i_a in 1:num_actions
					if !isapprox(q_est[i_a, i_s], q_max)
						δ_s_a = min(agent_policy_values[n][i_a], x3)
						π[i_a, i_s] -= δ_s_a
					else
						x = zero(T)
						@inbounds @simd for i_a′ in 1:i_a-1
							x += !max_action[i_a′]*min(agent_policy_values[n][i_a′], x3)
							# x += min(agent_policy_values[n][i_a′], x3)
						end

						@inbounds @simd for i_a′ in i_a+1:num_actions
							x += !max_action[i_a′]*min(agent_policy_values[n][i_a′], x3)
							# x += min(agent_policy_values[n][i_a′], x3)
						end
						
						π[i_a, i_s] += x / num_max
						# sum(min(π[i_a′, i_s], x3) for i_a′ in 1:i_a-1; init = zero(T)) + sum(min(π[i_a′, i_s], x3) for i_a′ in i_a+1:num_actions; init = zero(T))
					end
				end
			# else
				# @info "Not doing policy update in state $i_s because all $num_actions actions are maximizing"
			# end
		end
	

		if terminated
			i_s′ = game.initialize_state_index()
			ep += 1
			push!(episode_steps, step)
		end

		i_s = i_s′
		step += 1
		α_step *= α_decay
	end
	
	return (action_values = q_ests, policies = πs, reward_history = reward_history, episode_steps = episode_steps, avg_policies = π̄s, state_counts = state_counts)
end	

# ╔═╡ 26f22c5b-fcb1-43c5-9b2e-efa74c79f235
wolf_phc(game::TabularStochasticGame{T, S, A, N, P, F}, γ::T, max_episodes::Integer, max_steps::Integer; init_value::T = zero(T), q_ests::NTuple{N, Matrix{T}} = initialize_agent_action_values(game, init_value), πs::NTuple{N, Matrix{T}} = make_random_policies(game), π̄s::NTuple{N, Matrix{T}} = deepcopy(πs), state_counts::Vector{T} = zeros(T, length(game.states)), kwargs...) where {T<:Real, S, A, N, P<:AbstractTabularGameTransition{T, N}, F<:Function} = wolf_phc!(q_ests, πs, π̄s, state_counts, game, γ, max_episodes, max_steps; kwargs...)

# ╔═╡ b802988a-48c7-4de8-b37b-95b63884f79f
const soccer_wolf_phc = wolf_phc(soccer_game, 0.9f0, typemax(Int64), 1_000_000; α = .5f0, α_decay = 0.9999954f0, ϵ = 0.2f0, save_history = true, l_w = 0.01f0, l_l = 0.02f0)

# ╔═╡ 78e0333b-90c1-40c8-9c03-17759da296c5
display_soccer_statistics(soccer_wolf_phc.policies..., "WoLF-PHC", "WoLF-PHC")

# ╔═╡ 9515108e-2ab2-44f6-930a-05723cc4a07d
display_soccer_statistics(soccer_wolf_phc.policies[1], soccer_random_policy, "WoLF-PHC", "Random")

# ╔═╡ 4e4a131b-b824-491b-a0cb-012d5dd1afe6
const soccer_wolf_phc_vs_optimal = independent_q_learning(soccer_game, 0.9f0; α = .5f0, max_steps = 1_000_000, πs = (soccer_wolf_phc.policies[1], copy(soccer_random_policy)), train_policies = 2:2, save_history = true, ϵ = 0.2f0, α_decay = 0.9999954f0)

# ╔═╡ ab8d5a68-4ea8-4bec-8c2a-3f0727d05943
display_soccer_statistics(soccer_wolf_phc_vs_optimal.policies..., "WoLF-PHC", "Optimal")

# ╔═╡ a74c712b-5458-4d87-bdba-159617d30742
const lbf_wolf_phc = wolf_phc(lbf_game, 0.99f0, typemax(Int64), 4_000_000; α = .02f0, ϵ = 0.05f0, save_history = true, l_w = 0.01f0, l_l = .02f0)

# ╔═╡ 42a04270-323d-4d45-b025-d77694b5b982
runepisode(lbf_game; πs = lbf_wolf_phc.policies)

# ╔═╡ 4556459e-54d8-4b3c-9a22-96d0985176f0
#add a way to save the action probability history for games

# ╔═╡ 2cd6aa77-6828-44ef-b7d9-4f6cfc50e880
const rps_wolf_phc = wolf_phc(RockPaperScissors.non_repeated_game, 1f0, typemax(Int64), 200_000; α = .01f0, ϵ = .2f0, save_history = true, l_w = 0.000001f0, l_l = .000008f0)

# ╔═╡ 5da718f5-c940-49a3-9c38-7af26a930439
md"""
# Visualization Tools
"""

# ╔═╡ 52a51c78-4438-44b2-8315-9cffd8ba2581
md"""
## General
"""

# ╔═╡ 9a80e476-bad5-4a8f-a8e2-0008a1e5d698
#=╠═╡
function plot_reward_learning_curve(v; npoints = 1000)
	y = v |> cumsum |> x -> x ./ (1:length(x))
	l = length(y)
	inds = map(x -> round(Int64, x), LinRange(1, l, npoints))
	plot(scatter(x = view(eachindex(y), inds), y = view(y, inds)), Layout(xaxis_title = "Training Steps", yaxis_title = "Player A Reward"))
end
  ╠═╡ =#

# ╔═╡ 9d9db616-81b3-462b-9400-3cef27107c56
#=╠═╡
plot_reward_learning_curve(soccer_iql.reward_history)
  ╠═╡ =#

# ╔═╡ 2278415f-f4e5-4ed9-94ad-761941039b97
#=╠═╡
plot_reward_learning_curve(soccer_random_vs_optimal.reward_history)
  ╠═╡ =#

# ╔═╡ ed04bac9-32e1-4d26-b44a-39da7ed52abc
#=╠═╡
plot_reward_learning_curve(soccer_jal_am.reward_history)
  ╠═╡ =#

# ╔═╡ 07af85fb-cf84-4592-aeeb-49b8c37b5794
#=╠═╡
plot_reward_learning_curve(soccer_jal_am_vs_optimal.reward_history)
  ╠═╡ =#

# ╔═╡ 1ec55568-2331-40a2-8fa5-191cae1f6cc8
#=╠═╡
plot_reward_learning_curve([sum(r) for r in lbf_jal_am.reward_history])
  ╠═╡ =#

# ╔═╡ 6bf0c879-7808-4b47-aa06-1614a43bba4a
#=╠═╡
plot_reward_learning_curve(soccer_wolf_phc.reward_history)
  ╠═╡ =#

# ╔═╡ 882a36b8-89ef-4d8d-ac57-0f060b670bba
#=╠═╡
plot_reward_learning_curve(soccer_wolf_phc_vs_optimal.reward_history)
  ╠═╡ =#

# ╔═╡ 0e195ca4-9077-441e-8349-06b8824dce70
#=╠═╡
plot_reward_learning_curve([sum(a) for a in lbf_wolf_phc.reward_history])
  ╠═╡ =#

# ╔═╡ b053dc54-af90-463e-8510-38c33be32312
#=╠═╡
function addelements(e1, e2)
	@htl("""
	$e1
	$e2
	""")
end
  ╠═╡ =#

# ╔═╡ f338dda8-79b1-4c54-90ee-400e29b9d0b0
#=╠═╡
const rook_action_display = @htl("""
<div style = "display: flex; flex-direction: column; align-items: center; justify-content: center; color: black; background-color: rgba(100, 100, 100, 0.1);">
	<div style = "display: flex; align-items: center; justify-content: center;">
	<div class = "downarrow" style = "transform: rotate(90deg);"></div>
	<div class = "downarrow" style = "position: absolute; transform: rotate(180deg);"></div>
	<div class = "downarrow" style = "position: absolute; transform: rotate(270deg);"></div>
	<div class = "downarrow" style = "position: absolute;"></div>
	</div>
	<div>Actions</div>
</div>
""")
  ╠═╡ =#

# ╔═╡ 065ba539-ac92-4117-a1a3-d44d21237e34
HTML("""
<style>
	.downarrow {
		display: flex;
		justify-content: center;
		align-items: center;
		flex-direction: column;
	}

	.downarrow::before {
		content: '';
		width: 2px;
		height: 40px;
		background-color: black;
	}
	.downarrow::after {
		content: '';
		width: 0px;
		height: 0px;
		border-left: 5px solid transparent;
		border-right: 5px solid transparent;
		border-top: 10px solid black;
	}

	.idle {
	 	width: 15px;
	 	height: 15px;
	 	background-color: black;
	 	border-radius: 50%;
	}

	.gridcell {
			display: flex;
			justify-content: center;
			align-items: center;
			border: 1px solid black;
		}
</style>
""")

# ╔═╡ c6ca307f-9753-4b0d-8fd8-ee05086fd5ed
function join_str_vert(x::AbstractString, y::AbstractString)
"""$x
$y"""
end

# ╔═╡ 2f8b643d-0dbb-47b3-8644-eaaa1e5410cf
join_hoz(x, y) = """$x|$y"""

# ╔═╡ b8de4e87-0d28-4731-84f9-b0c8e799b9a5
form_str_col(v::AbstractVector{T}) where T<:Real = mapreduce(x -> string("|", round(Float64(x); sigdigits = 3), "|"), join_str_vert, v)

# ╔═╡ bfd96f31-f01f-4495-bde5-2ade7dcf2525
function form_str_row(v::AbstractVector{T}) where T<:Real 
	str = mapreduce(x -> round(Float64(x); sigdigits = 3), join_hoz, v)
	"""|$str|"""
end

# ╔═╡ 445b2fca-85cf-4937-a827-88453b96dba1
function form_str_row(v::AbstractVector)
	str = reduce(join_hoz, v)
	"""|$str|"""
end

# ╔═╡ 4d8eb87e-c43b-4b55-b632-79bca14b7759
function form_str_mat(x::Matrix)
	rows = [form_str_row(r) for r in eachrow(x)]
	reduce(join_str_vert, rows)
end

# ╔═╡ ab01fea4-2da7-47c6-81e7-847ffeda8b34
form_table_mat(x::Matrix) = form_table_mat(x, ["" for _ in 1:size(x, 2)])

# ╔═╡ 6eccc26c-85a5-4ee4-90e7-874c9b670081
function form_table_id(n::Integer)
	str = mapreduce(x -> "---", join_hoz, 1:n)
	"""|$str|"""
end

# ╔═╡ cd97f119-5eeb-424c-b73a-86c525cd3787
function form_table_mat(x::Matrix, header::AbstractVector)
	body = form_str_mat(x)
	head_str = form_str_row(header)
	id_str = form_table_id(length(header))
	Markdown.parse("""
	$head_str
	$id_str
	$body
	""")
end

# ╔═╡ 4ff02aaf-950e-463f-9c69-306a1ca5c4b0
form_table_id(5)

# ╔═╡ d23092be-c8b8-4723-a435-83e3e2e30650
function display_markdown_vector(v::AbstractVector{T}; name::AbstractString = "") where T<:Real
	str = form_str_col(v)
	Markdown.parse("""			   
|$name|
|---|			   
$str				   
""")
end

# ╔═╡ 6383ab5d-9d91-42f9-8a7f-427408ca356f
#=╠═╡
function form_table_mat(x::Matrix, rownames::AbstractVector, colnames::AbstractVector; str = "Policies")
	mat = form_table_mat(x, colnames)

	rowtab = display_markdown_vector(rownames; name = str)

	@htl("""
		
		 <div style = "display: flex; justify-content: center; font-size: 100%;">
		 <div style = "font-weight: bolder;">
		 $rowtab
		 </div>
		 $mat
		 </div>
		 """)
end
  ╠═╡ =#

# ╔═╡ 05aef224-961e-4ee9-84bb-6d6002c29e79
#=╠═╡
form_table_mat(rand(5, 5), rand(5), rand(5))
  ╠═╡ =#

# ╔═╡ 3c1bfb16-c9d9-4c88-8f0b-749526b0018e
display_markdown_vector(rand(5))

# ╔═╡ 400aea40-3d2c-4696-b31c-3e0518341998
display_markdown_vector(rand(9); name= "π_A")

# ╔═╡ 697e2312-c9a6-47b3-b13c-277ad4b0efb2
md"""
## Soccer Game
"""

# ╔═╡ 2b067fa4-3597-451b-8572-5ff877c68498
create_soccer_visualization() = create_soccer_visualization(soccer_random_policy, soccer_random_policy)

# ╔═╡ 322d6682-05db-41cc-b6d4-42d772b051ac
plot_soccer_solution(episode_length::Integer, final_reward::Real, s::TwoPlayerSoccer.State{W, H, GH}, a::Tuple{Int64, Int64}, πs::Tuple{Matrix, Matrix}, value_functions::Tuple{Matrix, Matrix}; kwargs...) where {W, H, GH} = plot_soccer_solution(episode_length, final_reward, s, a, πs, (maximum.(eachcol(value_functions[1])), maximum.(eachcol(value_functions[2]))); kwargs...)

# ╔═╡ d056d56d-3c46-4d3d-8789-e1ad4aa41bf1
plot_soccer_solution(episode_length::Integer, final_reward::Real, s::TwoPlayerSoccer.State{W, H, GH}, a::Tuple{Int64, Int64}, πs::Tuple{Matrix, Matrix}, value_function::Array; kwargs...) where {W, H, GH} = plot_soccer_solution(episode_length, final_reward, s, a, πs, (value_function, -value_function); kwargs...)

# ╔═╡ 7ab9b64b-6571-4cd4-bf9f-6ad00015dac8
#=╠═╡
function plot_soccer_state(s::TwoPlayerSoccer.State{W, H, GH}) where {W, H, GH}	
	tr1 = scatter(x = [s.agent_positions[1][1]], y = [s.agent_positions[1][2]], mode = "markers", marker_size = 30, name = "A", marker_color = s.agent1_ball ? "black" : "rgba(0, 0, 0, 0)", marker_line = attr(width = 3, color = "blue"))
	tr2 = scatter(x = [s.agent_positions[2][1]], y = [s.agent_positions[2][2]], mode = "markers", name = "B", marker_size = 30, marker_line = attr(width = 3, color = "red"), marker_color = !s.agent1_ball ? "black" : "rgba(0, 0, 0, 0)")

	field_traces = [
		scatter(x = [0.5, W+.5], y = [0.5, 0.5], mode = "lines", line_color = "black", line_width = 4, showlegend = false)
		scatter(x = [0.5, W+.5], y = [H+.5, H+.5], mode = "lines", line_color = "black", line_width = 4, showlegend = false)
		scatter(x = [0.5, 0.5], y = [0.5, H+.5], mode = "lines", line_color = "black", line_width = 4, showlegend = false)
		scatter(x = [W+.5, W+.5], y = [0.5, H+.5], mode = "lines", line_color = "black", line_width = 4, showlegend = false)
	]

	goal_traces = [
		scatter(x = [-.5, .5], y = [GH-0.5, GH-0.5], mode = "lines", line_color = "blue", line_width = 4, showlegend = false)
		scatter(x = [-.5, .5], y = [GH+1.5, GH+1.5], mode = "lines", line_color = "blue", line_width = 4, showlegend = false)
		scatter(x = [-.5, -.5], y = [GH-0.5, GH+1.5], mode = "lines", line_color = "blue", line_width = 4, showlegend = false)
		scatter(x = [W+.5, W+1.5], y = [GH-0.5, GH-0.5], mode = "lines", line_color = "red", line_width = 4, showlegend = false)
		scatter(x = [W+.5, W+1.5], y = [GH+1.5, GH+1.5], mode = "lines", line_color = "red", line_width = 4, showlegend = false)
		scatter(x = [W+1.5, W+1.5], y = [GH-0.5, GH+1.5], mode = "lines", line_color = "red", line_width = 4, showlegend = false)
	]

	plot([[tr1, tr2]; field_traces; goal_traces], Layout(xaxis_range = [0, W+1], xaxis_tickvals = 1:W, yaxis_range = [0, H+1], yaxis_tickvals = 1:H, yaxis_scaleanchor="x", yaxis_dtick = 1, xaxis_dtick = 1, width = 600))
end
  ╠═╡ =#

# ╔═╡ 50357183-c467-48c9-b1c7-05d7c2a5db55
plot_soccer_state(i_s::Integer) = plot_soccer_state(soccer_game.states[i_s])

# ╔═╡ ab3793ad-1069-4b77-a95a-4474fcd7c662
#=╠═╡
function display_soccer_policy(v::AbstractVector{T}; scale = 1.0) where T<:Real
	@htl("""
		<div style = "display: flex; align-items: center; justify-content: center; transform: scale($scale);">
		<div class = "downarrow" style = "position: absolute; transform: rotate(180deg); opacity: $(v[1]);"></div>	
		<div class = "downarrow" style = "position: absolute; opacity: $(v[2])"></div>
		<div class = "downarrow" style = "position: absolute; transform: rotate(90deg); opacity: $(v[3])"></div>
		<div class = "downarrow" style = "transform: rotate(-90deg); opacity: $(v[4])"></div>
		<div class = "idle" style = "position: absolute; opacity: $(v[5])"></div>
		</div>
	""")
end
  ╠═╡ =#

# ╔═╡ 1168356f-fedf-4e53-a761-89f99f6b3812
#=╠═╡
function display_soccer_policy(i_a::Integer; scale = 1.0)
	iszero(i_a) && return @htl("""""")
	v = zeros(5)
	v[i_a] = 1
	@htl("""
		<div style = "display: flex; align-items: center; justify-content: center; transform: scale($scale);">
		<div class = "downarrow" style = "position: absolute; transform: rotate(180deg); opacity: $(v[1]);"></div>	
		<div class = "downarrow" style = "position: absolute; opacity: $(v[2])"></div>
		<div class = "downarrow" style = "position: absolute; transform: rotate(90deg); opacity: $(v[3])"></div>
		<div class = "downarrow" style = "transform: rotate(-90deg); opacity: $(v[4])"></div>
		<div class = "idle" style = "position: absolute; opacity: $(v[5])"></div>
		</div>
	""")
end
  ╠═╡ =#

# ╔═╡ 59e334fd-0fdd-4c1c-a1e0-07777d233f6f
md"""
## Rock-Paper-Scissors Game
"""

# ╔═╡ 62d8a4f9-fd25-449c-acd0-7f8e34561c31
#=╠═╡
function plot_rps_policy_history(policy_history::Vector{NTuple{2, Matrix{Float32}}})
	l = length(policy_history)
	r1 = [a[1][1, 1] for a in view(policy_history, 1:2:l)]
	p1 = [a[1][2, 1] for a in view(policy_history, 1:2:l)]

	r2 = [a[2][1, 1] for a in view(policy_history, 1:2:l)]
	p2 = [a[2][2, 1] for a in view(policy_history, 1:2:l)]

	tr1 = scatter(x = p1, y = r1, name = "Agent 1 Empirical", mode = "lines+markers", marker_color = "rgba(0, 120, 255, .3)")
	tr2 = scatter(x = p2, y = r2, name = "Agent 2 Empirical", mode = "lines+markers", yaxis = "y2", xaxis = "x2", marker_color = "rgba(255, 130, 0, .3)", marker_symbol = "x")
	tr3 = scatter(x = [0, 1], y = [1, 0], mode = "lines", line_color = "black", line_width = "1", line_dash = "dot", showlegend = false)
	tr4 = scatter(x = [1/3], y = [1/3], mode = "markers", line_color = "rgb(0, 120, 255)", marker_symbol = "star", name = "Equilibrium Policy 1")
	tr5 = scatter(x = [1/3], y = [1/3], mode = "markers", line_color = "rgb(255, 130, 0)", marker_symbol = "star", name = "Equilibrium Policy 2", yaxis = "y2", xaxis = "x2")
	tickvals = 0.:0.2:1.
	ticktext = string.(tickvals)

	tickvals2 = 1.:-.2:0.
	ticktext2 = string.(tickvals2)
	plot([tr1, tr2, tr3, tr4, tr5], Layout(yaxis_scaleanchor = "x", yaxis2_scaleanchor = "x2", width = 600, xaxis = attr(tickvals = tickvals, ticktext = ticktext, title = "π(Rock)", range = [0, 1]), yaxis = attr(tickvals = tickvals, title = "π(Paper)", range = [0, 1]), yaxis2 = attr(range = [1, 0], side = "right", tickvals = tickvals, overlaying = "y", ticktext = ticktext), xaxis2 = attr(side = "top", overlaying = "x", range = [1, 0], tickvals = tickvals), legend_x = 1.1))
end
  ╠═╡ =#

# ╔═╡ 5eb49614-6762-4544-9bf1-bcfa3af62646
#=╠═╡
plot_rps_policy_history(jal_am_rps.policy_history)
  ╠═╡ =#

# ╔═╡ 9d54ee68-d60c-11f0-87d1-3be62822741b
md"""
# Dependencies
"""

# ╔═╡ a929c6cc-733b-44a1-bdbd-40a525bb5ffd
import HiGHS

# ╔═╡ 7b07aa13-b39a-4874-9de4-037685001e24
begin
	function solve_minimax_game!(reward_matrix::Matrix{T}, π::Matrix{T}, m::Array{T, 3}, i_s::Integer, player1::Bool) where T<:Real
		model = Model(HiGHS.Optimizer)
		(l, k) = size(reward_matrix)

		min_reward = typemax(T)
		@inbounds @simd for i_a1 in 1:l for i_a2 in 1:k
			r = -m[i_a1, i_a2, i_s]
			min_reward = min(r, min_reward)
			reward_matrix[i_a1, i_a2] = r
		end end

		reward_matrix .-= min_reward

		x_length = player1 ? l : k

		@variable(model, x[1:x_length])

		if player1
			@constraint(model, x' * reward_matrix .≤ one(T))
		else
			@constraint(model, reward_matrix * x .≤ one(T))
		end
		
		@constraint(model, x .>= zero(T))
		@objective(model, Max, sum(x))
		optimize!(model)
		modified_game_value = inv(sum(value.(x)))
		@inbounds @simd for i in 1:x_length
			π[i, i_s] = value(x[i]) * modified_game_value
		end
		
		-(modified_game_value + min_reward)
	end

	function solve_minimax_game!(reward_matrix::Matrix{T}, π::Matrix{T}, model::Model, x::Vector{VariableRef}, m::Array{T, 3}, i_s::Integer, player1::Bool) where T<:Real
		
		(l, k) = size(reward_matrix)
		x_length = player1 ? l : k

		c = (!player1 - player1)

		if all(x -> isapprox(x, zero(T); atol = eps(one(T))), view(m, :, :, i_s)) 
			p = one(T) / x_length
			@inbounds @simd for i in 1:x_length
				π[i, i_s] = p
			end
			return zero(T)
		end

		min_reward = typemax(T)
		@inbounds @simd for i_a1 in 1:l for i_a2 in 1:k
			r = c*m[i_a1, i_a2, i_s]
			min_reward = min(r, min_reward)
			reward_matrix[i_a1, i_a2] = r
		end end

		reward_matrix .-= (min_reward - one(T))

		if all(x -> isapprox(x, reward_matrix[1]; atol = eps(one(T))), reward_matrix) 
			p = one(T) / x_length
			@inbounds @simd for i in 1:x_length
				π[i, i_s] = p
			end
			return -reward_matrix[1]
		end

		if player1
			@constraint(model, con_s, x' * reward_matrix .≤ one(T))
		else
			@constraint(model, con_s, reward_matrix * x .≤ one(T))
		end
		
		optimize!(model)
		modified_game_value = inv(sum(value.(x)))
		@inbounds @simd for i in 1:x_length
			π[i, i_s] = max(zero(T), value(x[i]) * modified_game_value)
		end

		for c in con_s
			delete(model, c)
		end
		unregister(model, :con_s)
		
		-(modified_game_value + min_reward - one(T))
	end
end

# ╔═╡ 2eb915ed-16c2-4ea0-bf5f-5d1456249224
function value_iteration_game!(v_est::Vector{T}, πs::NTuple{2, Matrix{T}}, m_est::Array{T, 3}, ptf::TabularZeroSumGameTransition{T, ST, RT}, γ::T, nmax::Integer; show_message::Bool = true, θ::T = eps(zero(T))) where {T<:Real, ST, RT}
	(l, k, num_states) = size(m_est)
	reward_matrix = zeros(T, l, k)

	delta_max = typemax(T)
	sweep = 0
	
	state_transitions = ptf.state_transition_map
	reward_transitions = ptf.reward_transition_map

	#setup two models, one for each player, since in general the available actions could differ from one player to another
	model1 = Model(HiGHS.Optimizer)
	model2 = Model(HiGHS.Optimizer)

	@variable(model1, x[1:l])
	@constraint(model1, x .>= zero(T))
	@objective(model1, Max, sum(x))

	@variable(model2, y[1:k])
	@constraint(model2, y .>= zero(T))
	@objective(model2, Max, sum(y))

	while (sweep < nmax) && (delta_max > θ)
		for (i, inds) in enumerate(CartesianIndices(state_transitions))
			a = (inds[1], inds[2])
			i_s = inds[3]
			v = bellman_game_reward_value(ptf, i_s, a, γ, v_est)
			# if !iszero(v1)
			# 	@info "Non zero game value updated for player 1 state $i_s"
			# end

			# if !iszero(v2)
			# 	@info "Non zero game value updated for player2 state $i_s"
			# end
			
			m_est[i] = v
		end

		delta_max = typemin(T)
		for i_s in eachindex(v_est)
			v_old = v_est[i_s]
			v_est[i_s] = solve_minimax_game!(reward_matrix, πs[1], model1, x, m_est, i_s, true)
			
			delta_max = max(TabularRL.calc_pct_change(v_old, v_est[i_s]), delta_max)
		end
		sweep += 1
		show_message && @info "After $(sweep) sweeps, maximum value change is $delta_max"
	end

	for i_s in eachindex(v_est)
		solve_minimax_game!(reward_matrix, πs[2], model2, y, m_est, i_s, false)
	end

	return (values = v_est, total_sweeps = sweep, policies = πs, game_rewards = m_est)
end

# ╔═╡ af0265b8-7fe9-499a-b675-32bf5ba76f64
begin
	value_iteration_game(ptf::TabularZeroSumGameTransition{T, ST, RT}, γ::T; nmax::Integer = typemax(Int64), dims::NTuple{3, Int64} = size(ptf.state_transition_map), v_est::Vector{T} = zeros(T, dims[3]), πs::NTuple{2, Matrix{T}} = (zeros(T, dims[1], dims[3]), zeros(T, dims[2], dims[3])), m_est::Array{T, 3} = zeros(T, dims...), kwargs...) where {T<:Real, RT, ST} = value_iteration_game!(v_est, πs, m_est, ptf, γ, nmax; kwargs...)

	value_iteration_game(game::TabularStochasticGame, γ::T; kwargs...) where {T<:Real} = value_iteration_game(game.ptf, γ; kwargs...)
end

# ╔═╡ 439fff54-57e8-409e-994e-55e536d2ea0b
# ╠═╡ show_logs = false
const soccer_value_iter = value_iteration_game(soccer_game, 0.9f0; θ = 1f-5)

# ╔═╡ 30d70205-a091-4155-87f5-d0ada3b597e0
const minimax_vs_minimax_stats = display_soccer_statistics(soccer_value_iter.policies..., "Minimax", "Minimax")

# ╔═╡ 19d9e4b3-2c6b-4f33-a5be-32aa4a38513a
const minimax_vs_random_stats = display_soccer_statistics(soccer_value_iter.policies[1], soccer_random_policy, "Minimax", "Random")

# ╔═╡ 84fdc00e-43d1-44c4-b479-d7be79c2d4cd
const random_vs_minimax_stats = display_soccer_statistics(soccer_random_policy, soccer_value_iter.policies[2], "Random", "Minimax")

# ╔═╡ 0fc4963b-2100-4b5d-a663-1c5306f311ca
const iql_vs_minimax_stats = display_soccer_statistics(soccer_iql.policies[1], soccer_value_iter.policies[2], "IQL", "Minimax")

# ╔═╡ 140a1608-8105-4d9c-a839-12e94483eab4
const soccer_minimax_vs_optimal = independent_q_learning(soccer_game, 0.9f0; α = 1f0, max_steps = 1_000_000, πs = (soccer_value_iter.policies[1], copy(soccer_random_policy)), train_policies = 2:2, save_history = true, ϵ = 0.2f0, α_decay = 0.9999954f0)

# ╔═╡ d3a1b675-a218-4543-a501-df25b7255762
#=╠═╡
plot(soccer_minimax_vs_optimal.reward_history |> cumsum |> v -> v ./ (1:length(v)))
  ╠═╡ =#

# ╔═╡ 4b940a7f-21aa-44f3-9ecf-03e3ab0c2900
const minimax_vs_optimal_stats = display_soccer_statistics(soccer_minimax_vs_optimal.policies..., "Minimax", "Optimal")

# ╔═╡ ce4b2377-45e2-421d-b457-c40aaabee4e4
display_soccer_statistics(soccer_jal_am.policies[1], soccer_value_iter.policies[2], "JAL-AM", "Minimax")

# ╔═╡ 692cb71b-fb78-4a27-aad8-61c2b25a29c6
display_soccer_statistics(soccer_wolf_phc.policies[1], soccer_value_iter.policies[2], "WoLF-PHC", "Minimax")

# ╔═╡ 7ba3754f-4a2d-4f87-a303-e37ea2376947
#=╠═╡
function display_soccer_minimax(i_s::Integer; game_rewards = soccer_value_iter.game_rewards, πs = soccer_value_iter.policies, values = soccer_value_iter.values, str = "Mimimax Policies")
	reward_matrix = game_rewards[:, :, i_s]
	π1 = πs[1][:, i_s]
	π2 = πs[2][:, i_s]
	game_value = values[i_s]

	table = form_table_mat(reward_matrix, π1, π2; str = str)
	# (reward_matrix = reward_matrix, π_A = π1, π_B = π2, game_value = game_value)
	
	@htl("""
		 <div style = "display: flex; justify-content: space-around; color: black; background-color: white; font-size: 120%;">
		 	<div>
		 		<div style = "display: flex; justify-content: center; align-items: center;">
		 			$(display_soccer_policy(π2))
		 			<div style = "margin-left: 30px;">Player B Policy</div>
		 		</div>
		 		<div style = "display: flex; justify-content: space-between; align-items: center;">
		 			$table
		 			<div>
		 			<div style = "display: flex; justify-content: center; align-items: center; padding-left: 10%;">
		 				$(display_soccer_policy(π1))		
		 <div style = "margin-left: 30px;">Player A Policy</div>
		 				
		 			</div>
		 			Game Value: $(round(Float64(game_value); sigdigits = 3))
		 			</div>
		 		</div>
		 	</div>
			 
		 </div>
		 """)
end
  ╠═╡ =#

# ╔═╡ e3cc189b-80a8-44a9-b734-59eece616647
#=╠═╡
function plot_soccer_solution(episode_length::Integer, final_reward::Real, s::TwoPlayerSoccer.State{W, H, GH}, a::Tuple{Int64, Int64}, πs::Tuple{Matrix, Matrix}, value_functions::Tuple{Vector, Vector}; show_minimax::Bool = false, minimax2::Bool = false, kwargs...) where {W, H, GH}	
	state_plot = plot_soccer_state(s)
	i_s = soccer_game.state_index[s]
	v1 = πs[1][:, i_s]
	v2 = πs[2][:, i_s]
	π1_disp = display_soccer_policy(v1)
	a1_disp = display_soccer_policy(a[1])
	π2_disp = display_soccer_policy(v2)
	a2_disp = display_soccer_policy(a[2])
	outcome_str = if isone(final_reward)
		"won by A"
	elseif final_reward == -1
		"won by B"
	else
		"ending in a draw"
	end

	minimax_disp = show_minimax ? display_soccer_minimax(i_s) : @htl("""""")
	minimax_disp2 = minimax2 ? display_soccer_minimax(i_s; kwargs...) : @htl("""""")

	@htl("""
	<div style = "display: flex; align-items: center; justify-content: space-around;">
	<div style = "max-width: 50%; font-weight: bold; font-size: 125%;">
		 $episode_length step game $outcome_str
		<div style = "display: flex; justify-content: space-around; background-color: white; color: black;">
		 <div style = "color: blue; font-weight: bold;">
			Player A Value: $(value_functions[1][i_s])
			 <div style = "display: flex; justify-content: space-around;">
				 <div>
					 Policy
					 $π1_disp
				 </div>
				 <div>
					 Action
					 $a1_disp
				 </div>
			 </div>
		 </div>
		 <div style = "color: red; font-weight: bold;">
		 	Player B Value: $(value_functions[2][i_s])
				<div style = "display: flex; justify-content: space-around;">
				 <div>
					 Policy
					 $π2_disp
				 </div>
				 <div>
					 Action
					 $a2_disp
				 </div>
			 </div>
			 </div>
		</div>
		$state_plot
	</div>
	<div style = "width: 50%">
	$minimax_disp
	$minimax_disp2
	</div>
		 </div>
		 """)
end
  ╠═╡ =#

# ╔═╡ bf9e5075-3595-42ca-9b73-0e16e608b946
#=╠═╡
function plot_soccer_solution(episode_length::Integer, final_reward::Real, s::TwoPlayerSoccer.State{W, H, GH}, a::Tuple{Int64, Int64}, πs::Tuple{Matrix, Matrix}; show_minimax::Bool = false, minimax2::Bool = false, kwargs...) where {W, H, GH}	
	state_plot = plot_soccer_state(s)
	i_s = soccer_game.state_index[s]
	v1 = πs[1][:, i_s]
	v2 = πs[2][:, i_s]
	π1_disp = display_soccer_policy(v1)
	π2_disp = display_soccer_policy(v2)
	a1_disp = display_soccer_policy(a[1])
	a2_disp = display_soccer_policy(a[2])
	minimax_disp = show_minimax ? display_soccer_minimax(i_s) : @htl("""""")
	minimax_disp2 = minimax2 ? display_soccer_minimax(i_s; kwargs...) : @htl("""""")

	outcome_str = if isone(final_reward)
		"won by A"
	elseif final_reward == -1
		"won by B"
	else
		"ending in a draw"
	end
	
	@htl("""
	<div style = "display: flex; align-items: center; justify-content: space-around;">
	<div style = "width: 600px; font-weight: bold; font-size: 125%;">
		 $episode_length step game $outcome_str
		<div style = "display: flex; justify-content: space-around; background-color: white; color: black;">
		 <div style = "width: 33%; text-align: center; color: blue; font-weight: bold;">
			Player A
			 <div style = "display: flex; justify-content: space-around;">
				 <div>
					 Policy
					 $π1_disp
				 </div>
				 <div>
					 Action
					 $a1_disp
				 </div>
			 </div>
		 </div>
		 <div style = "width: 33%; text-align: center; color: red; font-weight: bold;">
		 	Player B
				<div style = "display: flex; justify-content: space-around;">
				 <div>
					 Policy
					 $π2_disp
				 </div>
				 <div>
					 Action
					 $a2_disp
				 </div>
			 </div>
			 </div>
		</div>
		$state_plot
	</div>
	<div style = "width: 50%">
	$minimax_disp
	$minimax_disp2
	</div>
	</div>
		 """)
end
  ╠═╡ =#

# ╔═╡ ee51c98e-444e-4a44-bb67-ea91f739d731
#=╠═╡
function create_soccer_visualization(π1, π2, vs...)
	ep = runepisode(soccer_game; πs = (π1, π2), max_steps = 1_000)
	el = Slider(1:length(ep[1])+1; show_value=true)

	function f(ep_step; kwargs...) 
		soccer_state_index = vcat(ep[1], ep[4])[ep_step]
		s = soccer_game.states[soccer_state_index]
		a = ep_step > length(ep[2]) ? (0, 0) : ep[2][ep_step]
		plot_soccer_solution(length(ep[1]), ep[3][end], s, a, (π1, π2), vs...; kwargs...)
	end

	return (slider = el, plot_function = f)
end
  ╠═╡ =#

# ╔═╡ 7c691c82-6de9-47ee-8b61-e3e239976ecc
#=╠═╡
const soccer_random_viz = create_soccer_visualization()
  ╠═╡ =#

# ╔═╡ dceebec1-bf10-4389-9ef8-864ea59e92e8
#=╠═╡
@bind soccer_rnd_step soccer_random_viz.slider
  ╠═╡ =#

# ╔═╡ 25c1c075-b10a-4bc1-b113-39b684d8f939
#=╠═╡
soccer_random_viz.plot_function(soccer_rnd_step)
  ╠═╡ =#

# ╔═╡ 74d539f4-defc-4e5e-a085-a30e96df75bf
#=╠═╡
const soccer_ideal_viz = create_soccer_visualization(soccer_value_iter.policies..., soccer_value_iter.values)
  ╠═╡ =#

# ╔═╡ 036e8dbe-9acb-41bd-83eb-957da19fd86d
#=╠═╡
@bind soccer_ideal_step soccer_ideal_viz.slider
  ╠═╡ =#

# ╔═╡ 37c4bbe2-f78e-4ad1-a0fb-a00da11631f1
#=╠═╡
soccer_ideal_viz.plot_function(soccer_ideal_step; show_minimax = true)
  ╠═╡ =#

# ╔═╡ b22df2e2-d76d-4426-a704-42a291103603
#=╠═╡
const soccer_minimax_vs_random_viz = create_soccer_visualization(soccer_value_iter.policies[1], soccer_random_policy, soccer_value_iter.values)
  ╠═╡ =#

# ╔═╡ b29a3234-8717-4b12-ac64-a79a0a74113a
#=╠═╡
@bind minimax_vs_random_step soccer_minimax_vs_random_viz.slider
  ╠═╡ =#

# ╔═╡ 46791653-7452-4afb-8639-ac753af9f249
#=╠═╡
soccer_minimax_vs_random_viz.plot_function(minimax_vs_random_step; show_minimax = true)
  ╠═╡ =#

# ╔═╡ feb35b37-c99f-40b7-b181-a86038a8ed92
#=╠═╡
iql_viz = create_soccer_visualization(soccer_iql.policies..., soccer_iql.value_estimates)
  ╠═╡ =#

# ╔═╡ f3566f79-350b-40ad-9817-af038202fd11
#=╠═╡
@bind iql_step iql_viz.slider
  ╠═╡ =#

# ╔═╡ 03178016-9014-45a8-a977-1e3c794d3db6
#=╠═╡
iql_viz.plot_function(iql_step)
  ╠═╡ =#

# ╔═╡ 50a06568-bd54-4f0c-82a1-532050980940
#=╠═╡
random_vs_optimal_viz = create_soccer_visualization(soccer_random_vs_optimal.policies...)
  ╠═╡ =#

# ╔═╡ 97e10e56-9990-4b53-bf01-109eda72bb1c
#=╠═╡
@bind random_vs_optimal_step random_vs_optimal_viz.slider
  ╠═╡ =#

# ╔═╡ d5351789-cd1a-4b20-9e96-8f3810b811f1
#=╠═╡
random_vs_optimal_viz.plot_function(random_vs_optimal_step)
  ╠═╡ =#

# ╔═╡ d6dc4e3c-faa7-426c-972e-dacf9bf6ba88
#=╠═╡
minimax_vs_optimal_viz = create_soccer_visualization(soccer_minimax_vs_optimal.policies..., soccer_minimax_vs_optimal.value_estimates)
  ╠═╡ =#

# ╔═╡ e071b761-5df2-4776-a959-2b8546027cd6
#=╠═╡
@bind minimax_vs_optimal_step minimax_vs_optimal_viz.slider
  ╠═╡ =#

# ╔═╡ b1f4df52-2049-43e6-bc46-fe7e192e74ca
#=╠═╡
minimax_vs_optimal_viz.plot_function(minimax_vs_optimal_step)
  ╠═╡ =#

# ╔═╡ cd3bb7ba-3560-40a9-b166-7cb5cb23f316
#=╠═╡
iql_vs_optimal_viz = create_soccer_visualization(soccer_iql_vs_optimal.policies..., soccer_iql_vs_optimal.value_estimates)
  ╠═╡ =#

# ╔═╡ 5cfcb48f-59af-435c-81ea-170eb8a94c78
#=╠═╡
@bind iql_vs_optimal_step iql_vs_optimal_viz.slider
  ╠═╡ =#

# ╔═╡ bf7a9be7-5e04-4059-b67e-01b30f929036
#=╠═╡
iql_vs_optimal_viz.plot_function(iql_vs_optimal_step; show_minimax=true)
  ╠═╡ =#

# ╔═╡ a1407418-2efb-4bee-9645-d029735d99f7
#=╠═╡
viz_test = create_soccer_visualization(soccer_value_iter.policies..., soccer_value_iter.values)
  ╠═╡ =#

# ╔═╡ 78815d3e-b1fa-4a60-b86d-07a5e40fea60
#=╠═╡
@bind test_step viz_test.slider
  ╠═╡ =#

# ╔═╡ 511e384b-697e-4397-8ec5-bf5449b9eee5
#=╠═╡
viz_test.plot_function(test_step; show_minimax = true)
  ╠═╡ =#

# ╔═╡ 95a3fcca-1f0c-49e9-9e9c-b1bea8cb76c4
#joint action learning with game theory
function jal_gt!(q_est::Array{T, 3}, πs::NTuple{2, Matrix{T}}, game::TabularStochasticGame{T, S, A, 2, P, F}, γ::T, max_episodes::Integer, max_steps::Integer; α::T = one(T) / 10, ϵ::T = one(T) / 10, α_decay::T = one(T)) where {T<:Real, S, A, P<:AbstractTabularZeroSumGameTransition{T}, F<:Function}
	ep = 1
	step = 1
	i_s = game.initialize_state_index()
	a = [0, 0]
	(l, k, num_states) = size(q_est)
	reward_matrix = zeros(T, l, k)

	model1 = Model(HiGHS.Optimizer)
	model2 = Model(HiGHS.Optimizer)

	@variable(model1, x[1:l])
	@constraint(model1, x .>= zero(T))
	@objective(model1, Max, sum(x))

	@variable(model2, y[1:k])
	@constraint(model2, y .>= zero(T))
	@objective(model2, Max, sum(y))

	α_step = α
	
	while (ep <= max_episodes) && (step <= max_steps)
		if rand() < ϵ
			a[1] = rand(eachindex(game.agent_actions[1]))
		else
			# solve_minimax_game!(reward_matrix, πs[1], model1, x, q_est, i_s, true)
			a[1] = sample_action(πs[1], i_s)
		end

		if rand() < ϵ
			a[2] = rand(eachindex(game.agent_actions[2]))
		else
			solve_minimax_game!(reward_matrix, πs[2], model2, y, q_est, i_s, false)
			a[2] = sample_action(πs[2], i_s)
		end

		(r, i_s′) = game.ptf(i_s, NTuple{2, Int64}(a))

		if game.terminal_states[i_s′]
			game_state_value = zero(T)
			i_s′ = game.initialize_state_index()
			solve_minimax_game!(reward_matrix, πs[1], model1, x, q_est, i_s′, true)
			ep += 1
		else
			game_state_value = solve_minimax_game!(reward_matrix, πs[1], model1, x, q_est, i_s′, true)
		end

		
		target = r + γ*game_state_value
		δ = target - q_est[a[1], a[2], i_s]
		q_est[a[1], a[2], i_s] += α_step * δ

		i_s = i_s′
		step += 1
		α_step *= α_decay
	end

	for i_s in eachindex(game.states)
		solve_minimax_game!(reward_matrix, πs[1], model1, x, q_est, i_s, true)
		solve_minimax_game!(reward_matrix, πs[2], model2, y, q_est, i_s, false)
	end
	
	return (joint_action_values = q_est, policies = πs)
end	

# ╔═╡ b0ef6a1d-cb67-44a3-b8da-4f56930f85d7
begin
	jal_gt(game::TabularStochasticGame{T, S, A, 2, P, F}, γ::T, max_episodes::Integer, max_steps::Integer; init_value::T = zero(T), q_est::Array{T, 3} = ones(T, length(game.agent_actions[1]), length(game.agent_actions[2]), length(game.states)) .* init_value, πs::NTuple{2, Matrix{T}} = make_random_policies(game), kwargs...) where {T<:Real, S, A, P<:AbstractTabularZeroSumGameTransition{T}, F<:Function} = 
	jal_gt!(q_est, πs, game, γ, max_episodes, max_steps; kwargs...)
end

# ╔═╡ e12a9ce4-1441-4863-9e31-19d04f512c61
# ╠═╡ show_logs = false
const soccer_jal = jal_gt(soccer_game, 0.9f0, typemax(Int64), 1_000_000; α = 1f0, ϵ = 0.2f0, α_decay = 0.9999954f0)

# ╔═╡ 5906e0f9-eaf6-44f0-bfee-fdb4dd02f08b
const jal_values = [soccer_jal.joint_action_values[:, :, i_s] .* (soccer_jal.policies[1][:, i_s] * soccer_jal.policies[2][:, i_s]') |> sum for i_s in eachindex(soccer_game.states)]

# ╔═╡ a86d8b48-6619-4f92-aa17-17e3c4a5f4f7
const soccer_jal_stats = display_soccer_statistics(soccer_jal.policies..., "Minimax Q", "Minimax Q")

# ╔═╡ a2f8445e-8b83-4903-b829-15b997e4c3fc
#=╠═╡
const soccer_jal_viz = create_soccer_visualization(soccer_jal.policies...)
  ╠═╡ =#

# ╔═╡ 98f6ddc3-c534-4ba2-bce6-97b29e1428e4
#=╠═╡
@bind soccer_jal_step soccer_jal_viz.slider
  ╠═╡ =#

# ╔═╡ a3fd5c60-105c-4c9d-909f-0a0af6c592c3
#=╠═╡
soccer_jal_viz.plot_function(soccer_jal_step; show_minimax=true, minimax2 = true, game_rewards = soccer_jal.joint_action_values, πs = soccer_jal.policies, values = jal_values, str = "Minimax Q πs")
  ╠═╡ =#

# ╔═╡ 1eaed379-6f4d-4634-aea6-a5b1941389d4
const soccer_jal_vs_random_stats = display_soccer_statistics(soccer_jal.policies[1], soccer_random_policy, "Minimax Q", "Random")

# ╔═╡ f6ea001e-38f6-428c-a107-4ddcf817f470
(exact_solution = minimax_vs_random_stats, estimated_solution = soccer_jal_vs_random_stats)

# ╔═╡ 46060259-a774-4ff1-b415-27a5c6e9fa40
#=╠═╡
const soccer_jal_vs_random_viz = create_soccer_visualization(soccer_jal.policies[1], soccer_random_policy)
  ╠═╡ =#

# ╔═╡ 59461d31-4d11-4802-8e58-996a0fa7bf67
#=╠═╡
@bind soccer_jal_vs_random_step soccer_jal_vs_random_viz.slider
  ╠═╡ =#

# ╔═╡ 0bb09a54-4d4b-472d-86cc-133a9d73349c
#=╠═╡
soccer_jal_vs_random_viz.plot_function(soccer_jal_vs_random_step)
  ╠═╡ =#

# ╔═╡ fb3e4d0f-bc6c-45a2-9491-225ef7b8f647
const soccer_jal_vs_minimax_stats = display_soccer_statistics(soccer_jal.policies[1], soccer_value_iter.policies[2], "Minimax Q", "Minimax")

# ╔═╡ 349e7689-24bd-467d-a7f0-73332fc93118
#=╠═╡
const soccer_jal_vs_minimax_viz = create_soccer_visualization(soccer_jal.policies[1], soccer_value_iter.policies[2])
  ╠═╡ =#

# ╔═╡ 1e15c352-c684-49ff-bbff-3cfeac7b41f6
#=╠═╡
@bind soccer_jal_vs_minimax_step soccer_jal_vs_minimax_viz.slider
  ╠═╡ =#

# ╔═╡ ffbd79f9-ac31-4ff4-ab8d-7c8603dfedf6
#=╠═╡
soccer_jal_vs_minimax_viz.plot_function(soccer_jal_vs_minimax_step; show_minimax=true, minimax2 = true, game_rewards = soccer_jal.joint_action_values, πs = soccer_jal.policies, values = jal_values, str = "Minimax Q πs")
  ╠═╡ =#

# ╔═╡ 879e7384-7786-43b2-a36f-229929e802cf
const iql_vs_jal_stats = display_soccer_statistics(soccer_iql.policies[1], soccer_jal.policies[2], "IQL", "Minimax Q")

# ╔═╡ 32bc7859-7503-46c6-b8e0-2c3c22b2acc9
const soccer_jal_vs_optimal = independent_q_learning(soccer_game, 0.9f0; α = 1f0, max_steps = 1_000_000, πs = (soccer_jal.policies[1], copy(soccer_random_policy)), train_policies = 2:2, save_history = true, ϵ = 0.2f0, α_decay = 0.9999954f0)

# ╔═╡ 72097ef2-0fcb-45c8-8537-253af364e35a
const jal_vs_optimal_stats = display_soccer_statistics(soccer_jal_vs_optimal.policies..., "Minimax Q", "Optimal")

# ╔═╡ cf478f51-42df-4f1f-a666-b470afe28767
#=╠═╡
soccer_jal_vs_optimal_viz = create_soccer_visualization(soccer_jal_vs_optimal.policies...)
  ╠═╡ =#

# ╔═╡ c110a6fc-e2a2-4d26-822e-6cfe799ee687
#=╠═╡
@bind jal_vs_optimal_step soccer_jal_vs_optimal_viz.slider
  ╠═╡ =#

# ╔═╡ dc1b48f8-54ec-46e1-84b5-d5ac5c276390
#=╠═╡
soccer_jal_vs_optimal_viz.plot_function(jal_vs_optimal_step)
  ╠═╡ =#

# ╔═╡ 0de23d03-fc68-4b8b-9d0a-468e40ceee9a
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
CSV = "~0.10.15"
DataFrames = "~1.8.1"
HiGHS = "~1.18.1"
HypertextLiteral = "~0.9.5"
JuMP = "~1.29.3"
LaTeXStrings = "~1.4.0"
PlutoDevMacros = "~0.9.2"
PlutoPlotly = "~0.6.5"
PlutoProfile = "~0.4.0"
PlutoUI = "~0.7.75"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.12.3"
manifest_format = "2.0"
project_hash = "23981a06a3b41b40e40bf9ba709fc6f6e77421b6"

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
git-tree-sha1 = "deddd8725e5e1cc49ee205a1964256043720a6c3"
uuid = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
version = "0.10.15"

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

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions"]
git-tree-sha1 = "cd33c7538e68650bd0ddbb3f5bd50a4a0fa95b50"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "1.3.0"

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
git-tree-sha1 = "4072498280282c7d80a139b1fbf26091e6c338ca"
uuid = "87dc4568-4c63-4d18-b0c0-bb2238e4078b"
version = "1.18.1"

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

[[deps.JSON3]]
deps = ["Dates", "Mmap", "Parsers", "PrecompileTools", "StructTypes", "UUIDs"]
git-tree-sha1 = "411eccfe8aba0814ffa0fdf4860913ed09c34975"
uuid = "0f8b85d8-7281-11e9-16c2-39a750bddbf1"
version = "1.14.3"

    [deps.JSON3.extensions]
    JSON3ArrowExt = ["ArrowTypes"]

    [deps.JSON3.weakdeps]
    ArrowTypes = "31f734f8-188a-4ce0-8406-c8a06bd891cd"

[[deps.JuMP]]
deps = ["LinearAlgebra", "MacroTools", "MathOptInterface", "MutableArithmetics", "OrderedCollections", "PrecompileTools", "Printf", "SparseArrays"]
git-tree-sha1 = "b76f23c45d75e27e3e9cbd2ee68d8e39491052d0"
uuid = "4076af6c-e467-56ae-b986-b466b2749572"
version = "1.29.3"

    [deps.JuMP.extensions]
    JuMPDimensionalDataExt = "DimensionalData"

    [deps.JuMP.weakdeps]
    DimensionalData = "0703355e-b756-11e9-17c0-8b28908087d0"

[[deps.JuliaInterpreter]]
deps = ["CodeTracking", "InteractiveUtils", "Random", "UUIDs"]
git-tree-sha1 = "80580012d4ed5a3e8b18c7cd86cebe4b816d17a6"
uuid = "aa1ae85d-cabe-5617-a682-6adf51b2e16a"
version = "0.10.9"

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
deps = ["BenchmarkTools", "CodecBzip2", "CodecZlib", "ForwardDiff", "JSON3", "LinearAlgebra", "MutableArithmetics", "NaNMath", "OrderedCollections", "PrecompileTools", "Printf", "SparseArrays", "SpecialFunctions", "Test"]
git-tree-sha1 = "1381263d06a9526c4449238b32b1e01bed6b7ce9"
uuid = "b8f27783-ece8-5eb3-8dc8-9495eed66fee"
version = "1.47.0"

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

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.12.0"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "f2685b435df2613e25fc10ad8c26dddb8640f547"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.6.1"

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
git-tree-sha1 = "a3c1536470bf8c5e02096ad4853606d7c8f62721"
uuid = "892a3eda-7b42-436c-8928-eab12a02cf0e"
version = "0.4.2"

[[deps.StructTypes]]
deps = ["Dates", "UUIDs"]
git-tree-sha1 = "159331b30e94d7b11379037feeb9b690950cace8"
uuid = "856f2bd8-1eba-4b0a-8007-ebc267875bd4"
version = "1.11.0"

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
# ╟─c62ca696-4fbe-42e2-ab00-4a916f9fac8d
# ╠═844c29f5-fe4d-49cc-9dab-4b0ed6256a85
# ╟─798386fc-f572-4cff-9222-00ab3452faae
# ╠═3120f952-d848-4f37-b20c-922b5e897587
# ╠═69ed465f-7d0b-480b-90a2-186f51258b38
# ╠═7c691c82-6de9-47ee-8b61-e3e239976ecc
# ╟─96b067f8-5e7e-459a-9e3f-5640aa18680b
# ╟─dceebec1-bf10-4389-9ef8-864ea59e92e8
# ╟─25c1c075-b10a-4bc1-b113-39b684d8f939
# ╟─164251a2-a42c-45d7-839f-bf06d082a971
# ╟─939f23a4-d089-49ab-807c-ac107c3664d4
# ╟─bd5e2748-4a79-4fab-8019-f6a733c9a653
# ╠═1ed1be16-d0d0-446f-b096-5cc5e48e2ab5
# ╠═fe4bd7bf-a73b-401c-8682-dc734122f5e8
# ╠═e7cc25e9-b4fe-42a6-b07f-1940e7018068
# ╟─af9ba099-95b3-40cb-b1fe-b6091cbe7cd2
# ╟─bd7ce958-162b-419f-a177-6d1b21905e1c
# ╟─a01f5f05-788d-491e-9e2f-a047a11fbc03
# ╠═7b07aa13-b39a-4874-9de4-037685001e24
# ╟─e527160e-6d64-48d2-ab4d-abc67afebb97
# ╠═54db08a2-b7c5-4311-8bca-fce64d665563
# ╠═2eb915ed-16c2-4ea0-bf5f-5d1456249224
# ╠═af0265b8-7fe9-499a-b675-32bf5ba76f64
# ╟─fe077be2-467e-43a7-8729-3f3d48b8c91b
# ╠═439fff54-57e8-409e-994e-55e536d2ea0b
# ╟─9a615e94-6be8-47f5-aa38-91088f3accc3
# ╟─30d70205-a091-4155-87f5-d0ada3b597e0
# ╠═74d539f4-defc-4e5e-a085-a30e96df75bf
# ╟─036e8dbe-9acb-41bd-83eb-957da19fd86d
# ╠═37c4bbe2-f78e-4ad1-a0fb-a00da11631f1
# ╠═bfd2d1a0-7b3e-4fbf-9a29-22de88bbc08d
# ╠═badfc16b-3b6e-471a-8e6e-d7ced4885770
# ╟─09e01730-c983-48bf-9e7a-a3d2ccc0a631
# ╠═19d9e4b3-2c6b-4f33-a5be-32aa4a38513a
# ╠═b22df2e2-d76d-4426-a704-42a291103603
# ╟─b29a3234-8717-4b12-ac64-a79a0a74113a
# ╟─46791653-7452-4afb-8639-ac753af9f249
# ╟─84fdc00e-43d1-44c4-b479-d7be79c2d4cd
# ╟─f31c6793-2bfb-4970-9b36-fbc95460be30
# ╠═95a3fcca-1f0c-49e9-9e9c-b1bea8cb76c4
# ╠═b0ef6a1d-cb67-44a3-b8da-4f56930f85d7
# ╟─2d8f6a70-1f8e-4ad6-a616-e725b5c0a377
# ╠═e12a9ce4-1441-4863-9e31-19d04f512c61
# ╠═5906e0f9-eaf6-44f0-bfee-fdb4dd02f08b
# ╟─9ea31e89-6279-456d-be11-e915686e0be4
# ╟─a86d8b48-6619-4f92-aa17-17e3c4a5f4f7
# ╠═a2f8445e-8b83-4903-b829-15b997e4c3fc
# ╟─98f6ddc3-c534-4ba2-bce6-97b29e1428e4
# ╠═a3fd5c60-105c-4c9d-909f-0a0af6c592c3
# ╟─3ec640b5-9c7a-4973-b128-ec438d45f5b7
# ╟─efcc1361-774a-4245-b034-090dc29ae131
# ╠═1eaed379-6f4d-4634-aea6-a5b1941389d4
# ╠═46060259-a774-4ff1-b415-27a5c6e9fa40
# ╟─59461d31-4d11-4802-8e58-996a0fa7bf67
# ╠═0bb09a54-4d4b-472d-86cc-133a9d73349c
# ╟─b2872d0f-f780-449a-b17d-72d1506f90c0
# ╠═f6ea001e-38f6-428c-a107-4ddcf817f470
# ╟─bbdece04-95cd-4333-8244-eaf6f0be4b18
# ╠═fb3e4d0f-bc6c-45a2-9491-225ef7b8f647
# ╠═349e7689-24bd-467d-a7f0-73332fc93118
# ╟─1e15c352-c684-49ff-bbff-3cfeac7b41f6
# ╠═ffbd79f9-ac31-4ff4-ab8d-7c8603dfedf6
# ╟─dfd93d6c-da3a-4f02-b8a4-6ebec3dc8284
# ╟─989c2b5b-1d33-4292-b372-6c9a1f13be20
# ╠═c4929e3a-c48f-4ba1-bf78-32f42fa7446d
# ╟─9d9db616-81b3-462b-9400-3cef27107c56
# ╠═e6f529bf-c71e-431a-90ae-884d59d12132
# ╟─448ed962-b29f-445a-bb62-29c617a0b12e
# ╠═feb35b37-c99f-40b7-b181-a86038a8ed92
# ╟─f3566f79-350b-40ad-9817-af038202fd11
# ╟─03178016-9014-45a8-a977-1e3c794d3db6
# ╠═eda86d7a-87ca-4f07-9ca9-e9045906da92
# ╠═0fc4963b-2100-4b5d-a663-1c5306f311ca
# ╠═879e7384-7786-43b2-a36f-229929e802cf
# ╟─33669b7d-4609-4a11-a6d2-7569a582353e
# ╟─7e0e10fe-2c5c-4c27-b21a-2568bebe8e6d
# ╠═c3418786-fa47-4eb3-b239-3183ff920cc7
# ╠═2278415f-f4e5-4ed9-94ad-761941039b97
# ╠═91b0b90b-a1d7-42ac-a493-257dcc29bc6a
# ╠═50a06568-bd54-4f0c-82a1-532050980940
# ╟─97e10e56-9990-4b53-bf01-109eda72bb1c
# ╠═d5351789-cd1a-4b20-9e96-8f3810b811f1
# ╟─b84645f7-7ce2-498a-9db7-0da603e2904a
# ╠═140a1608-8105-4d9c-a839-12e94483eab4
# ╠═d3a1b675-a218-4543-a501-df25b7255762
# ╠═4b940a7f-21aa-44f3-9ecf-03e3ab0c2900
# ╠═d6dc4e3c-faa7-426c-972e-dacf9bf6ba88
# ╟─e071b761-5df2-4776-a959-2b8546027cd6
# ╠═b1f4df52-2049-43e6-bc46-fe7e192e74ca
# ╠═32bc7859-7503-46c6-b8e0-2c3c22b2acc9
# ╠═72097ef2-0fcb-45c8-8537-253af364e35a
# ╠═cf478f51-42df-4f1f-a666-b470afe28767
# ╟─c110a6fc-e2a2-4d26-822e-6cfe799ee687
# ╠═dc1b48f8-54ec-46e1-84b5-d5ac5c276390
# ╠═63af1889-a4a6-4d0e-a21f-21107c0efed2
# ╠═bd291ecb-f529-46da-8fab-ae172535afa9
# ╠═7f486f74-6b68-4e03-9d09-858091d51784
# ╠═cd3bb7ba-3560-40a9-b166-7cb5cb23f316
# ╟─5cfcb48f-59af-435c-81ea-170eb8a94c78
# ╠═bf7a9be7-5e04-4059-b67e-01b30f929036
# ╟─b4380430-e223-44e9-96e7-e784de33020d
# ╟─0ada2900-e35b-42c4-8e35-f357b57a2579
# ╠═422d5853-3a5e-4f84-96a4-8b05a752c64b
# ╠═42e13eeb-b6e4-49da-81b4-e8c16ed74688
# ╠═9918d23e-a266-4c5c-b63b-f984c8db1bb5
# ╠═da51fa8e-02cc-43a8-be86-6d9e6514bd46
# ╟─1c18c351-9afe-41b0-8e12-c4a7441d8499
# ╠═9abf652e-7ab2-43a1-80ed-3a992bf720f5
# ╠═b199d65b-95ab-41ab-bce9-fd43bda79f35
# ╠═865b6b13-531e-4960-8bfb-2e22906b077d
# ╠═9cfe4831-18b1-4c16-aa79-ffcff88c3f4e
# ╠═c5cd53c2-9057-492e-b2e2-dfed2dce814b
# ╟─278ed71e-9872-472c-ab59-6541b16e13e8
# ╠═731008b1-2d23-4a8e-ab19-ac61e1f6a01b
# ╠═ed04bac9-32e1-4d26-b44a-39da7ed52abc
# ╠═8087e0c6-c0e2-499d-baec-499e23b3a8e5
# ╠═ce4b2377-45e2-421d-b457-c40aaabee4e4
# ╠═026c1c3e-8ae2-4acb-8f2e-286187a4e408
# ╠═7429193c-9bee-4031-8c4b-ccb1e8f6ad84
# ╠═07af85fb-cf84-4592-aeeb-49b8c37b5794
# ╠═f1255c1c-6435-4eca-8d84-4b7f3a505ace
# ╠═f3a8c840-8c0c-4bcd-a541-2801cddd529f
# ╠═1b671a28-ab2d-4a4f-ab4a-f6916daa34b6
# ╠═8878e879-6e70-4d21-8eb9-c4e1898f0f48
# ╠═1ec55568-2331-40a2-8fa5-191cae1f6cc8
# ╠═58cda3e3-9ba1-428b-bc80-b62951c4e193
# ╟─8adec11d-e873-49c7-8a49-3e584cf76720
# ╠═23db3972-76b7-4369-866b-d1f187ba7670
# ╟─5eb49614-6762-4544-9bf1-bcfa3af62646
# ╟─549e81d0-9dbe-4f89-9849-333d7bdc4d4f
# ╟─3ab9580f-0b52-4b96-a386-91783e598798
# ╠═abd8af73-c03d-468f-8bc3-5c5b5caaae44
# ╠═be5b8d2e-2d66-4291-93b4-056cdc68e3c4
# ╠═3efd8233-04da-4a6c-bbb7-ef7ed5c02057
# ╠═26f22c5b-fcb1-43c5-9b2e-efa74c79f235
# ╠═b802988a-48c7-4de8-b37b-95b63884f79f
# ╠═6bf0c879-7808-4b47-aa06-1614a43bba4a
# ╠═78e0333b-90c1-40c8-9c03-17759da296c5
# ╠═692cb71b-fb78-4a27-aad8-61c2b25a29c6
# ╠═9515108e-2ab2-44f6-930a-05723cc4a07d
# ╠═4e4a131b-b824-491b-a0cb-012d5dd1afe6
# ╠═882a36b8-89ef-4d8d-ac57-0f060b670bba
# ╠═ab8d5a68-4ea8-4bec-8c2a-3f0727d05943
# ╠═a74c712b-5458-4d87-bdba-159617d30742
# ╠═0e195ca4-9077-441e-8349-06b8824dce70
# ╠═42a04270-323d-4d45-b025-d77694b5b982
# ╠═4556459e-54d8-4b3c-9a22-96d0985176f0
# ╠═2cd6aa77-6828-44ef-b7d9-4f6cfc50e880
# ╟─5da718f5-c940-49a3-9c38-7af26a930439
# ╟─52a51c78-4438-44b2-8315-9cffd8ba2581
# ╠═9a80e476-bad5-4a8f-a8e2-0008a1e5d698
# ╠═b053dc54-af90-463e-8510-38c33be32312
# ╠═f338dda8-79b1-4c54-90ee-400e29b9d0b0
# ╠═065ba539-ac92-4117-a1a3-d44d21237e34
# ╠═c6ca307f-9753-4b0d-8fd8-ee05086fd5ed
# ╠═2f8b643d-0dbb-47b3-8644-eaaa1e5410cf
# ╠═b8de4e87-0d28-4731-84f9-b0c8e799b9a5
# ╠═bfd96f31-f01f-4495-bde5-2ade7dcf2525
# ╠═445b2fca-85cf-4937-a827-88453b96dba1
# ╠═4d8eb87e-c43b-4b55-b632-79bca14b7759
# ╠═cd97f119-5eeb-424c-b73a-86c525cd3787
# ╠═ab01fea4-2da7-47c6-81e7-847ffeda8b34
# ╠═6383ab5d-9d91-42f9-8a7f-427408ca356f
# ╠═05aef224-961e-4ee9-84bb-6d6002c29e79
# ╠═3c1bfb16-c9d9-4c88-8f0b-749526b0018e
# ╠═6eccc26c-85a5-4ee4-90e7-874c9b670081
# ╠═4ff02aaf-950e-463f-9c69-306a1ca5c4b0
# ╠═d23092be-c8b8-4723-a435-83e3e2e30650
# ╠═400aea40-3d2c-4696-b31c-3e0518341998
# ╟─697e2312-c9a6-47b3-b13c-277ad4b0efb2
# ╠═a1407418-2efb-4bee-9645-d029735d99f7
# ╟─78815d3e-b1fa-4a60-b86d-07a5e40fea60
# ╠═511e384b-697e-4397-8ec5-bf5449b9eee5
# ╠═ee51c98e-444e-4a44-bb67-ea91f739d731
# ╠═2b067fa4-3597-451b-8572-5ff877c68498
# ╠═e3cc189b-80a8-44a9-b734-59eece616647
# ╠═322d6682-05db-41cc-b6d4-42d772b051ac
# ╠═d056d56d-3c46-4d3d-8789-e1ad4aa41bf1
# ╠═bf9e5075-3595-42ca-9b73-0e16e608b946
# ╠═7ab9b64b-6571-4cd4-bf9f-6ad00015dac8
# ╠═50357183-c467-48c9-b1c7-05d7c2a5db55
# ╠═ab3793ad-1069-4b77-a95a-4474fcd7c662
# ╠═1168356f-fedf-4e53-a761-89f99f6b3812
# ╠═7ba3754f-4a2d-4f87-a303-e37ea2376947
# ╟─59e334fd-0fdd-4c1c-a1e0-07777d233f6f
# ╠═62d8a4f9-fd25-449c-acd0-7f8e34561c31
# ╟─9d54ee68-d60c-11f0-87d1-3be62822741b
# ╠═20a8c921-816f-4c5c-9341-b7749644d249
# ╠═c3e732e2-2a10-4de3-800f-0378e0acd0dc
# ╠═00999193-757c-4bea-a24c-a5ed8633d826
# ╠═e5a78e58-0208-4c6a-9f7f-4779e384855b
# ╠═a929c6cc-733b-44a1-bdbd-40a525bb5ffd
# ╠═db2d2393-e984-4828-9ec7-d9dfe076bc79
# ╠═0de23d03-fc68-4b8b-9d0a-468e40ceee9a
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
