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
# Stochastic Game Utilities
"""

# ╔═╡ 393419ba-28fb-417e-ab2a-fafe857d0814
md"""
## Tabular Games
"""

# ╔═╡ 6eebd434-1520-4885-a5ca-139fc916bd7a
function make_random_policies(game::TabularStochasticGame{T, S, A, N, P, F}) where {T<:Real, S, A, P, F, N}
	n_actions = Tuple(length(a) for a in game.agent_actions)
	n_states = length(game.states)
	Tuple(ones(T, n_actions[n], n_states) ./ n_actions[n] for n in 1:N)
end

# ╔═╡ 844c29f5-fe4d-49cc-9dab-4b0ed6256a85
begin
	function TabularRL.runepisode!((states, joint_actions, rewards)::Tuple{Vector{Int64}, Vector{NTuple{N, Int64}}, Vector{R}}, game::TabularStochasticGame{T, S, A, N, P, F}; i_s0::Integer = game.initialize_state_index(), πs::NTuple{N, M} = make_random_policies(game), a0::NTuple{N, Int64} = sample_joint_action(πs, i_s0), max_steps = Inf) where {T<:Real, T2<:Real, S, A, P, F, N, M <: AbstractMatrix{T2}, R<:Union{NTuple{N, T}, T}}
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

# ╔═╡ 3120f952-d848-4f37-b20c-922b5e897587
const soccer_game = TwoPlayerSoccer.make_distribution_environment()

# ╔═╡ 641255f6-19d8-401d-b522-7cf22577d87d
const soccer_rnd_ep = runepisode(soccer_game; max_steps = 1000)

# ╔═╡ c3e5baea-2b49-4084-83c2-ec0a9402d297
#=╠═╡
@bind ep_step Slider(1:length(soccer_rnd_ep[1])+1)
  ╠═╡ =#

# ╔═╡ 48df4121-7e87-4514-b852-ec1c00e7a232
# ╠═╡ disabled = true
#=╠═╡
@bind ep_step Clock(;max_value = length(soccer_rnd_ep[1])+1, repeat = true, interval = 0.1)
  ╠═╡ =#

# ╔═╡ 44027bb5-a9de-4448-8afc-a725b01557ca
#this game can also be written as a tabular stochastic game where each state/joint-action pair can result in two outcomes with equal probability depending on which player is randomly selected to go first.  There are also cases when the order doesn't matter.  If I create a game like that, then I could try value iteration to solve it.

# ╔═╡ af9ba099-95b3-40cb-b1fe-b6091cbe7cd2
md"""
# Solution Methods
"""

# ╔═╡ bd7ce958-162b-419f-a177-6d1b21905e1c
md"""
## Value Iteration for Stochastic Games
"""

# ╔═╡ e527160e-6d64-48d2-ab4d-abc67afebb97
#implement algorithm 6 from the book here.  The "value" calculation can be specialized to situations depending on the game transition type.  For the soccer game I need to focus on the minimax solution type which only works for a two player zero sum game

# ╔═╡ 9171a928-3c5a-4e3d-b758-89d060e4d26d
#I also want to create utilities to convert the game into an MDP from the perspective of one of the agents using some fixed policy for the other agents.

# ╔═╡ b53482aa-aa48-4846-ba9b-a9f7c2279fcd
#add version of bellman game reward function that works with non-deterministic game states like I intend to have for the soccer game

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

# ╔═╡ ad5613d9-4db4-44a3-98fe-fd0adeacac87
#change this function to take the model with the above settings already baked in.  that way we only need to add the reward matrix constraint and then solve the updated model

# ╔═╡ 393a2ab8-eda8-4197-a9b6-4f2e609dc936
#need to change this solver to use the reward matrix of the opposing player instead of the main player since this is the proper wya for the linear program to work.  That way the player policy is based on minimizing the expected reward for the other player based on that players reward estimation

# ╔═╡ ae3a776d-65a2-4797-935c-53e6c5bffa58
0.9^14

# ╔═╡ 2621b3be-cf6c-4431-8986-583446124fed
const i_s0_soccer = soccer_game.initialize_state_index()

# ╔═╡ 84f56ea1-9222-4fc2-a427-5c9b8b9468ed
soccer_game.states[i_s0_soccer]

# ╔═╡ 69ed465f-7d0b-480b-90a2-186f51258b38
const soccer_random_policies = make_random_policies(soccer_game)

# ╔═╡ bfd2d1a0-7b3e-4fbf-9a29-22de88bbc08d
#compare solution states with different discount rates to see which states are affected by that
#show policy and value as a grid with one agent in place and the other squqares show the value/policy for the first agent when the second agent is in that square, this would be more useful for value to see what is desirable for the position of the second agent when you are the first
#add value iteration for policy to train against one of these.  Need to create stochastic distribution MDP based on the other agent's policies
#note that the non-game based algorithms cannot reproduce stochastic policies like what is needed for some of the game states

# ╔═╡ 2a420894-33a5-401f-adf9-4253f648ae42
soccer_game.agent_actions

# ╔═╡ 5da718f5-c940-49a3-9c38-7af26a930439
md"""
# Visualization Tools
"""

# ╔═╡ 52a51c78-4438-44b2-8315-9cffd8ba2581
md"""
## General
"""

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

# ╔═╡ 697e2312-c9a6-47b3-b13c-277ad4b0efb2
md"""
## Soccer Game
"""

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

# ╔═╡ a5901beb-eac1-4056-b0e0-1add182b061d
#=╠═╡
plot_soccer_state(soccer_game.states[vcat(soccer_rnd_ep[1], soccer_rnd_ep[4])[ep_step]])
  ╠═╡ =#

# ╔═╡ 7eb2de3f-6a7d-485b-a740-9da4cc03484d
#=╠═╡
plot_soccer_state(soccer_game.states[59])
  ╠═╡ =#

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

# ╔═╡ d056d56d-3c46-4d3d-8789-e1ad4aa41bf1
#=╠═╡
function plot_soccer_solution(s::TwoPlayerSoccer.State{W, H, GH}, πs::Tuple{Matrix, Matrix}, value_function::Vector) where {W, H, GH}	
	state_plot = plot_soccer_state(s)
	i_s = soccer_game.state_index[s]
	v1 = πs[1][:, i_s]
	v2 = πs[2][:, i_s]
	π1_disp = display_soccer_policy(v1)
	π2_disp = display_soccer_policy(v2)

	@htl("""
	<div style = "width: 600px;">
		<div style = "display: flex; justify-content: space-around; background-color: white; color: black;">
		 <div>
		  Player 1 Value/Policy: $(value_function[i_s])
		 $π1_disp
		 </div>
		 <div>
		  Player 2 Value/Policy: $(-value_function[i_s])
		 $π2_disp
		 </div>
		 </div>
		 $state_plot
	</div>
		 """)
end
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

		if all(iszero, m) 
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

		if all(x -> isapprox(x, reward_matrix[1]), reward_matrix) 
			p = one(T) / x_length
			@inbounds @simd for i in 1:x_length
				π[i, i_s] = p
			end
			return -reward_matrix[1]
		end

		reward_matrix .-= (min_reward - one(T))

		if player1
			@constraint(model, con_s, x' * reward_matrix .≤ one(T))
		else
			@constraint(model, con_s, reward_matrix * x .≤ one(T))
		end
		
		optimize!(model)
		modified_game_value = inv(sum(value.(x)))
		@inbounds @simd for i in 1:x_length
			π[i, i_s] = value(x[i]) * modified_game_value
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
	π1 = zeros(T, l, num_states)
	π2 = zeros(T, k, num_states)

	delta_max = typemax(T)
	sweep = 0
	
	state_transitions = ptf.state_transition_map
	reward_transitions = ptf.reward_transition_map

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
			
			delta_max = max(abs(v_old - v_est[i_s]), delta_max)
		end
		sweep += 1
		show_message && @info "After $(sweep) sweeps, maximum value change is $delta_max"
	end

	for i_s in eachindex(v_est)
		solve_minimax_game!(reward_matrix, πs[2], model2, y, m_est, i_s, false)
	end

	return (final_values = v_est, total_sweeps = sweep, final_policies = πs, game_rewards = m_est)
end

# ╔═╡ af0265b8-7fe9-499a-b675-32bf5ba76f64
begin
	value_iteration_game(ptf::TabularZeroSumGameTransition{T, ST, RT}, γ::T; nmax::Integer = typemax(Int64), dims::NTuple{3, Int64} = size(ptf.state_transition_map), v_est::Vector{T} = zeros(T, dims[3]), πs::NTuple{2, Matrix{T}} = (zeros(T, dims[1], dims[3]), zeros(T, dims[2], dims[3])), m_est::Array{T, 3} = zeros(T, dims...), kwargs...) where {T<:Real, RT, ST} = value_iteration_game!(v_est, πs, m_est, ptf, γ, nmax; kwargs...)

	value_iteration_game(game::TabularStochasticGame, γ::T; kwargs...) where {T<:Real} = value_iteration_game(game.ptf, γ; kwargs...)
end

# ╔═╡ 439fff54-57e8-409e-994e-55e536d2ea0b
const soccer_value_iter = value_iteration_game(soccer_game, 0.9f0; θ = 1f-3)

# ╔═╡ 3d9d265a-35d3-4f23-adbc-99f32d19350e
filter(a -> maximum(a) != 1, eachcol(soccer_value_iter.final_policies[1]))

# ╔═╡ d9286880-143e-43e3-bff4-4e33cc3c10bd
soccer_value_iter.final_values[i_s0_soccer]

# ╔═╡ fd270bad-7ccd-46e0-ac22-30c6a3ae925f
const soccer_ideal_ep = runepisode(soccer_game; πs = (soccer_value_iter.final_policies[1], soccer_value_iter.final_policies[2]), max_steps = 10_000)

# ╔═╡ f6a1e935-52c3-4f8d-83a9-9a82850b38f9
#=╠═╡
@bind ep_step_ideal Slider(1:length(soccer_ideal_ep[1])+1)
  ╠═╡ =#

# ╔═╡ 4413ba09-3dbc-4e87-9adf-d9e62efa4aac
#=╠═╡
soccer_state_index = vcat(soccer_ideal_ep[1], soccer_ideal_ep[4])[ep_step_ideal]
  ╠═╡ =#

# ╔═╡ 2276c50d-b556-495c-9b0b-efc19b8f2a99
#=╠═╡
soccer_value_iter.final_policies[1][:, soccer_state_index]
  ╠═╡ =#

# ╔═╡ 1e13f0be-7b17-40e9-81e1-1b176869f175
#=╠═╡
plot_soccer_solution(soccer_game.states[soccer_state_index], soccer_value_iter.final_policies, soccer_value_iter.final_values)
  ╠═╡ =#

# ╔═╡ f0b2edf8-ba0a-464e-bc78-4035728b7c54
0:10000 |> Map(i -> runepisode(soccer_game; πs = (soccer_random_policies[2], soccer_value_iter.final_policies[2] .+ eps(1f0)), max_steps = 10000)[3][end]) |> foldxt(+) |> x -> x / 10000

# ╔═╡ 0f1dcee0-3e52-43d4-a79e-b76e81e7cc30
0:10000 |> Map(i -> runepisode(soccer_game; πs = (soccer_random_policies[2], soccer_value_iter.final_policies[2] .+ eps(1f0)), max_steps = 10000)[3] |> length) |> foldxt(+) |> x -> x / 10000

# ╔═╡ 93924c93-e859-471d-8505-f7213b1ef5f6
#=╠═╡
plot_soccer_solution(soccer_game.states[1], soccer_value_iter.final_policies, soccer_value_iter.final_values)
  ╠═╡ =#

# ╔═╡ 19b92b1c-df1f-43b6-9504-cfbfc6524465
# ╠═╡ show_logs = false
#=╠═╡
@plutoprofview value_iteration_game(soccer_game, 0.9f0; θ = 1f-3)
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

julia_version = "1.12.2"
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
# ╟─393419ba-28fb-417e-ab2a-fafe857d0814
# ╠═6eebd434-1520-4885-a5ca-139fc916bd7a
# ╠═844c29f5-fe4d-49cc-9dab-4b0ed6256a85
# ╠═3120f952-d848-4f37-b20c-922b5e897587
# ╠═641255f6-19d8-401d-b522-7cf22577d87d
# ╟─c3e5baea-2b49-4084-83c2-ec0a9402d297
# ╟─48df4121-7e87-4514-b852-ec1c00e7a232
# ╠═a5901beb-eac1-4056-b0e0-1add182b061d
# ╠═44027bb5-a9de-4448-8afc-a725b01557ca
# ╟─af9ba099-95b3-40cb-b1fe-b6091cbe7cd2
# ╟─bd7ce958-162b-419f-a177-6d1b21905e1c
# ╠═e527160e-6d64-48d2-ab4d-abc67afebb97
# ╠═9171a928-3c5a-4e3d-b758-89d060e4d26d
# ╠═b53482aa-aa48-4846-ba9b-a9f7c2279fcd
# ╠═54db08a2-b7c5-4311-8bca-fce64d665563
# ╠═ad5613d9-4db4-44a3-98fe-fd0adeacac87
# ╠═393a2ab8-eda8-4197-a9b6-4f2e609dc936
# ╠═7b07aa13-b39a-4874-9de4-037685001e24
# ╠═2eb915ed-16c2-4ea0-bf5f-5d1456249224
# ╠═af0265b8-7fe9-499a-b675-32bf5ba76f64
# ╠═439fff54-57e8-409e-994e-55e536d2ea0b
# ╠═3d9d265a-35d3-4f23-adbc-99f32d19350e
# ╠═19b92b1c-df1f-43b6-9504-cfbfc6524465
# ╠═ae3a776d-65a2-4797-935c-53e6c5bffa58
# ╠═2621b3be-cf6c-4431-8986-583446124fed
# ╠═84f56ea1-9222-4fc2-a427-5c9b8b9468ed
# ╠═d9286880-143e-43e3-bff4-4e33cc3c10bd
# ╠═69ed465f-7d0b-480b-90a2-186f51258b38
# ╠═fd270bad-7ccd-46e0-ac22-30c6a3ae925f
# ╠═bfd2d1a0-7b3e-4fbf-9a29-22de88bbc08d
# ╟─f6a1e935-52c3-4f8d-83a9-9a82850b38f9
# ╠═2276c50d-b556-495c-9b0b-efc19b8f2a99
# ╠═1e13f0be-7b17-40e9-81e1-1b176869f175
# ╠═2a420894-33a5-401f-adf9-4253f648ae42
# ╠═4413ba09-3dbc-4e87-9adf-d9e62efa4aac
# ╠═f0b2edf8-ba0a-464e-bc78-4035728b7c54
# ╠═0f1dcee0-3e52-43d4-a79e-b76e81e7cc30
# ╠═7eb2de3f-6a7d-485b-a740-9da4cc03484d
# ╟─5da718f5-c940-49a3-9c38-7af26a930439
# ╟─52a51c78-4438-44b2-8315-9cffd8ba2581
# ╠═b053dc54-af90-463e-8510-38c33be32312
# ╠═f338dda8-79b1-4c54-90ee-400e29b9d0b0
# ╠═065ba539-ac92-4117-a1a3-d44d21237e34
# ╟─697e2312-c9a6-47b3-b13c-277ad4b0efb2
# ╠═93924c93-e859-471d-8505-f7213b1ef5f6
# ╠═d056d56d-3c46-4d3d-8789-e1ad4aa41bf1
# ╠═7ab9b64b-6571-4cd4-bf9f-6ad00015dac8
# ╠═ab3793ad-1069-4b77-a95a-4474fcd7c662
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
