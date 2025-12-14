### A Pluto.jl notebook ###
# v0.20.21

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    return quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ 7336c97e-15d7-45ee-980c-5e90e9e324d8
using PlutoUI, PlutoPlotly, PlutoProfile, BenchmarkTools, Statistics, StatsBase, LinearAlgebra, JuMP, HypertextLiteral

# ╔═╡ be459828-55e3-499c-946d-1c510187db56
md"""
# Normal-Form Games

*A normal-form game consists of:*

- *Finite set of agents* ``I = \{1, \dots, n \}``
- *For each agent* ``i \in I:``
 - *Finite set of actions* ``A_i``
 - *Reward function* ``\mathcal{R}_i: A \rightarrow \mathbb{R}`` *where* ``A = A_1 \times \dots \times A_n``

Each agent ``i \in I`` selects a policy ``\pi_i: A_i \rightarrow [0, 1]`` which assigns probabilities to the actions available to the agent so that ``\sum_{a_i \in A_i} \pi_i(a_i) = 1``.  Each agent samples an action ``a_i \in A_i`` with probability ``\pi_i(a_i)`` given by its policy.

*Each agent has a policy that provides a probability distribution over actions*:
- ``\pi_i \implies \{\pi_i(a_1), \pi_i(a_2), \dots\}`` such that ``\sum_{a_i \in A_i} \pi_i(a_i) = 1`` ``\forall i \in I``
- Joint action: ``a = (a_1, \dots a_n)``
- Join policy: ``\pi = (\pi_1, \cdots, \pi_n)``

"""

# ╔═╡ 14115f9b-e92d-4bf2-aa08-c7e9e4b55e10
md"""
## Best Response Policy

Given a set of policies for all agents other than agent ``i``, denoted by ``\pi_{-i} = (\pi_1, \dots \pi_{i-1}, \pi_{i+1}, \dots, \pi_n)``, a best response for agent ``i`` to ``\pi_{-i}`` is a policy ``\pi_i`` that maximizes the expected return for ``i`` when played against ``\pi_{-i}``:

```math
\operatorname{BR}_{i}(\pi_{-i}) = \operatorname{argmax}_{\pi_i} U_i(\langle \pi_i, \pi_{-i} \rangle ) \tag{4.9}
```

where ``\langle \pi_i, \pi_{-i} \rangle`` denotes the complete joint policy consisting of ``\pi_i`` and ``\pi_{-i}``.  For convenience we will write this without the angled brackets often (e.g. ``U_i(\pi_i, \pi_{-i})``)
"""

# ╔═╡ 02890ffe-4000-4295-898c-769ed931354d
md"""
## Two Player Games

In a two player game there are just two agents, so we can represent the reward function for each agent as a 2x2 matrix with agent 1 represented by the rows and agent 2 represented by the columns.  For convenience, let's refer to agent one as ``X`` and agent 2 as ``Y``.  Let's say there are ``l`` actions available to ``X`` and ``k`` actions available to ``Y``.  Then each reward function is an ``l \times k`` matrix

```math
\mathcal{R}_x = \left [ \begin{matrix}
r_{11}^x \; &r_{12}^x \cdots &r_{1k}^x\\
r_{21}^x \; &r_{22}^x \dots &r_{2k}^x\\
\vdots &\ddots &\vdots \\
r_{l1}^x &  &r_{lk}^x
\end{matrix} \right ],

\quad
\mathcal{R}_y = \left [ \begin{matrix}
r_{11}^y \; &r_{12}^y \cdots &r_{1k}^y\\
r_{21}^y \; &r_{22}^y \dots &r_{2k}^y\\
\vdots &\ddots &\vdots\\
r_{l1}^y &  &r_{lk}^y
\end{matrix} \right ]
```

For example, ``r_{12}^x`` is the reward received by ``X`` when ``X`` takes action 1 and ``Y`` takes action 2.  For convenience we can refer to actions by their numerical index which just means the ``nth`` action available to that agent.

- Joint policy: ``\pi = (\pi_x, \pi_y)`` where each policy is a list of probabilities over the available actions
- ``\pi_x = [p_1^x, p_2^x, \dots, p_l^x]`` 
- ``\pi_y = [p_1^y, p_2^y, \dots, p_l^y]`` 

*A pure policy is one in which only a single action is selected*

- ``\pi_x = [1, 0, \dots, 0]`` is a pure policy for ``X`` selecting action 1
- ``\pi_y = [0, 1, \dots, 0]`` is a pure policy for ``Y`` selecting action 2

Pure policies are deterministic.  A *mixed policy* is stochastic meaning more than one action could be selected with non-zero probability.  Neither agent would know the outcome of a mixed policy ahead of time.
"""

# ╔═╡ aff36c99-8b0a-4b73-af43-30c10effd53a
md"""
## Two Player Zero-Sum Games

In a *zero-sum game* the reward matrix of ``X`` is always the negative of ``Y``'s.  In other words:

```math
\mathcal{R}_x = - \mathcal{R}_y \text{ and } \mathcal{R}_y = - \mathcal{R}_x
```

We can therefore represent the entire game with a single reward matrix and use the notation ``\mathcal{R} = \mathcal{R}_x = -\mathcal{R}_y``

```math
\mathcal{R}_x = \left [ \begin{matrix}
r_{11}^x \; &r_{12}^x \cdots &r_{1k}^x\\
r_{21}^x \; &r_{22}^x \dots &r_{2k}^x\\
\vdots &\ddots &\vdots\\
r_{l1}^x &  &r_{lk}^x
\end{matrix} \right ] \implies 

\mathcal{R}_y = \left [ \begin{matrix}
-r_{11}^x \; &-r_{12}^x \cdots &-r_{1k}^x\\
-r_{21}^x \; &-r_{22}^x \dots &-r_{2k}^x\\
\vdots &\ddots &\vdots\\
-r_{l1}^x &  &-r_{lk}^x
\end{matrix} \right ]
```

In other words, ``r_{ij}^x = -r_{ij}^y \; \forall \; i \in [1, l], j \in [1, k]``

For this special case, we can simplify the notation and refer to a single reward matrix and reward values without superscripts:

```math
\mathcal{R} = \left [ \begin{matrix}
r_{11} \; &r_{12} \cdots &r_{1k}\\
r_{21} \; &r_{22} \dots &r_{2k}\\
\vdots &\ddots &\vdots\\
r_{l1} &  &r_{lk}
\end{matrix} \right ]
```
"""

# ╔═╡ c3ea41a2-ba31-4004-aeb3-b91b044532d4
md"""
### Minimax Solution

*In a zero-sum game with two agents, a joint policy ``\pi = (\pi_i, \pi_j)`` is a* minimax solution *if*

```math
\begin{flalign}
U_i(\pi) &= \max_{\pi_i^\prime} \min_{\pi_j^\prime} U_i(\pi_i ^\prime, \pi_j ^\prime) \tag{4.10} \\
		&= \min_{\pi_j^\prime} \max_{\pi_i^\prime} U_i(\pi_i ^\prime, \pi_j ^\prime) \tag{4.11} \\
&= -U_j(\pi) 
\end{flalign}
```

In other words, each policy is a best response to the other simultaneously.
"""

# ╔═╡ e6df629a-dc22-43f5-8e01-9f8bdfe0d3ea
md"""
### Two-Action Case

Let's say each agent only has two available actions.  Our reward matrix will have four values and can be written as follows:

```math
\mathcal{R} = \left [ \begin{matrix} r_{11} \quad r_{12} \\ r_{21} \quad r_{22} \end{matrix} \right ]
```

Each player's policy will be a vector of two values: ``\pi_x = [p_1^x, p_2^x]`` and ``\pi_y = [p_1^y, p_2^y]``.

In order to analyze this game, we need to write down ``U_x(\pi)`` which is the expected reward for ``X`` as a function of the joint policy.  In the two-action case, the joint policy consists of four values just like the reward function.  The expected return must account for the four possible game outcomes:

```math
\begin{flalign}
U_x(\pi) &= r_{11} p_1^x p_1^y + r_{12} p_1^x p_2^y + r_{21} p_2^x p_1^y + r_{22} p_2^x p_2^y
\end{flalign}
```

Let's look more closely at (4.10) and replace terms with our notation:

```math
\begin{flalign}
U_i(\pi) &= \max_{\pi_i^\prime} \min_{\pi_j^\prime} U_i(\pi_i ^\prime, \pi_j ^\prime) \\
\downarrow \\
U_x(\pi) &= \max_{p_1^x, p_2^x} \min_{p_1^y, p_2^y} \left ( r_{11} p_1^x p_1^y + r_{12} p_1^x p_2^y + r_{21} p_2^x p_1^y + r_{22} p_2^x p_2^y \right)
\end{flalign}
```

We can actually make a further simplification since each policy is a probability distribution, we know that ``p_1^x + p_2^x = 1 \implies p_2^x = 1 - p_1^x`` and similarly for ``Y``.  Let's simplify the notation further and use ``x \doteq p_1^x`` and ``y \doteq p_1^y``.  Now we can further simplify the above expression as follows:

and just consider the second part assuming that ``\pi_x`` is fixed:

```math
\begin{flalign}
U_x(\pi) &= \max_{x} \min_{y} \left ( x y \times r_{11}  +  x (1-y) \times r_{12} +  (1-x) y \times r_{21} +  (1-x) (1-y) \times r_{22} \right) \\
&= \max_{x} \min_{y} \left ( x y (r_{11} - r_{12} - r_{21} + r_{22}) + x \times (r_{12} - r_{22}) + y(r_{21} - r_{22}) + r_{22} \right )\\
&= \max_{x} \min_{y} \left ( y \times \left [ x(r_{11} - r_{12} - r_{21} + r_{22}) + r_{21} - r_{22} \right ] + x (r_{12} - r_{22}) + r_{22} \right )\\
\end{flalign}
```

The purpose of the final expression is to write everything as a function of ``y`` with other variables taken as a constant.  That way we can consider some fixed ``x`` and then our goal is to minimize the ``y`` expression which is now linear.  We can visualize what this function looks like as follows:
"""

# ╔═╡ 37c563f3-c8af-4ffa-958c-1f77dfdfe530
@bind rewardmatrix2 PlutoUI.combine() do Child
	md"""
	``\mathcal{R} =``
	
	|Action|||
	|---|---|---|
	||1|2|
	|1|$(Child(:AA, NumberField(-10:10, default = 1)))|$(Child(:AB, NumberField(-10:10, default = -1)))|
	|2|$(Child(:BA, NumberField(-10:10, default = -1)))|$(Child(:BB, NumberField(-10:10, default = 1)))|
	"""
end

# ╔═╡ 755b5857-b1c3-4739-a8c7-5d373f011e06
md"""
In order to perform the minimax optimization, ``X`` must consider all of the above lines and extract from each one the minimum value.  Then ``X`` will select the line that produces the highest value out of all those minimums.  In reality, there is a continum of lines so we can better represent this function as a heatmap shown below.  The important point for ``X`` is that the minimum value will always occur at an endpoint of the line, so we only ever need to calculate values for ``y = 0`` and ``y = 1``.
"""

# ╔═╡ d1dc4cf0-5739-4562-817e-303d8971e74d
md"""
Returning to the expression, let's see how we could write down this procedure algebraically.  
```math
\begin{flalign}
U_x(\pi) &= \max_{x} \min_{y} \left ( y \times \left [ x(r_{11} - r_{12} - r_{21} + r_{22}) + r_{21} - r_{22} \right ] + x (r_{12} - r_{22}) + r_{22} \right )\\
\end{flalign}
```

We know from inspection that we can reduce the inner minimization to a check over two values of y, so let's do that now:

```math
\begin{flalign}
U_x(\pi) &= \max_{x} \left [ \min \left ( x (r_{12} - r_{22}) + r_{22}, \; \left [ x(r_{11} - r_{12} - r_{21} + r_{22}) + r_{21} - r_{22} \right ] + x (r_{12} - r_{22}) + r_{22} \right ) \right ] \\
&= \max_{x} \left [ \min \left ( x (r_{12} - r_{22}) + r_{22}, \; x(r_{11} - r_{21}) + r_{21} \right ) \right ] \\
\end{flalign}
```

There are two functions of ``x`` inside the minimum expression.  If we just select either one of them, then the maximization would be trivial since it would occur at either ``x = 0`` or ``x = 1``, but with the inner minimization, the calculation is slightly more complicated.  Let's look at what these lines actually are.
"""

# ╔═╡ 0b6c96e3-9fe4-4a09-9d3c-bc87966c10ab
@bind rewardmatrix PlutoUI.combine() do Child
	md"""
	|Action|||
	|---|---|---|
	||1|2|
	|1|$(Child(:AA, NumberField(-10:10, default = 1)))|$(Child(:AB, NumberField(-10:10, default = -1)))|
	|2|$(Child(:BA, NumberField(-10:10, default = -1)))|$(Child(:BB, NumberField(-10:10, default = 1)))|
	"""
end

# ╔═╡ 961c0bfa-9e2f-45aa-bd34-5cae074a0d0c
md"""
While I've plotted both lines in the minimization expression, x can only ever acheive the minimum value between the two of these.  The maximization operator needs to select the value along the x axis that produces the highest value which is the minimum value between the two available lines.  By examining different reward matrices, we can realize that this point can only occur at the following three cases:

- ``x = 0``
- ``x = 1``
- Intersection of the two lines

So apparently we can solve the problem exactly by checking three values of ``x``, two of which are trivial.  The third value requires calculating the intersection point as follows:

```math
\begin{flalign}
x (r_{12} - r_{22}) + r_{22} &= x(r_{11} - r_{21}) + r_{21} \\
x(r_{12} - r_{22} - r_{11} + r_{21}) &= r_{21} - r_{22} \\
x &= \frac{r_{21} - r_{22}}{r_{12} - r_{22} - r_{11} + r_{21}}
\end{flalign}
```

Finally, we can compare the reward value at each of these three points:

```math
\begin{flalign}
U_x^1 &=  \min \left ( r_{22}, \; r_{21} \right ) \tag{x = 0 case} \\
U_x^2 &=  \min \left ( r_{12}, \; r_{11} \right ) \tag{x = 1 case}  \\
U_x^3 &=  \frac{(r_{21} - r_{22})(r_{12} - r_{22})}{r_{12} - r_{22} - r_{11} + r_{21}} + r_{22}  \tag{mixed case}\\
\end{flalign}
```

In the mixed case, ``x = \frac{r_{21} - r_{22}}{r_{12} - r_{22} - r_{11} + r_{21}}``.  The minimax solution itself will be whichever of these cases has the largest value.  The reward for ``Y`` will always be the negative of this, but the actual policy needs to be computed in a similar process using (4.11) instead of (4.10).

Below one can see a plot of the constraint line as well as the solution point.
"""

# ╔═╡ 84aaf3ca-be83-4fa1-ad3b-f565c259bb00
@bind rewardmatrix3 PlutoUI.combine() do Child
	md"""
	``\mathcal{R}`` = 

	
	|Action|||
	|---|---|---|
	||1|2|
	|1|$(Child(:AA, NumberField(-10:10, default = 1)))|$(Child(:AB, NumberField(-10:10, default = -1)))|
	|2|$(Child(:BA, NumberField(-10:10, default = -1)))|$(Child(:BB, NumberField(-10:10, default = 1)))|
	"""
end

# ╔═╡ 92ef9f37-8a63-40f9-a70c-786f9794a355
function reward(x, y, r)
	x*y*(r.AA - r.BA - r.AB + r.BB) + y*(r.BA - r.BB) + x*(r.AB - r.BB) + r.BB
end

# ╔═╡ e8fb0a96-e602-491d-9464-a8dc9b7e5f63
function plot_y_reward(r; xvals = LinRange(0, 1, 10))
	yvals = LinRange(0, 1, 1000)
	trs = 
	[begin
		out = [reward(x, y, r) for y in yvals]
		scatter(x = yvals, y = out, name = "x = $(round(x; sigdigits = 2))")
	end
	for x in xvals]
	plot(trs, Layout(xaxis_title = "y", yaxis_title = "reward for X", title = "Rewards for Y Against Fixed X"))
end

# ╔═╡ 96c662b7-007a-4a7e-b826-4d86bb7f9ca8
plot_y_reward(rewardmatrix2)

# ╔═╡ c6f0f327-2ddf-497b-996d-cc8f296c8fde
#note minimax definition

#look at two player case with 2 actions and show what equations actually look like

#in 2 player 2 action case, can we see the equations directly from the definition or do we need to write down the constraints?

#then show the general technique and claim about linear programming

#then show actual linear programming and how to convert problem into that

# ╔═╡ 5f455edf-7cb8-4199-9f71-c31fd1b793f2
md"""
### Minimax Solution via Linear Programming

According to the book, we can solve the general minimax problem for a two player game and arbitrarily many actions via linear programming:

```math
\begin{alignat*}{3}
&\text{minimize} \quad &&U_j^* &&\tag{4.12} \\
&\text{subject to}\quad &&\sum_{a_i \in A_i} \mathcal{R}_j(a_i, a_j) x_{a_i} \leq U_j^* \quad &&\forall a_j \in A_j \tag{4.13} \\
& &&x_{a_i} \geq 0 &&\forall a_i \in A_i \tag{4.14} \\
& &&\sum_{a_i \in A_i} x_{a_i} = 1 &&\tag{4.15}
\end{alignat*}
```
"""

# ╔═╡ d0571165-6b22-43b6-a023-f14fb5575c68
md"""
### General Linear Programming

"linear programs" refer to optimization problems in which the objective function and all the constraints are linear in the variables being optimized.  We can write a linear program in the following form:

```math
\begin{alignat*}{2}
&\text{Find a vector} &&\mathbf{x} \\
&\text{That maximizes} \quad &&\mathbf{c}^\top \mathbf{x} \\
&\text{Subject to} &&A \mathbf{x} \leq \mathbf{b} \\
&\text{and} &&\mathbf{x} \geq 0
\end{alignat*}
```

Equivalently, we can eschew vector notion by writing the following: 

```math
\begin{alignat*}{3}
&\text{Find a set of variables} \quad &&\{x_1, x_2, \dots, x_n\} \\
&\text{That maximizes} &&\sum_{i = 1}^n c_i \times x_i \\
&\text{Subject to} &&\sum_{i = 1}^n a_{ji} x_i \leq b_j &&\forall j \\
&\text{and} &&x_i \geq 0 &&\forall i
\end{alignat*}
```

The number of constraints depends upon the number of ``j``s there are.  Imagine ``n = 2``.  The objective function can be visualized as a plane in 3D space.  The constraints will define some boundary in the XY plane over which we seek to find the maximum.  The desired point will always be somewhere on the boundary itself due to the linear nature of the problem.  If the constraints are few, it may just be a matter of checking some cases, but in general we need to apply optimization techniques that intelligently test points on the boundary and hone in on the optimum.  In higher dimensional cases, this process can become difficult and inefficient.  In a low dimensional problem, we may not need to use any such technique as a direct solution could be possible.
"""

# ╔═╡ 500ddfe9-4f4c-4411-8ca2-ce60f058a125
md"""
### Converting Minimax to Linear Program

Comparing this formulation to (4.12 - 4.15) makes it clear that the minimax solution is not in the traditional form of linear programming.  Note that there is not even an equation for the objective function.  In order to apply linear programming, we must perform a change of variables on the above equations.


Consider the following new set of variables: ``v_{a_i} = \frac{x_{a_i}}{U_j ^*}``.  Also note that ``x_{a_i} \geq 0 \: \forall a_i \in A_i`` so what does it mean in this case to minimize ``U_j^*``?  Well let's consider the sum of our new variables: ``\sum_{a_i \in A_i} v_{a_i} = \frac{\sum_{a_i \in A_i} x_{a_i}}{U_j*} = \frac{1}{U_j^*}``.  Let's say we know ahead of time that ``U_j^* \geq 0``.  Then minimizing it would be equivalent to maximizing the sum of the new variables.  Similarly, if ``U_j^* \leq 0``, then minimizing it would be equivalent to maximizing he sum of the new variables.  So we have an objective function which is just the sum of all the new variables weighted by 1.

```math
\begin{alignat*}{3}
&\text{Find a set of variables } \quad && \{v_{a_1}, v_{a_2}, \dots \} && \forall a_i \in A_i\\
&\text{maximize} &&\sum_{a_i \in A_i} v_{a_i} \\
&\text{subject to}\quad &&\sum_{a_i \in A_i} \mathcal{R}_j(a_i, a_j) v_{a_i} \leq 1 \quad &&\forall a_j \in A_j \\
& &&v_{a_i} \geq 0 &&\forall a_i \in A_i \\
\end{alignat*}
```

The above program assumes that the game value (minimax reward for player ``i``) is positive.  We can ensure this by considering a modified game which subtracts the minimum value in the reward matrix to produce a new reward matrix whose minimum value is 0.  The resulting policies for the new game will be the same, but the values will all be positive.  Then at the end we can recover the original game values by adding the constant back.
"""

# ╔═╡ 8b7fc1fe-2bf3-440d-993c-6560b1f53ec1
md"""
### Solving a Linear Program
"""

# ╔═╡ f2aecc45-5773-4ef5-bf61-99a868ac3548
@bind rewardmatrix4 PlutoUI.combine() do Child
	md"""
	``\mathcal{R}`` = 

	
	|Action|||
	|---|---|---|
	||1|2|
	|1|$(Child(:AA, NumberField(-10:10, default = 1)))|$(Child(:AB, NumberField(-10:10, default = -1)))|
	|2|$(Child(:BA, NumberField(-10:10, default = -1)))|$(Child(:BB, NumberField(-10:10, default = 1)))|
	"""
end

# ╔═╡ 4989b264-fbd5-42da-92a8-6edf266a87f8
const rock_paper_scissors_game_rewards = 
	[
		0 	-1 	1
		1 	0 	-1
		-1 	1 	0
	]

# ╔═╡ 61b80664-30bd-4f21-a69f-46a3f8804427
rand([-1, 1, 0], 100, 100)

# ╔═╡ e132c8d0-6bdd-40b7-b18c-9feb45ba0cee
md"""
## General Sum Games

### Nash Equilibrium

The *Nash equilibrium* solution concept applies the idea of mutual best response to general-sum games with two or more agents.

*In a general-sum game with n agents, a joint policy ``\pi = (\pi_1, \dots, \pi_n)`` is a* Nash equilibrium *if*

```math
\forall i, \pi_i^\prime: U_i(\pi_i^\prime, \pi_{-i}) \leq U_i(\pi) \tag{4.16}
```

In other words no agent can improve its expected return by changing its policy against all of the other fixed policies.
"""

# ╔═╡ fbc4f7f7-e8cf-474d-b8ee-1b08661c43aa
md"""
### Two-Player, Two-Action Case
"""

# ╔═╡ a679f649-a97b-42d1-b6e0-678909598471
@bind generalreward2 PlutoUI.combine() do Child
	m1 = md"""
	|X Rewards|||
	|---|---|---|
	||1|2|
	|1|$(Child(:xAA, NumberField(-10:10, default = -1)))|$(Child(:xAB, NumberField(-10:10, default = -5)))|
	|2|$(Child(:xBA, NumberField(-10:10, default = 0)))|$(Child(:xBB, NumberField(-10:10, default = -3)))|
	"""

	m2 = md"""
	|Y Rewards|||
	|---|---|---|
	||1|2|
	|1|$(Child(:yAA, NumberField(-10:10, default = -1)))|$(Child(:yAB, NumberField(-10:10, default = -0)))|
	|2|$(Child(:yBA, NumberField(-10:10, default = -5)))|$(Child(:yBB, NumberField(-10:10, default = -3)))|
	"""

	@htl("""
		 <div style = "display: flex;">
		 $m1
		 $m2
		 </div>
		 """)
end |> confirm

# ╔═╡ b8a39b43-52b0-4a39-8593-7f3b9a7daee6
@bind xy_tests PlutoUI.combine() do Child
	md"""
	x test value: $(Child(Slider(0:0.01:1; default = 0.5, show_value = true)))
	y test value: $(Child(Slider(0:0.01:1; default = 0.5, show_value = true)))
	"""
end

# ╔═╡ 321e7b12-bc86-48c6-bae1-db5e7ef667a0
md"""
Let's say we only have two agents and each one only has two available actions.  Our two reward matrices will have four values each and can be written as follows:

```math
\mathcal{R_x} = \left [ \begin{matrix} r_{11}^x \quad r_{12}^x \\ r_{21}^x \quad r_{22}^x \end{matrix} \right ]

\mathcal{R_y} = \left [ \begin{matrix} r_{11}^y \quad r_{12}^y \\ r_{21}^y \quad r_{22}^y \end{matrix} \right ]
```

Each player's policy can be represented with a single value between 0 and 1 that gives the probability of selecting action 1.  We can call these values ``x`` and ``y`` for each respective player.  The joint policy ``\pi``, can then be written as a function of ``x`` and ``y`` as ``\pi(x, y)``.

In order to analyze this game, we need to write down ``U_x(\pi)`` and ``U_y(\pi)`` which is the expected reward for ``X`` and ``Y`` as a function of the joint policy.  We can write each return as a function of ``x`` and ``y`` as well as the reward matrix values.

```math
\begin{flalign}
U_x(x, y) &= r_{11}^x x y + r_{12}^x x (1-y) + r_{21}^x (1-x) y + r_{22}^x (1-x)(1-y) \\
U_y(x, y) &= r_{11}^y x y + r_{12}^y x (1-y) + r_{21}^y (1-x) y + r_{22}^y (1-x)(1-y) \\
\end{flalign}
```
"""

# ╔═╡ 918b7950-70fe-4cb5-9e11-b8bf891a937b
md"""
Using this notation, what is the definition of a Nash equilibrium?  The policy space to consider is just a single value for each player, so we can simplify the expression to:

A joint policy ``(x, y)`` is a Nash equilibrium if

```math
\begin{flalign}
\forall x^\prime: U_x(x^\prime, y) &\leq U_x(x, y) \\
\forall y^\prime: U_y(x, y^\prime) &\leq U_y(x, y) \\

&\downarrow \\
\forall x^\prime: r_{11}^x x^\prime y + r_{12}^x x^\prime (1-y) + r_{21}^x (1-x^\prime) y + r_{22}^x (1-x^\prime)(1-y) &\leq r_{11}^x x y + r_{12}^x x (1-y) + r_{21}^x (1-x) y + r_{22}^x (1-x)(1-y) \\
\forall y^\prime: r_{11}^y x y^\prime + r_{12}^y x (1-y^\prime) + r_{21}^y (1-x) y^\prime + r_{22}^y (1-x)(1-y^\prime) &\leq r_{11}^y x y + r_{12}^y x (1-y) + r_{21}^y (1-x) y + r_{22}^y (1-x)(1-y)
\end{flalign}
```

Since the Nash equilibrium conditions suggest considering alternative values of ``x`` for ``U_x`` and ``y`` for ``U_y`` while keeping everything else fixed, let's rewrite each expression with terms factored as such:

```math
\begin{flalign}
U_x(x, y) &= x(r_{11}^x y + r_{12}^x (1-y) - r_{21}^x y - r_{22}^x (1-y)) + r_{21}^x y + r_{22}^x (1-y) \\
U_y(x, y) &= y(r_{11}^y x - r_{12}^y x + r_{21}^y (1-x) - r_{22}^y (1-x)) + r_{12}^y x + r_{22}^y (1-x) \\
\end{flalign}
```

Both expressions are lines with respect to the other fixed variable, so what would it mean for a particular value of ``(x, y)`` to be an argmax in both functions at the same time?  If a line has a non-zero slope, then the argmax must occur at 1 for a positive slope and 0 for a negative slope.  The only alternative is if the slope of the line is zero which depends on the opposing player.  So there might be a value from the opposing player that forces a slope of 0 meaning that the primary player is indifferent to actions and all of its values are equivalently argmax's.  If the primary player can force the same circumstance on the opponent, then both players can be in a mixed strategy equilibrium.  We can easily take the derivative of both functions to see the checks necessary to find all of the possible equilibria for a problem:

```math
\begin{flalign}
\frac{\partial U_x(x, y)}{\partial x} &= r_{11}^x y + r_{12}^x (1-y) - r_{21}^x y - r_{22}^x (1-y) \\
&= y(r_{11}^x - r_{12}^x - r_{21}^x + r_{22}^x) + r_{12}^x - r_{22}^x \\
\frac{\partial U_y(x, y)}{\partial y} &= r_{11}^y x - r_{12}^y x + r_{21}^y (1-x) - r_{22}^y (1-x) \\
&= x(r_{11}^y - r_{12}^y - r_{21}^y + r_{22}^y) + r_{21}^y - r_{22}^y \\
\end{flalign}
```
"""

# ╔═╡ 44e36257-2f59-4451-b559-a0fc692ccd3a
md"""
Let's consider the mixed case first and find under what conditions both players would be indifferent to action selection:

```math
\begin{flalign}
\frac{\partial U_x(x, y)}{\partial x} = 0 &\implies y(r_{11}^x - r_{12}^x - r_{21}^x + r_{22}^x) + r_{12}^x - r_{22}^x = 0  \implies y = \frac{r_{22}^x - r_{12}^x}{r_{11}^x - r_{12}^x - r_{21}^x + r_{22}^x}\\
\frac{\partial U_y(x, y)}{\partial y} = 0 &\implies x(r_{11}^y - r_{12}^y - r_{21}^y + r_{22}^y) + r_{21}^y - r_{22}^y = 0 \implies x = \frac{r_{22}^y - r_{21}^y}{r_{11}^y - r_{12}^y - r_{21}^y + r_{22}^y} \\
\end{flalign}
```

If both of these values are between 0 and 1, then it exists as a mixed Nash equilibrium.  The remaining cases to check are the four pure strategies:

---

```math
(x = 0, y = 0): \quad \begin{flalign}
\left. \frac{\partial U_x(x, y)}{\partial x} \right\vert_{y = 0} < 0 &\implies r_{12}^x < r_{22}^x \\
&\text{ and } \\
\left. \frac{\partial U_y(x, y)}{\partial y} \right\vert_{x = 0} < 0 &\implies r_{21}^y < r_{22}^y \\
\end{flalign}
```
---

```math
(x = 0, y = 1): \quad \begin{flalign}

\left. \frac{\partial U_x(x, y)}{\partial x} \right\vert_{y = 1} < 0 &\implies r_{11}^x < r_{21}^x \\
&\text{ and } \\
\left. \frac{\partial U_y(x, y)}{\partial y} \right\vert_{x = 0} > 0 &\implies r_{21}^y > r_{22}^y \\

\end{flalign}
```
---
```math
(x = 1, y = 0): \quad \begin{flalign}

\left. \frac{\partial U_x(x, y)}{\partial x} \right\vert_{y = 0} > 0 &\implies r_{12}^x > r_{22}^x \\
&\text{ and } \\
\left. \frac{\partial U_y(x, y)}{\partial y} \right\vert_{x = 1} < 0 &\implies r_{11}^y < r_{12}^y \\

\end{flalign}
```
---
```math
(x = 1, y = 1): \quad \begin{flalign}

\left. \frac{\partial U_x(x, y)}{\partial x} \right\vert_{y = 1} > 0 &\implies r_{11}^x > r_{21}^x \\
&\text{ and } \\
\left. \frac{\partial U_y(x, y)}{\partial y} \right\vert_{x = 1} > 0 &\implies r_{11}^y > r_{12}^y \\

\end{flalign}
```
---
"""

# ╔═╡ b0213f40-b353-49cf-b355-45a29bfd151e
function check_mixed_nash(r)
	y = (r.xBB - r.xAB) / (r.xAA - r.xAB - r.xBA + r.xBB)
	x = (r.yBB - r.yBA) / (r.yAA - r.yAB - r.yBA + r.yBB)

	(x = x, y = y)
end

# ╔═╡ 934fceb4-35ef-4db1-a379-9e2d6eb67780
function check_pure_nash(r)
	out = BitMatrix(undef, 2, 2)
	if r.xAB <= r.xBB
		if r.xAA <= r.xBA
			if r.yBA <= r.yBB
				out[1, 1] = true
			else
				out[1, 2] = true
			end
		else
			if (r.yAA >= r.yAB)
				out[1, 1] = true
			end
			if (r.yBA <= r.yBB)
				out[2, 2] = true
			end
		end
	else
		if r.xAA <= r.xBA
			if r.yBA >= r.yBB
				out[1, 2] = true
			end
			if r.yAA <= r.yAB
				out[2, 1] = true
			end
		else
			if r.yAA >= r.yAB
				out[2, 2] = true
			else
				out[2, 1] = true
			end
		end
	end
	return out
end

# ╔═╡ e0a8af0a-0d12-46a8-b0f6-c713d8929320
function check_nash(r)
	mixed_nash = check_mixed_nash(r)
	pure_nash = check_pure_nash(r)

	mixed_nash.x < 0 && return (mixed_nash = NamedTuple(), pure_nash = pure_nash)
	mixed_nash.x > 1 && return (mixed_nash = NamedTuple(), pure_nash = pure_nash)
	mixed_nash.y < 0 && return (mixed_nash = NamedTuple(), pure_nash = pure_nash)
	mixed_nash.y > 1 && return (mixed_nash = NamedTuple(), pure_nash = pure_nash)

	(mixed_nash = mixed_nash, pure_nash = pure_nash)
end

# ╔═╡ b9699472-7f3d-4226-b60f-a3e1fb583ca8
md"""
If we choose an order with which to perform these checks, we can rule out certain equilibria early.  There are four distinct inequality conditions, so let's start with the first one:


- if ``r_{12}^x < r_{22}^x`` 
  -  ``(x = 1, y = 0)`` is ruled out
  - if ``r_{11}^x < r_{21}^x``
    - ``(x = 1, y = 1)`` is ruled out
    - if ``r_{21}^y < r_{22}^y``
      - ``(x = 0, y = 1)`` is ruled out
      - 3 of 4 checks required
      - ``(x = 0, y = 0)`` is the unique pure equilibrium
    - else
      - 3 of 4 checks required
      - ``(x = 0, y = 0)`` is ruled out
      - ``(x = 0, y = 1)`` is the unique pure equilibrium   
  - else
      - ``(x = 0, y = 1)`` is ruled out
      - ``(x = 1, y = 1)`` and ``(x = 0, y = 0)`` could be simultaneous equilibria
      - 4 checks required
      - if ``r_{11}^y > r_{12}^y``
        - ``(x = 1, y = 1)`` is an equilibrium
      - if ``r_{21}^y < r_{22}^y``
        - ``(x = 0, y = 0)`` is an equilibrium
- else
  - ``(x = 0, y = 0)`` is ruled out
  - if ``r_{11}^x < r_{21}^x``
    - ``(x = 1, y = 1)`` is ruled out
    - ``(x = 0, y = 1)`` and ``(x = 1, y = 0)`` could be simultaneous equilibria
    - 4 checks required
    - if ``r_{21}^y > r_{22}^y``
      - ``(x = 0, y = 1)`` is an equilibrium
    - if ``r_{11}^y < r_{12}^y``
      - ``(x = 1, y = 0)`` is an equilibrium
  - else
    - ``(x = 0, y = 1)`` is ruled out
    - if ``r_{11}^y > r_{12}^y``
      - ``(x = 1, y = 0)`` is ruled out
      - 3 of 4 checks required
      - ``(x = 1, y = 1)`` is the unique pure equilibrium
    - else
      - ``(x = 1, y = 1)`` is ruled out
      - 3 of 4 checks required
      - ``(x = 1, y = 0)`` is the unique pure equibrium

So out of the six branching cases, two of them require all four checks to determine all possible pure equilibria.  Those are also the two cases where two equilibria are possible.  The other 4 cases only have one solution and require only 3 of the 4 total checks.
"""

# ╔═╡ 97397f02-e87d-429a-a7b3-3ac961f22f54
@bind generalreward1 PlutoUI.combine() do Child
	m1 = md"""
	|X Rewards|||
	|---|---|---|
	||1|2|
	|1|$(Child(:xAA, NumberField(-10:10, default = -1)))|$(Child(:xAB, NumberField(-10:10, default = -5)))|
	|2|$(Child(:xBA, NumberField(-10:10, default = 0)))|$(Child(:xBB, NumberField(-10:10, default = -3)))|
	"""

	m2 = md"""
	|Y Rewards|||
	|---|---|---|
	||1|2|
	|1|$(Child(:yAA, NumberField(-10:10, default = -1)))|$(Child(:yAB, NumberField(-10:10, default = -0)))|
	|2|$(Child(:yBA, NumberField(-10:10, default = -5)))|$(Child(:yBB, NumberField(-10:10, default = -3)))|
	"""

	@htl("""
		 <div style = "display: flex;">
		 $m1
		 $m2
		 </div>
		 """)
end |> confirm

# ╔═╡ 59a47bf3-a80d-4936-b079-40ffed47c98c
0.22222222*7 + 0.222222222*2 + 0.4444444444*6

# ╔═╡ c932baa2-44e8-4109-97f9-c6a72b10a6e5
nashes = check_nash(generalreward1)

# ╔═╡ e3c29301-44d2-4925-8ac8-8b2391106cdc
check_mixed_nash(generalreward1)

# ╔═╡ c27fab3c-c29d-40c4-be81-d2d60c57544e
function reward_xy(x, y, r)
	u_x = x*y*(r.xAA - r.xBA - r.xAB + r.xBB) + y*(r.xBA - r.xBB) + x*(r.xAB - r.xBB) + r.xBB
	u_y = x*y*(r.yAA - r.yBA - r.yAB + r.yBB) + y*(r.yBA - r.yBB) + x*(r.yAB - r.yBB) + r.yBB
	(u_x, u_y)
end

# ╔═╡ 3a751006-2c04-4236-980b-10b3299f1b84
function plot_best_response_xy(reward, xtest, ytest; npoints = 400)
	xvals = LinRange(0, 1, npoints)
	yvals = LinRange(0, 1, npoints)
	x_output = zeros(npoints, npoints)
	y_output = zeros(npoints, npoints)
	for i in 1:npoints
		for j in 1:npoints
			(u_x, u_y) = reward_xy(xvals[i], yvals[j], reward)
			x_output[j, i] = u_x
			y_output[j, i] = u_y
		end
	end

	x_response = [reward_xy(x, ytest, reward)[1] for x in xvals]
	y_response = [reward_xy(xtest, y, reward)[2] for y in yvals]

	xrange = extrema(x_output)
	yrange = extrema(y_output)

	# y_maxind = [argmax(y_output[:, i]) for i in 1:npoints]
	# y_maxvals = [yvals[y_maxind[i]] for i in 1:npoints]
	# x_maxrewards = [x_output[y_maxind[i], i] for i in 1:npoints]

	# x_maxind = [argmax(x_output[i, :]) for i in 1:npoints]
	# x_maxvals = [xvals[x_maxind[i]] for i in 1:npoints]
	# y_maxrewards = [y_output[i, x_maxind[i]] for i in 1:npoints]

	tr_x_value = heatmap(x = xvals, y = yvals, z = x_output, colorscale = "rb")
	# tr_y_maxval = scatter(x = xvals, y = y_maxvals, line_color = "yellow", line_width = 4, name = "Best Possible Value")
	tr_y_value = heatmap(x = xvals, y = yvals, z = y_output, colorscale = "rb")
	# tr_x_maxval = scatter(x = x_maxvals, y = yvals, line_color = "yellow", line_width = 4, name = "Best Possible Value")

	tr_x_response = scatter(x = xvals, y = fill(ytest, npoints))
	tr_y_response = scatter(x = fill(xtest, npoints), y = yvals)

	p1 = plot([tr_x_value, tr_x_response], Layout(xaxis_title = L"x_A", yaxis_title = L"y_A", title = "X Reward"))
	p2 = plot([tr_y_value, tr_y_response], Layout(xaxis_title = L"x_A", yaxis_title = L"y_A", title = "Y Reward"))
	tr_x_reward = scatter(x = xvals, y = x_response, name = "Player X")
	tr_y_reward = scatter(x = yvals, y = y_response, name = "Player Y")
	p3 = plot(tr_x_reward, Layout(yaxis_range = xrange, xaxis_title = L"x_A", yaxis_title = "X Reward"))
	p4 = plot(tr_y_reward, Layout(yaxis_range = yrange, xaxis_title = L"y_A", yaxis_title = "Y Reward"))
	# tr2 = scatter(x = xvals, y = minvals, line_color = "yellow", line_width = 4, name = "Best Possible Value")
	# p1 = plot([tr, tr2], Layout(xaxis_title = L"x_A", yaxis_title = L"y_A"))
	# p2 = plot(scatter(x = xvals, y = minrewards))
	@htl("""
	<div style = "display:flex;">
	$p1
	$p2
	</div>
	<div style = "display:flex;">
	$p3
	$p4
	</div>
	""")
end

# ╔═╡ 83be4347-ba46-45b5-9736-a36895c7a1a7
plot_best_response_xy(generalreward2, xy_tests...; npoints = 200)

# ╔═╡ 71fe7db2-448a-4921-8272-f34d887c3668
reward_xy(nashes.mixed_nash..., generalreward1)

# ╔═╡ 38543174-4a36-4718-b646-848602e15429
function plot_reward_xy(reward; npoints = 400)
	xvals = LinRange(0, 1, npoints)
	yvals = LinRange(0, 1, npoints)
	x_output = zeros(npoints, npoints)
	y_output = zeros(npoints, npoints)
	for i in 1:npoints
		for j in 1:npoints
			(u_x, u_y) = reward_xy(xvals[i], yvals[j], reward)
			x_output[j, i] = u_x
			y_output[j, i] = u_y
		end
	end

	y_maxind = [argmax(y_output[:, i]) for i in 1:npoints]
	y_maxvals = [yvals[y_maxind[i]] for i in 1:npoints]
	x_maxrewards = [x_output[y_maxind[i], i] for i in 1:npoints]

	x_maxind = [argmax(x_output[i, :]) for i in 1:npoints]
	x_maxvals = [xvals[x_maxind[i]] for i in 1:npoints]
	y_maxrewards = [y_output[i, x_maxind[i]] for i in 1:npoints]

	tr_x_value = heatmap(x = xvals, y = yvals, z = x_output, colorscale = "rb")
	tr_y_maxval = scatter(x = xvals, y = y_maxvals, line_color = "yellow", line_width = 4, name = "Best Possible Value")
	tr_y_value = heatmap(x = xvals, y = yvals, z = y_output, colorscale = "rb")
	tr_x_maxval = scatter(x = x_maxvals, y = yvals, line_color = "yellow", line_width = 4, name = "Best Possible Value")

	p1 = plot([tr_x_value, tr_x_maxval], Layout(xaxis_title = L"x_A", yaxis_title = L"y_A", title = "X Reward"))
	p2 = plot([tr_y_value, tr_y_maxval], Layout(xaxis_title = L"x_A", yaxis_title = L"y_A", title = "Y Reward"))
	tr_x_reward = scatter(x = xvals, y = x_maxrewards, name = "Player X")
	tr_y_reward = scatter(x = yvals, y = y_maxrewards, name = "Player Y")
	p3 = plot([tr_x_reward, tr_y_reward])
	# tr2 = scatter(x = xvals, y = minvals, line_color = "yellow", line_width = 4, name = "Best Possible Value")
	# p1 = plot([tr, tr2], Layout(xaxis_title = L"x_A", yaxis_title = L"y_A"))
	# p2 = plot(scatter(x = xvals, y = minrewards))
	@htl("""
	<div style = "display:flex;">
	$p1
	$p2
	</div>
	$p3
	""")
end

# ╔═╡ 9c03ba14-9e59-4024-a227-3b65d694eaea
plot_reward_xy(generalreward1)

# ╔═╡ f9451804-e106-4e14-8985-2d2a82abf241
md"""
Let's say we wish to find the minimax solution for ``X``, that is the best that ``X`` can do in terms of expected reward assuming ``Y`` does its best to thwart that.  Thus we seek to find a solution to ``\pi_X`` that maximizes the expected reward ``U_X`` under the following constraints.

```math
\begin{flalign}
x_A + x_B &= 1 \\
x_A, x_B &\geq 0 \\
x_A \times r_{AA} + x_B \times r_{BA} &\geq U_X \\
x_A \times r_{AB} + x_B \times r_{BB} &\geq U_X \\
\end{flalign}
```

Notice that in the case of two actions, the probabilities obey a simpler relationship under the constraints such that we only need to keep track of one variable ``x`` to represent the probability of action ``A``.  The probability of action ``B`` will simply be ``1-x``.  Using this change of variables we can rewrite the constraints as follows:

```math
\begin{flalign}
0 \leq x &\leq 1 \\
x \times r_{AA} + (1-x) \times r_{BA} &\geq U_X \\
x \times (r_{AA} - r_{BA}) + r_{BA} &\geq U_X \\
x \times r_{AB} + (1-x) \times r_{BB} &\geq U_X \\
x \times (r_{AB} - r_{BB}) + r_{BB} &\geq U_X \\
\end{flalign}
```
"""

# ╔═╡ 64a76ef6-a784-4997-8132-497c6659d0c6
md"""
We can repeat the same procedure for ``Y``

Let's say we wish to find the minimax solution for ``X``, that is the best that ``X`` can do in terms of expected reward assuming ``Y`` does its best to thwart that.  Thus we seek to find a solution to ``\pi_X`` that maximizes the expected reward ``U_X`` under the following constraints.

```math
\begin{flalign}
0 \leq y &\leq 1 \\
y \times -r_{AA} + (1-y) \times -r_{AB} &\geq U_Y \\
y \times (r_{AB} - r_{AA}) - r_{AB} &\geq U_Y \\
y \times -r_{BA} + (1-y) \times -r_{BB} &\geq U_Y \\
y \times (r_{BB} - r_{BA}) - r_{BB} &\geq U_Y \\
\end{flalign}
```
"""

# ╔═╡ d9a9f2fb-0841-4933-ab71-4a5564d5794c
md"""
## Reward Visualization

For any game of this structure, the reward for player ``X`` is a function of the strategy for both ``X`` and ``Y``.  We can write this function as follows:

``U_X(\pi_X, \pi_Y) = y_A (x_A r_{AA} + x_B r_{BA}) + y_B (x_A r_{AB} + x_B r_{BB})``

Using the above notation for each agent which represents its policy by a single value which is the probability of action A, we can simplify the formula to a function of two variables:

```math
\begin{flalign}
U_X(x, y) &= y(x \times r_{AA} + (1-x)r_{BA}) + (1-y)(x \times r_{AB} + (1-x) r_{BB}) \\
&= xy(r_{AA} - r_{BA} - r_{AB} + r_{BB}) + y(r_{BA} - r_{BB}) + x(r_{AB} - r_{BB}) + r_{BB}
\end{flalign}
```

The ``X`` player would like to maximize ``U_X`` by selecting an appropriate value of ``x``, but ``X`` is not free to control the behavior of ``Y`` and thus ``y``.  In a zero-sum game, ``U_Y \doteq -U_X`` so for any ``x`` we should expect ``Y`` to attempt to minimize ``U_X`` since that is equivalent to maximizing ``U_Y`` (we assume each player attempts to maximize the corresponding reward).  

To answer this question, we can ask for every ``x`` we consider, what is the value of ``y`` that minimizes ``U_X``.  Such a value for ``y`` would be called the "best response" strategy.  However, ``U_X`` is linear in ``y``, so this value will either be indifferent to ``y`` or occur at the endpoints of ``y = 0`` or ``y = 1``.

```math
\begin{flalign}
U_X(x, 0) &= x(r_{AB} - r_{BB}) + r_{BB} \\
U_X(x, 1) &= x(r_{AA} - r_{BA} - r_{AB} + r_{BB}) + (r_{BA} - r_{BB}) + x(r_{AB} - r_{BB}) + r_{BB} \\
&= x(r_{AA} - r_{BA}) + r_{BA}
\end{flalign}
```

We do not know ahead of time which constraint is smaller, but we know that ``Y`` will not allow ``X`` to exceed this reward under any circumstance.  Furthermore, we know that ``x \ge 0`` and ``x \leq 1`` due to its representation of a probability distribution.
"""

# ╔═╡ 107740d9-803a-434d-a7fe-e94abb06c5ed
md"""
These constraints together form a linear optimization problem for ``U_X(x)`` where we no longer need to refer to any particular value of ``y`` since the endpoints are sufficient to fully specify the problem due to the linearity.  Such problems are often called "linear programs" and take the form:

```math
\begin{matrix}
&\text{Find a vector} &\mathbf{x} \\
&\text{That maximizes} &\mathbf{c}^\top \mathbf{x} \\
&\text{Subject to} &A \mathbf{x} \leq \mathbf{b} \\
&\text{and} &\mathbf{x} \geq 0
\end{matrix}
```

To convert our problem into this form, we need to change a few variables.  Consider ``u = x / U_X``

```math
\begin{flalign}
U_X &\leq x(r_{AB} - r_{BB}) + r_{BB} \implies \frac{x}{U_X} (r_{AB} - r_{BB}) + \\
U_X &\leq x(r_{AA} - r_{BA}) + r_{BA} \\
x \leq 1 \\
x \geq 0 \\
\end{flalign}
```
"""

# ╔═╡ e37de208-1635-4d25-8dca-ab07eb8916a3
function plot_reward(rewardmatrix; npoints = 400)
	xvals = LinRange(0, 1, npoints)
	yvals = LinRange(0, 1, npoints)
	output = zeros(npoints, npoints)
	for i in 1:npoints
		for j in 1:npoints
			output[npoints - j + 1, i] = reward(xvals[i], yvals[j], rewardmatrix)
		end
	end

	minvals = [yvals[argmin(output[:, i])] for i in 1:npoints]
	minrewards = [minimum(output[:, i]) for i in 1:npoints]

	tr = heatmap(x = xvals, y = yvals, z = output, colorscale = "rb")
	tr2 = scatter(x = xvals, y = minvals, line_color = "yellow", line_width = 4, name = "Best Possible Value")
	p1 = plot([tr, tr2], Layout(xaxis_title = L"x_A", yaxis_title = L"y_A"))
	p2 = plot(scatter(x = xvals, y = minrewards))
	@htl("""
	<div style = "display:flex;">
	$p1
	$p2
	</div>
	""")
end

# ╔═╡ 2f912c93-87d9-40d7-bc59-2c56590238e1
plot_reward(rewardmatrix2)

# ╔═╡ 944447ae-5597-4998-ad34-23e8510b323f
md"""
## Constraint Visualization

If we take the two inequality constraints that involve the reward function, we can see that ``U_X`` will be maximized at the value of ``x`` between 0 and 1 where the minimum value from each curve is the largest.  This maximum point will only ever occur at the extreme ends of ``x=0`` and ``x = 1`` or where the two curves intersect.  Therefore, if we check these three points then simply select the one with the highest value we can always find the answer.  Moreover, if the lines do not intersect at all then we only need to check the endpoints.

If we check the values of the curve at both endpoints, we can see whether or not they intersect because if they do intersect, the maximum value will change from one curve to the other across the endpoints.

```math
\begin{flalign}
v_1 &= r_{AA} \\
v_2 &= r_{AB} \\ 
v_3 &= r_{BA} \\
v_4 &= r_{BB} \\
\end{flalign}
```

What about where the curves intersect?

```math
\begin{flalign}
x (r_{AA} - r_{BA}) + r_{BA} &= x(r_{AB} - r_{BB}) + r_{BB} \\
x(r_{AA} - r_{BA} - r_{AB} + r_{BB}) &= r_{BB} - r_{BA} \\
x = \frac{r_{BB} - r_{BA}}{r_{AA} - r_{BA} - r_{AB} + r_{BB}}
\end{flalign}
```

```math
\begin{flalign}
y (r_{AB} - r_{AA}) - r_{AB} &= y(r_{BB} - r_{BA}) - r_{BB} \\
y(r_{AB} - r_{AA} - r_{BB} + r_{BA}) &= r_{AB} - r_{BB} \\
y = \frac{r_{AB} - r_{BB}}{r_{AB} - r_{AA} - r_{BB} + r_{BA}}
\end{flalign}
```

And the value at the intersection?
```math
\begin{flalign}
x &= \frac{r_{BB} - r_{BA}}{r_{AA} - r_{BA} - r_{AB} + r_{BB}} \\
U_X &= x(r_{AA} - r_{BA}) + r_{BA} \\
&= \frac{r_{BB} - r_{BA}}{r_{AA} - r_{BA} - r_{AB} + r_{BB}} (r_{AA} - r_{BA}) + r_{BA}
\end{flalign}
```

"""

# ╔═╡ ad7a0f40-fc91-4b8c-83ca-871efb9f2c06
md"""
Notice that ``U_X = -U_Y`` for all solutions regardless of the reward matrix and the solution value is either at 0, 1 or the point where the two constraints intersect.
"""

# ╔═╡ f6aea589-b844-46d5-888b-e4236e71d4d2
function get_x_constraints(r)
	f1(x) = x*(r.AA - r.BA) + r.BA
	f2(x) = x*(r.AB - r.BB) + r.BB

	return f1, f2
end

# ╔═╡ efde1b80-0972-4652-803c-6f7ce278b43c
function plot_x_constraints(r)
	xvals = LinRange(0, 1, 1000)
	(f1, f2) = get_x_constraints(r)
	tr1 = scatter(x = xvals, y = f1.(xvals))
	tr2 = scatter(x = xvals, y = f2.(xvals))
	plot([tr1, tr2], Layout(xaxis_title = "x", yaxis_title = "reward", title = "Minimum Reward Lines for X"))
end

# ╔═╡ 77ce6f8a-dc52-4e3c-b599-cb11f4e48e99
plot_x_constraints(rewardmatrix)

# ╔═╡ ec46b9c9-0ddc-49d8-9bb9-6571ab9f09c4
plot_x_constraints(rewardmatrix3)

# ╔═╡ 4ac2c4c6-f6ff-4126-85ce-904a746a694f
get_x_constraints(rewardmatrix)

# ╔═╡ c636112d-b788-46e4-a15f-9a7e347e5516
function get_y_constraints(r)
	f1(x) = x*(-r.AA + r.AB) - r.AB
	f2(x) = x*(-r.BA + r.BB) - r.BB

	return f1, f2
end

# ╔═╡ ba6f2a22-df07-4677-9696-d53fa7fae133
function get_x_intersect(r)
	(r.BB - r.BA) / (r.AA - r.BA - r.AB + r.BB)
end

# ╔═╡ aef23141-e1ef-416f-91da-965ca1f69397
function get_y_intersect(r)
	(r.AB - r.BB) / (r.AB - r.AA - r.BB + r.BA)
end

# ╔═╡ dd2824e6-d74e-4625-b1b1-497a9bc3b662
function plot_constraints(rewardmatrix)
	f1, f2 = get_x_constraints(rewardmatrix)
	f3, f4 = get_y_constraints(rewardmatrix)
	xvals = LinRange(0., 1., 1_000)
	l1vals = f1.(xvals)
	l2vals = f2.(xvals)
	l3vals = f3.(xvals)
	l4vals = f4.(xvals)

	x1 = f1(0.)
	x2 = f2(0.)
	x3 = f1(1.)
	x4 = f2(1.)
	check1 = x1 > x2
	check2 = x3 > x4

	x0_value = min(x1, x2)
	x1_value = min(x3, x4)

	
	x_extr, u_extr = if x0_value > x1_value
			(0., x0_value)
		else
			(1., x1_value)
		end

	(x_sol, u_sol) = if !(check1 == check2)
		x_intr = get_x_intersect(rewardmatrix)
		u_intr = f1(x_intr)
		if u_intr > u_extr
			(x_intr, u_intr)
		else
			(x_extr, u_extr)
		end
	else
		(x_extr, u_extr)
	end

	y1 = f3(0.)
	y2 = f4(0.)
	y3 = f3(1.)
	y4 = f4(1.)
	check1 = y1 > y2
	check2 = y3 > y4

	y0_value = min(y1, y2)
	y1_value = min(y3, y4)

	
	y_extr, v_extr = if y0_value > y1_value
			(0., y0_value)
		else
			(1., y1_value)
		end

	(y_sol, v_sol) = if !(check1 == check2)
		y_intr = get_y_intersect(rewardmatrix)
		v_intr = f3(y_intr)
		if v_intr > v_extr
			(y_intr, v_intr)
		else
			(y_extr, v_extr)
		end
	else
		(y_extr, v_extr)
	end

	

	tr1 = scatter(x = xvals, y = min.(l1vals, l2vals), name = "X maximum constraint")
	# tr2 = scatter(x = xvals, y = l2vals, name = "X maximum constraint 2")
	tr3 = scatter(x = xvals, y = min.(l3vals, l4vals), name = "Y maximum constraint")
	# tr4 = scatter(x = xvals, y = l4vals, name = "Y maximum constraint 2")
	p1 = plot([tr1, tr3], Layout(xaxis_title = "Action A Probability", yaxis_title = "Expected Value", title = "Minimax Value of $(round(u_sol; sigdigits = 3)) Acheived at x = $(round(x_sol; sigdigits=3))", legend_orientation = "h", legend_y = -.2))

	tr3 = scatter(x = [x_sol], y = [u_sol], name = "Player X")
	tr4 = scatter(x = [y_sol], y = [v_sol], name = "Player Y")
	p2 = plot([tr3, tr4], Layout(xaxis_range = [-.1, 1.1], xaxis_title = "Action A Probability", yaxis_title = "Expected Value", title = "Minimax Solutions"))

	@htl("""
		 <div style = "display: flex;">
		 $p1

		 $p2
		 </div>
		 """)
	# [p1 p2]
end

# ╔═╡ b1750cc5-01d3-4fb3-925d-aaaae8fe0e67
plot_constraints(rewardmatrix3)

# ╔═╡ 429e4864-16c7-415a-b7eb-8fb09d36ab18
plot_constraints(rewardmatrix4)

# ╔═╡ b8aee310-20aa-4ef5-9cc9-32ff4d8d8035
#use this to solve the simple two player game and the rock paper scissors game and compare timing

# ╔═╡ b8f4e17b-72a3-401c-bef1-9d02e3a0c577
md"""
## Correlated Equilibrium

Nash equilibrium assumes that policies are probabilistically independent.  *Correlated equilibrium* generalizes Nash equilibrium by allowing for correlation between policies.  In the general definition of correlated equilibrium, each agent ``i``'s policy is additionally conditioned on the outcomes of a private random variable ``d_i`` for the agent, which are governed by a joint probability distribution over ``(d_1, \dots, d_n)`` that is commonly known by all agents.  Here, we will present a common version of correlated equilibrium for non-repeated normal form games, in which ``d_i`` corresponds to an action recommendation for agent ``i`` given by a joint policy ``\pi_c``.

### Definition

*In a general-sum normal-form game with n agents, let ``\pi_c(a)`` by a joint policy that assigns probabilities to joint actions ``a \in A``.  Then, ``\pi_c`` is a* correlated equilibrium *if for every agent ``i \in I`` and every action modifier ``\xi_i: A_i \rightarrow A_i``*:

```math
\sum_{a \in A}\pi_c(a)\mathcal{R}_i(\left < \xi_i(a_i), a_{-i} \right >) \leq \sum_{a \in A} \pi_c(a) \mathcal{R}_i(a) \tag{4.19}
```

(4.19) states that in a correlated equilibrium, in which every agent knows the probability distribution ``\pi_c(a)`` and its own recommmended action ``a_i`` (but not the recommended actions for other agents), no agent can unilaterally deviate from its recommended actions in order to increase its expected return.  Here, deviating from the recommended actions is represented by the action modifier ``\xi_i``.  In a central learning approach, an algorithm can train a single policy ``\pi_c`` directly over the join-action space and use this policy to dictate the actions of each agent.

It can be shown that the set of correlated equilibria contains the set of Nash equilibria.  In other words, Nash equilibrium is a special case of correlated equilibrium in which the joint policy ``\pi_c`` is factored into independent agent policies ``\pi_1, \dots \pi_n`` with ``\pi_c(a) = \prod_{i \in I} \pi_i(a_i)``.  In this case, since the agents' policies in a Nash equilibrium are independent, knowing one's own action ``a_i`` does not give any information about the action probabilities for the other agents ``j \neq i``. 
"""

# ╔═╡ 36a5c1bb-f26f-4eaf-b5df-7bfc01963911
md"""
### Two-Player, Two-Action Case

The joint policy ``\pi_c(a)`` must give a probability distribution over the joint-action space.  For two agents and two actions, this space has four actions: ``(1, 1), (1, 2), (2, 1), (2, 2)``.  We can, therefore, specify ``\pi_c`` with four values ``(\pi_{11}, \pi_{12}, \pi_{21}, \pi_{22})`` such that ``0 \leq \pi_{i, j} \leq 1 \;; \forall i \in \{1, 2\}, j \in \{1, 2\}`` and ``\sum_{i, j} \pi_{ij} = 1``.

The reward function for each player is a single value for each joint action, so for example the right hand side of (4.19) would become:

``\pi_{11} r_{11}^x + \pi_{12} r_{12}^x + \pi_{21} r_{21}^x + \pi_{22} r_{22}^x``

For the left hand side, we must consider the reward that each agent would earn under ``\pi_c`` by selecting an alternative individual action.  In the two-action case, these modifiers amount simply to switching an action choice from 1 to 2 or 2 to 1.  As such, there are two modifiers to consider:

``1 \rightarrow 2, 2 \rightarrow 2``

``1 \rightarrow 1, 2 \rightarrow 1``

Writing down the inequality condition for each yields:

```math
\begin{flalign}
\pi_{11}r_{21}^x + \pi_{12} r_{22}^x + \pi_{21} r_{21}^x + \pi_{22} r_{22}^x &\leq \pi_{11} r_{11}^x + \pi_{12} r_{12}^x + \pi_{21} r_{21}^x + \pi_{22} r_{22}^x \\
\pi_{11}r_{11}^x + \pi_{12} r_{12}^x + \pi_{21} r_{11}^x + \pi_{22} r_{12}^x &\leq \pi_{11} r_{11}^x + \pi_{12} r_{12}^x + \pi_{21} r_{21}^x + \pi_{22} r_{22}^x \\
\end{flalign}
```

The only difference between the sides of each respective equation above is the modification to the ``r_{ij}^x`` term for all of the action swaps that occur.  We can group terms to simplify the conditions:

```math
\begin{flalign}
\pi_{11}(r_{21}^x - r_{11}^x) &\leq \pi_{12} (r_{12}^x - r_{22}^x) \\
\pi_{21}(r_{11}^x - r_{21}^x) &\leq \pi_{22} (r_{22}^x - r_{12}^x) \\
\end{flalign}
```

Similarly, we can consider the situation for ``Y``:

```math
\begin{flalign}
\pi_{11}r_{12}^y + \pi_{12} r_{12}^y + \pi_{21} r_{22}^y + \pi_{22} r_{22}^y &\leq \pi_{11} r_{11}^y + \pi_{12} r_{12}^y + \pi_{21} r_{21}^y + \pi_{22} r_{22}^y \\
\pi_{11}r_{11}^y + \pi_{12} r_{11}^y + \pi_{21} r_{21}^y + \pi_{22} r_{21}^y &\leq \pi_{11} r_{11}^y + \pi_{12} r_{12}^y + \pi_{21} r_{21}^y + \pi_{22} r_{22}^y \\
&\downarrow \\
\pi_{11}(r_{12}^y - r_{11}^y) &\leq \pi_{21}(r_{21}^y - r_{22}^y) \\
\pi_{12} (r_{11}^y - r_{12}^y)  &\leq \pi_{22} (r_{22}^y - r_{21}^y)  \\
\end{flalign}
```

So now we have four inequality equations and four unknowns and should be able to find solutions.  To make progress towards a solution, consider writing these as equality conditions and worrying about the inequality cases later.
"""

# ╔═╡ fc41aa2b-83eb-4914-90c1-e1af40e94711
md"""
In order to simplify notation, consider the following four constants:

``X1 = r_{21}^x - r_{11}^x, \; X2 = r_{12}^x - r_{22}^x, \; Y1 = r_{12}^y - r_{11}^y, \; Y2 = r_{21}^y - r_{22}^y``

Then we can rewrite the inequalities as follows along with the constraint that together the policy forms a distribution:

```math
\begin{flalign}
\pi_{11}X1 &\leq \pi_{12} X2 \\
\pi_{21} X1 &\leq \pi_{22} X2 \\
\pi_{11}Y1 &\leq \pi_{21} Y2 \\
\pi_{12}Y1 &\leq \pi_{22} Y2 \\
\pi_{11} + \pi_{12} + \pi_{21} + \pi_{22} &= 1 \\
0 \leq \pi_{11} \leq 1, 0 \leq \pi_{12} &\leq 1, 0 \leq \pi_{21} \leq 1, 0 \leq \pi_{22} \leq 1\\
\end{flalign}
```

We can begin to analyze the possible solutions by considering cases which distinguish possible values.  For example, let's say that ``X2 < 0`` and ``X1 > 0``.  That means that the first condition could not be met unless both ``\pi_{11}=0`` and ``\pi_{12}=0``.  What if ``X2 > 0`` and ``X1 < 0``?  In that case the first condition would hold for any value of ``\pi_{11}`` and ``\pi_{12}`` meaning it adds nothing as a constraint.  If all the constants are positive values, then we can operate on the inequalities and get a range of solutions.
"""

# ╔═╡ a89ea983-2c60-468b-a7eb-6bcc82533fd8
md"""
Let's consider a new set of variables for the distribution which are unnormalized probabilities.  That is ``p_{11}, p_{12}, p_{21}, p_{22}`` such that ``\pi_{11} = \frac{p_{11}}{p_{11}+ p_{12}+ p_{21}+ p_{22}}, \dots`` and with the restriction that all values are positive.  All of the previous inequalities hold now with the new variables.

#### Case 1: All constants are positive

We can consider all the inequality constraints as equalities in order to get an expression for ``\pi_{11}``.  Then we can determine whether this is an upper or lower bound on the value and then derive the other values in the distribution.

```math
\begin{flalign}
\pi_{12} &\geq \pi_{11} \frac{X1}{X2} \tag{1}\\
\pi_{22} &\geq \pi_{12} \frac{Y1}{Y2} \tag{2}\\
\pi_{22} &\geq \pi_{21} \frac{X1}{X2} \tag{3}\\
\pi_{21} &\geq \pi_{11} \frac{Y1}{Y2} \tag{4}\\
\end{flalign}
```

We can put additional constraints on the solution from these inequalities because we know that every value of ``\pi`` must be less than 1.  That means that if ``\frac{X1}{X2} = 6``, for example, then ``\pi_{11} \leq \frac{1}{6}``.  Otherwise, one of the probabilities would be forced to exceed 1.  We can express these new constraints as follows:

```math
\begin{flalign}
\pi_{11} &\leq \frac{X2}{X1} \tag{1′}\\
\pi_{12} &\leq \frac{Y2}{Y1} \tag{2′}\\
\pi_{21} &\leq \frac{X2}{X1} \tag{3′}\\
\pi_{11} &\leq \frac{Y2}{Y1} \tag{4′}\\

\frac{Y2}{Y1} \geq \pi_{12} &\geq \pi_{11} \frac{X1}{X2}
\end{flalign}
```

We can also eliminate one of the variables by expressing it in terms of the other 3.  Let's take ``\pi_{11} = 1 - \pi_{12} - \pi_{21} - \pi_{22}``.

```math
\begin{flalign}
\pi_{12} &\geq (1 - \pi_{12} - \pi_{21} - \pi_{22}) \frac{X1}{X2} \tag{1}\\
\pi_{12}\left ( 1 + \frac{X1}{X2} \right ) &\geq (1 - \pi_{21} - \pi_{22}) \frac{X1}{X2} \\
\pi_{12} \frac{X2 +X1}{X2} &\geq (1 - \pi_{21} - \pi_{22}) \frac{X1}{X2} \\
\pi_{12} &\geq (1 - \pi_{21} - \pi_{22}) \frac{X1}{X2+X1} \\
\pi_{22} &\geq \pi_{12} \frac{Y1}{Y2} \tag{2}\\
\pi_{22} &\geq \pi_{21} \frac{X1}{X2} \tag{3}\\
\pi_{21} &\geq (1 - \pi_{12} - \pi_{21} - \pi_{22}) \frac{Y1}{Y2} \tag{4}\\
\pi_{21} \left ( 1 + \frac{Y1}{Y2} \right ) &\geq (1 - \pi_{12} - \pi_{22}) \frac{Y1}{Y2} \\
\pi_{21} \frac{Y1+Y2}{Y2} &\geq (1 - \pi_{12} - \pi_{22}) \frac{Y1}{Y2} \\
\pi_{21} &\geq (1 - \pi_{12} - \pi_{22}) \frac{Y1}{Y1+Y2} \\
\end{flalign}
```

Now we have four inequalities and three unknowns, so we should be able to place even more restrictions on the solution.
"""

# ╔═╡ e992fead-2863-4e4c-bb31-a8ed2b70b2e0
md"""
```math
\begin{flalign}
\pi_{11} + \pi_{12} + \pi_{21} + \pi_{22} &= 1 \tag{5} \\
\pi_{11} + \pi_{12} +\pi_{11} \frac{Y1}{Y2} + \pi_{22} &= 1 \tag{using (4)} \\
\pi_{11} + \pi_{12} +\pi_{11} \frac{Y1}{Y2} +\pi_{21} \frac{X1}{X2} &= 1 \tag{using (3)} \\
\pi_{11} + \pi_{12} +\pi_{11} \frac{Y1}{Y2} +\pi_{11} \frac{Y1}{Y2} \frac{X1}{X2} &= 1 \tag{using (4)} \\
\pi_{11} + \pi_{11} \frac{X1}{X2} +\pi_{11} \frac{Y1}{Y2} +\pi_{11} \frac{Y1}{Y2} \frac{X1}{X2} &= 1 \tag{using (1)} \\
\pi_{11}\left ( 1 + \frac{X1}{X2} + \frac{Y1}{Y2} + \frac{Y1}{Y2} \frac{X1}{X2} \right) &= 1 \\
\pi_{11} \frac{Y2 \times X2 + X1 \times Y2 + Y1 \times X2 + Y1 \times X1}{Y2 \times X2} &= 1 \\
\pi_{11} &= \frac{Y2 \times X2}{Y2 \times X2 + X1 \times Y2 + Y1 \times X2 + Y1 \times X1} \\
\pi_{12} &\geq \pi_{11} \frac{X1}{X2} = \frac{X1 \times Y2}{Y2 \times X2 + X1 \times Y2 + Y1 \times X2 + Y1 \times X1} \\
\pi_{21} &\geq \pi_{11} \frac{Y1}{Y2} = \frac{Y1 \times X2}{Y2 \times X2 + X1 \times Y2 + Y1 \times X2 + Y1 \times X1} \\
\pi_{22} &\geq \pi_{21} \frac{X1}{X2} = \frac{Y1 \times X1}{Y2 \times X2 + X1 \times Y2 + Y1 \times X2 + Y1 \times X1}
\end{flalign}
```

Notice that all four expressions share the same denominator which is a sum of the four distinct numerators.  We can therefore think of the numerators on their own as unnormalized probabilities which define the boundaries of the solution space with an upper bound on ``\pi_{11}`` and then lower bounds on the remaining three variables.

"""

# ╔═╡ c52f0d6d-d367-4a85-8f92-20baec85fce9
md"""

"""

# ╔═╡ 27b7bb79-81ba-4287-9230-f22f74f04f9a
function correlated_equilibrium(r)
	X1 = r.xAA - r.xBA
	X2 = r.xAB - r.xBB
	Y1 = r.yAA - r.yAB
	Y2 = r.yBA - r.yBB

	π11 = Y2 * X2 / (Y2*X2 - X1*Y2 - Y1*X2 + Y1*X1)
	π12 = -π11*X1 / X2
	π22 = -π12*Y1/Y2
	π21 = -π11*Y1 / Y2
	return (π11 = π11, π12 = π12, π21 = π21, π22 = π22)
end

# ╔═╡ 5a62ad34-cb96-4dc4-a9ef-4841126aea57
correlated_equilibrium(generalreward1)

# ╔═╡ 55abdda0-4dcc-4348-91fc-c3faeb3e83f2
md"""
Let's consider an alternative set of variables ``A, B, C, D`` such that ``\pi_{11} = \frac{A}{A+B+C+D}`` etc...  In other words, unnormalized probabilities as well as the fact that all are greater than 0.  The above inequalities do not change since multiplying each equation by a positive constant does not change the inequality condition.  Our new equations are thus:

```math
\begin{flalign}
A(r_{21}^x - r_{11}^x) &\leq B (r_{12}^x - r_{22}^x) \\ 
C(r_{11}^x - r_{21}^x) &\leq D (r_{22}^x - r_{12}^x) \\ 
\end{flalign}
```
"""

# ╔═╡ bbd07ba1-528d-4801-8dfe-9488ac60a11d
md"""

Note that ``\pi_{22} = 1 - \pi_{11} - \pi_{12} - \pi_{21}`` so we can eliminate one of the variables.

```math
\begin{flalign}
\pi_{11}(r_{21}^x - r_{11}^x) &\leq \pi_{12} (r_{12}^x - r_{22}^x) \\
\pi_{21}(r_{11}^x - r_{21}^x) &\leq (1 - \pi_{11} - \pi_{12} - \pi_{21}) (r_{22}^x - r_{12}^x) \\
\pi_{21}(r_{11}^x - r_{21}^x + r_{22}^x - r_{12}^x) &\leq (1 - \pi_{11}) (r_{22}^x - r_{12}^x) - \pi_{12}(r_{22}^x - r_{12}^x) \\
\end{flalign}
```

Let's consider the equality case first and worry about the sign later.  That way we can replace in the second equation the term for ``\pi_{11}`` in the first:

```math
\begin{flalign}
\pi_{21}(r_{11}^x - r_{21}^x + r_{22}^x - r_{12}^x) &= (1 - \pi_{11}) (r_{22}^x - r_{12}^x) + \pi_{11}(r_{21}^x - r_{11}^x) \\
\pi_{21}(r_{11}^x - r_{21}^x + r_{22}^x - r_{12}^x) &= \pi_{11}(r_{12}^x - r_{22}^x + r_{21}^x - r_{11}^x) + r_{22}^x - r_{12}^x \\
&= -\pi_{11}(-r_{12}^x + r_{22}^x - r_{21}^x + r_{11}^x) + r_{22}^x - r_{12}^x \\
\pi_{21} &= -\pi_{11} + \frac{r_{22}^x - r_{12}^x}{r_{11}^x - r_{21}^x + r_{22}^x - r_{12}^x} \\
\pi_{12} &= \pi_{11} \frac{r_{21}^x - r_{11}^x}{r_{12}^x - r_{22}^x}
\end{flalign}
```
"""

# ╔═╡ c8aec333-bf57-4a84-8403-b13e55973937
md"""
Let's look at an example like the Chicken matrix game:

```math
r_{11}^x = 0, r_{12}^x = 7, r_{21}^x = 2, r_{22}^x = 6 \quad
r_{11}^y = 0, r_{12}^y = 2, r_{21}^y = 7, r_{22}^y = 6
```

So we get the following conditions for the joint policy:

```math
\begin{flalign}
\pi_{21} &= -\pi_{11} + \frac{6 - 7}{0 - 2 + 6 - 7} = \frac{1}{3} - \pi_{11} \\
\pi_{12} &= \pi_{11} \frac{2 - 0}{7 - 6} = 2 \pi_{11} \implies \pi_{12} = 2 \left ( \frac{1}{3} - \pi_{21} \right )
\end{flalign}
```

Already we can get some constraints on the joint policy from our knowledge of the distribution.  In particular, we know that ``0 \leq \pi_{11} \leq \frac{1}{3}``, ``0 \leq \pi_{21} \leq \frac{1}{3}``, and ``0 \leq \pi_{12} \leq \frac{2}{3}``.  Presumably, we could select any joint policy that falls within these conditions and be okay.  Let's says we select ``\pi_{11} = \frac{1}{3}`` and ``\pi_{21} = \frac{1}{3}`` and ``\pi_{12} = \frac{1}{3}``.  Since these three values already add up to 1, the remaining joint action ``(2, 2)`` would have zero probability.
"""

# ╔═╡ d1de00b2-4642-421c-a88a-56b5af271705
md"""
### Solution via Linear Programming
"""

# ╔═╡ e60dd8ed-0ce5-4773-9dfd-27dcf5c37fc3
#add linear programming solution for coarse and regular correlated equilibria

# ╔═╡ c84877c6-cbd1-43b5-b3cd-10756cf15578
md"""
## Pareto Optimality

Add definition of pareto optimality and see what the equation implies in the 2 player 2 action case
"""

# ╔═╡ e5660881-3be9-44cc-ba21-497b3b8dadc3
md"""

## Interesting 2x2 Games

"""

# ╔═╡ 829a0e99-13e9-4195-b584-3feb29bad0ac
#add description of these games and show a triple plot with the solution for a given x/y value for the x and y player as well as the plot in reward space with individual dots like what is shown in the book.  Ideally also label within this plot the different types of equilibria

# ╔═╡ 1167020f-3784-4ae7-b211-bee12281c745
const game_names = ["Battle of the Sexes", "Chicken", "Stag Hunt", "Prisoner's Dilemma", "Deadlock"]

# ╔═╡ 9850d312-55d3-4883-a959-a38d41858e1c
md"""
### Battle of the Sexes

Two parties would like to agree on an event to attend but differ on their first choice.  X prefers a 11 outcome while Y prefers 22.  If the decision does not match, there is a slight preference for each party to attend their first choice event.  The worst outcome is when both parties go separately to their second choice event.

"""

# ╔═╡ 009a188c-738b-469b-b3de-be5520c03fae
const battle_of_sexes = (x = [10 2; 0 7], y = [7 2; 0 10])

# ╔═╡ fdba1b1b-ac31-48d8-9d1e-ffd9e2ade073
md"""
### Chicken

Two parties are on a collision course and would like to stay the course as long as the other party leaves.  The second best outcome for both parties is when they both agree to leave.  The worst outcome for both parties is neither agreeing to leave.
"""

# ╔═╡ ebd3b15c-c4fa-417a-bfab-ab7de0a2f116
const chicken = (x = [0 7; 2 6], y = [0 2; 7 6])

# ╔═╡ def7b3e2-b67b-4c2d-9c63-298a01fb0df2
md"""
### Stag Hunt

Two parties are hunting prey but have two options.  The first option, the stag, is more desirable, but can only be successfully hunted together.  The second option, the rabbit, is less rewarding but can be hunted individually.  The best outcome for both parties is when they agree to cooperate hunting the more rewarding and dangerous prey.  The worst individual outcome is when one party decides to hunt the stag but is not joined by the other.  The second worst outcome is when both hunters try to hunt the rabbit together.  The second best outcome for an individual hunter is deciding to hunt the rabbit while the other attempts and fails to hunt the stag.

"""

# ╔═╡ f7611ca8-5d5a-42f8-99af-3a4afb00daae
const stag_hunt = (x = [4 0; 3 2], y = [4 3; 0 2])

# ╔═╡ 2923aefc-2332-4dbd-9aca-62bdb04c3850
md"""
### Prisoner's Dilemma

Two prisoners are interrogated separately and can either stay loyal to their compariot or betray them.  If both parties choose to betray, then they suffer a negative outcome but the worst outcome for an individual is when they stay loyal while their partner betrays them.  The second best outcome for both prisoners involves neither betraying. 
"""

# ╔═╡ 36408451-3729-4cbe-b0f9-d27df2d9eaa7
const prisoners_dilemma = (x = [-1 -5; 0 -3], y = [-1 0; -5 -3])

# ╔═╡ 0b94fc9c-cbe3-46f3-a7b5-53c8cc51f2cd
md"""
### Deadlock

This scenario matches the prisoner's dilemma except, the most favorable shared outcome is mutual coorperation rather than mutual betrayal.  The unmatched outcome is still equally good/bad for the betrayer/loyalist.
"""

# ╔═╡ 825e4812-79b3-4966-9525-ffe845c93149
const deadlock = (x = [-3 -5; 0 -1], y = [-3 0; -5 -1])

# ╔═╡ d460584a-3db5-44f0-bb04-51ff3d97115e
const game_list = [battle_of_sexes, chicken, stag_hunt, prisoners_dilemma, deadlock]

# ╔═╡ 43e595e8-1208-4951-9bc6-034d6d836159
convert_2x2_game(r::@NamedTuple{x::Matrix{T}, y::Matrix{T}}) where T<:Real = (xAA = r.x[1, 1], xAB = r.x[1, 2], xBA = r.x[2, 1], xBB = r.x[2, 2], yAA = r.y[1, 1], yAB = r.y[1, 2], yBA = r.y[2, 1], yBB = r.y[2, 2])

# ╔═╡ 29b15cb8-3bae-46c3-bbe7-c3e98435346e
const game_list2 = convert_2x2_game.(game_list)

# ╔═╡ 92b11e87-76a5-481d-873e-64ad4e4103aa
function display_game_rewards(r)
	m1 = md"""
	|X Rewards|||
	|---|---|---|
	|Action|A|B|
	|A|$(r.xAA)|$(r.xAB)|
	|B|$(r.xBA)|$(r.xBB)|
	"""

	m2 = md"""
	|Y Rewards|||
	|---|---|---|
	|Action|A|B|
	|A|$(r.yAA)|$(r.yAB)|
	|B|$(r.yBA)|$(r.yBB)|
	"""

	@htl("""
		 <div style = "display: flex;">
		 $m1
		 $m2
		 </div>
		 """)
end

# ╔═╡ 98e255ae-bc5f-4f17-b39b-57c0f2df324c
function display_game_rewards(r::@NamedTuple{x::Matrix{T}, y::Matrix{T}}) where T<:Real
	game = convert_2x2_game(r)
	display_game_rewards(game)
end

# ╔═╡ e4ab0870-5442-4bf4-9ed9-90d5d8184ea7
md"""
## Game Analysis
"""

# ╔═╡ e867873e-7e1a-452f-a0ac-f23778d1479b
@bind game_select Select([game_list2[i] => game_names[i] for i in eachindex(game_names)])

# ╔═╡ 5d2585d6-9ba9-4858-b1d8-355372926448
display_game_rewards(game_select)

# ╔═╡ 817e05ef-9fc8-4380-9350-cf2ed56f0533
function calculate_xy_rewards(reward; npoints = 50)
	xvals = LinRange(0, 1, npoints)
	yvals = LinRange(0, 1, npoints)
	x_output = zeros(npoints, npoints)
	y_output = zeros(npoints, npoints)
	for i in 1:npoints
		for j in 1:npoints
			(u_x, u_y) = reward_xy(xvals[i], yvals[j], reward)
			x_output[j, i] = u_x
			y_output[j, i] = u_y
		end
	end

	welfare = x_output .+ y_output
	fairness = x_output .* y_output

	welfare_optimal = argmax(welfare)
	fairness_optimal = argmax(fairness)

	nashes = check_nash(reward)

	mixed_reward = if isempty(nashes.mixed_nash)
		NaN
	else
		reward_xy(nashes.mixed_nash.x, nashes.mixed_nash.y, reward)
	end

	return (xvals, yvals, x_output, y_output, welfare, fairness, welfare_optimal, fairness_optimal, nashes, mixed_reward)
end

# ╔═╡ 5febdc0d-dcc3-408f-9277-6a9ed704569c
const game_nashes = check_nash(game_select)

# ╔═╡ 9669c37c-6a73-4abc-a391-3134642f4a43
const game_values = calculate_xy_rewards(game_select)

# ╔═╡ fa6013b0-0992-4d61-b70f-b2e073bf8db8
@bind xy_ref PlutoUI.combine() do Child
	md"""
	X Probability of Action 1: $(Child(Slider(eachindex(game_values[1]), default = floor(Int64, length(eachindex(game_values[1]))/2))))
	
	Y Probability of Action 1: $(Child(Slider(eachindex(game_values[2]), default = floor(Int64, length(eachindex(game_values[2]))/2))))
	"""
end

# ╔═╡ 8e31007e-eec7-4921-a794-eedb98c0e4ec
function check_pareto(x_output::Matrix, y_output::Matrix)
#check to see if a given point in the solution space is pareto optimal meaning it is not pareto dominated by any other policy
	(n, m) = size(x_output)
	x1 = x_output[1, 1]
	x2 = x_output[2, 2]
end

# ╔═╡ 6c9f0834-f282-41fb-81ba-bdba6763d172
function plot_xy_rewards(xvals::AbstractVector, yvals::AbstractVector, x_output::Matrix, y_output::Matrix, welfare::Matrix, fairness::Matrix, welfare_optimal, fairness_optimal, nashes, mixed_reward, xind::Integer, yind::Integer)
	tr_x_value = heatmap(x = xvals, y = yvals, z = x_output, colorscale = "rb")
	tr_ref = scatter(x = xvals[[xind]], y = xvals[[yind]], mode = "markers", marker_symbol = "x", marker_color = "black", name = "Reference Point", showlegend = false)
	tr_y_value = heatmap(x = xvals, y = yvals, z = y_output, colorscale = "rb")

	tr_fairness_optimal = scatter(x = xvals[[fairness_optimal[1]]], y = yvals[[fairness_optimal[2]]], mode = "markers", marker_symbol = "diamond", marker_color = "pink", name = "Fairness Optimal")
	tr_welfare_optimal = scatter(x = xvals[[welfare_optimal[1]]], y = yvals[[welfare_optimal[2]]], mode = "markers", marker_color = "pink", marker_symbol = "square", name = "Welfare Optimal")

	prob_lookup = Dict([1 => 0, 2 => 1])
	ind_lookup = Dict([2 => length(xvals), 1 => 1])
	if iszero(sum(nashes.pure_nash))
		tr_nash_pure = [scatter()]
		tr_nash_reward_pure = [scatter()]
	else
		inds = findall(nashes.pure_nash)
		tr_nash_pure = [scatter(x = [prob_lookup[c[1]]], y = [prob_lookup[c[2]]], mode = "markers", marker_symbol = "triangle-up", marker_color = "green", name = "Deterministic NE") for c in inds]
		tr_nash_reward_pure = [scatter(x = [x_output[ind_lookup[c[1]], ind_lookup[c[2]]]], y = [y_output[ind_lookup[c[1]], ind_lookup[c[2]]]], mode = "markers", marker_symbol = "triangle-up", marker_color = "green", name = "Deterministic NE") for c in inds]
	end

	if isempty(nashes.mixed_nash)
		tr_nash_mixed = [scatter()]
		tr_nash_reward_mixed = [scatter()]
	else
		tr_nash_mixed = [scatter(x = [nashes.mixed_nash.x], y = [nashes.mixed_nash.y], mode = "markers", marker_symbol = "x", marker_color = "yellow", name = "Probabilistic NE")]
		tr_nash_reward_mixed = [scatter(x = [mixed_reward[1]], y = [mixed_reward[2]], mode = "markers", marker_symbol = "x", marker_color = "yellow", name = "Probabilistic NE")]
	end
		

	xref = x_output[xind, yind]
	yref = y_output[xind, yind]
	p1 = plot([[tr_x_value, tr_ref, tr_fairness_optimal, tr_welfare_optimal]; tr_nash_pure; tr_nash_mixed], Layout(xaxis_title = "x_A", yaxis_title = "y_A", title = "X Reward ($xref Reference)", legend_orientation = "h")) 
	p2 = plot([[tr_y_value, tr_ref, tr_fairness_optimal, tr_welfare_optimal]; tr_nash_pure; tr_nash_mixed], Layout(xaxis_title = "x_A", yaxis_title = "y_A", title = "Y Reward ($yref Reference)", legend_orientation = "h"))

	joint_reward_tr = scatter(x = x_output[:], y = y_output[:], mode = "markers", marker_size = 1)
	joint_reward_ref = scatter(x = [xref], y = [yref], mode = "markers", marker_color = "black", marker_symbol = "x", name = "Reference Point")
	welfare_reward_ref = scatter(x = [x_output[welfare_optimal]], y = [y_output[welfare_optimal]], name = "Welfare Optimal", mode = "markers", marker_symbol = "square", marker_color = "pink")
	fairness_reward_ref = scatter(x = [x_output[fairness_optimal]], y = [y_output[fairness_optimal]], name = "Fairness Optimal", mode = "markers", marker_symbol = "diamond", marker_color = "pink")
	p3 = plot([[joint_reward_tr, joint_reward_ref, welfare_reward_ref, fairness_reward_ref]; tr_nash_reward_pure; tr_nash_reward_mixed], Layout(showlegend = false))

	fairness_ref = fairness[xind, yind]
	welfare_ref = welfare[xind, yind]
	tr_fairness = heatmap(x = xvals, y = yvals, z = fairness, colorscale = "rb")
	tr_welfare = heatmap(x = xvals, y = yvals, z = welfare, colorscale = "rb")
	p4 = plot([[tr_welfare, tr_ref, tr_welfare_optimal]; tr_nash_pure; tr_nash_mixed], Layout(xaxis_title = "x_A", yaxis_title = "y_A", title = "Welfare ($welfare_ref Reference)", legend_orientation = "h"))
	p5 = plot([[tr_fairness, tr_ref, tr_fairness_optimal]; tr_nash_pure; tr_nash_mixed], Layout(xaxis_title = "x_A", yaxis_title = "y_A", title = "Fairness ($fairness_ref Reference)", legend_orientation = "h"))

	tbl = md"""
	|Solution Type|x probability|y probability|x reward|y reward|
	|---|---|---|---|---|
	|Fairness Optimal|$(xvals[fairness_optimal[1]])|$(yvals[fairness_optimal[2]])|$(x_output[fairness_optimal])|$(y_output[fairness_optimal])|
	|Welfare Optimal|$(xvals[welfare_optimal[1]])|$(yvals[welfare_optimal[2]])|$(x_output[welfare_optimal])|$(y_output[welfare_optimal])|
	"""
	
	@htl("""
	<div style = "display:flex;">
	$p1
	$p2
	$p3
	</div>
	<div style = "display:flex;">
	$p4
	$p5
	$tbl
	</div>
	""")
end

# ╔═╡ 42e5da05-d7ee-4152-808e-586dfe660448
plot_xy_rewards(game_values..., xy_ref...)

# ╔═╡ 73a11546-b50b-11f0-8a29-2b34ea925a35
md"""
# Dependencies
"""

# ╔═╡ 0acabece-2e91-40a5-990b-2b923d2cbeaa
import HiGHS

# ╔═╡ 8fc47d0c-bc4c-460e-a9d3-8c33fe797024
begin
	model = Model(HiGHS.Optimizer)
	minreward = -maximum(rewardmatrix4)
	@variable(model, x[1:2])
	@constraint(model, x .>= 0)
	@constraint(model, x[1]*(-rewardmatrix4.AA - minreward) + x[2]*(-rewardmatrix4.BA - minreward) ≤ 1)
	@constraint(model, x[1]*(-rewardmatrix4.AB - minreward) + x[2]*(-rewardmatrix4.BB - minreward) ≤ 1)
	@objective(model, Max, sum(x))
	optimize!(model)
	modified_game_value = inv(sum(value.(x)))
	x_policy1 = value.(x) .* modified_game_value
	game_value1 = -(modified_game_value + minreward)
end

# ╔═╡ a6fac14d-ab9b-49e2-b92a-6a204ae96477
(game_value1 = game_value1, x_policy1 = x_policy1)

# ╔═╡ b846df09-09d1-4947-b03e-8e64b8a159f1
print(model)

# ╔═╡ 06980c18-c294-4c8d-9783-0b75beedf13b
assert_is_solved_and_feasible(model)

# ╔═╡ bf19b54a-59af-4291-b2c5-d77dba40e03a
solution_summary(model)

# ╔═╡ 9c8099eb-a263-4487-863b-e34e1ba3bb99
begin
	model2 = Model(HiGHS.Optimizer)
	minreward2 = minimum(rewardmatrix4)
	@variable(model2, x2[1:2])
	@constraint(model2, x2 .>= 0)
	@constraint(model2, x2[1]*(rewardmatrix4.AA - minreward2) + x2[2]*(rewardmatrix4.AB - minreward2) ≤ 1)
	@constraint(model2, x2[1]*(rewardmatrix4.BA - minreward2) + x2[2]*(rewardmatrix4.BB - minreward2) ≤ 1)
	@objective(model2, Max, sum(x2))
	optimize!(model2)
	modified_game_value2 = inv(sum(value.(x2)))
	x_policy2 = value.(x2) .* modified_game_value2
	game_value2 = modified_game_value + minreward2
end

# ╔═╡ 8173f230-088e-4e5a-8bfc-a68f0e416da4
(game_value2 = game_value2, x_policy2 = x_policy2)

# ╔═╡ b4396fe9-c88b-4d17-b096-139c610f7a9d
function solve_minimax_game(reward_matrix::Matrix)
	(l, k) = size(reward_matrix)
	model = Model(HiGHS.Optimizer)
	minreward = minimum(reward_matrix)
	reward_matrix′ = reward_matrix .- minreward
	@variable(model, x[1:l])
	@constraint(model, x .>= 0.)
	@constraint(model, x' * reward_matrix′ .≤ 1.)
	@objective(model, Max, sum(x))
	optimize!(model)
	modified_game_value = inv(sum(value.(x)))
	x_policy = value.(x) .* modified_game_value
	game_value = modified_game_value + minreward
	return (game_value = game_value, x_policy = x_policy)
end

# ╔═╡ d3154123-62d0-43b3-8ec7-a73874a402a1
solve_minimax_game(rock_paper_scissors_game_rewards)

# ╔═╡ 1939bf92-0ba9-4392-929c-630de9c46be5
solve_minimax_game(rand([-1, 1, 0], 100, 100))

# ╔═╡ ff3c2978-468e-4f46-a6a0-cae743a0b2bd
function test_game(reward_matrix::Matrix)
	#the reward matrix contains the rewards for each agent in the rows of the matrix, so there is one row per agent and one column for each joint action
	(n, l) = size(reward_matrix)
	
	model = Model(HiGHS.Optimizer)
	
	@variable(model, x[1:l]) #create a variable with a probability for each joint action

	#maximize the expected reward summed across all agents
	@objective(model, Max, sum(reward_matrix*x))
	
	#ensure variables form a probability distribution
	@constraint(model, x .>= 0.)
	@constraint(model, sum(x) == 1)

	#ensure joint policy is a correlated equilibrium
	#--add constraint here
	
	optimize!(model)
	modified_game_value = inv(sum(value.(x)))
	x_policy = value.(x) .* modified_game_value
	game_value = modified_game_value + minreward
	return (game_value = game_value, x_policy = x_policy)
end

# ╔═╡ c2c5714a-276c-454e-936e-9b32eb617131
test_game(rand(2, 2))

# ╔═╡ f608b735-f145-442a-b4ac-1a0c61cb72aa
TableOfContents()

# ╔═╡ 4e962f2b-120d-4030-a591-cc4a87c9a200
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
HiGHS = "87dc4568-4c63-4d18-b0c0-bb2238e4078b"
HypertextLiteral = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
JuMP = "4076af6c-e467-56ae-b986-b466b2749572"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
PlutoPlotly = "8e989ff0-3d88-8e9f-f020-2b208a939ff0"
PlutoProfile = "ee419aa8-929d-45cd-acf6-76bd043cd7ba"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"

[compat]
BenchmarkTools = "~1.6.3"
HiGHS = "~1.18.1"
HypertextLiteral = "~0.9.5"
JuMP = "~1.26.0"
PlutoPlotly = "~0.6.5"
PlutoProfile = "~0.4.0"
PlutoUI = "~0.7.73"
StatsBase = "~0.34.7"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.12.2"
manifest_format = "2.0"
project_hash = "7befd4f8b8cca10d64e4cd6c6f8fa457290abcb8"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "6e1d2a35f2f90a4bc7c2ed98079b2ba09c35b83a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.3.2"

[[deps.AbstractTrees]]
git-tree-sha1 = "03e0550477d86222521d254b741d470ba17ea0b5"
uuid = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
version = "0.3.4"

[[deps.AliasTables]]
deps = ["PtrArrays", "Random"]
git-tree-sha1 = "9876e1e164b144ca45e9e3198d0b689cadfed9ff"
uuid = "66dad0bd-aa9a-41b7-9441-69ab47430ed8"
version = "1.1.3"

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

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "4e1fe97fdaed23e9dc21d4d664bea76b65fc50a0"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.22"

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
git-tree-sha1 = "910febccb28d493032495b7009dce7d7f7aee554"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "1.0.1"

    [deps.ForwardDiff.extensions]
    ForwardDiffStaticArraysExt = "StaticArrays"

    [deps.ForwardDiff.weakdeps]
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

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

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.IrrationalConstants]]
git-tree-sha1 = "b2d91fe939cae05960e760110b328288867b5758"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.6"

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
git-tree-sha1 = "90002c976264d2f571c98cd1d12851f4cba403df"
uuid = "4076af6c-e467-56ae-b986-b466b2749572"
version = "1.26.0"

    [deps.JuMP.extensions]
    JuMPDimensionalDataExt = "DimensionalData"

    [deps.JuMP.weakdeps]
    DimensionalData = "0703355e-b756-11e9-17c0-8b28908087d0"

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
deps = ["BenchmarkTools", "CodecBzip2", "CodecZlib", "DataStructures", "ForwardDiff", "JSON3", "LinearAlgebra", "MutableArithmetics", "NaNMath", "OrderedCollections", "PrecompileTools", "Printf", "SparseArrays", "SpecialFunctions", "Test"]
git-tree-sha1 = "2f2c18c6acab9042557bdb0af8c3a14dd7b64413"
uuid = "b8f27783-ece8-5eb3-8dc8-9495eed66fee"
version = "1.41.0"

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
git-tree-sha1 = "491bdcdc943fcbc4c005900d7463c9f216aabf4c"
uuid = "d8a4904e-b15c-11e9-3269-09a3773c0cb0"
version = "1.6.4"

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
git-tree-sha1 = "28278bb0053da0fd73537be94afd1682cc5a0a83"
uuid = "a03496cd-edff-5a9b-9e67-9cda94a718b5"
version = "0.8.21"

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
git-tree-sha1 = "3faff84e6f97a7f18e0dd24373daa229fd358db5"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.73"

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

[[deps.PtrArrays]]
git-tree-sha1 = "1d36ef11a9aaf1e8b74dacc6a731dd1de8fd493d"
uuid = "43287f4e-b6f4-7ad1-bb20-aadabca52c3d"
version = "1.3.0"

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

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "9d72a13a3f4dd3795a195ac5a44d7d6ff5f552ff"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.7.1"

[[deps.StatsBase]]
deps = ["AliasTables", "DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "a136f98cefaf3e2924a66bd75173d1c891ab7453"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.7"

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
git-tree-sha1 = "372b90fe551c019541fafc6ff034199dc19c8436"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.12"

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
# ╟─be459828-55e3-499c-946d-1c510187db56
# ╟─14115f9b-e92d-4bf2-aa08-c7e9e4b55e10
# ╟─02890ffe-4000-4295-898c-769ed931354d
# ╟─aff36c99-8b0a-4b73-af43-30c10effd53a
# ╟─c3ea41a2-ba31-4004-aeb3-b91b044532d4
# ╟─e6df629a-dc22-43f5-8e01-9f8bdfe0d3ea
# ╟─37c563f3-c8af-4ffa-958c-1f77dfdfe530
# ╟─96c662b7-007a-4a7e-b826-4d86bb7f9ca8
# ╟─755b5857-b1c3-4739-a8c7-5d373f011e06
# ╟─2f912c93-87d9-40d7-bc59-2c56590238e1
# ╟─d1dc4cf0-5739-4562-817e-303d8971e74d
# ╟─0b6c96e3-9fe4-4a09-9d3c-bc87966c10ab
# ╟─77ce6f8a-dc52-4e3c-b599-cb11f4e48e99
# ╟─961c0bfa-9e2f-45aa-bd34-5cae074a0d0c
# ╟─84aaf3ca-be83-4fa1-ad3b-f565c259bb00
# ╟─ec46b9c9-0ddc-49d8-9bb9-6571ab9f09c4
# ╟─b1750cc5-01d3-4fb3-925d-aaaae8fe0e67
# ╠═efde1b80-0972-4652-803c-6f7ce278b43c
# ╠═4ac2c4c6-f6ff-4126-85ce-904a746a694f
# ╠═e8fb0a96-e602-491d-9464-a8dc9b7e5f63
# ╠═92ef9f37-8a63-40f9-a70c-786f9794a355
# ╠═c6f0f327-2ddf-497b-996d-cc8f296c8fde
# ╟─5f455edf-7cb8-4199-9f71-c31fd1b793f2
# ╟─d0571165-6b22-43b6-a023-f14fb5575c68
# ╟─500ddfe9-4f4c-4411-8ca2-ce60f058a125
# ╟─8b7fc1fe-2bf3-440d-993c-6560b1f53ec1
# ╟─f2aecc45-5773-4ef5-bf61-99a868ac3548
# ╠═429e4864-16c7-415a-b7eb-8fb09d36ab18
# ╠═a6fac14d-ab9b-49e2-b92a-6a204ae96477
# ╠═8fc47d0c-bc4c-460e-a9d3-8c33fe797024
# ╠═8173f230-088e-4e5a-8bfc-a68f0e416da4
# ╠═9c8099eb-a263-4487-863b-e34e1ba3bb99
# ╠═b846df09-09d1-4947-b03e-8e64b8a159f1
# ╠═06980c18-c294-4c8d-9783-0b75beedf13b
# ╠═bf19b54a-59af-4291-b2c5-d77dba40e03a
# ╠═b4396fe9-c88b-4d17-b096-139c610f7a9d
# ╠═4989b264-fbd5-42da-92a8-6edf266a87f8
# ╠═d3154123-62d0-43b3-8ec7-a73874a402a1
# ╠═61b80664-30bd-4f21-a69f-46a3f8804427
# ╠═1939bf92-0ba9-4392-929c-630de9c46be5
# ╟─e132c8d0-6bdd-40b7-b18c-9feb45ba0cee
# ╟─fbc4f7f7-e8cf-474d-b8ee-1b08661c43aa
# ╟─a679f649-a97b-42d1-b6e0-678909598471
# ╟─b8a39b43-52b0-4a39-8593-7f3b9a7daee6
# ╠═83be4347-ba46-45b5-9736-a36895c7a1a7
# ╠═3a751006-2c04-4236-980b-10b3299f1b84
# ╟─321e7b12-bc86-48c6-bae1-db5e7ef667a0
# ╟─918b7950-70fe-4cb5-9e11-b8bf891a937b
# ╟─44e36257-2f59-4451-b559-a0fc692ccd3a
# ╠═b0213f40-b353-49cf-b355-45a29bfd151e
# ╠═934fceb4-35ef-4db1-a379-9e2d6eb67780
# ╠═e0a8af0a-0d12-46a8-b0f6-c713d8929320
# ╟─b9699472-7f3d-4226-b60f-a3e1fb583ca8
# ╟─97397f02-e87d-429a-a7b3-3ac961f22f54
# ╠═5a62ad34-cb96-4dc4-a9ef-4841126aea57
# ╠═59a47bf3-a80d-4936-b079-40ffed47c98c
# ╠═c932baa2-44e8-4109-97f9-c6a72b10a6e5
# ╠═71fe7db2-448a-4921-8272-f34d887c3668
# ╠═e3c29301-44d2-4925-8ac8-8b2391106cdc
# ╠═9c03ba14-9e59-4024-a227-3b65d694eaea
# ╠═c27fab3c-c29d-40c4-be81-d2d60c57544e
# ╠═38543174-4a36-4718-b646-848602e15429
# ╟─f9451804-e106-4e14-8985-2d2a82abf241
# ╟─64a76ef6-a784-4997-8132-497c6659d0c6
# ╟─d9a9f2fb-0841-4933-ab71-4a5564d5794c
# ╠═107740d9-803a-434d-a7fe-e94abb06c5ed
# ╠═e37de208-1635-4d25-8dca-ab07eb8916a3
# ╟─944447ae-5597-4998-ad34-23e8510b323f
# ╟─ad7a0f40-fc91-4b8c-83ca-871efb9f2c06
# ╠═f6aea589-b844-46d5-888b-e4236e71d4d2
# ╠═c636112d-b788-46e4-a15f-9a7e347e5516
# ╠═ba6f2a22-df07-4677-9696-d53fa7fae133
# ╠═aef23141-e1ef-416f-91da-965ca1f69397
# ╠═dd2824e6-d74e-4625-b1b1-497a9bc3b662
# ╠═b8aee310-20aa-4ef5-9cc9-32ff4d8d8035
# ╟─b8f4e17b-72a3-401c-bef1-9d02e3a0c577
# ╟─36a5c1bb-f26f-4eaf-b5df-7bfc01963911
# ╟─fc41aa2b-83eb-4914-90c1-e1af40e94711
# ╠═a89ea983-2c60-468b-a7eb-6bcc82533fd8
# ╠═e992fead-2863-4e4c-bb31-a8ed2b70b2e0
# ╠═c52f0d6d-d367-4a85-8f92-20baec85fce9
# ╠═27b7bb79-81ba-4287-9230-f22f74f04f9a
# ╠═55abdda0-4dcc-4348-91fc-c3faeb3e83f2
# ╠═bbd07ba1-528d-4801-8dfe-9488ac60a11d
# ╠═c8aec333-bf57-4a84-8403-b13e55973937
# ╟─d1de00b2-4642-421c-a88a-56b5af271705
# ╠═e60dd8ed-0ce5-4773-9dfd-27dcf5c37fc3
# ╠═ff3c2978-468e-4f46-a6a0-cae743a0b2bd
# ╠═c2c5714a-276c-454e-936e-9b32eb617131
# ╠═c84877c6-cbd1-43b5-b3cd-10756cf15578
# ╟─e5660881-3be9-44cc-ba21-497b3b8dadc3
# ╠═829a0e99-13e9-4195-b584-3feb29bad0ac
# ╠═d460584a-3db5-44f0-bb04-51ff3d97115e
# ╠═29b15cb8-3bae-46c3-bbe7-c3e98435346e
# ╠═1167020f-3784-4ae7-b211-bee12281c745
# ╟─9850d312-55d3-4883-a959-a38d41858e1c
# ╠═009a188c-738b-469b-b3de-be5520c03fae
# ╟─fdba1b1b-ac31-48d8-9d1e-ffd9e2ade073
# ╠═ebd3b15c-c4fa-417a-bfab-ab7de0a2f116
# ╟─def7b3e2-b67b-4c2d-9c63-298a01fb0df2
# ╠═f7611ca8-5d5a-42f8-99af-3a4afb00daae
# ╟─2923aefc-2332-4dbd-9aca-62bdb04c3850
# ╠═36408451-3729-4cbe-b0f9-d27df2d9eaa7
# ╟─0b94fc9c-cbe3-46f3-a7b5-53c8cc51f2cd
# ╠═825e4812-79b3-4966-9525-ffe845c93149
# ╠═43e595e8-1208-4951-9bc6-034d6d836159
# ╠═92b11e87-76a5-481d-873e-64ad4e4103aa
# ╠═98e255ae-bc5f-4f17-b39b-57c0f2df324c
# ╟─e4ab0870-5442-4bf4-9ed9-90d5d8184ea7
# ╟─e867873e-7e1a-452f-a0ac-f23778d1479b
# ╟─5d2585d6-9ba9-4858-b1d8-355372926448
# ╟─fa6013b0-0992-4d61-b70f-b2e073bf8db8
# ╠═42e5da05-d7ee-4152-808e-586dfe660448
# ╠═817e05ef-9fc8-4380-9350-cf2ed56f0533
# ╠═5febdc0d-dcc3-408f-9277-6a9ed704569c
# ╠═9669c37c-6a73-4abc-a391-3134642f4a43
# ╠═8e31007e-eec7-4921-a794-eedb98c0e4ec
# ╠═6c9f0834-f282-41fb-81ba-bdba6763d172
# ╟─73a11546-b50b-11f0-8a29-2b34ea925a35
# ╠═7336c97e-15d7-45ee-980c-5e90e9e324d8
# ╠═0acabece-2e91-40a5-990b-2b923d2cbeaa
# ╠═f608b735-f145-442a-b4ac-1a0c61cb72aa
# ╠═4e962f2b-120d-4030-a591-cc4a87c9a200
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
