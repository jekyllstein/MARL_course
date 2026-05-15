# Othello

A small Julia package implementing Othello/Reversi game logic with an RL-friendly board representation.

## Project structure

- `src/Othello.jl` — main game logic module
- `test/runtests.jl` — basic rule validation tests
- `Project.toml` — package manifest

## Features

- Immutable `OthelloBoard` representation
- `OthelloMove` validation and generation
- `make_move` with correct disc flipping and turn handling
- game-over detection and score/winner computation
- perspective-normalized feature vectors for RL input (both `NTuple` and `Vector{Float32}`)
- text-based board display via `show`

## Installation

For a quick trial, install Othello in a temporary Julia environment. This avoids adding the package to your default Julia environment:

```julia
using Pkg
Pkg.activate(; temp=true)
Pkg.add(url="https://github.com/jekyllstein/Othello")

using Othello

b = initialize_othello_board()
println(b)
println("Available moves: ", available_moves(b))
```

To add the package to your current Julia environment instead, run:

```julia
using Pkg
Pkg.add(url="https://github.com/jekyllstein/Othello")
```

For local development, clone the repository and activate it:

```bash
git clone https://github.com/jekyllstein/Othello.git
cd Othello
julia --project=.
```

Then instantiate the project in Julia:

```julia
using Pkg
Pkg.instantiate()
```

## Usage Examples

After installation, load the package with:

```julia
using Othello
```

Create a board:

```julia
b = initialize_othello_board()
println(b)
```

List legal moves:

```julia
moves = available_moves(b)
println("Available moves: ", moves)
```

Make a move:

```julia
move = OthelloMove(4, 3)
b = make_move(b, move)
println(b)
println("Score: ", score(b))
```

Use a 1-based action index:

```julia
action = Othello.board_index(4, 3)
move = OthelloMove(action)
b = make_move(b, action)
```

For RL-style action spaces, actions `1:64` are board placements and action `65` is `PASS_ACTION`:

```julia
move = Othello.action_from_index(PASS_ACTION)  # PassMove()
b = make_move(b, PASS_ACTION)
```

A pass move is only valid when the current player has no placement moves and the opponent has at least one placement move.

Check game state:

```julia
println("Has moves: ", has_moves(b))
println("Game over: ", game_over(b))
println("Winner: ", winner(b))
```

Build an RL feature vector:

```julia
features = Othello.initialize_feature_vector(b)
println(features)
```

## Running tests

Run the test file with:

```bash
julia --project=. test/runtests.jl
```

The implementation currently passes the provided core rule tests.

## API Overview

### Core types (exported)

- `OthelloPlayer` — enum with values accessible via `Othello.Black` or `Othello.White`
- `AbstractOthelloMove` — abstract supertype for Othello actions
- `OthelloMove` — a move with `row::Int, col::Int` (1-based)
- `PassMove` — a pass action
- `OthelloBoard` — immutable board state with cells and current player

### Main functions (exported)

- `initialize_othello_board()` — create a new board in starting position
- `available_moves(b::OthelloBoard)` — list legal moves
- `is_valid_move(b, move)` — check move legality
- `make_move(b, move)` — apply move, return new board state
- `make_move(b, index)` — apply the move at a 1-based action index
- `has_moves(b)` — check if current player has any placement moves
- `game_over(b)` — check if game has ended
- `score(b)` — returns `(black_count, white_count)`
- `winner(b)` — returns `Othello.Black`, `Othello.White`, or `nothing`
- `PASS_ACTION` — the 65th action index, representing a pass

### Helper functions (qualified access only)

- `Othello.initialize_feature_vector()` — allocate a new zeroed `Vector{Float32}` 
- `Othello.initialize_feature_vector(b::OthelloBoard)` — allocate and initialize with board state
- `Othello.update_feature_vector!(out::Vector{Float32}, b)` — fill existing vector in-place
- `Othello.board_index(row, col)` — convert coordinates to linear index
- `Othello.action_from_index(index)` — convert a 1-based action index to a move or pass
- `OthelloMove(index)` — convert a 1-based board index to a placement move

## Notes

- Perspective-normalized features: current player = `1.0`, opponent = `-1.0`, empty = `0.0`
- This implementation is designed to integrate easily into RL or search agents without extra dependencies
- Player enum values are not exported to avoid namespace conflicts; access as `Othello.Black` and `Othello.White`
