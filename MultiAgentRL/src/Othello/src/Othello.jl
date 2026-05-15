module Othello

export OthelloPlayer, AbstractOthelloMove, OthelloMove, PassMove, OthelloBoard,
       initialize_othello_board, available_moves, is_valid_move, make_move,
       has_moves, game_over, score, winner, PASS_ACTION

"""A player in Othello/Reversi."""
@enum OthelloPlayer Black=1 White=2

const Disc = Int8
const EMPTY = Disc(0)
const BoardArray = NTuple{64, Disc}
const DIRECTIONS = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
const BOARD_SIZE = 8
const PASS_ACTION = BOARD_SIZE^2 + 1

"""A move-like action in Othello, including board placements and pass actions."""
abstract type AbstractOthelloMove end

"""An immutable move described by 1-based row and column."""
struct OthelloMove <: AbstractOthelloMove
    row::Int
    col::Int
    function OthelloMove(row::Int, col::Int)
        if row < 1 || row > BOARD_SIZE
            throw(ArgumentError("row must be between 1 and $BOARD_SIZE"))
        end
        if col < 1 || col > BOARD_SIZE
            throw(ArgumentError("col must be between 1 and $BOARD_SIZE"))
        end
        new(row, col)
    end
end

"""A pass action, legal only when the current player has no placement moves."""
struct PassMove <: AbstractOthelloMove end

const INDEXED_MOVES = let
    moves = AbstractOthelloMove[OthelloMove(row, col) for row in 1:BOARD_SIZE for col in 1:BOARD_SIZE]
    push!(moves, PassMove())
    moves
end

"""Return the placement move represented by a 1-based board index."""
function OthelloMove(index::Int)
    if index < 1 || index > BOARD_SIZE^2
        throw(ArgumentError("index must be between 1 and $(BOARD_SIZE^2)"))
    end
    return INDEXED_MOVES[index]::OthelloMove
end

"""Return the move or pass represented by a 1-based action index."""
function action_from_index(index::Int)
    if index < 1 || index > PASS_ACTION
        throw(ArgumentError("index must be between 1 and $PASS_ACTION"))
    end
    if index == PASS_ACTION
        return INDEXED_MOVES[index]::PassMove
    end
    return INDEXED_MOVES[index]::OthelloMove
end

"""An immutable board state for Othello."""
struct OthelloBoard
    cells::BoardArray
    current::OthelloPlayer
end

Base.@propagate_inbounds function OthelloBoard(cells::AbstractVector{<:Integer}, current::OthelloPlayer=Black)
    if length(cells) != BOARD_SIZE^2
        throw(ArgumentError("cells must contain $(BOARD_SIZE^2) entries"))
    end
    OthelloBoard(ntuple(i -> Int8(cells[i]), BOARD_SIZE^2), current)
end

"""Return the linear index for a row/column pair."""
board_index(row::Int, col::Int) = 1 + (row - 1) * BOARD_SIZE + (col - 1)

"""Return true when coordinates are inside the board."""
inbounds(row::Int, col::Int) = 1 <= row <= BOARD_SIZE && 1 <= col <= BOARD_SIZE

"""Return the disc value for a player."""
player_disc(p::OthelloPlayer) = p === Black ? Disc(1) : Disc(-1)

"""Return the opponent player."""
opponent(p::OthelloPlayer) = p === Black ? White : Black

"""Create a new Othello board in the standard starting position."""
function initialize_othello_board()
    cells = fill(EMPTY, BOARD_SIZE^2)
    cells[board_index(4, 4)] = player_disc(White)
    cells[board_index(4, 5)] = player_disc(Black)
    cells[board_index(5, 4)] = player_disc(Black)
    cells[board_index(5, 5)] = player_disc(White)
    OthelloBoard(cells, Black)
end

"""Return the disc at the given board coordinates."""
function cell(b::OthelloBoard, row::Int, col::Int)
    if !inbounds(row, col)
        throw(BoundsError(b, (row, col)))
    end
    b.cells[board_index(row, col)]
end

"""Return all legal moves for the current player."""
function available_moves(b::OthelloBoard)
    moves = AbstractOthelloMove[]
    for row in 1:BOARD_SIZE, col in 1:BOARD_SIZE
        if cell(b, row, col) == EMPTY && is_valid_move(b, OthelloMove(row, col))
            push!(moves, OthelloMove(row, col))
        end
    end
    if isempty(moves) && has_moves(OthelloBoard(b.cells, opponent(b.current)))
        push!(moves, PassMove())
    end
    return moves
end

"""Return true if the current player may place a disc at the given move."""
function is_valid_move(b::OthelloBoard, move::OthelloMove)
    if cell(b, move.row, move.col) != EMPTY
        return false
    end
    player = player_disc(b.current)
    opponent_disc = -player

    for (dr, dc) in DIRECTIONS
        r, c = move.row + dr, move.col + dc
        had_opponent = false
        while inbounds(r, c) && cell(b, r, c) == opponent_disc
            had_opponent = true
            r += dr; c += dc
        end
        if had_opponent && inbounds(r, c) && cell(b, r, c) == player
            return true
        end
    end
    return false
end

"""Return true if the current player may pass."""
function is_valid_move(b::OthelloBoard, ::PassMove)
    if has_moves(b)
        return false
    end
    return has_moves(OthelloBoard(b.cells, opponent(b.current)))
end

"""Collect the indices of discs to flip when the current player plays a move."""
function capture_indices(b::OthelloBoard, move::OthelloMove)
    player = player_disc(b.current)
    opponent_disc = -player
    flips = Int[]

    for (dr, dc) in DIRECTIONS
        r, c = move.row + dr, move.col + dc
        line = Int[]
        while inbounds(r, c) && cell(b, r, c) == opponent_disc
            push!(line, board_index(r, c))
            r += dr; c += dc
        end
        if !isempty(line) && inbounds(r, c) && cell(b, r, c) == player
            append!(flips, line)
        end
    end
    return flips
end

"""Return true when the board has at least one valid move for the current player."""
has_moves(b::OthelloBoard) = any(cell(b, row, col) == EMPTY && is_valid_move(b, OthelloMove(row, col)) for row in 1:BOARD_SIZE, col in 1:BOARD_SIZE)

"""Apply a legal move and return the resulting state with turn handling."""
function make_move(b::OthelloBoard, move::OthelloMove)
    if !is_valid_move(b, move)
        throw(ArgumentError("Move $(move.row),$(move.col) is not legal for player $(b.current)"))
    end

    flips = capture_indices(b, move)
    updated = collect(b.cells)
    updated[board_index(move.row, move.col)] = player_disc(b.current)
    for idx in flips
        updated[idx] = player_disc(b.current)
    end

    next_player = opponent(b.current)
    next_state = OthelloBoard(updated, next_player)
    if !has_moves(next_state) && has_moves(OthelloBoard(updated, b.current))
        return OthelloBoard(updated, b.current)
    end
    return next_state
end

"""Apply the move represented by a 1-based action index."""
make_move(b::OthelloBoard, index::Int) = make_move(b, action_from_index(index))

"""Apply a legal pass move and return the board with the opponent to move."""
function make_move(b::OthelloBoard, move::PassMove)
    if !is_valid_move(b, move)
        throw(ArgumentError("Pass is only legal when the current player has no moves and the opponent has at least one move"))
    end
    return OthelloBoard(b.cells, opponent(b.current))
end

"""Return true when the game has ended."""
game_over(b::OthelloBoard) = !has_moves(b) && !has_moves(OthelloBoard(b.cells, opponent(b.current)))

"""Return the raw score pair (black, white)."""
function score(b::OthelloBoard)
    black = count(==(player_disc(Black)), b.cells)
    white = count(==(player_disc(White)), b.cells)
    return (black, white)
end

"""Return the winner or nothing for a tie."""
function winner(b::OthelloBoard)
    black, white = score(b)
    if black > white
        return Black
    elseif white > black
        return White
    else
        return nothing
    end
end

"""Return a perspective-normalized board vector (NTuple) for RL features."""
function board_feature_vector(b::OthelloBoard)
    reference = player_disc(b.current)
    ntuple(i -> b.cells[i] == EMPTY ? EMPTY : Disc(b.cells[i] * reference), BOARD_SIZE^2)
end

"""Allocate a new zeroed Float32 feature vector."""
function initialize_feature_vector()
    return zeros(Float32, BOARD_SIZE^2)
end

"""Allocate and return a new Float32 feature vector initialized with board state."""
function initialize_feature_vector(b::OthelloBoard)
    out = Vector{Float32}(undef, BOARD_SIZE^2)
    update_feature_vector!(out, b)
    return out
end

"""Update an existing Float32 feature vector in-place with board state."""
function update_feature_vector!(out::Vector{Float32}, b::OthelloBoard)
    if length(out) != BOARD_SIZE^2
        throw(ArgumentError("output vector must have length $(BOARD_SIZE^2)"))
    end
    reference = player_disc(b.current)
    for i in 1:BOARD_SIZE^2
        out[i] = b.cells[i] == EMPTY ? Float32(0) : Float32(b.cells[i] * reference)
    end
    return out
end

"""Visualize the board as a text grid."""
function Base.show(io::IO, b::OthelloBoard)
    for row in 1:BOARD_SIZE
        for col in 1:BOARD_SIZE
            ch = cell(b, row, col) == player_disc(Black) ? 'B' : cell(b, row, col) == player_disc(White) ? 'W' : '.'
            print(io, ch, ' ')
        end
        println(io)
    end
    println(io, "Current player: ", b.current)
end

end # module Othello
