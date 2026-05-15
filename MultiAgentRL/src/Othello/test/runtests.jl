using Test
using Othello

@testset "Othello core rules" begin
    b0 = initialize_othello_board()
    @test score(b0) == (2, 2)
    @test length(available_moves(b0)) == 4
    @test has_moves(b0)

    move = OthelloMove(4, 3)
    b1 = make_move(b0, move)
    @test score(b1) == (4, 1)
    @test b1.current == Othello.White
    @test !is_valid_move(b0, OthelloMove(4, 4))

    @test OthelloMove(1) == OthelloMove(1, 1)
    @test OthelloMove(64) == OthelloMove(8, 8)
    @test @inferred(OthelloMove(1)) == OthelloMove(1, 1)
    @test_throws ArgumentError OthelloMove(PASS_ACTION)
    @test Othello.action_from_index(PASS_ACTION) == PassMove()
    @test Othello.INDEXED_MOVES[PASS_ACTION] == PassMove()
    action_lookup(index::Int) = Othello.action_from_index(index)
    @test only(Base.return_types(action_lookup, Tuple{Int})) == Union{OthelloMove, PassMove}
    @test OthelloMove(Othello.board_index(4, 3)) == OthelloMove(4, 3)
    @test make_move(b0, Othello.board_index(4, 3)) == b1
    @test @inferred(make_move(b0, Othello.board_index(4, 3))) == b1
    @test_throws ArgumentError OthelloMove(0)
    @test_throws ArgumentError OthelloMove(PASS_ACTION + 1)
    @test_throws ArgumentError OthelloMove(0, 1)
    @test !is_valid_move(b0, PassMove())
    @test_throws ArgumentError make_move(b0, PASS_ACTION)

    pass_cells = fill(Int8(0), 64)
    pass_cells[Othello.board_index(1, 1)] = Int8(1)
    pass_cells[Othello.board_index(1, 2)] = Int8(-1)
    b_pass = OthelloBoard(pass_cells, Othello.White)
    @test !has_moves(b_pass)
    @test available_moves(b_pass) == AbstractOthelloMove[PassMove()]
    @test is_valid_move(b_pass, PassMove())
    @test make_move(b_pass, PassMove()) == OthelloBoard(pass_cells, Othello.Black)
    @test make_move(b_pass, PASS_ACTION) == OthelloBoard(pass_cells, Othello.Black)
    @test @inferred(make_move(b_pass, PASS_ACTION)) == OthelloBoard(pass_cells, Othello.Black)

    # Confirm perspective normalization keeps the current player positive
    fv = Othello.initialize_feature_vector(b1)
    @test fv[Othello.board_index(5, 5)] == 1.0f0
    @test fv[Othello.board_index(4, 3)] == -1.0f0

    # Game over detection on a full-sweep position
    full_cells = fill(Int8(1), 64)
    b_full = OthelloBoard(full_cells, Othello.Black)
    @test game_over(b_full)
    @test winner(b_full) == Othello.Black
    @test !is_valid_move(b_full, PassMove())
    @test available_moves(b_full) == AbstractOthelloMove[]
end
