using Othello

# Test blank vector initialization
blank = Othello.initialize_feature_vector()
println("Blank feature vector: ", blank)

# Test board-based initialization
b = initialize_othello_board()
fv = Othello.initialize_feature_vector(b)
println("Feature vector from board (first 10): ", fv[1:10])
println("Non-zero indices: ", findall(!iszero, fv))

# Test in-place update
out = zeros(Float32, 64)
Othello.update_feature_vector!(out, b)
println("In-place updated (first 10): ", out[1:10])
println("Sum of absolute values: ", sum(abs.(out)))
