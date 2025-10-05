def game_of_stones(N):
    """
    Alice and Bob play optimally to form a sequence of N stones where each player can take exactly two consecutive stones,
    until there are no consecutive stones left. The first player wins if the number of stone left is odd, otherwise Bob wins.
    
    Args:
    N (int): The number of stones on the ground.
    
    Returns:
    str: "Alice" or "Bob".
    """
    # Initialize variables
    current = 1
    result = None
    
    # Check if there are any stones left to play
    while current < N:
        # If there's only one stone left, Alice wins
        if current == N - 1:
            return "Alice"
        
        # Take the next two consecutive stones
        current += 2
        
        # Check if the number of stones is odd
        if current % 2 != 0:
            result = "Bob"
    
    return result

# Read input from stdin
N = int(input())
# Call the game_of_stones function with N and print the result
print(game_of_stones(N))