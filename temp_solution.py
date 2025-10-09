# Read input
import sys

def main():
    try:
        # Read n and k from stdin
        n = int(sys.stdin.readline())
        k = int(sys.stdin.readline())
        
        # Check for edge cases
        if n <= 0 or k <= 0:
            print("Invalid input: n must be greater than 0.")
            return
        
        # Calculate q
        q = n // k
        
        # Initialize an empty list to store the result
        result = []
        
        # Loop through the list and extract every kth element
        for i in range(n):
            if (i + 1) % k == 0:
                result.append(x[i])
                
        # Print the result
        print(" ".join(map(str, result)))

    except ValueError:
        print("Invalid input: Please enter valid integers.")
        return

if __name__ == "__main__":
    main()