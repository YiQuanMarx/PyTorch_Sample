def minMoves(maze, x, y):
    n, m = len(maze), len(maze[0])
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    
    def is_valid_move(i, j):
        return 0 <= i < n and 0 <= j < m and maze[i][j] != 1
    
    def bfs(start, end):
        queue = [(start, 0)]
        visited = set()
        visited.add(start)
        
        while queue:
            (i, j), steps = queue.pop(0)
            if (i, j) == end:
                return steps
            
            for di, dj in directions:
                ni, nj = i + di, j + dj
                if is_valid_move(ni, nj) and (ni, nj) not in visited:
                    queue.append(((ni, nj), steps + 1))
                    visited.add((ni, nj))
                    
        return -1
    
    # Find positions of gold coins
    gold_positions = [(i, j) for i in range(n) for j in range(m) if maze[i][j] == 2]
    
    total_steps = 0
    current_position = (0, 0)
    
    for gold_position in gold_positions:
        steps = bfs(current_position, gold_position)
        if steps == -1:
            return -1
        total_steps += steps
        current_position = gold_position
    
    final_steps = bfs(current_position, (x, y))
    if final_steps == -1:
        return -1
    
    return total_steps + final_steps

# Example usage:
maze = [[0, 2, 1], [1, 2, 0], [1, 0, 0]]
x = 1
y = 1
result = minMoves(maze, x, y)
print(result)  # Output: 2
