"""

Welcome to GDB Online.
GDB online is an online compiler and debugger tool for C, C++, Python, Java, PHP, Ruby, Perl,
C#, OCaml, VB, Swift, Pascal, Fortran, Haskell, Objective-C, Assembly, HTML, CSS, JS, SQLite, Prolog.
Code, Compile, Run and Debug online from anywhere in world.

"""
import collections


class Solution:
    def islandPerimeter(self, grid) -> int:
        # https://leetcode.com/problems/island-perimeter/
        # https://neetcode.io/practice
        visit = set()

        def dfs(i, j):
            if i < 0 or j < 0 or i >= len(grid) or j >= len(grid[0]) or grid[i][j] == 0:
                return 1
            if (i, j) in visit:
                return 0
            visit.add((i, j))
            perim = dfs(i + 1, j)
            perim += dfs(i, j + 1)
            perim += dfs(i - 1, j)
            perim += dfs(i, j - 1)
            return perim

        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == 1:
                    return dfs(i, j)

    def isAlienSorted(self, words, order):
        orderedInd = {ch: ind for ind, ch in enumerate(order)}
        for i in range(len(words) - 1):
            w1, w2 = words[i], words[i + 1]
            for j in range(len(w1)):
                # second word length is smaller than first return False
                if j == len(w2):
                    return False
                # Found first mismatch character and first word character order is more than second then return False
                if w1[j] != w2[j]:
                    if orderedInd[w1[j]] > orderedInd[w2[j]]:
                        return False
                    break
        return True

    def findJudge(self, n: int, trust) -> int:
        delta = defaultdict(int)
        for src, dest in trust:
            delta[src] -= 1
            delta[dest] += 1
        for i in range(1, n + 1):
            if delta[i] == n - 1:
                return i
        return -1

    def numIslands(self, grid) -> int:
        # https://neetcode.io/problems/count-number-of-islands
        numIslands = 0
        visit = set()
        rows = len(grid)
        cols = len(grid[0])

        def Bfs(r, c):
            queue = collections.deque()
            queue.append((r, c))
            visit.add((r, c))
            while queue:
                row, col = queue.popleft()
                directions = [[1, 0], [-1, 0], [0, 1], [0, -1]]
                for dr, dc in directions:
                    new_r, new_c = row + dr, col + dc
                    if (
                        new_r in range(rows)
                        and new_c in range(cols)
                        and grid[new_r][new_c] == "1"
                        and (new_r, new_c) not in visit
                    ):
                        queue.append((new_r, new_c))
                        visit.add((new_r, new_c))

        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == "1" and (r, c) not in visit:
                    Bfs(r, c)
                    numIslands += 1
        return numIslands

    def maxAreaOfIsland(self, grid) -> int:
        # https://neetcode.io/problems/max-area-of-island
        # T(n)=O(V*E)  S(n)=O(V*E)
        rows = len(grid)
        cols = len(grid[0])
        visit = set()

        def dfs(r, c):
            if (
                r < 0
                or r == rows
                or c < 0
                or c == cols
                or grid[r][c] == 0
                or (r, c) in visit
            ):
                return 0
            visit.add((r, c))
            return 1 + dfs(r + 1, c) + dfs(r - 1, c) + dfs(r, c + 1) + dfs(r, c - 1)

        area = 0
        for r in range(rows):
            for c in range(cols):
                area = max(area, dfs(r, c))
        return area

    def cloneGraph(self, node):
        # https://neetcode.io/problems/clone-graph
        # T(n)=O(V+E)
        oldToNew = {}

        def dfs(node):
            if node in oldToNew:
                return oldToNew[node]
            copy = Node(node.val)
            oldToNew[node] = copy
            for neighbors in node.neighbors:
                copy.neighbors.append(dfs(neighbors))
            return copy

        return dfs(node) if node else None

    def islandsAndTreasure(self, grid) -> None:
        # https://neetcode.io/problems/islands-and-treasure
        rows = len(grid)
        cols = len(grid[0])
        visit = set()
        queue = deque()

        def AddRooms(r, c):
            if (
                r < 0
                or r == rows
                or c < 0
                or c == cols
                or (r, c) in visit
                or grid[r][c] == -1
            ):
                return
            visit.add((r, c))
            queue.append([r, c])

        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == 0:
                    visit.add((r, c))
                    queue.append([r, c])
        dist = 0
        while queue:
            for i in range(len(queue)):
                r, c = queue.popleft()
                grid[r][c] = dist
                AddRooms(r + 1, c)
                AddRooms(r - 1, c)
                AddRooms(r, c + 1)
                AddRooms(r, c - 1)
            dist += 1

    def orangesRotting(self, grid):
        # https://neetcode.io/problems/rotting-fruit
        nrow, ncol = len(grid), len(grid[0])
        fresh = 0
        time = 0
        q = collections.deque()
        for r in range(nrow):
            for c in range(ncol):
                if grid[r][c] == 2:
                    q.append([r, c])
                if grid[r][c] == 1:
                    fresh += 1
        directions = [[1, 0], [-1, 0], [0, 1], [0, -1]]
        while q and fresh > 0:
            for i in range(len(q)):
                r, c = q.popleft()
                for dr, dc in directions:
                    nr = r + dr
                    nc = c + dc

                    if (
                        nr < 0
                        or nr == nrow
                        or nc < 0
                        or nc == ncol
                        or grid[nr][nc] != 1
                    ):
                        continue
                    grid[nr][nc] = 2
                    q.append([nr, nc])
                    fresh -= 1
            time += 1
        return time if fresh == 0 else -1

    def countSubIslands(self, grid1, grid2) -> int:
        rows = len(grid1)
        cols = len(grid1[0])
        count = 0
        visit = set()

        def dfs(r, c):
            if (
                r < 0
                or r == rows
                or c < 0
                or c == cols
                or grid2[r][c] == 0
                or (r, c) in visit
            ):
                return True

            visit.add((r, c))
            res = True
            if grid1[r][c] == 0:
                res = False
            res = dfs(r + 1, c) and res
            res = dfs(r - 1, c) and res
            res = dfs(r, c + 1) and res
            res = dfs(r, c - 1) and res
            return res

        for r in range(rows):
            for c in range(cols):
                if grid2[r][c] == 1 and (r, c) not in visit and dfs(r, c):
                    count += 1
        return count

    def pacificAtlantic(self, heights):
        # T(n)=O(m*n)
        # https://neetcode.io/problems/pacific-atlantic-water-flow
        # https://leetcode.com/problems/pacific-atlantic-water-flow/description/
        rows = len(heights)
        cols = len(heights[0])
        pac = set()
        atl = set()

        def dfs(r, c, visit, prevHeight):
            # New height get from moving to cell from ocean must be more if less than not consider
            if (
                r < 0
                or c < 0
                or r == rows
                or c == cols
                or (r, c) in visit
                or heights[r][c] < prevHeight
            ):
                return
            visit.add((r, c))
            dfs(r + 1, c, visit, heights[r][c])
            dfs(r - 1, c, visit, heights[r][c])
            dfs(r, c + 1, visit, heights[r][c])
            dfs(r, c - 1, visit, heights[r][c])

        # Start with first and last row
        for c in range(cols):
            dfs(0, c, pac, heights[0][c])
            dfs(rows - 1, c, atl, heights[rows - 1][c])
        # Start with first and last col
        for r in range(rows):
            dfs(r, 0, pac, heights[r][0])
            dfs(r, cols - 1, atl, heights[r][cols - 1])
        res = []
        for r in range(rows):
            for c in range(cols):
                if (r, c) in pac and (r, c) in atl:
                    res.append([r, c])
        return res

    def solve(self, board) -> None:
        # https://neetcode.io/problems/surrounded-regions
        # T(n)=O(m*n)
        # https://leetcode.com/problems/surrounded-regions/description/
        """
        Do not return anything, modify board in-place instead.
        """
        rows = len(board)
        cols = len(board[0])
        # Apply dfs
        def capture(r, c):
            if r < 0 or r == rows or c < 0 or c == cols or board[r][c] != "O":
                return
            board[r][c] = "T"
            # Capturing 'O' vertically and horizontally to boundary 'O'
            capture(r + 1, c)
            capture(r - 1, c)
            capture(r, c + 1)
            capture(r, c - 1)

        # Capture unsurrounded region(O-> T) 'O' present at boundary and connected '0' to 'T'
        for r in range(rows):
            for c in range(cols):
                if r in [0, rows - 1] or c in [0, cols - 1]:
                    capture(r, c)
        # Capture Surrounded region(O->X)
        for r in range(rows):
            for c in range(cols):
                if board[r][c] == "O":
                    board[r][c] = "X"
        # UnCapture unsurrounded region(T->O)
        for r in range(rows):
            for c in range(cols):
                if board[r][c] == "T":
                    board[r][c] = "O"
        return board

    def minReorder(self, n: int, connections) -> int:
        edges = {(a, b) for a, b in connections}
        neighbors = {city: [] for city in range(n)}
        changes = 0
        visit = set()
        for a, b in connections:
            neighbors[a].append(b)
            neighbors[b].append(a)

        def dfs(city):
            nonlocal edges, neighbors, visit, changes
            for neighbor in neighbors[city]:
                if neighbor in visit:
                    continue
                if (neighbor, city) not in edges:
                    changes += 1
                visit.add(neighbor)
                dfs(neighbor)

        visit.add(0)
        dfs(0)
        return changes

    def openLock(self, deadends, target):
        if "0000" in deadends:
            return -1
        queue = collections.deque()
        queue.append(["0000", 0])  # wheels,turns
        visit = set(deadends)

        def Children(wheel):
            # res contain all children of the wheel by changing digit by +1 and -1
            res = []
            for i in range(4):
                digit = str((int(wheel[i]) + 1) % 10)
                # wheel=wheel[:i]+digit+wheel[i+1:]
                res.append(wheel[:i] + digit + wheel[i + 1 :])
                digit = str((int(wheel[i]) + 10 - 1) % 10)
                # wheel=wheel[:i]+digit+wheel[i+1:]
                res.append(wheel[:i] + digit + wheel[i + 1 :])
            return res

        while queue:
            wheel, turn = queue.popleft()
            if wheel == target:
                return turn
            for child in Children(wheel):
                if child not in visit:
                    queue.append([child, turn + 1])
                    visit.add(child)
        return -1

    def eventualSafeNodes(self, graph):
        #https://leetcode.com/problems/find-eventual-safe-states/description/ 
        #https://neetcode.io/practice
        #T(n)=O(V+E)
        safe = {}
        n = len(graph)
        res = []

        def dfs(i):
            if i in safe:
                return safe[i]
            safe[i] = False
            for neigh in graph[i]:
                # If any of the neighbor is not safe then that node safe value not change to True and it returns and if all neighbor is safe then its value change to True and return True
                if not dfs(neigh):
                    return False
            safe[i] = True
            return True

        for i in range(n):
            if dfs(i):
                res.append(i)
        return res
    
    def canFinish(self, numCourses: int, prerequisites) -> bool:
        preMap={i:[] for i in range(numCourses)}
        visit=set()
        for crs,preq in prerequisites:
            preMap[crs].append(preq)
        def dfs(crs):
            if crs in visit:
                return False 
            if preMap[crs]==[]:
                return True 
            visit.add(crs)
            for pre in preMap[crs]:
                if not dfs(pre):
                    return False 
            visit.remove(crs)
            preMap[crs]=[]
            return True 
        for c in range(numCourses):
            if not dfs(c):
                return False 
        return True
    def validTree(self, n: int, edges) -> bool:
        #https://neetcode.io/problems/valid-tree
        #T(n)=O(V+E)
        if not n:
            return True 
        adj={i:[] for i in range(n)}
        
        for v1,v2 in edges:
            adj[v1].append(v2)
            adj[v2].append(v1)
        visit=set()
        def dfs(cur,prev):
            if cur in visit:
                return False 
            visit.add(cur)
            for neigh in adj[cur]:
                if neigh==prev:
                    continue 
                if not dfs(neigh,cur):
                    return False  
            return True 
        return dfs(0,-1) and n==len(visit)
    def checkMove(self, board, rMove: int, cMove: int, color: str) -> bool:
        #https://leetcode.com/problems/check-if-move-is-legal/description/
        #T(n)=O(n)
        ROWS=len(board)
        COLS=len(board[0])
        directions=[[1,0],[-1,0],[0,1],[0,-1],[1,1],[-1,-1],[-1,1],[1,-1]]
        board[rMove][cMove]=color
        def legal(row,col,color,direc):
            #Check for legal line in a given direction
            length=1 
            dr,dc=direc
            row+=dr
            col+=dc
            while(0<=row<ROWS and 0<=col<COLS):
                length+=1
                if board[row][col]=='.':
                    return False 
                if board[row][col]==color:
                    return length>=3 
                row+=dr 
                col+=dc 
            return False
        for d in directions:
            if legal(rMove,cMove,color,d):
                return True 
        return False
        
    def shortestBridge(self, grid) -> int:
        #T(n)=O(V^2)
        #https://leetcode.com/problems/shortest-bridge/description/
        #https://neetcode.io/practice
        N=len(grid)
        direct=[[1,0],[-1,0],[0,1],[0,-1]]
        visit=set()
        def invalid(r,c):
            return r<0 or r>=N or c<0 or c>=N
        def dfs(r,c):
            #For finding first island and add its vertices in visit set
            if invalid(r,c) or not grid[r][c] or (r,c) in visit :
                return 
            visit.add((r,c))
            for dr,dc in direct:
                dfs(r+dr,c+dc)
        def bfs(r,c):
            q=collections.deque(visit)
            res=0
            while q:
                for i in range(len(q)):
                    r,c=q.popleft()
                    for dr,dc in direct:
                        new_r=r+dr 
                        new_c=c+dc 
                        if invalid(new_r,new_c) or (new_r,new_c) in visit:
                            continue
                        if grid[new_r][new_c]:
                            return res 
                        q.append([new_r,new_c])
                        visit.add((new_r,new_c))
                res+=1 
        for r in range(N):
            for c in range(N):
                if grid[r][c]:
                    dfs(r,c)
                    return bfs(r,c)
                    
    def shortestPathBinaryMatrix(self, grid) -> int:
        #https://leetcode.com/problems/shortest-path-in-binary-matrix/description/ 
        #https://neetcode.io/practice
        N=len(grid)
        visit=set((0,0))
        queue=collections.deque([(0,0,1)])
        direct=[[1,0],[-1,0],[0,1],[0,-1],[1,1],[1,-1],[-1,-1],[-1,1]]
        while queue:
            r,c,length=queue.popleft()
            if min(r,c)<0 or max(r,c)>=N or grid[r][c]:
                continue 
            if r==N-1 and c==N-1:
                return length 
            for dr,dc in direct:
                new_r=r+dr 
                new_c=c+dc 
                if (new_r,new_c) not in visit:
                    queue.append([new_r,new_c,length+1])
                    visit.add((new_r,new_c))
        return -1
    def countComponents(self, n, edges) -> int:
        #https://neetcode.io/problems/count-connected-components
        #Method1-> DFS
        adjacencyList={i:[] for i in range(n)}
        count_component=0
        for v1,v2 in edges:
            adjacencyList[v1].append(v2)
            adjacencyList[v2].append(v1)
        visited=[False]*n 
        def dfs(node,visited):
            visited[node]=True 
            for neighbor in adjacencyList[node]:
                if not visited[neighbor]:
                    dfs(neighbor,visited)

        for i in range(n):
            if not visited[i]:
                count_component+=1 
                dfs(i,visited)
        return count_component
    
        
        
        



solution = Solution()
# Rotten Oranges:------------------------------
# grid=[[2,1,1],[1,1,0],[0,1,1]]
# print("Time for rotten=",solution.orangesRotting(grid))


# #Count SubIsland------------------------------------------
# grid1 = [[1,1,1,0,0],[0,1,1,1,1],[0,0,0,0,0],[1,0,0,0,0],[1,1,0,1,1]]
# grid2 = [[1,1,1,0,0],[0,0,1,1,1],[0,1,0,0,0],[1,0,1,1,0],[0,1,0,1,0]]
# print("Count SubLsland=",solution.countSubIslands(grid1,grid2))


# #pacificAtlantic-------------------------------
# heights = [[1,2,2,3,5],[3,2,3,4,4],[2,4,5,3,1],[6,7,1,4,5],[5,1,1,2,4]]
# print("pacificAtlantic=",solution.pacificAtlantic(heights))


# unsurrounded Region
# board = [["X","X","X","X"],["X","O","O","X"],["X","X","O","X"],["X","O","X","X"]]

# print("Surrounded region=",solution.solve(board))

# #Reorder Routes------------------------------
# connections=[[0,1],[1,3],[2,3],[4,0],[4,5]]
# print("Number of changes=",solution.minReorder(6,connections))


# #openLock-------------------------------------------
# deadends = ["0201","0101","0102","1212","2002"]
# target = "0202"
# print("openLock  matched turns=",solution.openLock(deadends,target))


# eventualSafeNodes-------------------------------------------------
# graph = [[1, 2], [2, 3], [5], [0], [5], [], []]
# print("Safe Nodes=", solution.eventualSafeNodes(graph))

#canFinish--------------------------------------------
# numCourses = 2
# prerequisites = [[1,0],[0,1]]
# print("canFinish=",solution.canFinish(numCourses,prerequisites))

#validTree------------------------------------------------
# n = 5
# edges = [[0, 1], [0, 2], [0, 3], [1, 4]]
# print("validTree=",solution.validTree(n,edges))


#checkMove---------------------------------------------------
# board = [[".",".",".",".",".",".",".","."],[".","B",".",".","W",".",".","."],[".",".","W",".",".",".",".","."],[".",".",".","W","B",".",".","."],[".",".",".",".",".",".",".","."],[".",".",".",".","B","W",".","."],[".",".",".",".",".",".","W","."],[".",".",".",".",".",".",".","B"]]
# rMove = 4
# cMove = 4
# color = "W"
# print("checkMove=",solution.checkMove(board,rMove,cMove,color))

#shortestBridge---------------------------------------------
# grid = [[0,1,0],[0,0,0],[0,0,1]]
# print("shortestBridge=",solution.shortestBridge(grid))

#shortestPathBinaryMatrix-----------------------------------
# grid = [[0,0,0],[1,1,0],[1,1,0]] 
# print("shortestPathBinaryMatrix=",solution.shortestPathBinaryMatrix(grid))

#countComponents-------------------------------------------
n=6
edges=[[0,1], [1,2], [2,3], [4,5]]
print("countComponents=",solution.countComponents(n,edges))