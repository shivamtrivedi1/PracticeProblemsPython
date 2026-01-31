'''

Welcome to GDB Online.
GDB online is an online compiler and debugger tool for C, C++, Python, Java, PHP, Ruby, Perl,
C#, OCaml, VB, Swift, Pascal, Fortran, Haskell, Objective-C, Assembly, HTML, CSS, JS, SQLite, Prolog.
Code, Compile, Run and Debug online from anywhere in world.

'''
#When revise question check time complexity and space complexity also
from collections import deque
class Node:
    def __init__(self,data):
        self.data=data
        self.left=None 
        self.right=None 
class Solution:
    def getHeight(self,root):
        if not root:
            return 0 
        left_ht=self.getHeight(root.left)
        right_ht=self.getHeight(root.right)
        return 1+max(left_ht,right_ht)
        
        
    def CheckBalanced(self,root):
        if not root:
            return True 
        left_ht=self.getHeight(root.left) 
        right_ht=self.getHeight(root.right)
        if left_ht-right_ht<=1 and self.CheckBalanced(root.left) and self.CheckBalanced(root.right):
            return True 
        return False 
    def InvertTree(self,root):
        #T(n)=O(n) and S(n)=O(n)
        if not root : return None 
        temp=root.left 
        root.left=root.right 
        root.right=temp 
        self.InvertTree(root.left)
        self.InvertTree(root.right)
        return root
    def PrintInorder(self,root):
        if root:
            self.PrintInorder(root.left)
            print(root.data,end=",")
            self.PrintInorder(root.right)
    def maxDepth1(self, root):
        #https://neetcode.io/problems/depth-of-binary-tree
        if not root:
            return 0 
        left=self.maxDepth1(root.left)
        right=self.maxDepth1(root.right)
        return 1+max(left,right)
    def maxDepth2(self, root) -> int:
        if not root:
            return 0 
        level=0
        queue=deque()
        queue.append(root)
        while queue:
            for  i in range(len(queue)):
                node=queue.popleft()
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            level+=1 
        return level
        
    def diameterOfBinaryTree(self, root):
        self.res=0
        def dfs(curr):
            if not curr:
                return 0
            left=dfs(curr.left)
            right=dfs(curr.right)
            #nonlocal res
            self.res=max(self.res,left+right)
            return 1+max(left,right)
        dfs(root)
        return self.res
        
    def isBalanced(self, root):
        #https://neetcode.io/problems/balanced-binary-tree
        def dfs(root):
            if not root:
                return [True,0]
            left=dfs(root.left)
            right=dfs(root.right)
            balanced=left[0] and right[0] and abs(left[1]-right[1])<=1 
            return [balanced,1+max(left[1],right[1])]
        return dfs(root)[0]
    def SameTree(self,r1,r2):
        #https://neetcode.io/problems/same-binary-tree
        #T(n)=O(n+m)-> Sum of sizes of tree
        if r1 is None and r2 is None:
            return True 
        if r1 and r2 and r1.data==r2.data:
            return self.SameTree(r1.left,r1.left) and self.SameTree(r1.right,r2.right)
        else:
            return False
    def isSubtree(self, root, subRoot) -> bool:
        if not subRoot:
            return True 
        if not root:
            return False 
        if self.SameTree(root,subRoot):
            return True 
        return (self.isSubtree(root.left,subRoot) or self.isSubtree(root.right,subRoot))
        
    def sortedArrayToBST(self, nums):
        #https://leetcode.com/problems/convert-sorted-array-to-binary-search-tree/description/
        n=len(nums)
        if not n:
            return None
        mid=(n)//2    #(n-1)//2 is more efficient 
        root=Node(nums[mid])
        root.left=self.sortedArrayToBST(nums[:mid])
        root.right=self.sortedArrayToBST(nums[mid+1:])
        return root
        
    def mergeTrees(self, root1, root2):
        #https://leetcode.com/problems/merge-two-binary-trees/ 
        #T(n)=O(n+m)
        if not root1 and not root2:
            return None 
        v1=root1.data if root1 else 0 
        v2=root2.data if root2 else 0 
        root=Node(v1+v2)
        root.left=self.mergeTrees(root1.left if root1 else None,root2.left if root2 else None)
        root.right=self.mergeTrees(root1.right if root1 else None,root2.right if root2 else None)
        return root
    def hasPathSum(self, root, targetSum):
        #https://leetcode.com/problems/path-sum/
        #https://neetcode.io/practice
        #T(n)=O(n)
        def dfs(node,csum):
            if node is None:
                return False 
            csum+=node.val
            if not node.left and not node.right:
                return targetSum==csum
           
            left=dfs(node.left,csum)
            right=dfs(node.right,csum)
            return left or right 
        return dfs(root,0)
        
    def RangeSum(self,root,low,high):
        if not root:
            return 0 
        if root.data<low:
            return self.RangeSum(root.right,low,high)
        if root.data>high:
            return  self.RangeSum(root.left,low,high)
        return root.data+self.RangeSum(root.right,low,high)+self.RangeSum(root.left,low,high)
        
    def leafSimilar(self, root1, root2) -> bool:
        #https://leetcode.com/problems/leaf-similar-trees/
        #https://neetcode.io/practice
        #T(n)=O(n+m)
        def dfs(root,leaf):
            if not root:
                return 
            if not root.left and not root.right:
                leaf.append(root.data)
                return 
            dfs(root.left,leaf)
            dfs(root.right,leaf)
        leaf1=[]
        leaf2=[]
        dfs(root1,leaf1)
        dfs(root2,leaf2)
        return leaf1==leaf2
        
    def evaluateTree(self, root) -> bool:
        if not root.left and not root.right:
            return root.data==1
        if root.data==2:
            return self.evaluateTree(root.left) or self.evaluateTree(root.right)
        elif root.data==3:
            return self.evaluateTree(root.left) and self.evaluateTree(root.right)
            
    def insertIntoBST(self, root, val):
        #T(n)=O(h)
        #https://leetcode.com/problems/insert-into-a-binary-search-tree/
        if not root:
            return TreeNode(val)
        if root.val>val:
            #If the left child is None, the recursive call creates a new node and returns it.
            #If the left child is not None, the recursive call continues to traverse the left subtree and returns the modified subtree.
            root.left=self.insertIntoBST(root.left,val)
        elif root.val<val:
            root.right=self.insertIntoBST(root.right,val)
        return root
    def deleteNode(self, root, key: int):
        #https://leetcode.com/problems/delete-node-in-a-bst/
        #https://neetcode.io/practice
        if not root:
            return 
        if key>root.data:
            root.right=self.deleteNode(root.right,key)
        elif key<root.data:
            root.left=self.deleteNode(root.left,key)
        else:
            if not root.left:
                return root.right
            elif not root.right:
                return root.left 
            else:
                cur=root.right
                while cur.left:
                    cur=cur.left 
                root.val=cur.val 
                root.right=self.deleteNode(root.right,root.val)
        return root
        
    def levelOrder(self, root):
        #https://neetcode.io/problems/level-order-traversal-of-binary-tree
        queue=collections.deque()
        res=[]
        if root:
            queue.append(root)
        while queue:
            lev_ele=[]
            que_len=len(queue)
            for i in range(que_len):
                node=queue.popleft()
                if node:
                    lev_ele.append(node.data)
                if node and  node.left:
                    queue.append(node.left)
                if node and node.right:
                    queue.append(node.right)
            res.append(lev_ele)
        return res 
        
    def rightSideView(self, root):
        #https://neetcode.io/problems/binary-tree-right-side-view
        res=[]
        queue=collections.deque([root])
        while queue:
            len_q=len(queue)
            rightSideView=None
            for i in range(len_q):
                node=queue.popleft()
                if node:
                    rightSideView=node 
                    queue.append(node.left)
                    queue.append(node.right)
            if rightSideView:
                res.append(rightSideView.val)
        return res
        
    def minDiffInBST(self, root) -> int:
        diff=10000000
        prev=None 
        def dfs(node):
            if not node:
                return 
            dfs(node.left)
            nonlocal diff,prev
            if prev:
                diff=min(diff,node.val-prev.val)
            prev=node
            dfs(node.right)
        dfs(root)
        return diff
        
        
    def isSymmetric(self, root) -> bool:
        #https://leetcode.com/problems/symmetric-tree/
        #https://neetcode.io/practice
        if root is None:
            return True 
        else:
            return  self.dfs(root.left,root.right)
        
    def dfs(self,left,right):
        if not left and not right:
            return True 
        if not left or not right:
            return False 
        return left.val==right.val and self.dfs(left.left,right.right) and self.dfs(left.right,right.left)
        
    def goodNodes(self, root):
        #https://neetcode.io/problems/count-good-nodes-in-binary-tree
        def dfs(node,maxVal):
            if not node:
                return 0 
            res=1 if node.val>=maxVal else 0 
            maxVal=max(maxVal,node.val)
            res+=dfs(node.left,maxVal)
            res+=dfs(node.right,maxVal)
            return res 
        return dfs(root,root.val)
        
    def buildTree(self, inorder, postorder):
        #https://leetcode.com/problems/construct-binary-tree-from-inorder-and-postorder-traversal/
        #T(n)=O(n^2)
        if not inorder:
            return None 
        root=Node(postorder.pop())
        idx=inorder.index(root.val)
        root.right=self.buildTree(inorder[idx+1:],postorder)
        root.left=self.buildTree(inorder[:idx],postorder)
        return root
        
        
    def buildTree(self,inorder,postorder):
        #T(n)=O(n)
        idxMap={v:i for i,v in enumerate(inorder)}
        def helper(l,r):
            if l>r:
                return None 
            root=Node(postorder.pop())
            idx=idxMap[root.val]
            root.right=helper(idx+1,r)
            root.left=helper(l,idx-1)
            return root 
        return helper(0,len(inorder)-1)
        
    def widthOfBinaryTree(self, root) -> int:
        queue=collections.deque()
        prevNum=1
        prevlevel=0
        res=0
        queue.append([root,1,0])   #(node,indexNUm,level)
        while queue:
            node,num,level=queue.popleft()
            #level changes to find start node index of that level
            if level>prevlevel:
                prevlevel=level 
                prevNum=num 
            res=max(res,num-prevNum+1)
            if node.left:
                queue.append([node.left,2*num,level+1])
            if node.right:
                queue.append([node.right,2*num+1,level+1])
        return res
        
    def lowestCommonAncestor(self, root, p, q):
        #https://neetcode.io/problems/lowest-common-ancestor-in-binary-search-tree
        #T(n)=O(logn)
        #S(n)=O(1)
        cur=root 
        while cur:
            if p.val>cur.val and q.val>cur.val:
                cur=cur.right
            elif p.val<cur.val and q.val<cur.val:
                cur=cur.left 
            else:
                return cur
    def isValidBST(self, root): 
        #https://neetcode.io/problems/valid-binary-search-tree  
        #T(n)=O(n)
        def Valid(node,left_val,right_val):
            if not node:
                return True 
            if not(node.val>left_val  and node.val<right_val):
                return False 
            return Valid(node.left,left_val,node.val) and Valid(node.right,node.val,right_val)
            return Valid(root,float("-inf"),float("inf"))
            
    def kthSmallest(self, root, k):
        #https://neetcode.io/problems/kth-smallest-integer-in-bst  
        #T(n)=O(n)
        stack=[]
        cnt=0 
        cur=root
        while stack or cur:
            while cur:
                stack.append(cur)
                cur=cur.left 
            cur=stack.pop()
            cnt+=1
            if cnt==k:
                return cur.val 
            cur=cur.right
            
            
    def numTrees(self, n: int) -> int:
        #https://leetcode.com/problems/unique-binary-search-trees/description/  
        #https://neetcode.io/practice
        # for n=3(0 to 3) numTrees[1,1,1,1]->[1,1,2,5]
        #numTrees(4)=numTrees(0)*numTrees(3)+numTrees(1)*numTrees(2)+numTrees(2)*numTrees(1)+numTrees(3)*numTrees(0)
        numTree=[1]*(n+1)
        #nodes-> Number of nodes choosen for getting number of BST 
        for nodes in range(2,n+1):
            total=0 
            # out of given nodes which one is consider as root then 
            # find number of BST if that node is root and keep changing root and 
            # find total number of BST for given number of nodes
            for root in range(1,nodes+1):
                left=root-1 
                right=nodes-root 
                total+=numTree[left]*numTree[right]
            numTree[nodes]=total 
        return numTree[n]
        
        
    def minTime(self, n: int, edges, hasApple) -> int:
        #https://leetcode.com/problems/minimum-time-to-collect-all-apples-in-a-tree/description/`` 
        adjList={i:[] for i in range(n)}
        for par,child in edges:
            adjList[par].append(child)
            adjList[child].append(par)
        def dfs(cur,par):
            time=0 
            for child in adjList[cur]:
                if child==par:
                    continue 
                childTime=dfs(child,cur)
                if childTime>0 or hasApple[child]:
                    time+=2+childTime 
            return time 
        return dfs(0,-1)
        
    def sumNumbers(self, root) -> int:
        #https://leetcode.com/problems/sum-root-to-leaf-numbers/----------
        def dfs(cur,num):
            if not cur:
                return 0 
            num=num*10+cur.data 
            if not cur.left and not cur.right:
                return num 
            return dfs(cur.left,num)+dfs(cur.right,num)
        return dfs(root,0)
    def rob(self, root) -> int:
        #https://leetcode.com/problems/house-robber-iii/
        def dfs(root):
            if not root:
                return [0,0]
            leftPair=dfs(root.left)
            rightPair=dfs(root.right)
            withRoot=root.data+leftPair[1]+rightPair[1]
            withoutRoot=max(leftPair)+max(rightPair)
            return [withRoot,withoutRoot]
        return max(dfs(root))
        
        
    def flipEquiv(self, root1, root2) -> bool:
        #https://leetcode.com/problems/flip-equivalent-binary-trees/
        if not root1 or not root2:
            return not root1 and not root2 
        if root1.val!=root2.val:
            return False 
        a=self.flipEquiv(root1.left,root2.left) and self.flipEquiv(root1.right,root2.right)
        return a or self.flipEquiv(root1.left,root2.right) and self.flipEquiv(root1.right,root2.left)
        
    def allPossibleFBT(self, n: int):
        #https://leetcode.com/problems/all-possible-full-binary-trees/
        dp={0:[],1:[TreeNode()]}
        
        def backtrack(n):
            if n in dp:
                return dp[n]
            res=[]

            for l in range(n):
                r=n-l-1
                leftTree,rightTree=backtrack(l),backtrack(r)
                for t1 in leftTree:
                    for t2 in rightTree:
                        res.append(TreeNode(0,t1,t2))
            dp[n]=res
            return res 
        return backtrack(n)
        
    def findBottomLeftValue(self, root) -> int:
        #https://leetcode.com/problems/find-bottom-left-tree-value/
        queue=deque([root])
        while queue:
            node=queue.popleft()
            if node.right:queue.append(node.right)
            if node.left:queue.append(node.left)
        return node.val
        
        



                
    

        
        
    
        
            
    
        
    
    
        
root = Node(1)
root.left = Node(2)
root.right = Node(3)
root.left.left = Node(4)
root.left.right = Node(5)
root.left.right.right = Node(6)
root.left.right.right.right = Node(7)
solution=Solution()
print("Height of leaf=",solution.getHeight(root.left.left ))
#Check Balanced -------------------------------
# if solution.CheckBalanced(root):
#     print("Tree Is Balanced")
# else:
#     print("Tree Is not Balanced")
    
#INvert Tree
# solution.PrintInorder(root)
# invertedRoot=solution.InvertTree(root)
# solution.PrintInorder(invertedRoot)

# #Max Depth-----------------------------
# #Method1(Recursion)-------------
# print("Max Depth=",solution.maxDepth1(root))
# #Method2(BFS)----
# print("Max Depth=",solution.maxDepth2(root))

#Diameter of Tree----------------------------------
# print("Diameter of tree=",solution.diameterOfBinaryTree(root))


#isBalanced-------------------------------------------------
# print("Is tree Balanced=",solution.isBalanced(root))

#SameTree--------------------------------------------
# root2 = Node(1)
# root2.left = Node(2)
# root2.right = Node(3)
# root2.left.left = Node(4)
# root2.left.right = Node(5)
# root2.left.right.right = Node(6)
# root2.left.right.right.right = Node(7)
# print("Tree is same=",solution.SameTree(root,root2))

#isSubtree------------------------
# root3 = Node(2)
# root3.left = Node(4)
# root3.right = Node(5)
# print("Tree is subtree=",solution.isSubtree(root,root3))

#RangeSum----------------------------
# root4=Node(10)
# root4.left=Node(3)
# root4.right=Node(12)
# root4.left.left=Node(4)
# root4.left.right=Node(7)
# root4.right.left=Node(11)
# root4.right.right=Node(13)
# print("Range Sum=",solution.RangeSum(root4,3,6))

#numTrees--------------------------------------
# n=3
# print("numTrees=",solution.numTrees(n))


#minTime----------------------------------------
# n = 7
# edges = [[0,1],[0,2],[1,4],[1,5],[2,3],[2,6]]
# hasApple = [False,False,True,False,False,True,False]
# print("Minimum Time to collect all apples=",solution.minTime(n,edges,hasApple))

#sumNumbers--------------------------------------------
# print("Sum of root to leaf numbers=",solution.sumNumbers(root))

#House Robber III
root5=Node(3)
root5.left=Node(4)
root5.right=Node(5)
root5.left.left=Node(1)
root5.left.right=Node(3)
root5.right.right=Node(1)
print("Maximum amount of money the thief can rob=",solution.rob(root5))

        
        