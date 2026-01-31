import sys
def Fibonacci(dp,n):
    dp=[-1]*(n+1)
    if(n<=1):
        return n 
    if dp[n]!=-1:
        return dp[n]
    dp[n]=Fibonacci(dp,n-1)+Fibonacci(dp,n-2)
    return dp[n]
def Fibonacci2(n):
    prev1=0
    prev2=1 
    for i in range(2,n+1):
        cur=prev1+prev2
        prev1=prev2
        prev2=cur
    return prev2
  #Tabulation Method--Climbing Stairs  
def ClimbingStairs(n):
    dp=[-1]*(n+1)
    dp[0]=1 
    dp[1]=1

    for i in range(2,n+1):
        dp[i]=dp[i-1]+dp[i-2]
   

    return dp[n]
#Frog Jumps-------------------
#Method1-Memorization(Recursive)---------------------
def FrogJumps(arr,dp,ind):
    #base condition
    if ind==0:
        return 0
    if dp[ind]!=-1:
        return dp[ind]
    jump_two=sys.maxsize
    jump_one=FrogJumps(arr,dp,ind-1)+abs(arr[ind]-arr[ind-1])
    if ind>1:
        jump_two=FrogJumps(arr,dp,ind-2)+abs(arr[ind]-arr[ind-2])
    dp[ind]=min(jump_one,jump_two)
    return dp[ind]

#Method2-Tabulation(Iterative)Frog Jump-----------------------------------
def FrogJumps(arr,n):
    dp=[-1 for i in range(n+1)]
    dp[0]=0 #Base Condition of Recursive approach 
    for i in range(1,n+1):
        jump_two=float("inf")
        jump_one=dp[i-1]+abs(arr[i]-arr[i-1])
        if i>1:
            jump_two=dp[i-2]+abs(arr[i]-arr[i-2])
        dp[i]=min(jump_one,jump_two)
    return dp[n]
    
#Maximum Sum of Non Adjacent Elements------------------

def MaxSum(arr,n):
    dp=[-1 for i in range(n)]
    return MaxSumDP(n-1,arr,dp)
def MaxSumDP(ind,arr,dp):
    if dp[ind]!=-1:
        return dp[ind]
    if ind==0:
        return arr[ind]
    if ind<0:
        return 0 
    pick=arr[ind]+MaxSumDP(ind-2,arr,dp)
    notpick=0+MaxSumDP(ind-1,arr,dp)
    dp[ind]=max(pick,notpick)
    return dp[ind]
    
    
def MaxSumTabulation(arr,n,dp):
    dp[0]=arr[0]
    for i in range(1,n):
        pick=arr[i]
        if i>1:
            pick+=dp[i-2]
        notpick=0+dp[i-1]
        dp[i]=max(pick,notpick)
    return dp[n-1]
    
def MaxSumSpaceOptimized(arr,n):
    prev2=0 
    prev1=arr[0]
    for i in range(1,n):
        pick=arr[i]
        if i>1:
            pick+=prev2
        notpick=prev1 
        curr=max(pick,notpick)
        prev2=prev1
        prev1=curr
    return prev1 
    
#House Robber----------------------------------------------
#https://takeuforward.org/data-structure/dynamic-programming-house-robber-dp-6/--------------
def SolveRobber(arr):
    prev1=arr[0]
    prev2=0 
    n=len(arr)
    for i in range(1,n):
        pick=arr[i]
        if i>1:
            pick+=prev2 
        notpick=prev1 
        cur=max(pick,notpick)
        prev2=prev1 
        prev1=cur 
    return prev1
        

def HouseRobber(arr,n):
    arr_st=[]
    arr_end=[]
    if(n==1):
        return arr[0]
    for i in range(n):
        if(i!=n-1):
            arr_st.append(arr[i])
        if(i!=0):
            arr_end.append(arr[i])
    max1=SolveRobber(arr_st)
    max2=SolveRobber(arr_end)
    return max(max1,max2) 
    
def Dp_Training(day,last,points,dp):
    if(day==0):
        maxi=-1 
        for i in range(3):
            if i!=last:
                maxi=max(maxi,points[0][i])
        dp[day][last]=maxi
        return dp[day][last] 
    maxi=-1 
    for i in range(3):
        if i!=last:
            train_points=points[day][i]+Dp_Training(day-1,i,points,dp)
            maxi=max(maxi,train_points)
    dp[day][last]=maxi
    return dp[day][last]
            
    
def LastTraining(points,n):
    dp=[[-1 for i in range(4)] for i in range(n)]
    return Dp_Training(n-1,3,points,dp)
    

        
    
            
"""
#Fibonacci-----------------------------------
x=int(input("Enter the term of fibonacci"))
dp=[]
#print(f"Fibonacci number at {x} index= {Fibonacci(dp,x)}")
print(f"Fibonacci number at {x} index= {Fibonacci2(x)}")
"""

"""
# Climbing Series-------------------------------------
x=int(input("Enter the stair"))

#ClimbingStairs(x)
print(f"Ways to reach at stair {x}= {ClimbingStairs(x)}")
"""
#Frog Jumps---------------------------------------------

# arr=[30, 10, 60, 10, 60, 50]
# n=len(arr)
# dp=[-1]*(n)
# #print(f"Number of jumps={FrogJumps(arr,dp,n-1)}")
# print(f"Number of jumps={FrogJumps(arr,n-1)}")


#MaxSum----------------------------------------
# arr=[30, 10, 60, 10, 60, 50]
# n=len(arr)
# #Method1-> Memorization-------------------
# print("Maxum of subsequence non adjacent=",MaxSum(arr,n))
# #Method2 Tabulation--------------------
# dp=[-1 for _ in range(n)]
# print("Maxum of subsequence non adjacent=",MaxSumTabulation(arr,n,dp))

# #Method3 Space Optimized--------------------
# print("Maxum of subsequence non adjacent=",MaxSumSpaceOptimized(arr,n))

#HouseRobber-----------------------------------------------
#T(n)=O(n)   S(n)=O(1)
# arr = [1, 5, 1, 2, 6]
# n = len(arr)
# print("Max Robber Amount=",HouseRobber(arr,n))


#Last Training-------------------------------------------------------------
#https://takeuforward.org/data-structure/dynamic-programming-ninjas-training-dp-7/
points = [[10, 40, 70],
              [20, 50, 80],
              [30, 60, 90]]
n=len(points)
print("Max Last TRaining Points=",LastTraining(points,n))










