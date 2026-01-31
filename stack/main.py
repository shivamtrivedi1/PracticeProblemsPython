'''

Welcome to GDB Online.
GDB online is an online compiler and debugger tool for C, C++, Python, Java, PHP, Ruby, Perl,
C#, OCaml, VB, Swift, Pascal, Fortran, Haskell, Objective-C, Assembly, HTML, CSS, JS, SQLite, Prolog.
Code, Compile, Run and Debug online from anywhere in world.

'''
class StackSolution:
    def NextGreaterElement(self,list1):
        stack=[]
        n=len(list1)
        nge=[-1]*n
        for i in range(2*n-1,-1,-1):
            while(stack and stack[-1]<=list1[i%n]):
                stack.pop()
            if(i<n):
                if stack:
                    nge[i]=stack[-1]

            stack.append(list1[i%n])
        return nge
        
    def TrappingRainWater1(self,arr):
        n=len(arr)
        trapped_water=0
        for i in range(n):
            left_max=0
            right_max=0
            #left_max
            j=i
            while(j>=0):
                left_max=max(left_max,arr[j])
                j-=1 
            #right_max
            j=i
            while(j<n):
                right_max=max(right_max,arr[j])
                j+=1 
            trapped_water+=min(left_max,right_max)-arr[i]
        return trapped_water
    def TrappingRainWater2(self,arr):
        n=len(arr)
        pre=[0]*n
        suf=[0]*n
        trapped_water=0
        # Prefix 
        pre[0]=arr[0]
        for i in range(1,n):
            
            pre[i]=max(pre[i-1],arr[i])
        
        # Suffix 
        suf[n-1]=arr[n-1]
        for i in range(n-2,-1,-1):
            suf[i]=max(suf[i+1],arr[i])
        for i in range(n):
            trapped_water+=min(pre[i],suf[i])-arr[i]
        return trapped_water
        
    def TrappingRainWater3(self,A):
        l=0
        r=len(A)-1
        l_max=0
        r_max=0
        trap_water=0
        while(l<=r):
            if(A[l]<=A[r]):
                if(A[l]>=l_max):
                    l_max=A[l]
                else:
                    trap_water+=l_max-A[l] 
                l+=1
            else:
                if(A[r]>=r_max):
                    r_max=A[r]
                else:
                    trap_water+=r_max-A[r]
                r-=1
        return trap_water
        
    def largestRectangleArea(self, heights):

     
        n=len(heights)
        lsmall=[]
        rsmall=[0]*n
        stack=[]
        #Create lsmall->Boundary at left
        for i in range(n):
            print(f"i1={i}")
            print(stack)
            while(len(stack)!=0 and heights[stack[-1]]>=heights[i]):
                # print(f"i1={i}")
                # print(f"heights[i]={heights[i]}")
                print(f"stack[-1]={stack[-1]}")
                 
                print(f"heights[stack[-1]]={heights[stack[-1]]}")
                stack.pop()
                #stack become 
            if not stack: 
                lsmall.insert(i,0)
            else:
                lsmall.insert(i,stack[-1]+1)
            stack.append(i)
        while len(stack)>0:
            stack.pop()

        #Create rsmall->Boundary at right
        print(f"Stack after lsmall{stack}")
        for i in range(n-1,-1,-1):
            while(stack and heights[stack[-1]]>=heights[i]):
                stack.pop()
                #stack become 
            if len(stack)==0: 
                rsmall[i]=n-1
              
            else:
                rsmall[i]=stack[-1]-1
               
            stack.append(i)
        maxArea=0
        print("lsmall=")
        print(lsmall)
        print("rsmall=")
        print(rsmall)
        print("heights")
        print(heights)
        for i in range(n):
            print(f"i={i}")
            # print(f"heights[i]={heights[i]}")
            # print(f"rsmall[i]={rsmall[i]}")
            # print(f"lsmall[i]={lsmall[i]}")
            maxArea=max(maxArea,heights[i]*(rsmall[i]-lsmall[i]+1))
            print(f"maxArea={maxArea}")
        return maxArea
        
    def generateParenthesis(self, n):
        stack = []
        res = []

        def backtrack(openN,closedN):
            if openN==closedN==n:
                res.append("".join(stack))
                return 

            if openN<n:
                stack.append("(")
                backtrack(openN+1,closedN)
                stack.pop()
            if openN>closedN:
                stack.append(")")
                backtrack(openN,closedN+1)
                stack.pop()
        backtrack(0,0)
        return res
        
        
        
    def dailyTemperatures(self, temperatures):
        res = [0]*len(temperatures)
        stack = []

        for ind,t in enumerate(temperatures):
            while stack and t>stack[-1][0]:
                stackTemp,stackInd = stack.pop()
                indDiff = ind - stackInd 
                res[stackInd]= indDiff 
            stack.append((t,ind)) 
        return res
        
        

    def carFleet(self, target, position, speed):
        pair = [(p,s) for p,s in zip(position,speed)]
        timeStack=[]
        for p,s in sorted(pair)[::-1]:
            diff= (target-p)/s
            timeStack.append(diff)
            #Pop when old(rightmost) coming time is more(speed less) than new one
            if len(timeStack)>=2 and timeStack[-2]>=timeStack[-1]:
                timeStack.pop() 
        return len(timeStack)




        
            
                    
            
                
                
            
            

            
        
        
        
        
if __name__=="__main__":
    #Next Greater Element----------------------------------------------
    stack_obj=StackSolution()
    # print("Number of elements in list")
    # n=int(input())
    # print("Enter the elements in list")
    # l=[int(input()) for i in range(n)]
    
    # print(stack_obj.NextGreaterElement(l))
    
    #Trapped Rain water---------------------------------
    
    
    # print("Number of elements in list")
    # n=int(input())
    # print("Enter the elements in list")
    # l=[int(input()) for i in range(n)]
    l = [0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]
    
     #Method1-> T(n)=O(n^2) S(n)=0(1)
   # print("Trapped Water="+str(stack_obj.TrappingRainWater1(l)))
    
     #Method2-> T(n)=O(n) S(n)=0(n)
    #print("Trapped Water="+str(stack_obj.TrappingRainWater2(l)))
    
         #Method3-> T(n)=O(n) S(n)=0(1)
    # print("Trapped Water="+str(stack_obj.TrappingRainWater3(l)))
    
    
    #Largest Histogram Area(Error)-----------------------------------
    # l2=[2,1,5,6,2,3,1]
    # print("largestRectangleArea="+str(stack_obj.largestRectangleArea(l2)))
    
    #Generate Parenthesis--------------------------------------------
    
    # n=3 
    # print(stack_obj.generateParenthesis(n))
    
    
    #dailyTemperatures--------------------------------------------------
    
    # temperatures = [30,38,30,36,35,40,28]
    # print(stack_obj.dailyTemperatures(temperatures))
    
    #CarFleet----------------------------------------------------------
    target = 12
    position = [10,8,0,5,3]
    speed = [2,4,1,1,3]

    print("CarFleet count=",stack_obj.carFleet(target,position,speed))





