'''

Welcome to GDB Online.
GDB online is an online compiler and debugger tool for C, C++, Python, Java, PHP, Ruby, Perl,
C#, OCaml, VB, Swift, Pascal, Fortran, Haskell, Objective-C, Assembly, HTML, CSS, JS, SQLite, Prolog.
Code, Compile, Run and Debug online from anywhere in world.

'''
def IsValidPalindrome(str1):
    l=0 
    r=len(str1)-1
    while(l<r):
        while l<r and not isAlphaNum(s[l]):
            l+=1 
        while l<r and not isAlphaNum(s[r]):
            r-=1 
        if s[l].lower()!=s[r].lower():
            return False 
        l+=1 
        r-=1 
    return True
    
def isAlphaNum(ch):
    return ord('a')<=ord(ch)<=ord('z')  or \
    ord('A')<=ord(ch)<=ord('Z') or \
    ord('0')<=ord(ch)<=ord('9') 
    
def twoSum(numbers, target):
    l=0 
    r=len(numbers)-1
    while(l<r):
        while(l<r and numbers[l]+numbers[r]>target):
            r-=1 
        while(l<r and numbers[l]+numbers[r]<target):
            l+=1 
        if numbers[l]+numbers[r]==target:
            return [l+1,r+1]
            
def maxArea(heights):
    maxArea=0 
    l=0 
    r=len(heights)-1 
    while(l<r):
        width=r-l 
        height=min(heights[l],heights[r])
        area=width*height 
        maxArea=max(area,maxArea)
        if(heights[l]>heights[r]):
            r-=1
        else:
            l+=1 
    return maxArea
           
           
#IsValidPalindrome---------------------------
#https://neetcode.io/problems/is-palindrome
s="Madam, in Eden, I'm Adam"
if(IsValidPalindrome(s)==True):
    print(f"{s} -> is valid palindrome")
else:
      print(f"{s} -> is not valid palindrome")
      
#twoSum---------------------------
arr=[1,2,3,4] 
target=3
print("Indexes for 2 sum is=",twoSum(arr,target))

#maxArea-------------------------------------------------
#https://neetcode.io/problems/max-water-container
#T(n)=O(n)
heights=[1,7,2,5,4,7,3,6]
print("Max Heights=",maxArea(heights))

    
    
        
    