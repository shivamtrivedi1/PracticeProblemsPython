'''

Welcome to GDB Online.
GDB online is an online compiler and debugger tool for C, C++, Python, Java, PHP, Ruby, Perl,
C#, OCaml, VB, Swift, Pascal, Fortran, Haskell, Objective-C, Assembly, HTML, CSS, JS, SQLite, Prolog.
Code, Compile, Run and Debug online from anywhere in world.

'''
def maxProfit(prices):
    #https://neetcode.io/problems/buy-and-sell-crypto
    #T(n)=O(n)
    #S(n)=O(1)
    maxP=0 
    l=0
    r=1 
    while r<len(prices):
        #Profit 
        if prices[l]<prices[r]:
            profit=prices[r]-prices[l]
            maxP=max(maxP,profit)
        else:
            l=r 
        r+=1
    return maxP
    
def LengthOfLongestSubstrWithoutDuplicate(str1):
    #https://neetcode.io/problems/longest-substring-without-duplicates
    l=0 
    charSet=set()
    maxlen=0 
    for r in range(len(str1)):
        while str1[r] in charSet:
            charSet.remove(str1[l])
            l+=1 
        charSet.add(str1[r])
        maxlen=max(maxlen,r-l+1)
    return maxlen
        
    
    
    
    
    
#maxProfit------------------------------------------    
# prices=[7,1,5,3,6,4] 
# print("Max Profit in buy and selling stock=",maxProfit(prices))

#LengthOfLongestSubstrWithoutDuplicate-----------------
s="zxyzxyz"
print("Max Length of Non Duplicate Substring=",LengthOfLongestSubstrWithoutDuplicate(s))