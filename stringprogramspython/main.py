'''

Welcome to GDB Online.
GDB online is an online compiler and debugger tool for C, C++, Python, Java, PHP, Ruby, Perl,
C#, OCaml, VB, Swift, Pascal, Fortran, Haskell, Objective-C, Assembly, HTML, CSS, JS, SQLite, Prolog.
Code, Compile, Run and Debug online from anywhere in world.

'''
from collections import deque
from collections import defaultdict
#https://www.geeksforgeeks.org/string-data-structure/?ref=shm
#https://www.geeksforgeeks.org/top-50-string-coding-problems-for-interviews/
#https://www.geeksforgeeks.org/explore?page=2&category=Strings&sprint=a663236c31453b969852f9ea22507634&sortBy=difficulty&sprint_name=SDE%20Sheet&itm_medium=main_header&itm_campaign=practice_header
def StringSame(s1,s2):
    n1=len(s1)
    n2=len(s2)
    if n1!=n2:
        return False 
    for i in range(n1):
        s1=s1[1:]+s1[0]
        if s1==s2:
            return True 
    return False 

def findLongestWord (S, d):
    d.sort(key=lambda x:(-len(x),x))
    def Helper(sub,whole):
        n_sub=len(sub)
        n_wh=len(whole)
        i=0
        j=0 
        while(i<n_wh and j<n_sub):
            if(sub[j]==whole[i]):
                j+=1 
                i+=1 
        return j==n_sub
    for w in d:
        if Helper(w,S):
            return w 
    return ""
def AllSubstring(s):
    n=len(s)
    for i in range(n):
        for j in range(i,n):
            print(s[i:j+1])
def AllSubstring1(s):
    n=len(s)
    for i in range(n):
        temp=""
        for j in range(i,n):
            temp+=s[j]
            print(temp)

def AllSubsequence1(input1,output):
    if(len(input1)==0):
        print(output,end=" ")
        return
    AllSubsequence1(input1[1:],output+input1[0])
    AllSubsequence1(input1[1:],output)
    
    
def RemoveAllOccurenceSubstring(string,sub):
    check=True 
    while(check):
        check=False 
        ind=string.find(sub)
        if ind!=-1:
            string=string[:ind]+string[ind+len(sub):]
            check=True
    return string

def RemoveAllOccurenceSubstring1(string,sub):
    stack=[]
    for char in string:
        stack.append(char)
        if "".join(stack[-len(sub):])==sub:
            for _ in range(len(sub)):
                stack.pop()
    return "".join(stack)
    
    
def CheckStringSubsequence(string,sub):
    i=0
    k=0 
    n1=len(string)
    n2=len(sub)
    while(i<n1 and k<n2):
        if(sub[k]==string[i]):
             
            k+=1 
            print("k=",k)
            if k==len(sub):
                return True
        i+=1
    return False 
    
def CheckRepeatedSubstring1(s):
    N=len(s)
    def CheckPattern(n):
        for i in range(0,N-n,n):
            if s[i:i+n]!=s[i+n:i+2*n]:
                return False 
        return True 
    for i in range(1,N//2):
        if(CheckPattern(i)==True):
            return True
    return False

def CheckRepeatedSubstring2(s):
    ds=(s+s)[1:-1]
    return s in ds
    
def StringDiff1(s1,s2):
    out=0
    #^->XOR-> I/p->Equal cancel each other
    for ch in s1:
        out^=ord(ch)
    for ch in s2:
        out^=ord(ch)
    return chr(out)
    
def StringDiff2(s1,s2):
    out=ord(s1[len(s1)-1])
    #^->XOR-> I/p->Equal cancel each other
    for i in range(len(s2)):
        out^=ord(s1[i])^ord(s2[i])


    return chr(out)
    
    
def FindNumberOfUnattendedCustomer(sequence,capacity):
    visited=set()
    allocated=set()
    num_unattended=0 
    for ch in sequence:
        if ch not in visited:
            visited.add(ch)
            if len(allocated)<capacity:
                allocated.add(ch)
                
            else:
                num_unattended+=1 
        else:
            visited.remove(ch)
            if ch in allocated:
                allocated.remove(ch)
    return num_unattended
    
def RemoveAB_C(s):
    l=list(s)
    i=0
    k=0
    n=len(s)
    while(i<n):
        if(l[i]=='C'):
            i=i+1
        if(l[i]=='B' and l[k-1]=='A'):
            k=k-1 
            i=i+1 
        else:
            l[k]=l[i]
            i=i+1 
            k=k+1 
    return "".join(s[:k])
    
def RemoveDuplicateCharacter1(s):
    l=list(s)
    i=0 
    n=len(l)
    while(i<n-1):
        ch=l[i]
        while(i<n-1 and ch==l[i+1]):
            del l[i+1]
            n=n-1
        i+=1 
    print("".join(l))
    
def RemoveDuplicateCharacter2(s):
    #https://www.techiedelight.com/remove-adjacent-duplicates-characters-string/
    char=[]
    prev=None 
    for ch in s:
        if ch !=prev:
            char.append(ch)
            prev=ch 
    print("".join(char))
    
def CheckPalindrome(sub):
    l=0 
    r=len(sub)-1
    
    is_palindrome=True
    while(l<=r):
        if(sub[l]!=sub[r]):
            is_palindrome=False 
            break 
        l+=1 
        r-=1 
    return is_palindrome
            
        
        
    
    
def LongestPalindromic1(s):
    max_palindrome=""
    n=len(s)
    max_len=0
    for i in range(n):
        l1=0
        for j in range(i+1,n):
            sub=s[i:j+1]
         
            if(CheckPalindrome(sub)):
                
             
                l1=len(sub)
                
                if(l1>max_len):
                    max_len=l1
                    max_palindrome=sub 
    print("max_palindrome=",max_palindrome)
    
    
def LongestPalindromic2(s):
    max_pal=""
    max_len=0
    n=len(s)
    for i in range(n):
        odd_pal=Palindrome(s,i,i)
        len1=len(odd_pal)
        if max_len<len1:
            max_len=len1
            max_pal=odd_pal
        even_pal=Palindrome(s,i,i+1)
        len1=len(even_pal)
        if max_len<len1:
            max_len=len1
            max_pal=even_pal 
    return max_pal 
def Palindrome(s,low,high):
    n=len(s)
    while(low>=0 and high<n and s[low]==s[high]):
        low-=1 
        high+=1 
    return s[low+1:high]
    
def CheckIsRotatedPalindrome1(s):
    #https://www.techiedelight.com/check-given-string-rotated-palindrome-not/
    n=len(s)
    for i in range(n):
        s=s[1:]+s[0]
        if isPalindrome2(s,0,n-1):
            return True 
    return False 

def isPalindrome1(s,low,high):
    while(low<=high):
        if(s[low]!=s[high]):
            return  False
            
        low+=1 
        high-=1 
    return True
def isPalindrome2(s,low,high):
    return low>high or(s[low]==s[high] and isPalindrome2(s,low+1,high-1))
    
    
def CheckIsRotatedPalindrome2(s):
    n=len(s)
    s=s+s 
    return isPalindrome3(s,n)

def isPalindrome3(s,n):
    n1=len(s)
    for i in range(n1):
        if expandSubstring(s,i,i,n) or expandSubstring(s,i,i+1,n):
            return True 
    return False
def expandSubstring(s,low,high,k):
    while(low<=high and s[low]==s[high]):
        if high-low+1==k:
            return True 
        low-=1 
        high+=1
    return True 

def patternMatch(lst,pattern):

    out=[]
    for word in lst:
        n=len(word)
        if n==len(pattern):
            d1={} 
            d2={}
            i=0
            while(i<n):
                pat=pattern[i]
                w=word[i]
                if pat not in d1:
                    d1[pat]=w
                else:
                    if d1[pat]!=w:
                       
                        break 
                if w not in d2:
                    d2[w]=pat
                else:
                    if d2[w]!=pat:
                     
                        break
                i+=1
            if(i==n):
              
                out.append(word)
            
    print(out)
    
    
def GroupAnagrams1(words):
    anagrams=[]
    d={}
    sorted_words=["".join(sorted(word))  for word in words]
    for i,e in enumerate(sorted_words):
        d.setdefault(e,[]).append(i)
    for index in d.values():
        collection=tuple(words[i] for i in index)
        print("collection=",collection)
        if len(collection)>1:
            anagrams.append(collection)
    return anagrams
    
def GroupAnagrams2(words):
    d1={} 
    for word in words:
        sortedword="".join(sorted(word))
        if sortedword not in d1.keys():
            d1[sortedword]=[word]
        else:
            d1[sortedword].append(word)
    res=[]
    for value in d1.values():
        res.append(value)
    return res 
def GroupAnagrams3(words):
    #https://neetcode.io/problems/anagram-groups
    d1=defaultdict(list)
    for word in words:
        freq=[0]*26  #A-> 0th index ,B->1st index
        for ch in word:
            freq[ord(ch)-ord('A')]+=1 
        d1[tuple(freq)].append(word)
    res=[]
    for value in d1.values():
        res.append(value)
    return res
            
    
def SingleEditStringMatchCheck(first,second):
    n=len(first)
    m=len(second)
    edit=0
    if abs(n-m)>1:
        return False 
    i=0
    j=0 
    while(i<n and j<m):
        if first[i]!=second[j]:
            if n>m:
                i+=1 
            elif m>n:
                j+=1 
            else:
                i+=1 
                j+=1 
            edit+=1 
        else:
            i+=1 
            j+=1 
    if i<n:
        edit+=1 
    elif j<m:
        edit+=1 
    return edit==1
    
def LongestPalindromicSumString1(s):
    n=len(s)
    total=[0]*(n+1)
    max1=0
    for i in range(1,n+1):
        total[i]=total[i-1]+int(s[i-1])
    for i in range(0,n-1):
        for j in range(i+1,n,2):
            len1=j-i+1
            mid=i+len1//2 
            if total[mid]-total[i]==total[j+1]-total[mid] and len1>max1:
                max1=len1 
    return max1
    
def LongestPalindromicSumString2(s):
    max1=0
    n=len(s)
    for i in range(n-1):
        max1=expand(s,i,i+1,max1)
    return max1

def expand(s,low,high,max1):
    lsum=0 
    rsum=0 
    n=len(s)
    while(low>=0 and high<n):
        lsum+=int(s[low])
        rsum+=int(s[high])
        if lsum==rsum and high-low+1>max1:
            max1=high-low+1
        low-=1 
        high+=1
    return max1
    
    
def LongestNonRepeatedSubstring(s):
    check_win={}
    #substring index 
    begin,end=0,0 
    #window index
    low,high=0,0
    n=len(s)
    while(high<n):
        if check_win.get(s[high]):
            while s[low]!=s[high]:
                check_win[s[low]]=False 
                low+=1 
            low+=1 
        else:
            check_win[s[high]]=True 
            if end-begin<high-low:
                begin=low 
                end=high 
        high+=1 
    return s[begin:end+1]

def LeftRotation2(s,d):
    str1=s+s 
    n=len(s)
    return str1[d:d+n]
def RightRotation2(s,d):
    return LeftRotation2(s,len(s)-d)
    
def LeftRotation3(s,d):
    char_deque=deque(s)
    char_deque.rotate(-d)
    return "".join(char_deque)
def RightRotation3(s,d):
    char_deque=deque(s)
    char_deque.rotate(d)
    return "".join(char_deque) 
    
def PrefixMatch(s1,s2):
    res=""
    n=len(s1)
    m=len(s2)
    i=0 
    j=0 
    while(i<=n-1 and j<=m-1):
        if s1[i]!=s2[j]:
            break
        res+=s1[i]
        i+=1 
        j+=1 
    return res
    
    
    
def LCPWordMatch(arr,n):
    prefix=arr[0]
    for i in range(1,n):
        prefix=PrefixMatch(prefix,arr[i])
    return prefix 
    
def LCPrefix1(arr):
    n=len(arr)
    prefix=""
    arr.sort()
    for i in range(len(arr[0])):
        if all(arr[0][i]==x[i] for x in arr):
            prefix+=arr[0][i] 
        else:
            break 
    return prefix 
    
def LCPrefix2(arr):
    n=len(arr)
    prefix=""
    minlen=FindMinLen(arr)
    for i in range(minlen):
        current=arr[0][i]
        for j in range(1,n):
            if(current!=arr[j][i]):
                return  prefix
        prefix+=current 
    return prefix 
    
def FindMinLen(arr):
    n=len(arr)
    minlen=len(arr[0])
    for i in range(1,n):
        if minlen>len(arr[i]):
            minlen=len(arr[i]) 
    return minlen
    
def LCPrefixStr(s1,s2):
    n1=len(s1)
    n2=len(s2)
    prefix=""
    i=0 
    j=0 
    while(i<n1 and j<n2):
        if s1[i]!=s2[j]:
            break
        prefix+=s1[i]
        i+=1 
        j+=1
    return prefix
        
    
    
def LCPrefix3(arr,low,high):
    if low==high:
        return arr[low]
    if low<high:
        mid=low+(high-low)//2 
        s1=LCPrefix3(arr,low,mid)
        s2=LCPrefix3(arr,mid+1,high)

        return LCPrefixStr(s1,s2)
        
def CheckStringConstruct(s1,s2):
    freq={}
    for ele in s1:
        freq[ele]=freq.get(ele,0)+1 
    for ele2 in s2:
        freq[ele]=freq.get(ele,0)-1 
        if freq[ele2]<0:
            return False 
    return True
    
#Reverse string------------------------------------
#Method1-> Recursion
def StringReverse(str1):
    if str1=="":
        return str1 
    return StringReverse(str1[1:])+str1[:1]
    
def StringReverseExplicitStack(str1):
    #https://www.techiedelight.com/reverse-a-string-using-stack-data-structure/
    stack=deque(str1)
    reverse="".join(stack.pop() for _ in range(len(str1)))
    return reverse
    
def LongestCommonPrefix(str_arr):
    #https://www.geeksforgeeks.org/problems/longest-common-prefix-in-an-array5129/1?page=1&category=Strings&sortBy=difficulty
    res=""
    if not str_arr:
        return "-1"
    for i in range(len(str_arr[0])):
        for s in str_arr:
            if i==len(s) or s[i]!=str_arr[0][i]:
                return res if res else "-1"
        res+=str_arr[0][i]
    return res if res else "-1"
    
def CheckAnagram(s1,s2):
    if(len(s1)!=len(s2)):
        return False 
    d1={}
    d2={}
    for i in range(len(s1)):
        d1[s1[i]]=d1.get(s1[i],0)+1
        d2[s2[i]]=d2.get(s2[i],0)+1
    for ch in d1:
        if d1[ch]!=d2.get(ch,0):
            return False 
    return True 
    
def isSubsequence(s, t):
    i=0 
    j=0 
    while i<len(s) and j<len(t):
        if s[i]==t[j]:
            i+=1 
        j+=1 
    return True if i==len(s) else False
    
def isIsomorphic(s: str, t: str) -> bool:
    mapST={}
    mapTS={}
    for i in range(len(s)):
        c1=s[i]
        c2=t[i]
        if (c1 in mapST and mapST[c1]!=c2) or (c2 in mapTS and mapTS[c2]!=c1):
            return False 
        mapST[c1]=c2 
        mapTS[c2]=c1 
    return True
    
def maxNumberOfBalloons(text):
    #https://leetcode.com/problems/maximum-number-of-balloons/ 
    #T(n)=O(n)
    countText={}
    balloon={}
    for ch in text:
        countText[ch]=countText.get(ch,0)+1 
    for ch in "balloon":
        balloon[ch]=balloon.get(ch,0)+1 
    res=float("inf")
    # Find minimum ratio value of frequency of character of "balloon" in given Text
    for ch in balloon:
        if ch in countText:
            res=min(res,countText[ch]//balloon[ch])
        else:
            res=0
    return res
    
def Count(s):
        count1=0
        for i in range(len(s)):
            if s[i]=='1':
                count1+=1 
        return count1

def maxScore(s: str) -> int:
    #https://leetcode.com/problems/maximum-score-after-splitting-a-string/
    #https://neetcode.io/practice
    #T(n)=O(n)
    zeroes=0 
    ones=Count(s)
    score=0
    for i in range(len(s)-1):
        if s[i]=='0':
            zeroes+=1 
        else:
            ones-=1 
        score=max(score,ones+zeroes)
    return score
    
    
def isPathCrossing( path: str) -> bool:
    dir={'N':[0,1],'S':[0,-1],'E':[1,0],'W':[-1,0]}
    visit=set()
    x=0
    y=0 
    for ch in path:
        visit.add((x,y))
        dx,dy=dir[ch]
        x+=dx 
        y+=dy 
        if (x,y) in visit:
            return True 
    return False
    
    
def minOperations(s: str) -> int:
    #https://leetcode.com/problems/minimum-changes-to-make-alternating-binary-string/description/
    #https://neetcode.io/practice
    count=0 
    n=len(s)
    #Need to compare string with alternating string starts with 0 like length 4 string compare with "0101" if change occur count+=1 and other alternating string(1010)  digit differ calculate by n-count
    for i in range(n):
        #Even Index value of string is 1 then operation required (Count+1) 
        if i%2==0:
            count+=1 if s[i]=='1' else 0 
        else:
             #Odd Index value of string is 0 then operation required (Count+1) 
            count+=1 if s[i]=='0' else 0
    return min(count,n-count)
    
def maxLengthBetweenEqualCharacters(s: str) -> int:
    char_IndMap={}
    res=-1
    for ind,ch in enumerate(s):
        if ch not in char_IndMap:
            char_IndMap[ch]=ind 
        else:
            res=max(res,ind-char_IndMap[ch]-1)
    return res
        
        
        
            
        
    
    
    
        
            
    
        
    
        
            
    
        
            
  
   
                    
    
    
    
        
    
    
    
            
    
            
    
    
        
            
            
                
                
        
    
        

    
    

        
    
    
    
    
if __name__ == '__main__':
    
    
    #String Same------------------------------------------------------
    #https://www.techiedelight.com/check-strings-can-derived-circularly-rotating/
    # X = 'ABCD'
    # Y = 'DABC'
 
    # if StringSame(X, Y):
    #     print('Given strings can be derived from each other')
    # else:
    #     print('Given strings cannot be derived from each other')
    
    #All Substring------------------
    # s=input("Enter the string for generating substring")
    # # AllSubstring(s)
    # AllSubstring1(s)
    
    #AllSubsequence-------------------------------------
#     s=input("Enter the string for generating subsequence")
#   #Method1->Pick Donot Pick
#     #T(n)=O(2^n)
#     #S(n)=O(n)
   
#     AllSubsequence1(s,"")
    #Method2
    
    
    
    #RemoveAllOccurenceSubstring------------------------------
    #s="ABCABAAB"
    # substring="AB"
    # print("String after removal of substring")
    # # print(RemoveAllOccurenceSubstring(s,substring))
    # print(RemoveAllOccurenceSubstring1(s,substring))
    
    #CheckStringSubsequence--------------------------
    # first = 'abcde'
    # second = 'abd'
    # if CheckStringSubsequence(first, second):
    #     print('Subsequence found')
    # else:
    #     print('Not a subsequence')
        
    #Repeated Substring in string------------------------------------
    #https://www.techiedelight.com/check-string-for-repeated-substrings/
    #https://www.youtube.com/watch?v=2qEmA76Unm4
    # s="ABCDABCDABCDABCD"
    # if(CheckRepeatedSubstring1(s)==True):
    #     print("Repeated Substring present")
    # else:
    #     print("Repeated Substring not present")
        
    # if(CheckRepeatedSubstring2(s)==True):
    #     print("Repeated Substring present")
    # else:
    #     print("Repeated Substring not present")
    
    # String Diff-----------------------
    # first = 'abc'
    # second = 'ac'
    # # print(StringDiff1(first, second))
    # print(StringDiff2(first, second))
    
    #FindNumberOfUnattendedCustomer----------------------------------
    
    # sequence = 'ABCDDCEFFEBGAG'
    # capacity = 3
 
    # print(FindNumberOfUnattendedCustomer(sequence, capacity))
    
    
    #RemoveAB_C(s)-----------------------------------------------------------
    # s="ABCACBCAAB"
    # print(RemoveAB_C(s))
    
    
    #RemoveDuplicateCharacter---------------------------------------------------------
    # s="AABBBCCCCDDDDDDED"
    # # RemoveDuplicateCharacter1(s)
    # RemoveDuplicateCharacter2(s)
    
    #LongestPalindromic------------------------------------------------------------
    #s="ABDCBCDBDCBBC"
    #LongestPalindromic1(s)-------------------------------------------------------------
    #https://www.techiedelight.com/longest-palindromic-substring-non-dp-space-optimized-solution/
    #print("LongestPalindromic=",LongestPalindromic2(s))
    
    #Check Rotated Palindrome--------------------------------------------
    #https://www.techiedelight.com/check-given-string-rotated-palindrome-not/
    # s = 'ABCDCBA'
 
    # # rotate it by 2 units
    # s = s[2:] + s[:2]
 
    # if CheckIsRotatedPalindrome2(s):
    #     print('The string is a rotated palindrome')
    # else:
    #     print('The string is not a rotated palindrome')
        
        
     #Find all words that follow the same order of characters as given pattern------------------
     #https://www.techiedelight.com/find-words-that-follows-given-pattern/
    # words = ['leet', 'abcd', 'loot', 'geek', 'cool', 'for', 'peer', 'dear', 'seed',
    #         'meet', 'noon', 'otto', 'mess', 'loss']
 
   
    # pattern = 'moon'
 
    # patternMatch(words, pattern)
    
    #Group Anagrams--------------------------------------------------------
    # words = ['CARS', 'REPAID', 'DUES', 'NOSE', 'SIGNED', 'LANE', 'PAIRED', 'ARCS',
    #          'GRAB', 'USED', 'ONES', 'BRAG', 'SUED', 'LEAN', 'SCAR', 'DESIGN']
 
    # # anagrams = GroupAnagrams1(words)
    # # print("Anagrams")
    # # for anagram in anagrams:
    # #     print(anagram)
    # #Method2--
    # #T(n)=O(mnlogn)
    # print(GroupAnagrams2(words))
    # print(GroupAnagrams3(words))

    
    
    
    #Determine whether string matches after single edit--------------
    #https://www.techiedelight.com/determine-string-transformed-into-another-string-single-edit/
    # print(SingleEditStringMatchCheck("xyz", "xz"))       # True
    # print(SingleEditStringMatchCheck("xyz", "xyyz"))     # True
    # print(SingleEditStringMatchCheck("xyz", "xyx"))      # True
    # print(SingleEditStringMatchCheck("xyz", "xxx")) 
    
    #longest palindromic sum substring-----------------------------------------
    #https://www.techiedelight.com/longest-even-length-palidromic-sum-substring/------------
    # s = '13267224'
    # #Method1-- T(n)=O(n^2) ,S(n)=O(n)
    # print('The length of the longest palindromic sum substring is',
    #         LongestPalindromicSumString1(s))
            
            
            
    #Minimum Window Substring----------------------------------------------
    #https://www.youtube.com/watch?v=jSto0O4AJbM&list=PLot-Xpze53leOBgcVsJBEGrHPd_7x_koV&index=3
    
    
    
    
    #LongestNonRepeatedSubstring---------------------------------------------------
    #https://www.techiedelight.com/find-longest-substring-given-string-containing-distinct-characters/
    
    # s = 'abbcdafeegh'
    # print(LongestNonRepeatedSubstring(s))
    
    
    #StringLeftRotation----------------------------------------------------------------------
    # s="abcefg"
    # #Method2  Extended String--------------
    # print(LeftRotation3(s,2))
    # print(RightRotation3(s,2))
    
    #Method3  Deque--------------------------
    
    
    
    
    #LOngest Common Prefix-----------------------------------------------------
    # arr = ["geeksforgeeks", "geeks",
    #                 "geek", "geezer"]
    # n = len(arr)
    #Method 1------------------------------------------------------------------
    # ans = LCPWordMatch(arr, n)
    # if(len(ans)>0):
    #     print("longest common prefix is",ans)
    # else:
    #     print("No common") 
    #method2-----------------------------------------------------
    #https://www.geeksforgeeks.org/longest-common-prefix-using-word-by-word-matching/
    #T(N)=O(NlogN + NM) S(N)=O(1)
    
    # print("longest common prefix is=",LCPrefix1(arr))  
    
    # #Method 3------------------------------------
    # print("longest common prefix is=",LCPrefix2(arr))
    
    # #Method4---------------------------------------------------
    # print("longest common prefix is=",LCPrefix3(arr,0,n-1))
    
    
    #CheckStringConstruct--------------------------------
    #https://www.techiedelight.com/check-string-can-be-constructed-from-another-string/---
    # s1="ab"
    # s2="abb"
    # print(CheckStringConstruct(s1,s2))
    
    #StringReverse--------------------------------------
    # s="code"
    # #Method1-> Recursion
    # print("String Reverse=",StringReverse(s))
    # #Method2-> Explicit Stack 
    # print("String Reverse=",StringReverseExplicitStack(s))
    
    
    #LongestCommonPrefix------------------------------------
    # str_arr=["flower","glow","sly"]
    # print(LongestCommonPrefix(str_arr))
    
    #CheckAnagram-------------------------------------------
    # s1="bat"
    # s2="tat"
    # print("Strings are Anagram=",CheckAnagram(s1,s2))
    
    #isSubsequence-------------------------------
    # s1="abc"
    # s2="ahbgdc"
    # print("isSubsequence=",isSubsequence(s1,s2))
    
    
    #isIsomorphic--------------------------------------------------------
    # s1="add"
    # s2="egg"
    # print("IsImorphic",isIsomorphic(s1,s2))
    #maxNumberOfBalloons-------------------------------
    # str1="loonbalxballpoon"
    # print("Max number of balloon string in text=",maxNumberOfBalloons(str1))
    
    #maxScore-----------------------------------------------------------
    # s = "011101"
    # print("Max Score=",maxScore(s))
    
    #isPathCrossing----------------------------------------
    # path = "NESWW"
    # print("Is Path Crossing=",isPathCrossing(path))
    
    #minOperations--------------------------------------------
    # s = "0100"
    # print("minOperations=",minOperations(s))
    
    #maxLengthBetweenEqualCharacters-----------------------------------------
     s = "abca"
     print("maxLengthBetweenEqualCharacters=",maxLengthBetweenEqualCharacters(s))
    

    
    
 
    
        
 
    

































