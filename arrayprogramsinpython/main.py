# Link: https://takeuforward.org/strivers-a2z-dsa-course/strivers-a2z-dsa-course-sheet-2/
from collections import defaultdict
from collections import Counter

def SecondLargestSmallest(arr, n):
    max1, max2 = arr[0], arr[1]
    min1, min2 = arr[0], arr[1]
    if max2 > max1:
        max1, max2 = max2, max1
    if min2 < min1:
        min1, min2 = min2, min1
    for i in range(2, n):
        if arr[i] > max1:
            max2 = max1
            max1 = arr[i]
        elif arr[i] > max2 and arr[i] < max1:
            max2 = arr[i]
        if arr[i] < min1:
            min2 = min1
            min1 = arr[i]
        elif arr[i] > max2 and arr[i] < max1:
            min2 = arr[i]
    print(f"Second largest={max2} and Second Smallest={min2}")


def RemoveDuplicates(arr, n):
    b = []
    j = 0
    for i in range(n):
        if arr[i] not in b:
            b.insert(j, arr[i])
            j += 1
    print(b)


def RemoveDuplicates2(arr, n):
    s = set()
    for x in arr:
        s.add(x)
    j = 0
    for ele in s:
        arr[j] = ele
        j += 1
    print(list(s))


def MoveZeroesEnd(arr, n):
    b = []
    j = 0
    for i in range(n):
        if arr[i] != 0:

            b.insert(j, arr[i])
            j += 1

    for i in range(n):
        if arr[i] == 0:
            b.append(arr[i])
    print(b)


def MoveZeroesEnd2(arr, n):
    zeros = [x for x in arr if x == 0]
    non_zero = [x for x in arr if x != 0]
    b = non_zero + zeros
    print(b)


def MaxConsecutiveOne(arr, n):
    cnt = 0
    for x in arr:
        if x == 1:
            cnt += 1
        else:
            cnt = 0
    print(f"Max consecutive 1={cnt}")


# Important
def SubarraySumK1(arr, k):
    n = len(arr)
    len1 = 0
    for i in range(n):
        sum1 = 0
        for j in range(i, n):
            sum1 += arr[j]
            if sum1 == k:
                len1 = max(len1, j - i + 1)
    print(f"Max length of subarray sum={len1}")


def SubarraySumK2(arr, k):
    sum_map = {}  # key:sum,value:index
    max_len = 0
    sum1 = 0
    n = len(arr)
    for i in range(n):
        sum1 += arr[i]
        if sum1 not in sum_map:
            sum_map[sum1] = i
        if sum1 == k:
            max_len = max(max_len, i + 1)
        rem = sum1 - k
        if rem in sum_map:
            max_len = max(max_len, i - sum_map[rem])
    print(f"Max length of subarray sum={max_len}")


def SubarraySumK3(arr, k):
    l = 0
    r = 0
    sum1 = arr[0]
    n = len(arr)
    max_len = 0
    while r < n:
        while l <= r and sum1 > k:
            sum1 -= arr[l]
            l += 1
        if sum1 == k:
            max_len = max(max_len, r - l + 1)
        r += 1
        if r < n:
            sum1 += arr[r]

    print(f"Max length of subarray sum={max_len}")


def NumberOccurOnce1(arr, n):
    freq = {}
    for x in arr:
        if x not in freq.keys():
            freq[x] = 1
        else:
            freq[x] += 1
    for key, value in freq.items():
        if value == 1:
            print(f"Element occur once={key}")


def NumberOccurOnce2(arr, n):
    x = arr[0]
    for i in range(1, n):
        x ^= arr[i]
    print(f"Element occur once={x}")


def Sort012_1(arr, n):
    cnt0, cnt1, cnt2 = 0, 0, 0
    for i in range(n):
        if arr[i] == 0:
            cnt0 += 1
        if arr[i] == 1:
            cnt1 += 1
        if arr[i] == 2:
            cnt2 += 1
    for i in range(cnt0):
        arr[i] = 0
    for i in range(cnt0, cnt0 + cnt1):
        arr[i] = 1
    for i in range(cnt0 + cnt1, len(arr)):
        arr[i] = 2
    print(arr)


def Sort012_2(arr, n):
    low, mid = 0, 0
    high = n - 1
    while mid <= high:
        if arr[mid] == 0:
            arr[low], arr[mid] = arr[mid], arr[low]
            low += 1
            mid += 1
        if arr[mid] == 1:
            mid += 1
        else:
            arr[mid], arr[high] = arr[high], arr[mid]
            high -= 1
    print(arr)


def MajorityElement1(arr, n):
    for i in range(n):
        if arr[i] != -100:
            cnt = 1
            for j in range(i + 1, n):
                if arr[i] == arr[j]:
                    cnt += 1
                    arr[j] = -100
            if cnt >= n / 2:
                return arr[i]
    return -1


def MajorityElement2(arr, n):
    d1 = {}
    for x in arr:
        if x not in d1.keys():
            d1[x] = 1
        else:
            if d1[x] > (n / 2):
                return x

            else:
                d1[x] += 1

    return -1


# Moorebs Voting Algorithm:
def MajorityElement3(arr, n):
    cnt = 0
    for x in arr:
        if cnt == 0:
            ele = x
            cnt = 1
        if ele == x:
            cnt += 1
        if ele != x:
            cnt -= 1
    cnt1 = 0
    for x in arr:
        if x == ele:
            cnt1 += 1
        if cnt1 > n / 2:
            return x
    return -1


def AlterPosNeg1(arr, n):
    new_arr = []
    pos = []
    neg = []
    for x in arr:
        if x >= 0:
            pos.append(x)
        else:
            neg.append(x)

    for i in range(len(pos)):
        new_arr.insert(2 * i, pos[i])
    for i in range(len(neg)):
        new_arr.insert(2 * i + 1, neg[i])
    print(new_arr)


# Used when positive and negative numbers are equal
def AlterPosNeg2(arr, n):
    new_arr = []
    pos_ind = 0
    neg_ind = 1
    for x in arr:
        if x >= 0:
            new_arr.insert(pos_ind, x)
            pos_ind += 2
        else:
            new_arr.insert(neg_ind, x)
            neg_ind += 2
    print(new_arr)


def AlterPOsNeg3(arr, n):
    setPositive = True
    setNegative = False
    pos = []
    neg = []
    res = []

    for i in range(n):
        if arr[i] > 0:
            pos.append(arr[i])
        else:
            neg.append(arr[i])
    n1 = len(pos)
    n2 = len(neg)
    l1 = 0
    l2 = 0
    while l1 < n1 and l2 < n2:
        if setPositive == True:
            res.append(pos[l1])
            l1 += 1
            setPositive = False
            setNegative = True
        if setNegative == True:
            res.append(neg[l2])
            l2 += 1
            setNegative = False
            setPositive = True
    while l1 < n1:
        res.append(pos[l1])
        l1 += 1
    while l2 < n2:
        res.append(neg[l2])
        l2 += 1
    print(res)


def Partition(arr):
    j = 0
    pivot = 0
    for i in range(n):
        if arr[i] < pivot:
            temp = arr[i]
            arr[i] = arr[j]
            arr[j] = temp
            j += 1

    return j


def AlterPOsNeg4(arr, n):
    p = Partition(arr)
    neg = 0
    while p < n and neg < p and arr[neg] < 0:
        temp = arr[p]
        arr[p] = arr[neg]
        arr[neg] = temp
        neg += 2
        p += 1
    print(arr)


def Leader1(arr, n):
    leader_arr = []

    for i in range(n):
        leader_chk = True
        for j in range(i + 1, n):
            if arr[j] > arr[i]:
                leader_chk = False
                break
        if leader_chk == True:
            leader_arr.append(arr[i])
    print(leader_arr)


def Leader2(arr, n):
    max1 = arr[n - 1]

    leader_arr = []
    leader_arr.append(arr[n - 1])
    for i in range(n - 1, -1, -1):
        if arr[i] > max1:
            leader_arr.append(arr[i])
            max1 = arr[i]
    print(leader_arr)


def MaxSubArraySum1(arr, n):
    max_sum = -1000
    for i in range(n):
        sum1 = 0
        for j in range(i, n):
            sum1 += arr[j]
            max_sum = max(sum1, max_sum)

    print(f"Max sum ={max_sum}")


def EquilibriumIndex(arr, n):
    sum_arr = []
    sum_arr.insert(0, arr[0])
    for i in range(1, n):
        sum_arr.insert(i, arr[i] + sum_arr[i - 1])
    for i in range(1, n - 1):
        left_sum = sum_arr[i] - arr[i]
        right_sum = sum_arr[n - 1] - sum_arr[i]
        print("compare")
        print(left_sum)
        print(right_sum)
        if left_sum == right_sum:
            print(f"Index={i}")
            break


def FirstRepeatedEle(arr, n):
    d1 = {}
    repeat = -1
    for x in arr:
        if x not in d1:
            d1[x] = 1
        else:
            repeat = x
            break
    if repeat == -1:
        print("No repeated element")
    else:
        print(f"First Repeated element={repeat}")


def overlappedInterval(Intervals):
    # Code here
    Intervals.sort()
    n = len(Intervals)
    res = []
    for i in range(0, n):
        start = Intervals[i][0]
        end = Intervals[i][1]
        if res and end <= res[-1][1]:
            continue
        for j in range(i + 1, n):
            if Intervals[j][0] <= end:
                end = max(end, Intervals[j][1])
            else:
                break
        res.append([start, end])
    return res


# https://www.geeksforgeeks.org/problems/overlapping-intervals--170633/1
# https://takeuforward.org/data-structure/merge-overlapping-sub-intervals/
def overlappedInterval1(self, Intervals):
    # Code here
    Intervals.sort()
    n = len(Intervals)
    res = []
    for i in range(n):
        start = Intervals[i][0]
        end = Intervals[i][1]
        if len(res) == 0 or Intervals[i][0] > res[-1][1]:
            res.append([start, end])
        else:
            res[-1][1] = max(Intervals[i][1], res[-1][1])
    return res


def MaktrixZero1(A, m, n):
    for i in range(n):
        for j in range(m):
            if A[i][j] == 0:
                MakeRowZero(A, m, i)
                MakeColZero(A, n, j)
    for i in range(n):
        for j in range(m):
            if A[i][j] == -1:
                A[i][j] = 0
    return A


def MakeRowZero(A, m, i):
    for j in range(m):
        if A[i][j] != 0:
            A[i][j] = -1


def MakeColZero(A, n, j):
    for i in range(n):
        if A[i][j] != 0:
            A[i][j] = -1


def MaktrixZero2(A, m, n):
    row = [0] * n
    col = [0] * m

    for i in range(n):
        for j in range(m):
            if A[i][j] == 0:
                row[i] = 1
                col[i] = 1
    for i in range(n):
        for j in range(m):
            if row[i] or col[j]:
                A[i][j] = 0

    return A
    # Need To revise


# https://takeuforward.org/data-structure/program-to-generate-pascals-triangle/
def PascalElementSpecificPosition(C, R):
    # Formula=> (Row-1)C(Col-1) here C means combination
    # Time=> O(C)
    xc = C - 1
    xr = R - 1
    res = 1
    for i in range(xc):
        res = res * (xr - i)
        res = res / (i + 1)
    return res


def PascalSpecificRow(nrow):
    res = 1
    print(res, end="\t")
    for i in range(1, nrow):
        res = res * (nrow - i)
        res = res / i
        print(res, end="\t")
    print()


def PascalGenearteRow(nrow):
    row = []
    row.append(1)
    res = 1
    for i in range(1, nrow):
        res = res * (nrow - i)
        res = res / i
        row.append(res)
    return row


def GeneratePascalTriangle(num_rows):
    ans = []
    for i in range(1, num_rows + 1):
        ans.append(PascalGenearteRow(i))
    return ans


def MergeSortedArray1(A, B):
    n1 = len(A)
    n2 = len(B)
    i = n1 - 1
    j = 0
    while i >= 0 and j < n2:
        if A[i] > B[j]:
            temp = A[i]
            A[i] = B[j]
            B[j] = temp
            i -= 1
            j += 1
        else:
            break
    A.sort()
    B.sort()
    print(f"A={A}")
    print(f"B={B}")


def RotateMatrix90_1(A):
    n = len(A)
    B = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            B[j][n - i - 1] = A[i][j]
    return B


def RotateMatrix90_2(A):
    # First Transpose
    n = len(A)
    for i in range(n):
        for j in range(i + 1, n):
            A[i][j], A[j][i] = A[j][i], A[i][j]

    print("After transpose")
    str1 = "".join(map(str, A))

    print(str1)

    # Reverse Each row
    for i in range(n):
        for j in range(n // 2):
            temp = 0
            temp = A[i][j]
            A[i][j] = A[i][n - i - 1]
            A[i][n - i - 1] = temp
    print("After reverse each row")
    str1 = "".join(map(str, A))
    print("Rotated matrix is")
    print(str1)
    return A


def SpiralMatrix(A):
    res = []
    n = len(A)
    m = len(A[0])
    left = 0
    right = m - 1
    top = 0
    bottom = n - 1
    while left <= right and top <= bottom:
        # left->right
        for j in range(left, right + 1):
            res.append(A[top][j])
        top += 1
        # top->bottom
        for i in range(top, bottom + 1):
            res.append(A[i][right])
        right -= 1

        # if condition->Single row is there-> top>bottom as top increase after left->right move
        if top <= bottom:
            for j in range(right, left - 1, -1):
                res.append(A[bottom][j])
            bottom -= 1
            # if condition->Single column is there-> left>right as left increase after top->bottom move
        if left <= right:
            for i in range(bottom, top - 1, -1):
                res.append(A[i][left])
            left += 1
    return res


def Reverse(A, ind):

    left = ind
    right = len(A) - 1
    while left <= right:
        temp = A[left]
        A[left] = A[right]
        A[right] = temp
        left += 1
        right -= 1


def NextGreaterPermutation(A):
    n = len(A)
    bpoint = -1
    for i in range(n - 2, -1, -1):
        if A[i] < A[i + 1]:
            bpoint = i
            break
    if bpoint == -1:
        # Reverse(A,0)
        A.reverse()
        return A
    for i in range(n - 1, bpoint, -1):
        if A[i] > A[bpoint]:
            temp = A[i]
            A[i] = A[bpoint]
            A[bpoint] = temp
            break
    # Reverse(A,bpoint+1)
    A[bpoint + 1 :] = reversed(A[bpoint + 1 :])
    return A


def CountSubArraySum(A, k):
    pre_sum = 0
    count = 0
    n = len(A)
    d1 = defaultdict(int)
    d1[0] = 1
    for i in range(n):
        pre_sum += A[i]
        remove = pre_sum - k
        count += d1[remove]
        d1[pre_sum] += 1
    return count


def MaxHistogramArea1(A):
    max_Area = 0
    n = len(A)
    for i in range(n):
        minHeight = A[i]
        for j in range(i, n):
            minHeight = min(minHeight, A[j])
            max_Area = max(max_Area, minHeight * (j - i + 1))
    return max_Area


def MajorityElement1_n3(l):
    d1 = {}
    n = len(l)
    maj_arr = []
    for i in range(n):
        if l[i] in d1.keys():
            d1[l[i]] += 1
            if d1[l[i]] > len(l) // 3:
                maj_arr.append(l[i])
        else:
            d1[l[i]] = 1
        if len(maj_arr) == 2:
            break
    return maj_arr


def MajorityElement2_n3(l):
    cnt1 = 0
    cnt2 = 0
    maj_ls = []
    ele1 = float("-inf")
    ele2 = float("-inf")
    n = len(l)
    for i in range(n):
        if cnt1 == 0 and l[i] != ele2:
            ele1 = l[i]
            cnt1 += 1
        elif cnt2 == 0 and l[i] != ele1:
            ele2 = l[i]
            cnt2 += 1
        elif l[i] == ele1:
            cnt1 += 1
        elif l[i] == ele2:
            cnt2 += 1
        else:
            cnt1 -= 1
            cnt2 -= 1
    cnt1 = 0
    cnt2 = 0
    for i in range(n):
        if ele1 == l[i]:
            cnt1 += 1
        elif ele2 == l[i]:
            cnt2 += 1
    min_count = (n / 3) + 1
    if cnt1 >= min_count:
        maj_ls.append(ele1)
    if cnt2 >= min_count:
        maj_ls.append(ele2)
    return maj_ls


def KandaneAlgo(A):
    n = len(A)
    csum = 0
    msum = float("-inf")
    for i in range(n):
        csum += A[i]
        if csum < 0:
            csum = 0
        msum = max(csum, msum)
    return msum


def KandaneAlgoPrintSubArray(A):
    n = len(A)
    csum = 0
    msum = float("-inf")
    start_ind = -1
    end_ind = -1
    for i in range(n):
        if csum == 0:
            start_ind = i
        csum += A[i]
        if csum < 0:
            csum = 0
        if csum > msum:
            msum = csum
            end_ind = i
            start_ind = start_ind
    print("Max Sub Array=" + str(A[start_ind : end_ind + 1]))
    return msum


def LengthOfLongestSubarray(arr):
    d1 = {}
    max_len = 0
    csum = 0
    n = len(arr)
    for i in range(n):
        csum += arr[i]
        if csum in d1:
            max_len = max(max_len, i - d1[csum])
        elif csum == 0:
            max_len = max(max_len, i + 1)
        else:
            d1[csum] = i
    return max_len


def BinarySubSumLEK(arr, k):
    l = 0
    r = 0
    sum1 = 0
    cnt = 0
    n = len(arr)
    while r < n:
        sum1 += arr[r]
        while sum1 > k:
            sum1 -= arr[l]
            l += 1
        cnt += r - l + 1
        r += 1
    return cnt


def LongestCommonSequence(nums):
    # https://neetcode.io/problems/longest-consecutive-sequence
    nums_set = set(nums)
    longest = 0
    for n in nums:
        # Check first number of a sequence
        if n - 1 not in nums_set:
            length = 0
            while n + length in nums_set:
                length += 1
            longest = max(longest, length)
    return longest


def TopKFrequent(nums, k):
    # https://neetcode.io/problems/top-k-elements-in-list
    count = {}
    freq = [[] for i in range(len(nums) + 1)]
    for n in nums:
        count[n] = count.get(n, 0) + 1
    for n, c in count.items():
        freq[c].append(n)
    res = []
    for i in range(len(freq) - 1, 0, -1):
        for n in freq[i]:
            res.append(n)
            if len(res) == k:
                return res


# https://neetcode.io/problems/string-encode-and-decode
def encode(arr):
    str1 = ""
    for ele in arr:
        str1 += str(len(ele)) + "#" + ele
    return str1


def decode(str1):
    res = []
    i = 0
    while i < len(str1):
        j = i
        while str1[j] != "#":
            j += 1
        length = int(str1[i:j])
        res.append(str1[j + 1 : j + 1 + length])
        i = j + 1 + length
    return res


def CheckDuplicates(arr_nums):
    hashset = set()
    for num in arr_nums:
        if num in hashset:
            return True
        hashset.add(num)
    return False


def Sum2Problem(arr, target):
    # https://neetcode.io/problems/two-integer-sum
    prevMap = {}
    for i, num in enumerate(arr):
        diff = target - num
        if diff in prevMap:
            return [prevMap[diff], i]
        prevMap[num] = i


def ProductExceptItself2(arr):
    n = len(arr)
    res = [1] * n
    prefix = 1
    for i in range(n):
        res[i] = prefix
        prefix *= arr[i]
    postfix = 1
    for i in range(n - 1, -1, -1):
        res[i] *= postfix
        postfix *= arr[i]
    return res

#Error
def ProductExceptItself1(arr):
    n = len(arr)
    res = []
    right_prod = [1] * n
    right_prod[n - 1] = arr[n - 1]
    for i in range(n - 2, -1, -1):
        right_prod[i] = right_prod[i + 1] * arr[i]
    prod = 1
    for i in range(n - 1):
        res[i] = prod * right_prod[i + 1]
        prod *= arr[i]

    return res


def threeSum(nums):
    nums.sort()
    res = []
    for ind, a in enumerate(nums):
        if ind > 0 and a == nums[ind - 1]:
            continue
        l = ind + 1
        r = len(nums) - 1
        #Apply 2 Sum Logic
        while l < r:
            Sum3 = a + nums[l] + nums[r]
            if Sum3 > 0:
                r -= 1
            elif Sum3 < 0:
                l += 1
            else:
                res.append([a, nums[l], nums[r]])
                # l only update r automatic update by Sum3>0 condition
                l += 1
                while nums[l] == nums[l - 1] and l < r:
                    l += 1
    return res
    
    
def replaceElements(arr):
    rightMax=-1 
    n=len(arr)
    for i in range(n-1,-1,-1):
        newMax=max(arr[i],rightMax)
        arr[i]=rightMax 
        rightMax=newMax 
    return arr
    
def longestCommonPrefix(strs):
    #https://leetcode.com/problems/longest-common-prefix/description/
    if not  strs or "" in strs:
        return ""
    lcp=""
    n=len(strs)
    for i in range(len(strs[0])):
        ch=strs[0][i]
        for j in range(1,n):
            if i>=len(strs[j]) or ch!=strs[j][i]:
                return lcp 
        lcp+=ch
    return lcp
    
def GeneratePascalTriangle2(numRows):
    res=[[1]]
    for i in range(numRows-1):
        row=[]
        temp=[0]+res[-1]+[0]
        for j in range(len(res[-1])+1):
            row.append(temp[j]+temp[j+1])
        res.append(row)
    return res
    
def canPlaceFlowers1(flowerbed, n) -> bool:
    #https://leetcode.com/problems/can-place-flowers/description/
    #T(n)=O(n)   S(n)=O(n)
    f=[0]+flowerbed+[0]
    for i in range(1,len(f)-1):
        if f[i-1]==0 and f[i]==0 and f[i+1]==0:
            f[i]=1 
            n-=1 
    return n<=0
    
def findDisappearedNumbers(nums):
    #T(N)=O(N) S(N)=O(1)
    #https://leetcode.com/problems/find-all-numbers-disappeared-in-an-array/description/
        res=[]
        for n in nums:
            ind=abs(n)-1 
            nums[ind]=-1*abs(nums[ind])
        for ind,n in enumerate(nums):
            if n>0:
                res.append(ind+1)
        return res
def wordPattern(pattern, s) -> bool:
    #https://leetcode.com/problems/word-pattern/description/ 
    #https://neetcode.io/practice
    #T(n)=O(m+n)
    words=s.split(" ")
    if len(pattern)!=len(words):
        return False 
    charWord={}
    wordChar={}
    for ch,word in zip(pattern,words):
        if ch in charWord and word!=charWord[ch]:
            return False 
        if word in wordChar and ch!=wordChar[word]:
            return False 
        charWord[ch]=word 
        wordChar[word]=ch 
    return True
def isMonotonic1(nums):
    n=len(nums)
    #Chance of monotonic creasing then reverse it and apply only monotonic increasing
    if nums[n-1]-nums[0]<0:
        nums.reverse()
    for i in range(n-1):
        if nums[i]>nums[i+1]:
            return False 
    return True

def isMonotonic2(nums):
    n=len(nums)
    #Intially Take it as True as if it become False while looping then it cannot become true again 
    
    increase=True
    decrease=True 
    for i in range(n-1):
        if nums[i]>nums[i+1]:
            increase=False 
        if nums[i]<nums[i+1]:
            decrease=False 
    return increase or decrease
    
def numIdenticalPairs1(nums):
    res=0
    count=Counter(nums)
    for n,c in count.items():
        res+=(c)*(c-1)//2 
    return res

def numIdenticalPairs2(nums) :
    res=0
    d1={}
    for n in nums:
        if n in d1:
            res+=d1[n]
            d1[n]+=1 
        else:
            d1[n]=1 
    return res
    
    
def countCharacters(words, chars):
    #https://leetcode.com/problems/find-words-that-can-be-formed-by-characters/
    #T(N)=O(n*K+m) n-> number of words  k-> avergae length of each word  m-> length of chars
    count=Counter(chars)
    res=0
    for word in words:
        word_freq=defaultdict(int)
        good=True 
        for ch in word:
            word_freq[ch]+=1 
            if ch not in count or word_freq[ch]>count[ch]:
                good=False 
                break 
        if good:
            res+=len(word)
    return res
    
def largestGoodInteger(num: str) -> str:
    #https://leetcode.com/problems/largest-3-same-digit-number-in-string/description/ 
    #https://neetcode.io/practice
    res="0"
    n=len(num)
    for i in range(n-2):
        if num[i]==num[i+1]==num[i+2]:
            res=max(res,num[i:i+3])
    return  "" if res=="0" else res
    
    
def DestinationCity(paths):
    source=set()
    for p in paths:
        source.add(p[0])
    for p in paths:
        if p[1] not in source:
            return p[1]
            
            
def maxProductDifference(nums) -> int:
    #https://leetcode.com/problems/maximum-product-difference-between-two-pairs/description/
    max1=0 
    max2=0 
    min1=float("inf")
    min2=float("inf")
    for n in nums:
        if n>max2:
            if n>max1:
                max2=max1
                max1=n 
            else:
                max2=n 
        if n<min2:
            if n<min1:
                min2=min1 
                min1=n 
            else:
                min2=n 
    return (max1*max2-min1*min2)
    
def makeEqual(words) -> bool:
    freq_ch=defaultdict(int)
    for w in words:
        for ch in w:
            freq_ch[ch]+=1 
    for ch in freq_ch:
        #If count of each word not gives remainder 0 it means it cannot equally distribute within list of words
        if freq_ch[ch]%len(words):
            return False 
    return True
    
    
def findErrorNums(nums):
    freq=defaultdict(int)
    for num in nums:
        freq[num]+=1 
    res=[0,0]
    for i in range(1,len(nums)+1):
        if freq[i]==2:
            res[0]=i 
        if freq[i]==0:
            res[1]=i 
    return res
    
    
def intersection(nums1, nums2):
    seen=set(nums1)
    res=[]
    for n in nums2:
        if n in seen:
            res.append(n)
            seen.remove(n)
    return res

def countStudents(students, sandwiches) -> int:
    #https://leetcode.com/problems/number-of-students-unable-to-eat-lunch/description/
    #https://neetcode.io/practice
    stuLikes={}
    num_stu=len(students)
    for s in students:
        stuLikes[s]=stuLikes.get(s,0)+1
    for sand in sandwiches:
        if stuLikes.get(sand, 0)>0:
            stuLikes[sand]-=1 
            num_stu-=1 
        else:
            return num_stu
    return num_stu if num_stu > 0 else 0 
def timeRequiredToBuy(tickets, k: int) -> int:
    res=0 
    n=len(tickets)
    for i in range(n):
        if i<=k:
            res+=min(tickets[i],tickets[k])
        else:
            res+=min(tickets[i],tickets[k]-1)
    return res
    
    
def specialArray(nums) -> int:
    #https://leetcode.com/problems/special-array-with-x-elements-greater-than-or-equal-x/description/
    n=len(nums)
    for i in range(n+1):
        cnt=0 
        for x in nums:
            if x>=i:
                cnt+=1 
        if cnt==i:
            return i 
    return -1
    
def specialArray2(nums) -> int:
    nums.sort()
    prev=-1  
    tot_right=len(nums)
    i=0 
    n=len(nums)
    while i<n:
        if nums[i]==tot_right or (prev<tot_right<nums[i]):
            return tot_right 
        while i+1<n and nums[i]==nums[i+1]:
            i+=1 
        prev=nums[i]
        i+=1 
        tot_right=n-i
    return -1

def sortColors(nums) -> None:
    """
    Do not return anything, modify nums in-place instead.
    """
    zeroCount=0
    oneCount=0
    twoCount=0
    for n in nums:
        if n==0:
            zeroCount+=1 
        if n==1:
            oneCount+=1 
        if n==2:
            twoCount+=1 
    for i in range(zeroCount):
        nums[i]=0 
    for i in range(zeroCount,zeroCount+oneCount):
        nums[i]=1 
    for i in range(zeroCount+oneCount,len(nums)):
        nums[i]=2 
    return nums
    
    
def leastBricks(wall) -> int:
    #https://leetcode.com/problems/brick-wall/  
    #https://neetcode.io/practice
    countGap={0:0}  #{POsition Gap Exist:NUmber of gaps at given position}
    for row in wall:
        total=0 
        for brick in row[:-1]:
            total+=brick
            countGap[total]=countGap.get(total,0)+1 
    return len(wall)-max(countGap.values())   #Minimum Cut brick if  number of gaps are there
    
    
def maxProfit(prices):
    #https://leetcode.com/problems/best-time-to-buy-and-sell-stock-ii/description/
    profit=0 
    n=len(prices)
    for i in range(1,n):
        if prices[i]>prices[i-1]:
            profit+=(prices[i]-prices[i-1])
    return profit
    
def subarraySum(nums, k) -> int:
    #https://leetcode.com/problems/subarray-sum-equals-k/
    #T(n)=o(n)
    csum=0 
    count=0 
    preMap={0:1} #{prefixSum:count of that prefixSum}
    for  n in nums:
        csum+=n 
        diff=csum-k 
        count+=preMap.get(diff,0) # To get sum k we can check how many diff is there as prefix sum so it can be removed that number of sum 
        preMap[csum]=preMap.get(csum,0)+1 
    return count  
    
    
def countPalindromicSubsequence(s: str) -> int:
    res=set() #(mid,common at left and right char)
    left=set()
    right=Counter(s)
    for i in range(len(s)):
        mid=s[i]
        right[mid]-=1 
        if right[mid]==0:
            right.pop(mid)
        for j in range(26):
            ch=chr(ord('a')+j)
            if ch in left and ch in right:
                res.add((s[i],ch))
        left.add(s[i])
    return len(res)
    
    
def minSwaps(s: str) -> int:
    #https://leetcode.com/problems/minimum-number-of-swaps-to-make-the-string-balanced/description/
    close=0 
    maxClose=0 
    for ch in s:
        if ch=="[":
            close-=1 
        else:
            close+=1 
        maxClose=max(close,maxClose)
    return (maxClose+1)//2
    
    
    
def interchangeableRectangles(rectangles):
    count={}
    res=0 
    for width,height in rectangles:
        count[width/height]=count.get(width/height,0)+1 
    for cnt in count.values():
        if cnt>1:
            res+=(cnt*(cnt-1))//2 
    return res

    
    

    

        
        



"""
n=int(input("Enter the size of array"))
print("Enter the elements in array")
arr=[]
arr=[int(input()) for i in range(n)]
SecondLargestSmallest(arr,n)
"""
"""
# Remove Duplicates-------------------------------------
n1=int(input("Enter the size of array"))
print("Enter the elements in array")
arr1=[]
arr1=[int(input()) for i in range(n1)]
RemoveDuplicates(arr1,n1)
RemoveDuplicates2(arr1,n1)
"""

"""
# Move zeroes to end-------------------------------------
n2=int(input("Enter the size of array"))
print("Enter the elements in array")
arr2=[]
arr2=[int(input()) for i in range(n2)]
MoveZeroesEnd(arr2,n2)

MoveZeroesEnd2(arr2,n2)
"""

"""
#Max Consecutive 1-----------------------
n3=int(input("Enter the size of array"))
print("Enter the elements in array")
arr3=[]
arr3=[int(input()) for i in range(n3)]
MaxConsecutiveOne(arr3,n3)
"""

# Max Subarray sum k
"""
n4=int(input("Enter the size of array"))
print("Enter the elements in array")
arr4=[]
arr4=[int(input()) for i in range(n4)]
k=int(input("Enter the required sum"))
#SubarraySumK1(arr4,k)

#SubarraySumK2(arr4,k)
SubarraySumK3(arr4,k)
"""
"""
#NUmber Occur Once----------------------------------------
n5=int(input("Enter the size of array"))
print("Enter the elements in array")
arr5=[]
arr5=[int(input()) for i in range(n5)]
#NumberOccurOnce1(arr5,n5)
NumberOccurOnce2(arr5,n5)
"""
"""
#Sort 0,1,2----------------------------------------
n6=int(input("Enter the size of array"))
print("Enter the elements in array either 0,1,2")
arr6=[]
arr6=[int(input()) for i in range(n6)]
#Sort012_1(arr6,n6)
Sort012_2(arr6,n6)
"""
"""
#Majority Element n/2----------------------------------------
n7=int(input("Enter the size of array"))
print("Enter the elements in array")
arr7=[]
arr7=[int(input()) for i in range(n7)]
#val1=MajorityElement1(arr7,n7)
#val2=MajorityElement2(arr7,n7)
val3=MajorityElement3(arr7,n7)

if(val1!=-1):
    print(f"Majority element={val1}")

if(val2!=-1):
    print(f"Majority element={val2}")
if(val2==-1):
    print("No majority Element")

    
if(val3!=-1):
    print(f"Majority element={val3}")
if(val3==-1):
    print("No majority Element")
    """

# Alternating POsitive And Negative-----------------------------
# n8=int(input("Enter the size of array"))
# print("Enter the elements in array")
# n = 6
# A = [1, 2, -4, -5, 3, 4]

# # arr8=[]
# # arr8=[int(input()) for i in range(n8)]
# # #AlterPosNeg1(arr8,n8)
# # AlterPosNeg2(arr8,n8)
# # AlterPOsNeg3(A,n)
# AlterPOsNeg4(A,n)

"""
#Leader---------------------------------------------------------
n9=int(input("Enter the size of array"))
print("Enter the elements in array")
arr9=[]
arr9=[int(input()) for i in range(n9)]
#Leader1(arr9,n9)
Leader2(arr9,n9)
"""

"""
#MaxSubArraySum-------------------------------------------------
n10=int(input("Enter the size of array"))
print("Enter the elements in array")
arr10=[]
arr10=[int(input()) for i in range(n10)]
MaxSubArraySum1(arr10,n10)
"""
"""
#EquilibriumIndex---------------------------
n11=int(input("Enter the size of array"))
print("Enter the elements in array")
arr11=[]
arr11=[int(input()) for i in range(n11)]
EquilibriumIndex(arr11,n11)
 """

# Repeated Element----------------------------------------
# T(n)=O(n^2)  S(n)=O(n^2)
# n12=int(input("Enter the size of array"))
# print("Enter the elements in array")
# arr12=[]
# arr12=[int(input()) for i in range(n12)]
# FirstRepeatedEle(arr12,n12)


# overlappedInterval--------------------------------

# l=[[1,3],[2,4],[6,5],[9,10]]
# print("overlappedInterval--------------------------------")
# print(",".join(map(str,overlappedInterval(l))))


# Set Matrix Zero------------------------------------
# https://takeuforward.org/data-structure/set-matrix-zero/
# Met1->T(n)=(n*m)(n+m)+(n*m)=O(n^3)
# Met2-> T(n)=O(n*m)+O(n*m)=O(n*m)

# l=[[1,1,1],[1,0,1],[1,1,1]]
# nrows=len(l)
# ncols=len(l[0])
# #print("".join(map(str,MaktrixZero1(l,ncols,nrows))))
# print("".join(map(str,MaktrixZero2(l,ncols,nrows))))


# Pascal Specific element
# row_no=int(input("Enter row number"))
# col_no=int(input("Enter column number"))
# print(f"Pascal value at {row_no} and {col_no} is {PascalElementSpecificPosition(col_no,row_no)}")

# Pascal Specific row print
# row_no=int(input("Enter row number"))
# PascalSpecificRow(row_no)

# Pascal Triangle Generate

# num_row=int(input("Enter number of rows"))
# # print(GeneratePascalTriangle(num_row))
# print(GeneratePascalTriangle2(num_row))


# Merge Sorted MaxSubArraySum-------------------------------------------------
# https://takeuforward.org/data-structure/merge-two-sorted-arrays-without-extra-space/
# n1=int(input("Enter the size of array1="))
# n2=int(input("Enter the size of array2="))
# print("enter the elements in array1",end="\n")
# a1=[int(input()) for i in range(n1)]
# print("enter the elements in array2",end="\n")
# a2=[int(input()) for i in range(n2)]
# MergeSortedArray1(a1,a2)

# Rotate Matrix 90-------------------------------------------------------------------
# arr = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
# rotated=RotateMatrix90_1(arr)
# str1="".join(map(str,rotated))
# print("Rotated matrix is")
# print(str1)


# rotated=RotateMatrix90_2(arr)
# str1="".join(map(str,rotated))
# print("Rotated matrix is")
# print(str1)


# Spiral matrix------------------------------------

# mat = [[1, 2, 3, 4],
#       [5, 6, 7, 8],
#       [9, 10, 11, 12],
#       [13, 14, 15, 16]]

# ans = SpiralMatrix(mat)
# #need to check ->Wrong output
# print(ans)


# Next Greater Permutation------------------
# A = [2, 1, 5, 4, 3, 0, 0]
# print(f"Next greater permutation after {A}")
# ans = NextGreaterPermutation(A)

# print("".join(map(str,ans)))


# arr = [3, 1, 2, 4]
# k = 6
# cnt = CountSubArraySum(arr, k)
# print("The number of subarrays is:", cnt)

# Max MaxHistogramArea---------------------------------------------------
# Method1-> T(n)=O(n^2)
# arr=[2, 1, 5, 6, 2, 3, 1]
# print(f"Max MaxHistogramArea={MaxHistogramArea1(arr)}")

# Majority Element n/3-----------------------------------------------------------------
# n11=int(input("Enter the size of array"))
# print("Enter the elements in array")
# arr11=[]
# arr11=[int(input()) for i in range(n11)]
# Method1(Dictionary)---------------------------
# val1=MajorityElement1_n3(arr11)
# print(val1)
# Method1(Extended Boyer Moore's Voting Algorithm)------
# arr = [11, 33, 33, 11, 33, 11]
# maj_arr=[]
# maj_arr=MajorityElement2_n3(arr)
# print("Majority element for n/3")
# print(maj_arr)
# val3=MajorityElement3(arr7,n7)

# KandaneAlgo-------------------------------------------

# A=[-2, 1, -3, 4, -1, 2, 1, -5, 4]
# # print("Max subarray sum="+str(KandaneAlgo(A)))
# print("Max subarray sum="+str(KandaneAlgoPrintSubArray(A)))

# LengthOfLongestSubarray-----------------------
# A=[9, -3, 3, -1, 6, -5]
# # print("Max subarray sum="+str(KandaneAlgo(A)))
# print("Max subarray sum="+str(LengthOfLongestSubarray(A)))

# BinarySubSumLEK-------------------------------------------------
# https://www.youtube.com/watch?v=xvNwoz-ufXA----------------------------
# #T(n)=O(2*(2N)) S(n)=O(1)
# arr=[1,0,0,1,1,0]
# k=2
# sumLessThanEqual_k=BinarySubSumLEK(arr,k)
# sumLessThan_k=BinarySubSumLEK(arr,k-1)
# sumEqual_k=sumLessThanEqual_k-sumLessThan_k
# print("sumEqual_k=",sumEqual_k)


# LongestCommonSequence-------------------------------------
# nums=[2,100,3,1,200]
# print("Length of LongestCommonSequence=",LongestCommonSequence(nums))

# TopKFrequent------------------------------------------
# nums=[1,1,1,2,2,100]
# k=2
# print("Top K Frequent items=",TopKFrequent(nums,k))

# Encode And Decode String------------------------------------------
# str_arr=["neet","code"]
# encoded_str=encode(str_arr)
# decoded_res=decode(encoded_str)
# print("Decoded list=",decoded_res)


# CheckDuplicates-----------------------------------------------
# arr_nums=[1, 2, 3, 3]
# print("Check Duplicates=",CheckDuplicates(arr_nums))

# #Sum2Problem-------------------------
# arr=[2,1,5,3]
# target=7
# print(f"Indexes where {target} is there {Sum2Problem(arr,target)} ")

# ProductExceptItself2---------------------
# arr = [1, 2, 3, 4]
# # Method2->https://neetcode.io/problems/products-of-array-discluding-self
# print("Product Except Itself=", ProductExceptItself2(arr))
# print("Product Except Itself=", ProductExceptItself1(arr))

# threeSum------------------------------------------------
# nums = [-1, 0, 1, 2, -1, -4]
# print("Triplets=",threeSum(nums))

#replaceElements----------------------------------
# arr = [17,18,5,4,6,1]
# print("Replace Elements=",replaceElements(arr))

#longestCommonPrefix--------------------------
# arr=["flower","flow","flight"]  
# print("Longest Common Prefix=",longestCommonPrefix(arr))


#canPlaceFlowers---------------------------------------
# flowerbed = [1,0,0,0,1]
# n = 1
# print("Can Place flower=",canPlaceFlowers1(flowerbed,n))


# #findDisappearedNumbers---------------------------------
# nums = [4,3,2,7,8,2,3,1]
# print("findDisappearedNumbers=",findDisappearedNumbers(nums))


#wordPattern-----------------------------------------------------
# pattern = "abba"
# s = "dog cat cat dog"
# print("wordPattern=",wordPattern(pattern,s))


#isMonotonic------------------------------------
# nums = [6,5,4,4] 
# print("MOnotonic Increasing=",isMonotonic1(nums))
# print("MOnotonic Increasing=",isMonotonic2(nums))

#numIdenticalPairs----------------------------------------------------------------
# nums = [1,2,3,1,1,3]
# print("Number of numIdenticalPairs=",numIdenticalPairs1(nums))
# print("Number of numIdenticalPairs=",numIdenticalPairs2(nums))


#countCharacters-------------------------------------------
# words = ["cat","bt","hat","tree"]
# chars = "atach"
# print("countCharacters=",countCharacters(words,chars))

#largestGoodInteger-----------------------
# num = "6777133339"
# print("Largets Good Integer Present=",largestGoodInteger(num))


#DestinationCity------------------------------------------------
# paths=[["London","New York"],["New York","Lima"],["Lima","Sao Paulo"]]
# print("DestinationCity=",DestinationCity(paths))

#Maximum Product Difference
# nums = [4,2,5,9,7,4,8] 
# print("Maximum Product Difference=",maxProductDifference(nums))DestinationCity------------------------------------------------

#makeEqual------------------------------------------------
# words = ["abc","aabc","bc"]
# print("Make Equal=",makeEqual(words))

#findErrorNums--------------------------------------
# nums = [1,2,2,4]
# print("findErrorNums=",findErrorNums(nums))

#intersection--------------------------------------
# nums1 = [4,9,5]
# nums2 = [9,4,9,8,4]
# print("intersection=",intersection(nums1,nums2))


#countStudents-----------------------------------------------
# students = [1,1,0,0]
# sandwiches = [0,1,0,1]
# print("NUmber of students =",countStudents(students,sandwiches))

#timeRequiredToBuy---------------------------------------
# tickets = [2,3,2]
# k = 2
# print("timeRequiredToBuy=",timeRequiredToBuy(tickets,k))

#specialArray-------------------------------------------------
# nums = [0,4,3,0,4]
# print("specialArray=",specialArray(nums))

#sortColors--------------------------------
# nums = [2,0,2,1,1,0] 
# print("sortColors=",sortColors(nums))


# #leastBricks------------------------------------------
# wall = [[1,2,2,1],[3,1,2],[1,3,2],[2,4],[3,1,2],[1,3,1,1]]
# print("Minimum Cut=",leastBricks(wall))


#maxProfit---------------------------------
# prices = [7,1,5,3,6,4]
# print("Max Profit=",maxProfit(prices))

# #subarraySum-------------------------------------------------
# nums = [1,2,3]
# k=3 
# print("MaxSubArraySum=",subarraySum(nums,k))


#countPalindromicSubsequence-----------------------------------
s = "aabca"
print("countPalindromicSubsequence =",countPalindromicSubsequence(s))



#minSwaps------------------------------------------------------
s = "]]][[["
print("Minimum number of swaps=",minSwaps(s))


#interchangeableRectangles-------------------------------------------
rectangles = [[4,8],[3,6],[10,20],[15,30]]
print("interchangeableRectangles=",interchangeableRectangles(rectangles))





