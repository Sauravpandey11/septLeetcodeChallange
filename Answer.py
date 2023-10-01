#338. Counting Bits
class Solution:
    def countBits(self, n: int) -> List[int]:
        ans = [0] * (n + 1)
        for i in range(1, n + 1):
            ans[i] = ans[i >> 1] + (i & 1)
        return ans
# 2707. Extra Characters in a String
class Solution:
    def minExtraChar(self, s: str, dictionary: List[str]) -> int:
        max_val = len(s) + 1
        dp = [max_val] * (len(s) + 1)
        dp[0] = 0 
        dictionary_set = set(dictionary)
        for i in range(1, len(s) + 1):
            dp[i] = dp[i - 1] + 1
            for l in range(1, i + 1): 
                if s[i-l:i] in dictionary_set:
                    dp[i] = min(dp[i], dp[i-l])
        return dp[-1]
# 62. Unique Paths
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        return math.comb(m+n-2, m-1)

## 141. Linked List Cycle
class Solution:
    def hasCycle(self, head: Optional[ListNode]) -> bool:
        fast=head
        while fast and fast.next:
            head=head.next
            fast=fast.next.next
            if head is fast:
                return True
        return False
## 138. Copy List with Random Pointer
class Solution:
    def copyRandomList(self, head: 'Optional[Node]') -> 'Optional[Node]':
        old_to_copy={None:None}
        cur=head
        while cur:
            copy=Node(cur.val)
            old_to_copy[cur]=copy
            cur=cur.next
        cur=head
        while cur:
            copy=old_to_copy[cur]
            copy.next=old_to_copy[cur.next]
            copy.random=old_to_copy[cur.random]
            cur=cur.next
        return old_to_copy[head]                
        
## 725. Split Linked List in Parts
class Solution:
    def splitListToParts(self, head: Optional[ListNode], k: int) -> List[Optional[ListNode]]:
        length=0
        curr=head
        prev=None
        res=[]
        while curr:
            length+=1
            curr=curr.next 
        curr=head
        parts,left=length//k,length%k
        for _ in range(k):
            res.append(curr)
            for _ in range(parts):
                if curr:
                    prev=curr
                    curr=curr.next
            if left and curr:
                prev=curr
                curr=curr.next
                left-=1
            if prev: prev.next=None
        return res
## Reverse Linked List II
class Solution:
    def reverseBetween(self, head: Optional[ListNode], left: int, right: int) -> Optional[ListNode]:
        if not head or left == right:
            return head
        dummy = ListNode(0, head)
        prev = dummy
        for _ in range(left - 1):
            prev = prev.next
        current = prev.next
        for _ in range(right - left):
            next_node = current.next
            current.next, next_node.next, prev.next = next_node.next, prev.next, next_node
        return dummy.next
## 118. Pascal's Triangle
class Solution:
    def generate(self, numRows: int) -> List[List[int]]:
        List=[]
        list_0=[1]
        list_1=[1,1]
        List.append(list_0)
        List.append(list_1)
        if numRows == 1:
            return [[1]]
        if numRows == 2:
            return [[1],[1,1]]
        if numRows>=3:
            for i in range(2,numRows):
                list_i=[1]
                for j in range (1,i):
                    list_i.append(List[i-1][j-1]+List[i-1][j])
                list_i.append(1)
                List.append(list_i)
        return List
## 377. Combination Sum IV
class Solution:
    def combinationSum4(self, nums: List[int], target: int) -> int:
        dp = [0] * (target+1)
        dp[0] = 1 
        for i in range(1, target+1):
            for n in nums:
                if n <= i:
                    dp[i] += dp[i-n]
        return dp[-1]
## 1359. Count All Valid Pickup and Delivery Options
class Solution:
    def countOrders(self, n: int) -> int:
        mod=10**9+7
        ans=math.factorial(n*2)>>n
        return ans%mod
## 1282. Group the People Given the Group Size They Belong To
class Solution:
    def groupThePeople(self, groupSizes: List[int]) -> List[List[int]]:
        groups = {}
        result = [] 
        for i, size in enumerate(groupSizes):
            if size not in groups:
                groups[size] = []
            groups[size].append(i) 
            if len(groups[size]) == size:
                result.append(groups[size])
                groups[size] = []
        return result
## 1647. Minimum Deletions to Make Character Frequencies Unique
class Solution:
    def minDeletions(self, s: str) -> int:
        cnt = Counter(s)
        deletions = 0
        used_frequencies = set() 
        sorted_freqs = sorted(cnt.values(), reverse=True) 
        for freq in sorted_freqs:
            if freq not in used_frequencies:  # Early exit condition
                used_frequencies.add(freq)
                continue  
            while freq > 0 and freq in used_frequencies:
                freq -= 1
                deletions += 1
            used_frequencies.add(freq) 
        return deletions
## 135. Candy
from collections import defaultdict
class Solution:
    def candy(self, ratings: List[int]) -> int:
        candies = len(ratings)
        def get_sum(count):
            return int((count+1) * (count/2)) 
        i = 0
        while i < len(ratings)-1:
            if ratings[i] == ratings[i+1]:
                i += 1
                continue
            going_up = 0
            while i < len(ratings) -1 and ratings[i] < ratings[i+1]:
                going_up += 1
                i += 1
            going_down = 0
            while i < len(ratings)-1 and ratings[i] > ratings[i+1]:
                going_down += 1
                i += 1
            candies += get_sum(max(going_up, going_down))
            candies += get_sum(min(going_up-1, going_down-1))
        return candies
## 332. Reconstruct Itinerary
class Solution:
    def __init__(self):
        self.flight_graph = defaultdict(list)
        self.itinerary = []
    def dfs(self, airport:str) -> None:
        destinations = self.flight_graph[airport]
        while destinations:
            next_destination = destinations.pop()
            self.dfs(next_destination)
        self.itinerary.append(airport)
    def findItinerary(self, tickets: List[List[str]]) -> List[str]:
        for ticket in tickets:
            from_airport, to_airport = ticket
            self.flight_graph[from_airport].append(to_airport)
        for destinations in self.flight_graph.values():
            destinations.sort(reverse=True)
        self.dfs("JFK")
        self.itinerary.reverse()
        return self.itinerary
## 1584. Min Cost to Connect All Points
class Solution:
    def minCostConnectPoints(self, points: List[List[int]]) -> int:
        n = len(points)
        if n == 1:
            return 0
        dist_arr = []
        for i in range(n - 1):
            x1, y1 = points[i]
            for j in range(i + 1, n):
                x2, y2 = points[j]
                dist = abs(x1 - x2) + abs(y1 - y2)
                dist_arr.append((dist, i, j))
        heapq.heapify(dist_arr)
        root = [i for i in range(n)]
        def find(n):
            if root[n] != n:
                root[n] = find(root[n])
            return root[n]
        def union(x, y):
            s1 = find(x)
            s2 = find(y)
            if s1 != s2:
                root[s2] = root[s1]
                return 1
            return 0
        res = 0
        count = 0
        while count < n - 1:
            dist, x, y = heapq.heappop(dist_arr)
            if union(x, y) == 1:
                res += dist
                count += 1
        return res
## 1631. Path With Minimum Effort
class Solution:
    def minimumEffortPath(self, heights: List[List[int]]) -> int:
        
        M, N = map(len, (heights, heights[0]))
        heap = [(0, 0, 0)]
        seen = set()
        result = 0 
        while heap:
            effort, i, j = heapq.heappop(heap)
            seen.add((i, j))
            result = max(result, effort)
            if i == M-1 and j == N-1:
                break
            for x, y in [(i+1, j), (i-1, j), (i, j+1), (i, j-1)]:
                if not (x >= 0 <= y): continue
                if x >= M or y >= N: continue
                if (x, y) in seen: continue
                effort = abs(heights[i][j] - heights[x][y])
                heapq.heappush(heap, (effort, x, y)) 
        return result
## 847. Shortest Path Visiting All Nodes
from collections import deque, namedtuple
class Solution:
    def shortestPathLength(self, graph):
        n = len(graph)
        all_mask = (1 << n) - 1
        visited = set()
        Node = namedtuple('Node', ['node', 'mask', 'cost'])
        q = deque()
        for i in range(n):
            mask_value = (1 << i)
            this_node = Node(i, mask_value, 1)
            q.append(this_node)
            visited.add((i, mask_value))
        while q:
            curr = q.popleft()
            if curr.mask == all_mask:
                return curr.cost - 1
            for adj in graph[curr.node]:
                both_visited_mask = curr.mask | (1 << adj)
                this_node = Node(adj, both_visited_mask, curr.cost + 1)

                if (adj, both_visited_mask) not in visited:
                    visited.add((adj, both_visited_mask))
                    q.append(this_node)
        return -1
## 1337. The K Weakest Rows in a Matrix
class Solution:
    def kWeakestRows(self, mat: List[List[int]], k: int) -> List[int]:
        row_strength = [(sum(row), i) for i, row in enumerate(mat)]
        row_strength.sort(key=lambda x: (x[0], x[1]))
        return [row[1] for row in row_strength[:k]]
## 287. Find the Duplicate Number
class Solution:
    def findDuplicate(self, nums: List[int]) -> int:
        seen = set()
        for num in nums:
            if num in seen:
                return num
            seen.add(num)
## 1658. Minimum Operations to Reduce X to Zero
class Solution:
    def minOperations(self, nums: List[int], x: int) -> int:
        target, n = sum(nums) - x, len(nums) 
        if target == 0:
            return n 
        max_len = cur_sum = left = 0 
        for right, val in enumerate(nums):
            cur_sum += val
            while left <= right and cur_sum > target:
                cur_sum -= nums[left]
                left += 1
            if cur_sum == target:
                max_len = max(max_len, right - left + 1) 
        return n - max_len if max_len else -1
## 4. Median of Two Sorted Arrays
class Solution:
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        merged = []
        i, j = 0, 0 
        while i < len(nums1) and j < len(nums2):
            if nums1[i] < nums2[j]:
                merged.append(nums1[i])
                i += 1
            else:
                merged.append(nums2[j])
                j += 1 
        while i < len(nums1):
            merged.append(nums1[i])
            i += 1 
        while j < len(nums2):
            merged.append(nums2[j])
            j += 1 
        mid = len(merged) // 2
        if len(merged) % 2 == 0:
            return (merged[mid-1] + merged[mid]) / 2
        else:
            return merged[mid]
## 1048. Longest String Chain
class Solution:
    def longestStrChain(self, words: List[str]) -> int:       
        n = len(words)
        words.sort(key = lambda x : len(x)) 
        dit = {w:1 for w in words} 
        for i in range(1,n):
            w = words[i]
            for j in range(len(w)):
                new_w = w[:j]+w[j+1:] 
                if new_w in dit and dit[new_w]+1>dit[w]:
                    dit[w] = dit[new_w]+1 
        return max(dit.values())    
## 799. Champagne Tower
class Solution:
    def champagneTower(self, poured: int, query_row: int, query_glass: int) -> float:
        q = [poured]
        for r in range(query_row):
            q2 = [0]*(1+ len(q))
            for i, amount in enumerate(q):
                if amount <= 1:
                    continue
                tmp = (amount - 1 ) / 2
                q2[i] += tmp 
                q2[i+1] += tmp
            q = q2 
        return min(q[query_glass], 1.0)
## 389. Find the Difference
class Solution:
    def findTheDifference(self, s: str, t: str) -> str:
        return list((Counter(t)-Counter(s)).keys())[0]
## 316. Remove Duplicate Letters
class Solution:
    def removeDuplicateLetters(self, s: str) -> str:
        stack = []
        seen = set() 
        last_occ = {c: i for i, c in enumerate(s)} 
        for i, c in enumerate(s):
            if c not in seen: 
                while stack and c < stack[-1] and i < last_occ[stack[-1]]:
                    seen.discard(stack.pop())
                seen.add(c)
                stack.append(c) 
        return ''.join(stack)
## 880. Decoded String at Index
class Solution:
    def decodeAtIndex(self, s: str, k: int) -> str:
        length = 0
        i = 0 
        while length < k:
            if s[i].isdigit():
                length *= int(s[i])
            else:
                length += 1
            i += 1 
        for j in range(i-1, -1, -1):
            char = s[j]
            if char.isdigit():
                length //= int(char)
                k %= length
            else:
                if k == 0 or k == length:
                    return char
                length -= 1
## 905. Sort Array By Parity
class Solution:
    def sortArrayByParity(self, nums: List[int]) -> List[int]:
        left_index, right_index = 0, len(nums) - 1
        while(left_index < right_index):
            if(nums[left_index] % 2 != 0):
                if(nums[right_index] % 2 == 0):
                    nums[left_index], nums[right_index] = nums[right_index], nums[left_index]
                    left_index += 1
                    right_index -= 1
                else:
                    right_index -=1
            else:
                left_index += 1
        return nums
## 896. Monotonic Array
class Solution:
    def isMonotonic(self, nums: List[int]) -> bool:
        isIncrease, isDecrease = True, True
        val = nums.pop()
        while nums:
            if nums[-1] > val:
                isIncrease = False
            if nums[-1] < val:
                isDecrease = False
            val = nums.pop()
        return isDecrease or isIncrease
## 456. 132 Pattern
class Solution:
    def find132pattern(self, nums: List[int]) -> bool:
        stack, third = [], float('-inf') 
        for num in reversed(nums):
            if num < third:
                return True
            while stack and stack[-1] < num:
                third = stack.pop()
            stack.append(num)
        return False
