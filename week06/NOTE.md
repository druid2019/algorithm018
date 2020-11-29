学习笔记

```java
day36
给定一个包含非负整数的 m x n 网格 grid ，请找出一条从左上角到右下角的路径，使得路径上的数字总和为最小。
说明：每次只能向下或者向右移动一步。 
示例1:
输入：grid = [[1,3,1],[1,5,1],[4,2,1]]
输出：7
解释：因为路径 1→3→1→1→1 的总和最小
示例2:
输入：grid = [[1,2,3],[4,5,6]]
输出：12
public int minPathSum(int[][] grid) {
    int rows = grid.length;
    int cols = grid[0].length;
    // 用来记录当前节点到左上角的最短路径
    int[][] dp = new int[rows][cols];
    dp[0][0] = grid[0][0];
    for (int i = 0;i < rows;i++) {
        for (int j = 0;j < cols;j++) {
            if (i == 0 && j == 0) {
                continue;
            } else if (i == 0 && j != 0) {
                dp[i][j] = dp[i][j-1] + grid[i][j]; //对第一行的处理
            } else if (i != 0 && j == 0) {
                dp[i][j] = dp[i-1][j] + grid[i][j]; //对第一列的处理
            } else {
                 //对于普通位置，等于本身的值加上上面或者左边之间的最小值
                dp[i][j] = Math.min(dp[i-1][j],dp[i][j-1]) + grid[i][j];
            }
        }
    }
    return dp[rows-1][cols-1];
}

```

```java
day37
322. 零钱兑换
给定不同面额的硬币 coins 和一个总金额 amount。编写一个函数来计算可以凑成总金额所需的最少的硬币个数。如果没有任何一种硬币组合能组成总金额，返回 -1。
你可以认为每种硬币的数量是无限的。
示例 1：
输入：coins = [1, 2, 5], amount = 11
输出：3 
解释：11 = 5 + 5 + 1
示例 2：
输入：coins = [2], amount = 3
输出：-1
思路一：动态规划
F(S)=F(S−C)+1
F(S)表示组成金额S最少的硬币数，C为最后一枚硬币的面值。
时间复杂度O(Sn),空间复杂度O(S)
F(i) = min F(i-cj) + 1,cj为最后一枚硬币的面额
public int coinChange(int[] coins, int amount) {
    // dp[i]表示钱数为i时最小硬币数
	int[] dp = new int[amount + 1];
    
    // 初始化钱数为i时的硬币数
    Arrays.fill(dp,amount + 1);
    dp[0] = 0;
    // 钱数为0时的最小硬币数也为0
    for (int i = 0; i <= amount; i++) {
        for (int j = 0; j < coins.length; j++) {
            if (i >= coins[j]) {
                dp[i] = Math.min(dp[i], dp[i-coins[j]] + 1);
            }
        }
    }
    return dp[amount] > amount ? -1:dp[amount];
}


```

```python 
day38
213. 打家劫舍 II
围成一个圈，环
示例 1：
输入：nums = [2,3,2]
输出：3
解释：你不能先偷窃 1 号房屋（金额 = 2），然后偷窃 3 号房屋（金额 = 2）, 因为他们是相邻的。
示例 2：
输入：nums = [1,2,3,1]
输出：4
解释：你可以先偷窃 1 号房屋（金额 = 1），然后偷窃 3 号房屋（金额 = 3）。
偷窃到的最高金额 = 1 + 3 = 4 。
思路一:
时间复杂度O(N)，空间复杂度O(N)
1.不偷第一个房子nums[1:],最大金额p1;
2.不偷最后一个房子(nums[:n-1]),最大金额p2
综合两个取最大值
def rob(self, nums):
    # 边界条件
    n = len(nums)
    if n == 0:
        return 0
    if n <= 2:
        return max(nums)
    dp1 = [0] * n
    dp1[0] = 0
    dp1[1] = nums[1]
    # 不偷第一个，不偷最后一个
    for i in range(2,n):
        dp1[i] = max(dp1[i-1],dp1[i-2]+nums[i])
    # 偷第一个，不偷最后一个
    dp2 = [0] * n
    dp2[0] = nums[0]
    dp2[1] = max(nums[0],nums[1])
    for i in range(2,n-1):
        dp2[i] = max(dp2[i-1],dp2[i-2]+nums[i])
    return max(dp1[n-1],dp2[n-2])
```

```java
// 对应的java代码
public int rob(int[] nums) {
    if (nums == null || nums.length == 0) {
        return 0;
    }
    if (nums.length == 1) {
        return nums[0];
    }
    int n = nums.length;
    // 偷第一个，不偷最后一个
    int[] dp1 = new int[n];
    dp1[0] = 0;
    dp1[1] = nums[1];
    for (int i=2; i<n; i++){
        dp1[i] = Math.max(dp1[i-1], dp1[i-2] + nums[i]);
    }
    // 偷最后一个，不偷第一个
    int[] dp2 = new int[n];
    dp2[0] = nums[0];
    dp2[1] = Math.max(nums[0],nums[1]);
    for (int i=2; i<n-1; i++) {
        dp2[i] = Math.max(dp2[i-1], dp2[i-2] + nums[i]);
    }
    return Math.max(dp1[n-1],dp2[n-2]);
}
```

```java
day39
589. N叉树的前序遍历
给定一个 N 叉树，返回其节点值的前序遍历。
例如，给定一个 3叉树 :
返回其前序遍历: [1,3,5,6,2,4]
思路:递归
List<Integer> alst = new ArrayList<>();
public List<Integer> preorder(Node root) {
	// 终止条件
	if (root == null) {
		return alst;
	}
	// 递归过程
	alst.add(root.val);
	// 便利
	for (Node node:root.children) {
		preorder(node);
	}
	return alst;
}
```



```java
day40
363. 矩形区域不超过 K 的最大数值和
给定一个非空二维矩阵 matrix 和一个整数 k，找到这个矩阵内部不大于 k 的最大矩形和。
示例:
输入: matrix = [[1,0,1],[0,-2,3]], k = 2
输出: 2 
解释: 矩形区域 [[0, 1], [-2, 3]] 的数值和是 2，且 2 是不超过 k 的最大数字（k = 2）。
public int maxSumSubmatrix(int[][] matrix, int k) {
    int m = matrix.length;
    int n = matrix[0].length;
    int[][] sums = new int[m][n];
    // 按列求和
    for(int col = 0; col < n; col++) {
        for (int row = 0; row < m; row++) {
            // 第一行
            if (row == 0) {
                sums[row][col] = matrix[row][col];
            } else {
                sums[row][col] = sums[row-1][col] + matrix[row][col];
            }        
        }
    }
    // col1为矩阵起始列，col2为矩阵结尾列
    int result = Integer.MIN_VALUE;
    for (int col1 = 0; col1 < n; col1++) {
        for (int col2 = col1; col2 < n; col2++) {
            // set中都是存放的startCol和endCol相同的矩阵的和
            TreeSet<Integer> set = new TreeSet<>();
            set.add(0);
            for (int row = 0; row < m; row++) {
                // 子矩阵的和
                int sum = 0;
                for (int i = col1; i <= col2; i++) {
                    sum += sums[row][i];
                }
                
                // 求出set中大于等于(sum-k)最小值
                if (set.ceiling(sum -k) != null) {
                    int max = sum -set.ceiling(sum - k);
                    result = result > max ? result : max;
                }
                set.add(sum);
            }
        }
    }
    return result;
}

day41:
33. 搜索旋转排序数组
给你一个整数数组 nums ，和一个整数 target 。
该整数数组原本是按升序排列，但输入时在预先未知的某个点上进行了旋转。（例如，数组 [0,1,2,4,5,6,7] 可能变为 [4,5,6,7,0,1,2] ）。
请你在数组中搜索 target ，如果数组中存在这个目标值，则返回它的索引，否则返回 -1 。
示例 1：
输入：nums = [4,5,6,7,0,1,2], target = 0
输出：4
示例 2：
输入：nums = [4,5,6,7,0,1,2], target = 3
输出：-1
示例 3：
输入：nums = [1], target = 0
输出：-1
题目出的不严谨，实则是想考二分法
思路一：直接一个for循环就可以找出来，时间复杂度O(N)，空间复杂度O(1)
但这样出题毫无意义
public int search(int[] nums, int target) {
    for (int i = 0; i < nums.length; i++) {
        if (nums[i] == target) {
            return i;
        }
    }
    return -1;
}
思路二:出题者的意思，考二分法
分两种情况，一种是nums[mid] >= nums[left]，第二种nums[mid] <= nums[right]
时间复杂度O(logn),空间复杂度：O(1)
public static int search(int[] nums, int target) {
    if (nums == null || nums.length < 1) {
        return -1;
    }
    int left = 0;
    int right = nums.length -1;
    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (target == nums[mid]) {
            return mid;
        }
        // 中间的值大于等于左边的值
        if (nums[mid] >= nums[left]) {
            if (target < nums[mid] && target >= nums[left]) {
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }

        // 中间的值小于等于右边的值
        if (nums[mid] <= nums[right]) {
            if (target > nums[mid] && target <= nums[right]) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
    }
    return -1;
}

day42
212. 单词搜索 II
给定一个 m x n 二维字符网格 board 和一个单词（字符串）列表 words，找出所有同时在二维网格和字典中出现的单词。
单词必须按照字母顺序，通过 相邻的单元格 内的字母构成，其中“相邻”单元格是那些水平相邻或垂直相邻的单元格。同一个单元格内的字母在一个单词中不允许被重复使用。
示例 1：
输入：board = [["o","a","a","n"],["e","t","a","e"],["i","h","k","r"],["i","f","l","v"]], words = ["oath","pea","eat","rain"]
输出：["eat","oath"]
示例 2：
输入：board = [["a","b"],["c","d"]], words = ["abcb"]
输出：[]
思路一：
这是一道hard题，原因是考察了：字典树和回溯法+动态规划
具体思路就是先实现字典树的数据结构，再用回溯求出结果。
public List<String> findWords(char[][] board, String[] words) {
    int row = board.length;
    int col = board[0].length;
    List<String> res = new ArrayList<String>();
    // 先判断遍历的入口，再判断这个入口是否包含这个单词
    // 包含则在res中加入这个单词
    for (int i = 0; i < words.length; i++) {
        if (exist(board,words[i])) {
            res.add(words[i]);
        }
    }
    return res;    
}

public boolean exist(char[][] board, String word) {
    for (int i=0; i < board.length; i++) {
        for (int j = 0; j < board[0].length; j++) {
            if (dfs(board, i, j, word, 0)) {
                return true;
            }
        }
    }
    return false;
}

private boolean dfs(char[][] board, int i, int j, String word, int index) {
    // 边界条件
    if (i < 0 || i >= board.length || j < 0 || j > board[0].length || board[i][j] != word.charAt(index)) {
        return false;
    }
    if (index == word.length()-1) {
        return true;
    }
    char temp = board[i][j];
    board[i][j] = '$';
    boolean up = dfs(board, i-1, j, word, index+1);
    if (up) {
        board[i][j] = temp;
        return true;
    }
    boolean down = dfs(board, i+1, j, word, index+1);
    if (down) {
        board[i][j] = temp;
        return true;
    }
    boolean left = dfs(board, i, j-1, word, index+1);
    if (left) {
        board[i][j] = temp;
        return true;
    }
    boolean right = dfs(board, i, j+1, word, index+1);
    if (right) {
        board[i][j] = temp;
        return true;
    }
    board[i][j] = temp;
    return false;
} 
```



```python
课后作业：
120. 三角形最小路径和
给定一个三角形，找出自顶向下的最小路径和。每一步只能移动到下一行中相邻的结点上。
相邻的结点 在这里指的是 下标 与 上一层结点下标 相同或者等于 上一层结点下标 + 1 的两个结点。
例如，给定三角形：
[
     [2],
    [3,4],
   [6,5,7],
  [4,1,8,3]
]
自顶向下的最小路径和为 11（即，2 + 3 + 5 + 1 = 11）。
思路一：分治的思想
1.brute-force,递归, n层：left or right
2.dp
a.重复性（分治） problem(i,j) = min(sub(i+1,j), sub(i+1,j+1)) + a[i,j]
b.定义状态数组: f[i,j] = 
c.dp方程:f[i,j] = min(f[i+1,j], f[i+1,j+1]) + a[i,j]
def minimumTotal(self, triangle: List[List[int]]) -> int:
    // 将triangle值赋给dp
    dp = triangle
    for i in range(len(triangle)-2, -1, -1):
        for j in range(len(triangle[i])):
            dp[i][j] += min(dp[i+1][j],dp[i+1][j+1])
    print(triangle[0][0])
	return dp[0][0]

def minimumTotal(self, triangle: List[List[int]]) -> int:
    // 初始化0矩阵
    f = [0] * (len(triangle) + 1)
    for row in triangle[::-1]:
        for i in range(len(row)):
            f[i] = row[i] + min(f[i], f[i+1])
    return f[0]

java
时间复杂度O(Nk),空间复杂度O(N)
public int minimumTotal(List<List<Integer>> triangle) {
    int[] A = new int[triangle.size() + 1];
    for (int i = triangle.size()-1;i>=0;i--) {
        for (int j = 0; j<triangle.get(i).size();j++) {
            A[j] = Math.min(A[j],A[j+1]) + triangle.get(i).get(j);
        }
    }
    return A[0];
}

递归：不建议 ，超时
int row;
public int minimumTotal(List<List<Integer>> triangle) {
    row = triangle.size();
    return helper(0,0,triangle);
}

private int helper(int level, int c, List<List<Integer>> triangle) {
    if (level == row - 1) {
        return triangle.get(lever).get(c);
    }
    int left = helper(lever+1,c,triangle);
    int right = helper(lever+1,c+1,triangle);
    return Math.min(left, right) + triangle.get(level).get(c);
}
```



```java

1143.最长公共子序列
给定两个字符串 text1 和 text2，返回这两个字符串的最长公共子序列的长度。
一个字符串的 子序列 是指这样一个新的字符串：它是由原字符串在不改变字符的相对顺序的情况下删除某些字符（也可以不删除任何字符）后组成的新字符串。
例如，"ace" 是 "abcde" 的子序列，但 "aec" 不是 "abcde" 的子序列。两个字符串的「公共子序列」是这两个字符串所共同拥有的子序列。
若这两个字符串没有公共子序列，则返回 0。

示例 1:
输入：text1 = "abcde", text2 = "ace" 
输出：3  
解释：最长公共子序列是 "ace"，它的长度为 3。

示例 2:
输入：text1 = "abc", text2 = "abc"
输出：3
解释：最长公共子序列是 "abc"，它的长度为 3。

示例 3:
输入：text1 = "abc", text2 = "def"
输出：0
解释：两个字符串没有公共子序列，返回 0。

思路一：
暴力求解，text1的所有子序列，看是否在text2中，时间复杂度可高达O(2^N).

思路二：
找重复性.

def longestCommonSubsequence(self, text1: str, text2: str) -> int:
	if not text1 or not text2:
		return 0
	m = len(text1)
	n = len(text2)
	dp = [[0]*(n+1) for _ in range(m+1)]
	for i in range(1, m+1):
		for j in range(1, n+1):
			if text1[i-1] == text2[j-1]:
				dp[i][j] = 1 + dp[i-1][j-1]
			else:
				dp[i][j] = max(dp[i-1][j],dp[i][j-1])
	return dp[m][n]	

public int longestCommonSubsequence(String text1, String text2) {
	int m = text1.length();
	int n = text2.length();
	int[][] dp = new int[m+1][n+1];
	for (int i = 1; i < m+1; i++) {
		for (int j = 1; j < n+1; j++) {
			if (text1.charAt(i-1) == text2.charAt(j-1)) {
				dp[i][j] = dp[i-1][j-1] + 1;
			} else {
				dp[i][j] = Math.max(dp[i-1][j],dp[i][j-1]);
			}
		}
	}
	return dp[m][n];
}
```

```java
198.打家劫舍
  你是一个专业的小偷，计划偷窃沿街的房屋。每间房内都藏有一定的现金，影响你偷窃的唯一制约因素就是相邻的房屋装有相互连通的防盗系统，如果两间相邻的房屋在同一晚上被小偷闯入，系统会自动报警。
  给定一个代表每个房屋存放金额的非负整数数组，计算你 不触动警报装置的情况下 ，一夜之内能够偷窃到的最高金额。
示例 1：
输入：[1,2,3,1]
输出：4
解释：偷窃 1 号房屋 (金额 = 1) ，然后偷窃 3 号房屋 (金额 = 3)。
偷窃到的最高金额 = 1 + 3 = 4 。
示例 2：
输入：[2,7,9,3,1]
输出：12
解释：偷窃 1 号房屋 (金额 = 2), 偷窃 3 号房屋 (金额 = 9)，接着偷窃 5 号房屋 (金额 = 1)。
偷窃到的最高金额 = 2 + 9 + 1 = 12 。
思路一：动态规划,时间复杂度O(N)，空间复杂度O(N)
增加一维，代表偷与不偷
a[i][0]:不偷，a[i][1]:偷
dp方程：
a[0][0] = 0; a[0][1] = nums[0]
a[i][0] = max(a[i-1][0],a[i-1][1])
a[i][1] = a[i-1][0] + nums[i]（相邻不能被偷）
public int rob(int[] nums) {
    // 边界条件
	if (nums == null || nums.length == 0) {
		return 0;
	}
    // 长度
	int n = nums.length;
    // 初始化
	int[][] a = new int[n][2];
	a[0][0] = 0;
    a[0][1] = nums[0];
    for (int i=1; i< n; i++){
        // 第i个未被偷，取i-1被偷和不被偷的最大值
        a[i][0] = Math.max(a[i-1][0], a[i-1][1]);
        a[i][1] = a[i-1][0] + nums[i];
    }
    return Math.max(a[n-1][0],a[n-1][1]);
}

思路二:简化思路一代码
定义一维数组表示结果,时间复杂度O(N)
a[i]:0..i能偷到的最大值，第i个可偷可不偷
a[i-1]偷，a[i-2]不偷，则第i个可偷
a[i] = Math.max(a[i-1], nums[i] + a[i-2]);
python(代码简洁，比较耗时)
def rob(self, nums: List[int]) -> int:
	pre = 0
    now = 0
    for i in nums:
		pre,now = now,max(pre+i,now)
    return now

```

```python
53. 最大子序和
给定一个整数数组 nums ，找到一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。

示例:

输入: [-2,1,-3,4,-1,2,1,-5,4]
输出: 6
解释: 连续子数组 [4,-1,2,1] 的和最大，为 6。
思路一:暴力，算出所有的和，然后取最小值，时间复杂度O(N**2),不推荐
思路二:分治，dp:
a.子问题 max_sum(i) = Max(max_sun(i-1), 0) + a[i]
b.状态数组定义 f[i]
c.dp方程 f[i] = Max(max_sun(i-1), 0) + a[i]
最大子序和 = 当前元素自身最大，或者包含之前 后最大
def maxSubArray(self, nums: List[int]) -> int:
    for i in range(1, len(nums)):
        # nums[i-1]代表dp[i-1]
        nums[i] = max(nums[i-1] + nums[i], nums[i])
    return max(nums)

def maxSubArray(self, nums: List[int]) -> int:
    for i in range(1, len(nums)):
        dp = nums
        # nums[i-1]代表dp[i-1]
        dp[i] = nums[i] + max(dp[i-1], 0)
    return max(dp)
```

```
322. 零钱兑换
给定不同面额的硬币 coins 和一个总金额 amount。编写一个函数来计算可以凑成总金额所需的最少的硬币个数。如果没有任何一种硬币组合能组成总金额，返回 -1。
你可以认为每种硬币的数量是无限的。
示例 1：
输入：coins = [1, 2, 5], amount = 11
输出：3 
解释：11 = 5 + 5 + 1
1.暴力:递归，指数
2.BFS
3.DP
a.子问题
b.dp array: f(n) = min{(f(n-k), for k in [1, 2, 5])} + 1
c.dp方程
def coinChange(self, coins: List[int], amount: int) -> int:
	MAX = float('inf')
	dp = [0] + [MAX] * amount
	
	for i in range(1, amount+1):
		dp[i] = min(dp[i - c] if i - c >=0 else MAX for c in coins) + 1
	
	return [dp[amount],-1][dp[amount] == MAX]
	

```

