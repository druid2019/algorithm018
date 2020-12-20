学习笔记

```python
day 57
387. 字符串中的第一个唯一字符
给定一个字符串，找到它的第一个不重复的字符，并返回它的索引。如果不存在，则返回 -1。
示例：
s = "leetcode"
返回 0
s = "loveleetcode"
返回 2
方法一：直接调用python的count方法，但是比较耗时
时间复杂度O(n)
def firstUniqChar(self, s: str) -> int:
    for i in s:
        if s.count(i) == 1:
            return s.index(i)
    return -1


# collections.Counter(s)似乎更快
def firstUniqChar(self, s: str) -> int:
        """
        :type s: str
        :rtype: int
        """
        # build hash map : character and how often it appears
        count = collections.Counter(s)
        
        # find the index
        for idx, ch in enumerate(s):
            if count[ch] == 1:
                return idx
        return -1

```

```java
方法二：
用长度26的数组存放重复次数
public int firstUniqChat(String s) {
	int[] nums = new int[26];
	char[] chars = s.toCharArray();
	for (char c: chars) {
		nums[c - 'a']++;
	}
	for (int i = 0; i < chars.length; i++) {
		if (nums[chars[i] - 'a'] == 1) {
			return i;
		}
	}
	return -1;
}
```



```
day58
一个机器人位于一个 m x n 网格的左上角 （起始点在下图中标记为 “Start” ）。
机器人每次只能向下或者向右移动一步。机器人试图达到网格的右下角（在下图中标记为 “Finish” ）。
问总共有多少条不同的路径？
示例 1：
输入：m = 3, n = 7
输出：28
示例 2：
输入：m = 3, n = 2
输出：3
解释：
从左上角开始，总共有 3 条路径可以到达右下角。
1. 向右 -> 向右 -> 向下
2. 向右 -> 向下 -> 向右
3. 向下 -> 向右 -> 向右

方法一:dp
dp[i][j] = dp[i-1][j] + dp[i][j-1]
public int uniquePaths(int m, int n) {
	int dp[][] = new int[m][n];
	// 边界值
	for (int i = 0; i < m; i++) {
		dp[i][0] = 1;
	}
	for (int j = 0; j < n; j++) {
		dp[0][j] = 1;
	}
	for (int i = 1; i < m; i++) {
		for (int j = 1; j < n; j++) {
			dp[i][j] = dp[i-1][j] + dp[i][j-1];
		}
	}
	return dp[m-1][n-1];
}
```

```python
day59
541. 反转字符串 II
给定一个字符串 s 和一个整数 k，你需要对从字符串开头算起的每隔 2k 个字符的前 k 个字符进行反转。
示例:
输入: s = "abcdefg", k = 2
输出: "bacdfeg"
方法一:暴力
可以直接调python的reverse函数,时间及空间复杂度都为O(N)
代码比较简洁
def reverseStr(self, s, k):
    a = list(s)
    for i in range(0, len(a), 2*k):
        a[i:i+k] = reversed(a[i:i+k])
    return "".join(a)
```

```java
方法二:
交换元素，i + k - 1,与n - 1做比较
public String reverseStr(String s, int k) {
	char[] chars = s.toCharArray();
	int len = chars.length;
	for (int i = 0; i < len; i += 2 * k) {
		reverse(chars, i, Math.min(i + k - 1, len - 1));
	}
	return new String(chars);
}

public void reverse(char[] chars, int i, int j) {
	for (int t = i; t < j ; t++) {
		char tmp = chars[t];
        chars[t] = chars[j];
        chars[j] = tmp;
        j--;
	}
}
```

```java
day60
300. 最长递增子序列
给你一个整数数组 nums ，找到其中最长严格递增子序列的长度。
子序列是由数组派生而来的序列，删除（或不删除）数组中的元素而不改变其余元素的顺序。例如，[3,6,2,7] 是数组 [0,3,1,6,2,2,7] 的子序列。
示例 1：
输入：nums = [10,9,2,5,3,7,101,18]
输出：4
解释：最长递增子序列是 [2,3,7,101]，因此长度为 4 。
示例 2：
输入：nums = [0,1,0,3,2,3]
输出：4
示例 3：
输入：nums = [7,7,7,7,7,7,7]
输出：1
思路一：dp方程
dp[i] = dp[i - 1] + 1 
public int lengthOfLIS(int[] nums) {
	if (nums == null || nums.length == 0) return 0;
    int[] dp = new int[nums.length];
    Arrays.fill(1, dp);
    for (int i = 1; i < nums.length; i++) {
        for (int j = 0; j < i; j++) {
            if (nums[i] >  nums[j]) {
                dp[i] = Math.max(dp[i], dp[j] + 1);
            }
        }
    }
    int res = 1;
    for (int i =0; i < dp.length; i++) {
        res = Math.max(res, dp[i]);
    }
    return res;
}
```

```java
day61
15. 三数之和
给你一个包含 n 个整数的数组 nums，判断 nums 中是否存在三个元素 a，b，c ，使得 a + b + c = 0 ？请你找出所有满足条件且不重复的三元组。
示例：
给定数组 nums = [-1, 0, 1, 2, -1, -4]，
满足要求的三元组集合为：
[
  [-1, 0, 1],
  [-1, -1, 2]
]
思路一:
public List<List<Integer>> threeSum(int[] nums) {
	Arrays.sort(nums);
    List<List<Integer>> res = new ArrayList<>();
    int len = nums.length;
    for (int i = 0; i < len; i++) {
        if (nums[i] > 0) break;
        if (i > 0 && nums[i] == nums[i-1]) continue;
        int left = i + 1;
        int right = len - 1;
        int sum = -nums[i];
        while (left < right) {
            int count = nums[left] + nums[right];
            if (count == sum) {
                res.add(Arrays.asList(nums[left], nums[right],nums[i]));
                // 相邻重复元素
                while (left < right && nums[left] == nums[left + 1]) left++;
                while (left < right && nums[right] == nums[right - 1]) right--;
                left++;
                right--;
            } else if (count < sum) {
                left++;
            } else {
                right--;
            }
        }
    }
    return result;
}
```

```python
day62
680. 验证回文字符串 Ⅱ
给定一个非空字符串 s，最多删除一个字符。判断是否能成为回文字符串。
示例 1:
输入: "aba"
输出: True
示例 2:
输入: "abca"
输出: True
解释: 你可以删除c字符。
思路一：
如果回文，则返回
如果非回文，去掉回文，然后比较删除一个字符后是否回文
def validPalindrome(self, s):
    """
    :type s: str
    :rtype: bool
    """
    if s == s[::-1]:
        return True
    l, r = 0, len(s) - 1
    while l < r:
        if s[l] == s[r]:
            l, r = l + 1, r - 1
            else:
                a = s[l + 1 : r + 1]
                b = s[l:r]
                return a == a[::-1] or b==b[::-1]
```

```java
day63
32. 最长有效括号
给定一个只包含 '(' 和 ')' 的字符串，找出最长的包含有效括号的子串的长度。
示例 1:
输入: "(()"
输出: 2
解释: 最长有效括号子串为 "()"
示例 2:
输入: ")()())"
输出: 4
解释: 最长有效括号子串为 "()()"
思路一: 栈方法
public int longestValidParentheses(String s) {
	int res = 0, start = 0;
	if (s == null) return 0;
	int len = s.length();
	//遇左括号(，压栈(栈中元素为当前位置所处的下标)
	Stack<Integer> stack = new Stack<>();
	for (int i = 0; index < len; i++) {
		if ('(' == s.charAt(i)) {
			stack.push(i);
			continue;
		} else {
			if (stack.isEmpty()) {
				start = i + 1;
				continue;
			} else {
				stack.pop();
				if (stack.isEmpty()) {
					res = Math.max(res, i - start + 1);
				} else {
					res = Math.max(res, i - stack.peek());
				}
			}
		}
	}	
	return res;
}
思路二：
动态规划
当前字符下标为index，若当前字符为左括号'('，判断index+1+d[index+1]位置的字符是否为右括号')'，若为右括号，则d[index] = d[index+1]+2，并且判断index+1+d[index+1]+1位置的元素是否存在，若存在，则d[index] += d[index+1+d[index+1]+1]（解决上述两个有效括号子串直接相邻的情况）
public int longestValidParentheses(String s) {
	if (s == null) return 0;
	
	int res = 0,len = s.length();
	int[] d = new int[len];
	
	for (int i = len - 2; i >= 0; i--) {
		int symIndex = i + 1 + d[i + 1];
		if ('(' == s.charAt(i) && symIndex < len && ')' == s.charAt(symIndex)) {
			d[i] = d[i + 1] + 2;
			if (symIndex + 1 < len) {
				d[i] += d[symIndex + 1];
			}
		}
		res = Math.max(res, d[i]);
	}
	return res;
} 
```



```java
课后作业
72. 编辑距离
给你两个单词 word1 和 word2，请你计算出将 word1 转换成 word2 所使用的最少操作数 。
示例 1：
输入：word1 = "horse", word2 = "ros"
输出：3
解释：
horse -> rorse (将 'h' 替换为 'r')
rorse -> rose (删除 'r')
rose -> ros (删除 'e')
示例 2：
输入：word1 = "intention", word2 = "execution"
输出：5
解释：
intention -> inention (删除 't')
inention -> enention (将 'i' 替换为 'e')
enention -> exention (将 'n' 替换为 'x')
exention -> exection (将 'n' 替换为 'c')
exection -> execution (插入 'u')

1.BFS, two-ended BFS
2.DP
edit_dist(i, j) = edit_dist(i - 1, j - 1) if w1[i] == w2[j] // 分治
dp[i][j]表示word1.substr(0,i)与word2.substr(0,j)之间的距离
edit_dist(i, j) = 
MIN(edit_dist(i - 1, j - 1) + 1,
    edit_dist(i - 1, j) + 1,
    edit_dist(i, j - 1) + 1)
```

```python
def minDistance(self, word1:str, word2:str) -> int:
	n1 = len(word1)
	n2 = len(word2)
	dp = [[0] * (n2 + 1) for _ in range(n1 + 1)]
	# 第一行
    for j in range(1, n2 + 1):
        dp[0][j] = dp[0][j-1] + 1
    # 第一列
    for i in range(1, n1 + 1):
        dp[i][0] = dp[i - 1][0] + 1
    for i in range(1, n1 + 1):
        for j in range(1, n2 + 1):
            if word1[i - 1] == word2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i][j - 1], dp[i - 1][j], dp[i - 1][j - 1]) + 1
    return dp[-1][-1] 
```

```java
5. 最长回文子串
给定一个字符串 s，找到 s 中最长的回文子串。你可以假设 s 的最大长度为 1000。
示例 1：
输入: "babad"
输出: "bab"
注意: "aba" 也是一个有效答案。
示例 2：
输入: "cbbd"
输出: "bb"
思路一：
暴力算法，从中间向两边扩散
private int lo, maxLen;
public String longestPalindrome(String s) {
	int len = s.length();
    if (len < 2) return s;
    
    for (int i = 0; i < len - 1; i++) {
        extendPalindrome(s, i, i); // odd length
        extendPalindrome(s, i, i+1); // even length
    }
    return s.substring(lo, lo + maxLen);
}

private void extendPalindrome(String s, int j, int k) {
	while (j >= 0 && k < s.length() && s.charAt(j) == s.charAt(k)) {
        j--; k++
    }
    
    if (maxLen < k - j - 1) {
        lo = j + 1;
        maxLen = k - j - 1;
    }
}

思路二:
动态规划
P(i, j) = (P(i+1, j-1) && s[i] == s[j])
public String longestPalindrome(String s) {
	int n = s.length();
	String res = "";
	boolean[][] dp = new boolean[n][n];
	for (int i = n - 1; i >= 0; i--) {
		for (int j = i; j < n; j++) {
			dp[i][j] = s.charAt(i) == s.charAt(j) && (j - i < 2 || dp[i + 1][j - 1]);
			if (dp[i][j] && j - i + 1 > res.length()) {
				res = s.substring(i, j + 1);
			}
		}
	}
	return res;
}
```



```java
115. 不同的子序列
给定一个字符串 s 和一个字符串 t ，计算在 s 的子序列中 t 出现的个数。
字符串的一个 子序列 是指，通过删除一些（也可以不删除）字符且不干扰剩余字符相对位置所组成的新字符串。（例如，"ACE" 是 "ABCDE" 的一个子序列，而 "AEC" 不是）
题目数据保证答案符合 32 位带符号整数范围。
示例 1：

输入：s = "rabbbit", t = "rabbit"
输出：3
解释：
如下图所示, 有 3 种可以从 s 中得到 "rabbit" 的方案。
(上箭头符号 ^ 表示选取的字母)
rabbbit
^^^^ ^^
rabbbit
^^ ^^^^
rabbbit
^^^ ^^^

示例 2：

输入：s = "babgbag", t = "bag"
输出：5
解释：
如下图所示, 有 5 种可以从 s 中得到 "bag" 的方案。 
(上箭头符号 ^ 表示选取的字母)
babgbag
^^ ^
babgbag
^^    ^
babgbag
^    ^^
babgbag
  ^  ^^
babgbag
    ^^^

思路一:
dp
public int numDistinct(String s, String t) {
	int[][] dp = new int[t.length() + 1][s.length() + 1];
	for (int i = 0 ; i < s.length() + 1; i++) {
		dp[0][i] = 1;
	}
	for (int i = 1; i < t.length() + 1; i++) {
		for (int j = i ; j < s.length() + 1; j++) {
			if (t.charAt(i - 1) == s.charAt(j - 1)) {
				dp[i][j] = dp[i][j - 1] + dp[i - 1][j - 1];
			} else {
				dp[i][j] = dp[i][j - 1];
			}
		}
	}
	return dp[t.length()][s.length()];
}

```

```
def numDistinct(self, s: str, t: str) -> int:
	n1 = len(s)
	n2 = len(t)
	dp = [[0] * (n1 + 1) for _ in range(n2 + 1)]
	for j in range(n1 + 1):
		dp[0][j] = 1
	for i in range(1, n2 + 1):
		for j in range(1, n1 + 1):
			if t[i - 1]  == s[j - 1]:
				dp[i][j] = dp[i - 1][j - 1] + dp[i][j - 1]
			else:
				dp[i][j] = dp[i][j - 1]
	return dp[-1][-1]
		
```



