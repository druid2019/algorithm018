学习笔记

```java
课后作业：
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

