学习笔记

```java
day43
74. 搜索二维矩阵
编写一个高效的算法来判断 m x n 矩阵中，是否存在一个目标值。该矩阵具有如下特性：
每行中的整数从左到右按升序排列。
每行的第一个整数大于前一行的最后一个整数。
示例1:
输入：matrix = [[1,3,5,7],[10,11,16,20],[23,30,34,50]], target = 3
输出：true
示例2:
输入：matrix = [[1,3,5,7],[10,11,16,20],[23,30,34,50]], target = 13
输出：false
public boolean searchMatrix(int[][] matrix, int target) {
	// 行
    int r = matrix.length;
	if (r == 0) return false;
    // 列
    int c = matrix[0].length - 1;
    if (c == -1) return false;
    for (int i = 0; i < r; i++) {
        int low = 0;
        int high = c;
        // 中间值
        int mid = 0;
        while (low <= high) {
            mid = (low + high) / 2;
            if (target == matrix[i][mid]) return true;
            if (target < matrix[i][mid]) high = mid - i;
            if (target > matrix[i][mid]) low = mid + 1;
        }
    }
    return false;
}
```



```java
day44
208. 实现 Trie (前缀树)
实现一个 Trie (前缀树)，包含 insert, search, 和 startsWith 这三个操作。
示例:
Trie trie = new Trie();

trie.insert("apple");
trie.search("apple");   // 返回 true
trie.search("app");     // 返回 false
trie.startsWith("app"); // 返回 true
trie.insert("app");   
trie.search("app");     // 返回 true
解这题前，先了解一下概念:
Trie树，又叫字典树、前缀树（Prefix Tree）、单词查找树或键树，是一种多叉树结构。
Trie树的基本性质： 
①根节点不包含字符，除根节点外的每一个子节点都包含一个字符。 
②从根节点到某一个节点，路径上经过的字符连接起来，为该节点对应的字符串。 
③每个节点的所有子节点包含的字符互不相同。 
④从第一字符开始有连续重复的字符只占用一个节点。

前缀树的应用
1、前缀匹配 
2、字符串检索 
3、词频统计 
4、字符串排序
class Trie {

    private class Node {
        // 子节点
        public HashMap<Character,Node> childs;
        public boolean isLeaf;

        public Node() {
            // 叶子
            this.isLeaf = false; 
            // 它的孩子
            this.childs = new HashMap<>();
        }
    }
    
    private Node root;

    /** Initialize your data structure here. */
    public Trie() {
        root = new Node();
    }
    
    /** Inserts a word into the trie. */
    public void insert(String word) {
        insert(root,word);
    }

    private void insert(Node root, String word) {
        if (word == null || word.length() == 0) return;
        char[] chars = word.toCharArray();
        Node cur = root;
        for (int i = 0; i < chars.length; i++) {
            // 如果不包含字符，就插入
            if (!cur.childs.containsKey(chars[i])) {
                cur.childs.put(chars[i], new Node());
            }
            // 获取子节点
            cur = cur.childs.get(chars[i]);
        }
        // 遍历完成，更新叶子位置，叶子主要用来search
        if(!cur.isLeaf) {
            cur.isLeaf = true;
        }
    }
    
    /** Returns if the word is in the trie. */
    public boolean search(String word) {
        return search(root,word);
    }

    public boolean search(Node root, String word) {
        if (word == null || word.length() == 0) return false;
        char[] chars = word.toCharArray();
        Node cur = root;
        for (int i = 0; i < chars.length; i++) {
            // 其中一个字符不匹配就不行
            if(!cur.childs.containsKey(chars[i])) {
                return false;
            }
            cur = cur.childs.get(chars[i]);
        }
        //前边匹配了 ，还要这个还是叶子才行
        return cur.isLeaf;
    }
    
    /** Returns if there is any word in the trie that starts with the given prefix. */
    public boolean startsWith(String prefix) {
        if(prefix == null || prefix.length() == 0) return false;
        char[] chars = prefix.toCharArray();
        Node cur = root;
        for (int i = 0; i < chars.length; i++) {
            if(!cur.childs.containsKey(chars[i])) return false;
            cur = cur.childs.get(chars[i]);
        }
        return true;
    }
}
```

```java
day45
200. 岛屿数量
给你一个由 '1'（陆地）和 '0'（水）组成的的二维网格，请你计算网格中岛屿的数量。
岛屿总是被水包围，并且每座岛屿只能由水平方向和/或竖直方向上相邻的陆地连接形成。
此外，你可以假设该网格的四条边均被水包围。
示例 1：

输入：grid = [
  ["1","1","1","1","0"],
  ["1","1","0","1","0"],
  ["1","1","0","0","0"],
  ["0","0","0","0","0"]
]
输出：1
示例 2：

输入：grid = [
  ["1","1","0","0","0"],
  ["1","1","0","0","0"],
  ["0","0","1","0","0"],
  ["0","0","0","1","1"]
]
输出：3
思路一：
dfs，尝试深度优先遍历,每次遍历1时，赋值给0，直到全部为0，遍历求数量
private int m;
private int n;

public int numIslands(char[][] grid) {
	int count = 0;
	// 行长度
	m = grid.length;
	if (m == 0) return 0;
	// 列长度
	n = grid[0].length;
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			if (grid[i][j] == '1') {
                dfsHaddle(grid, i, j);
                count++;
			}			
		}
	}
    return count;
}

public void dfsHaddle (char[][] grid, int i, int j) {
	// 遍历终止条件
	if (i < 0 || j < 0 || i >= m || j >= n || grid[i][j] == '0') {
		return 0;
	}
	grid[i][j] = '0';
	// 遍历过程，相邻的值处理
	dfsHaddle(grid, i + 1, j);
	dfsHaddle(grid, i - 1, j);
	dfsHaddle(grid, i, j + 1);
	dfsHaddle(grid, i, j - 1);
}

```

```java
day46
53. 最大子序和
给定一个整数数组 nums ，找到一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。
示例:
输入: [-2,1,-3,4,-1,2,1,-5,4]
输出: 6
解释: 连续子数组 [4,-1,2,1] 的和最大，为 6。
思路一：动态规划，取上一个最大值,时间复杂度O(N),空间复杂度O(N)
求各个阶段子序列和，取最大值
public int maxSubArray(int[] nums) {
	int len = nums.length;
	int[] dp = new int[len];
	dp[0] = nums[0];
	int result = nums[0];
	for (int i = 1; i < len; i++) {
		dp[i] = Math.max(nums[i], nums[i] + dp[i-1]);
		result = Math.max(result,dp[i]);
	}
	return result;
}

```

