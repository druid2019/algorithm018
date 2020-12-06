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

```java
day47
127. 单词接龙
给定两个单词（beginWord 和 endWord）和一个字典，找到从 beginWord 到 endWord 的最短转换序列的长度。转换需遵循如下规则：

每次转换只能改变一个字母。
转换过程中的中间单词必须是字典中的单词。
说明:

如果不存在这样的转换序列，返回 0。
所有单词具有相同的长度。
所有单词只由小写字母组成。
字典中不存在重复的单词。
你可以假设 beginWord 和 endWord 是非空的，且二者不相同。
示例 1:

输入:
beginWord = "hit",
endWord = "cog",
wordList = ["hot","dot","dog","lot","log","cog"]
输出: 5
解释: 一个最短转换序列是 "hit" -> "hot" -> "dot" -> "dog" -> "cog",返回它的长度 5。
示例 2:
输入:
beginWord = "hit"
endWord = "cog"
wordList = ["hot","dot","dog","lot","log"]
输出: 0
解释: endWord "cog" 不在字典中，所以无法进行转换。
思路一：求最短长度，用bfs,逐层寻找与上一层只差一个字符的单词，当单词与目标单词一致时返回当前层数。
可采取队列来实现广度优先
时间复杂度：O(N×C²)。其中 N为 wordList 的长度，C为列表中单词的平均长度。
空间复杂度：O(N×C²)。
public int ladderLength(String beginWord, String endWord, List<String> wordList) {
	if(!wordList.contains(endWord)) {
		return 0;
	}
	// 初始化序列长度
	int res = 1;
	// 在原字典中去掉beginWord
	wordList.remove(beginWord);
	 // 用队列去存储每层的单词
	Queue<String> queue = new LinkedList<String>();
	queue.add(beginWord);
	while(!queue.isEmpty()) {
		// 读取当前层有多少单词需要遍历
		int size = queue.size();
		while (size > 0) {
			String temp = queue.poll();
			// 如果和目标单词一致则返回层数
			if(temp.equals(endWord)) {
				return res;
			}
			// 找到wordList中和当前单词只差一位的加入队列
			find(queue, temp, wordList);
			// 每次遍历减一
			size--;
		}
		// 遍历次数
		res++;
	}
	return 0;
}

public void find(Queue<String> queue, String temp, List<String> wordList) {
	for (int i = 0; i < wordList.size(); i++) {
		// 获取wordList的元素
		String s = wordList.get(i);
		// 初始化相同元素的数量
		int count = 0;
		// 比较字典中的单词元素和所给的单词匹配度
		for (int m = 0; m < s.length(); m++) {
			if (temp.charAt(m) != s.charAt(m)) {
				count++;
			}
		}
		// 如果当前单词只与temp差一位
		if (count == 1) {
			// 将其加入队列
			queue.add(s);
			// 并从wordList中移除
			wordList.remove(s);
			// 移除后wordList中所有的会向前移一位，故i--保证从下一个开始遍历
			i--;
		}
	}
}
```

```java
day48
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
采用双指针，先排序
时间复杂度O(n^2)
public List<List<Integer>> threeSum(int[] nums) {
	// 排序
	Arrays.sort(nums);
	// 定义结果集
	List<List<Integer>> result = new ArrayList<>();
	// 长度
	int len = nums.length;
	for (int i = 0; i < len; i++) {
		// 当第i个数开始大于0时，跳出循环
		if (nums[i] > 0) break;
		// 当相邻数相等时，跳过该数，避免重复计算
		if (i > 0 && nums[i] == nums[i-1]) {
			continue;
		}
		// 使用双指针，向中间靠拢
		int left = i + 1;
		int right = len - 1;
		int sum = -nums[i];
		while(left < right) {
			int count = nums[left] + nums[right];
			if (count == sum) {
				result.add(Arrays.asList(nums[i], nums[left], nums[right]));
				// 相邻重复元素
				while (left < right && nums[left] == nums[left+1]) left++;
				while (left < right && nums[right] == nums[right-1]) right--;
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

```java
day49
22. 括号生成
数字 n 代表生成括号的对数，请你设计一个函数，用于能够生成所有可能的并且 有效的 括号组合。
示例：
输入：n = 3
输出：[
       "((()))",
       "(()())",
       "(())()",
       "()(())",
       "()()()"
     ]
思路一:
递归
private List<String> result;
public List<String> generateParenthesis(int n) {
    result = new ArrayList();
    generate(0, 0, n, "");
    return result;
}

public void generate(int left, int right, int n, String s) {
    // 边界条件
    if (left == n && right == n) {
        result.add(s);
        return;
    }
    
    // 当前处理逻辑
    if (left < n) {
        generate(left+1, right, n, s + "(");
    }
    if (right < left) {
        generate(left, right + 1, n, s + ")");
    }
}
```



```python
课后作业
212. 单词搜索 II
给定一个 m x n 二维字符网格 board 和一个单词（字符串）列表 words，找出所有同时在二维网格和字典中出现的单词。

单词必须按照字母顺序，通过 相邻的单元格 内的字母构成，其中“相邻”单元格是那些水平相邻或垂直相邻的单元格。同一个单元格内的字母在一个单词中不允许被重复使用。
示例1
输入：board = [["o","a","a","n"],["e","t","a","e"],["i","h","k","r"],["i","f","l","v"]], words = ["oath","pea","eat","rain"]
输出：["eat","oath"]
示例2
输入：board = [["a","b"],["c","d"]], words = ["abcb"]
输出：[]
思路一：
1.word遍历 --  board search
O(N*m*m*4^k)
思路二:
trie
a.all words -- Trie 构建prefix
b.board, DFS
dx = [-1,1,0,0]
dy = [0,0,-1,1]
END_OF_WORD = "#"
class Solution:
    def findWords(self, board, words):
        if not board or not board[0]:return []
        if not words:return []
        self.result = set()

        #构建trie
        root = collections.defaultdict()
        for word in words:
            node = root
            for char in word:
                node = node.setdefault(char, collections.defaultdict())
            node[END_OF_WORD] = END_OF_WORD

        self.m, self.n = len(board), len(board[0])
        for i in xrange(self.m):
        	for j in xrange(self.n):
            	if board[i][j] in root:
                	self._dfs(board, i, j, "", root)
       return list(self.result)

	def _dfs(self,board,i,j,cur_word,cur_dict):
        cur_word += board[i][j]
        cur_dict = cur_dict[board[i][j]]
        if END_OF_WORD in cur_dict:
            self.result.add(cur_word)
        tmp,board[i][j] = board[i][j],'@'
        for k in xrange(4):
            x,y = i + dx[k],j + dy[k]
            if 0 <= x <self.m and 0 <= y < self.n and board[x][y] !='@' and board[x][y] in cur_dict:
                self._dfs(board,x,y,cur_word,cur_dict)
        board[i][j] = tmp
       
```

```java
547.朋友圈
班上有 N 名学生。其中有些人是朋友，有些则不是。他们的友谊具有是传递性。如果已知 A 是 B 的朋友，B 是 C 的朋友，那么我们可以认为 A 也是 C 的朋友。所谓的朋友圈，是指所有朋友的集合。

给定一个 N * N 的矩阵 M，表示班级中学生之间的朋友关系。如果M[i][j] = 1，表示已知第 i 个和 j 个学生互为朋友关系，否则为不知道。你必须输出所有学生中的已知的朋友圈总数。

示例 1：
输入：
[[1,1,0],
 [1,1,0],
 [0,0,1]]
输出：2 
解释：已知学生 0 和学生 1 互为朋友，他们在一个朋友圈。
第2个学生自己在一个朋友圈。所以返回 2 。
示例 2：
输入：
[[1,1,0],
 [1,1,1],
 [0,1,1]]
输出：1
解释：已知学生 0 和学生 1 互为朋友，学生 1 和学生 2 互为朋友，所以学生 0 和学生 2 也是朋友，所以他们三个在一个朋友圈，返回 1 。
1.DFS,BFS (类似岛屿问题)
DFS
public int findCircleNum(int[][] M) {
	/**
	使用visited数组，一次判断每个节点
	如果未访问，朋友圈加一并堆该节点进行dfs搜索标记所有访问到的节点
	*/	
	boolean[] visited = new boolean[M.length];
	int ret = 0;
	for (int i = 0; i < M.length; i++) {
		if (!visited[i]) {
			dfs(M, visited, i);
			ret++;
		}
	}
	return ret;
}

private void dfs(int[][] m, boolean[] visited, int i) {
	for (int j = 0; j < m.length; j++) {
		if (m[i][j] == 1 && !visited[j]) {
			visited[j] = true;
			dfs(m, visited, j);
		}
	}
}

2.并查集
```



```python
37. 解数独
编写一个程序，通过填充空格来解决数独问题。
一个数独的解法需遵循如下规则：
数字 1-9 在每一行只能出现一次。
数字 1-9 在每一列只能出现一次。
数字 1-9 在每一个以粗实线分隔的 3x3 宫内只能出现一次。
空白格用 '.' 表示。
一个数独。
def solveSudoku(self, board:List[List[str]]) -> None:
    # 行剩余可用数字
    row = [set(range(1, 10)) for _ in range(9)]
    # 列剩余可用数字
    col = [set(range(1, 10)) for _ in range(9)]
    # 块剩余可用数字
    block = [set(range(1, 10)) for _ in range(9)]
    
    # 收集需填数的位置
    empty = []
    for i in range(9):
        for j in range(9):
            # 更新可用数字
            if board[i][j] != '.':
                val = int(board[i][j])
                row[i].remove(val)
                col[j].remove(val)
                block[(i // 3)*3 + j // 3].remove(val)
            else:
                empty.append((i, j))
                
     def backtrack(iter = 0):
        # 处理完empty代表找到了答案
        if iter == len(empty):
            return True
        i, j = empty[iter]
        b = (i // 3) * 3 + j // 3
        for val in row[i] & col[j] & block[b]:
            row[i].remove(val)
            col[j].remove(val)
            block[b].remove(val)
            board[i][j] = str(val)
            if backtrack(iter + 1):
                return True
            # 回溯
            row[i].add(val)
            col[j].add(val)
            block[b].add(val)
        return False
    backtrack()
```
