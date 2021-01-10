```java
day76
509. 斐波那契数
斐波那契数，通常用 F(n) 表示，形成的序列称为 斐波那契数列 。该数列由 0 和 1 开始，后面的每一项数字都是前面两项数字的和。也就是：
F(0) = 0，F(1) = 1
F(n) = F(n - 1) + F(n - 2)，其中 n > 1
给你 n ，请计算 F(n) 。
示例 1：
输入：2
输出：1
解释：F(2) = F(1) + F(0) = 1 + 0 = 1

示例 2：
输入：3
输出：2
解释：F(3) = F(2) + F(1) = 1 + 1 = 2

示例 3：
输入：4
输出：3
解释：F(4) = F(3) + F(2) = 2 + 1 = 3
思路一：
迭代，容易堆栈溢出
时间复杂度O(2^n)
public int fib(int n) {
	if (n == 0 || n == 1) {
		return n;
	}
    return fib(n - 1) + fib(n - 2);
}
思路二：
记忆化递归
时间复杂度O(n)
public int fib(int n) {
	if (n == 0 || n == 1) {
		return n;
	}
	int first = 0;
	int second = 1;
	int result = 1;
	for (int i = 2; i < n; i++) {
		first = second;
		second = result;
		result = first + second;
	}
	return result;
}
```

```java
day77
830. 较大分组的位置
在一个由小写字母构成的字符串 s 中，包含由一些连续的相同字符所构成的分组。
例如，在字符串 s = "abbxxxxzyy" 中，就含有 "a", "bb", "xxxx", "z" 和 "yy" 这样的一些分组。
分组可以用区间 [start, end] 表示，其中 start 和 end 分别表示该分组的起始和终止位置的下标。上例中的 "xxxx" 分组用区间表示为 [3,6] 。
我们称所有包含大于或等于三个连续字符的分组为 较大分组 。
找到每一个 较大分组 的区间，按起始位置下标递增顺序排序后，返回结果。
示例 1：
输入：s = "abbxxxxzzy"
输出：[[3,6]]
解释："xxxx" 是一个起始于 3 且终止于 6 的较大分组。

示例 2：
输入：s = "abc"
输出：[]
解释："a","b" 和 "c" 均不是符合要求的较大分组。

示例 3：
输入：s = "abcdddeeeeaabbbcd"
输出：[[3,5],[6,9],[12,14]]
解释：较大分组为 "ddd", "eeee" 和 "bbb"
思路一：
一次遍历
如果下一个字符与当前字符不同，或者已经枚举到字符串尾部，就说明当前字符为当前分组的尾部。每次找到当前分组的尾部时，如果该分组长度达到 33，我们就将其加入结果集中。
public List<List<Integer>> largeGroupPositions(String s) {
	List<List<Integer>> res = new ArrayList<List<Integer>>;
	int n = s.length();
	int num = 1;
	for (int i = 0; i < n; i++) {
		if (i == n - 1 || s.charAt(i) != s.charAt(i + 1)) {
			if (num >= 3) {
				res.add(Arrays.asList(i - num + 1, i))
			}
			num = 1;
		} else {
			num++;
		}
	}
	return res;
}
```



```java
day78
399. 除法求值
给你一个变量对数组 equations 和一个实数值数组 values 作为已知条件，其中 equations[i] = [Ai, Bi] 和 values[i] 共同表示等式 Ai / Bi = values[i] 。每个 Ai 或 Bi 是一个表示单个变量的字符串。
另有一些以数组 queries 表示的问题，其中 queries[j] = [Cj, Dj] 表示第 j 个问题，请你根据已知条件找出 Cj / Dj = ? 的结果作为答案。
返回 所有问题的答案 。如果存在某个无法确定的答案，则用 -1.0 替代这个答案。
示例 1：
输入：equations = [["a","b"],["b","c"]], values = [2.0,3.0], queries = [["a","c"],["b","a"],["a","e"],["a","a"],["x","x"]]
输出：[6.00000,0.50000,-1.00000,1.00000,-1.00000]
解释：
条件：a / b = 2.0, b / c = 3.0
问题：a / c = ?, b / a = ?, a / e = ?, a / a = ?, x / x = ?
结果：[6.0, 0.5, -1.0, 1.0, -1.0 ]
示例 2：
输入：equations = [["a","b"],["b","c"],["bc","cd"]], values = [1.5,2.5,5.0], queries = [["a","c"],["c","b"],["bc","cd"],["cd","bc"]]
输出：[3.75000,0.40000,5.00000,0.20000]
示例 3：
输入：equations = [["a","b"]], values = [0.5], queries = [["a","b"],["b","a"],["a","c"],["x","y"]]
输出：[0.50000,2.00000,-1.00000,-1.00000]
思路一：
并查集
public double[] calcEquation(List<List<String>> equations, double[] values, List<List<String>> queries) {
	int equationsSize = equations.size();
	
	UnionFind unionFind = new UnionFind(2 * equationsSize);
	// 第一步：预处理，将变量的值与id进行映射，使得并查集的底层使用数组实现，方便编码
	Map<String, Integer> hashMap = new HashMap<>(2 * equationsSize);
	int id = 0;
	for (int i = 0; i < equationsSize; i++) {
		List<String> equation = equations.get(i);
		String var1 = equation.get(0);
		String var2 = equation.get(1);
		
		if (!hashMap.containsKey(var1)) {
			hashMap.put(var1, id);
			id++;
		}
		if (!hashMap.containsKey(var2)) {
			hashMap.put(var2, id);
			id++;
		}
		unionFind.union(hashMap.get(var1), hashMap.get(var2), values[i]);
	}
	
	// 第2步：做查询
	int queriesSize = queries.size();
	double[] res = new double[queriesSize];
	for (int i = 0; i < queriesSize; i++) {
		String var1 = queries.get(i).get(0);
		String var2 = queries.get(i).get(1);
		
		Integer id1 = hashMap.get(var1);
		Integer id2 = hashMap.get(var2);
		
		if (id1 == null || id2 == null) {
			res[i] = -1.0d;
		} else {
			res[i] = unionFind.isConnected(id1, id2);
		}
	}
	return res;
}

private class UnionFind {
	private int[] parent;
	/**
	 * 指向的父节点的权值
	*/
	private double[] weight;
	
	public UnionFind(int n) {
		this.parent = new int[n];
		this.weight = new double[n];
		for (int i = 0; i < n; i++) {
			parent[i] = i;
			weight[i] = 1.0d;
		}
	}
	
	public void union(int x, int y, double value) {
		int rootX = find(x);
		int rootY = find(y);
		if (rootX == rootY) {
			return;
		}
		
		parent[rootX] = rootY;
		// 关系式的推导请见[参考代码]下方的示意图
		weight[rootX] = weight[y] * value / weight[x];
	}
	
	/**
    * 路径压缩
    *
    * @param x
    * @return 根结点的 id
    */
    public int find(int x) {
    	if (x != parent[x]) {
    		int origin = parent[x];
    		parent[x] = find(parent[x]);
    		weight[x] *= weight[origin];
    	}
    	return parent[x];
    }

	public double isConnected(int x, int y) {
		int rootX = find(x);
		int rootY = find(y);
		if (rootX == rootY) {
			return weight[x] / weight[y];
		} else {
			return -1.0d;
		}
	}	
}
```



```java
day79
547. 省份数量
有 n 个城市，其中一些彼此相连，另一些没有相连。如果城市 a 与城市 b 直接相连，且城市 b 与城市 c 直接相连，那么城市 a 与城市 c 间接相连。
省份是一组直接或间接相连的城市，组内不含其他没有相连的城市。
给你一个 n x n 的矩阵 isConnected ，其中 isConnected[i][j] = 1 表示第 i 个城市和第 j 个城市直接相连，而 isConnected[i][j] = 0 表示二者不直接相连。
返回矩阵中省份的数量。
示例 1：
输入：isConnected = [[1,1,0],[1,1,0],[0,0,1]]
输出：2
示例 2：
输入：isConnected = [[1,0,0],[0,1,0],[0,0,1]]
输出：3
public int findCircleNum(int[][] M) {
	boolean[] visited = new boolean[M.length];
	int res = 0;
	for (int i = 0; i < M.length; i++) {
		if (!visited[i]) {
			dfs(M, visited, i);
			res++;
		}
	}
	return res;
}

public void dfs(int[][] m, boolean[] visited, int i) {
	for (int j = 0; j < m.length; j++) {
		if (m[i][j] == 1 && !visited[j]) {
			visited[j] = true;
			dfs(m, visited, j);
		}
	}
}
```



```java
day80
189. 旋转数组
给定一个数组，将数组中的元素向右移动 k 个位置，其中 k 是非负数。
示例 1:
输入: nums = [1,2,3,4,5,6,7], k = 3
输出: [5,6,7,1,2,3,4]
解释:
向右旋转 1 步: [7,1,2,3,4,5,6]
向右旋转 2 步: [6,7,1,2,3,4,5]
向右旋转 3 步: [5,6,7,1,2,3,4]

示例 2:
输入：nums = [-1,-100,3,99], k = 2
输出：[3,99,-1,-100]
解释: 
向右旋转 1 步: [99,-1,-100,3]
向右旋转 2 步: [3,99,-1,-100]
方法一：
使用额外的数组
时间复杂度： O(n)，其中 n 为数组的长度。
空间复杂度： O(n)。
public void rotate(int[] nums, int k) {
	int len = nums.length;
	int[] arr = new int[len];
	for (int i = 0; i < len; i++) {
		arr[(i + k) % len] = nums[i];
	}
	System.arraycopy(arr, 0, nums, 0, len);
} 
```



```java
day81
123. 买卖股票的最佳时机 III
给定一个数组，它的第 i 个元素是一支给定的股票在第 i 天的价格。
设计一个算法来计算你所能获取的最大利润。你最多可以完成 两笔 交易。
示例 1:
输入：prices = [3,3,5,0,0,3,1,4]
输出：6
解释：在第 4 天（股票价格 = 0）的时候买入，在第 6 天（股票价格 = 3）的时候卖出，这笔交易所能获得利润 = 3-0 = 3 。
     随后，在第 7 天（股票价格 = 1）的时候买入，在第 8 天 （股票价格 = 4）的时候卖出，这笔交易所能获得利润 = 4-1 = 3 。
示例 2：
输入：prices = [1,2,3,4,5]
输出：4
解释：在第 1 天（股票价格 = 1）的时候买入，在第 5 天 （股票价格 = 5）的时候卖出, 这笔交易所能获得利润 = 5-1 = 4 。   
注意你不能在第 1 天和第 2 天接连购买股票，之后再将它们卖出。   
因为这样属于同时参与了多笔交易，你必须在再次购买前出售掉之前的股票。
思路一：
动态规划
每天都会有五种状态：不买、买一次、卖一次、买两次、卖两次。
而这五种状态赚的钱的转换规律为：

总不买 = 0；
买一次 = 买过一次，这次不卖或者没买过，　这次买。
卖一次 = 卖过一次，这次不买或者买过一次，这次卖。
买两次 = 买过两次，这次不卖或者卖过一次，这次买。
卖两次 = 卖过两次，没机会了或者买过两次，这次卖。

0：没买过。　 => dp[i][0] = 0
1：买过 1 次。=> dp[i][1] = Math.max(dp[i - 1][0] - price[i], dp[i - 1][1])
2：卖过 1 次。=> dp[i][2] = Math.max(dp[i - 1][1] + price[i], dp[i - 1][2])
3：买过 2 次。=> dp[i][3] = Math.max(dp[i - 1][2] - price[i], dp[i - 2][3])
4：买过 2 次。=> dp[i][4] = Math.max(dp[i - 1][3] + price[i], dp[i - 2][4])

public int maxProfit(int[] prices) {
	long[][] dp = new long[prices.length + 1][5];
    for (long[] dayStatus : dp) {
        Arrays.fill(dayStatus, Integer.MIN_VALUE);
    }
    // 第零天的“没买过”状态的收入值是 0，其他都为 Integer.MIN_VALUE 表示没计算过
    dp[0][0] = 0; 
    for (int i = 1; i < dp.length; i++) {
        dp[i][0] = 0;
        dp[i][1] = Math.max(dp[i - 1][1], dp[i - 1][0] - prices[i - 1]);
        dp[i][2] = Math.max(dp[i - 1][2], dp[i - 1][1] + prices[i - 1]);
        dp[i][3] = Math.max(dp[i - 1][3], dp[i - 1][2] - prices[i - 1]);
        dp[i][4] = Math.max(dp[i - 1][4], dp[i - 1][3] + prices[i - 1]);
    }
    int res = Integer.MIN_VALUE;
    for (int i = 0; i < dp[prices.length].length; i++) {
        res = Math.max(res, (int)dp[prices.length][i]);
    }
    return res;
}
```



```java
day82
228. 汇总区间
给定一个无重复元素的有序整数数组 nums 。
返回 恰好覆盖数组中所有数字 的 最小有序 区间范围列表。也就是说，nums 的每个元素都恰好被某个区间范围所覆盖，并且不存在属于某个范围但不属于 nums 的数字 x 。
列表中的每个区间范围 [a,b] 应该按如下格式输出：
"a->b" ，如果 a != b
"a" ，如果 a == b
示例 1：	
输入：nums = [0,1,2,4,5,7]
输出：["0->2","4->5","7"]
解释：区间范围是：
[0,2] --> "0->2"
[4,5] --> "4->5"
[7,7] --> "7"

示例 2：
输入：nums = [0,2,3,4,6,8,9]
输出：["0","2->4","6","8->9"]
解释：区间范围是：
[0,0] --> "0"
[2,4] --> "2->4"
[6,6] --> "6"
[8,9] --> "8->9"

示例 3：
输入：nums = []
输出：[]

示例 4：	
输入：nums = [-1]
输出：["-1"]
思路一：
双指针，i 指向每个区间的起始位置，j 从 i 开始向后遍历直到不满足连续递增（或 j 达到数组边界），则当前区间结束；然后将 i 指向更新为 j + 1，作为下一个区间的开始位置，j 继续向后遍历找下一个区间的结束位置，如此循环，直到输入数组遍历完毕。
public List<String> summaryRanges(int[] nums) {
	List<String> res = new ArrayList<>();
	// i初始指向第一个区间的起始位置
	int i = 0;
	for (int j = 0; j < nums.length; j++) {
		// j向后遍历，直到不满足连续递增(nums[j] + 1 != nums[j + 1])
		// 或者j达到数组边界，则当前连续区间[i, j]遍历完毕，将其写入结果列表
		if (j + 1 == nums.length || nums[j] + 1 != nums[j + 1]) {
			// 将当前区间[i, j]写入结果列表
			StringBuilder sb = new StringBuilder();
			sb.append(nums[i]);
			if (i != j) {
				sb.append("->").append(nums[j]);
			}
			res.add(sb.toString());
			// 将i指向更新为j + 1, 作为下一个区间的起始位置
			i = j + 1;
		}
	}
	return res;
}
```
