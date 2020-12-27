```java
day64
83. 删除排序链表中的重复元素
给定一个排序链表，删除所有重复的元素，使得每个元素只出现一次。

示例 1:
输入: 1->1->2
输出: 1->2
示例 2:
输入: 1->1->2->3->3
输出: 1->2->3
public ListNode deleteDuplicates(ListNode head) {
    ListNode cur = head;
    while (cur != null && cur.next != null) {
        if (cur.val == cur.next.val) {
            cur.next = cur.next.next;
        } else {
            cur = cur.next;
        }
    }
    return head;
}
```



```java
day65
120. 三角形最小路径和
给定一个三角形，找出自顶向下的最小路径和。每一步只能移动到下一行中相邻的结点上。
例如，给定三角形：
[
     [2],
    [3,4],
   [6,5,7],
  [4,1,8,3]
]
思路一:
记忆化递归:
时间复杂度O(N²),空间复杂度O(N)
public int minimumTotal(List<List<Integer>> triangle) {
	int[] A = new int[triangle.size() + 1];
    for (int i = triangle.size() - 1; i >=0; i--) {
        for (int j = 0; j < triangle.get(i).size(); j++) {
            A[j] = Math.min(A[j], A[j + 1]) + triangle.get(i).get(j);
        }
    }
    return A[0];
}
```



```java
day66
127. 单词接龙
给定两个单词（beginWord 和 endWord）和一个字典，找到从 beginWord 到 endWord 的最短转换序列的长度。转换需遵循如下规则：
每次转换只能改变一个字母。
转换过程中的中间单词必须是字典中的单词。
示例 1:
输入:
beginWord = "hit",
endWord = "cog",
wordList = ["hot","dot","dog","lot","log","cog"]
输出: 5
解释: 一个最短转换序列是 "hit" -> "hot" -> "dot" -> "dog" -> "cog",返回它的长度 5。
思路一:
队列
public int ladderLength(String beginWord, String endWord, List<String> wordList) {
	if(!wordList.contains(endWord)) {
		return 0;
	}
	// 初始化序列长度
	int res = 1;
	wordList.remove(beginWord);
	Queue<String> queue = new LinkedList<String>();
	queue.add(beginWord);
	while(!queue.isEmpty()) {
		int size = queue.size();
		while (size > 0) {
			String temp = queue.poll();
			if (temp.equals(endWord)) {
				return res;
			}
			find(queue, temp, wordList);
			size--;
		}
		res++;
	}
	return 0;
}

public void find(Queue<String> queue, String temp, List<String> wordList) {
	for (int i = 0; i < wordList.size(); i++) {
		String s = wordList.get(i);
		int count = 0;
		for (int m = 0; m < s.length(); m++) {
			if (temp.charAt(m) != s.charAt(m)) {
				count++;
			}
		}
		if (count == 1) {
			queue.add(s);
			wordList.remove(s);
			i--;
		}
	}
}
```



```java
day67
46. 全排列
给定一个 没有重复 数字的序列，返回其所有可能的全排列。
示例:
输入: [1,2,3]
输出:
[
  [1,2,3],
  [1,3,2],
  [2,1,3],
  [2,3,1],
  [3,1,2],
  [3,2,1]
]
思路一：
递归
public List<List<Integer>> permute(int[] nums) {
	List<List<Integer>> res = new ArrayList<List<Integer>>();
	
	List<Integer> output = new ArrayList<Integer>();
	for (int num : nums) {
		output.add(num);
	}
	
	int n = nums.length;
	backtrack(n, output, res, 0);
	return res;
}

public void backtrack(int n, List<Integer> output, List<List<Integer>> res, int first) {
	// 递归终止条件
	if (first == n) {
		res.add(new ArrayList<Integer>(output));
	}
	for (int i = first; i < n; i++) {
		// 动态维护数组 swap
		// 从first第一个数开始填，保持左边的已经填完的右边都是待填
		Collections.swap(output, first, i);
		// 继续递归填下一个数
		backtrack(n, output, res, first + 1);
		// 撤销操作，搜索回溯的时候要撤销这一个位置填的数以及标记，并继续尝试其他没被标记过的数
		Collections.swap(output, first, i);
	}
}
```



```java
day68
122. 买卖股票的最佳时机 II
给定一个数组，它的第 i 个元素是一支给定股票第 i 天的价格。
设计一个算法来计算你所能获取的最大利润。你可以尽可能地完成更多的交易（多次买卖一支股票）。
注意：你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。
示例 1:
输入: [7,1,5,3,6,4]
输出: 7
解释: 在第 2 天（股票价格 = 1）的时候买入，在第 3 天（股票价格 = 5）的时候卖出, 这笔交易所能获得利润 = 5-1 = 4 。
     随后，在第 4 天（股票价格 = 3）的时候买入，在第 5 天（股票价格 = 6）的时候卖出, 这笔交易所能获得利润 = 6-3 = 3 
思路一：
因为不限交易次数，最大值当然是可以取每次差异为正值的情况
public int maxProfit(int[] prices) {
	int res = 0;
	for (int i = 0; i < prices.length - 1; i++) {
		if (prices[i+1] > prices[i]) {
			res += prices[i+1] - prices[i];
		}
	}
	return res;
}
```



```java
day69
238. 除自身以外数组的乘积
给你一个长度为 n 的整数数组 nums，其中 n > 1，返回输出数组 output ，其中 output[i] 等于 nums 中除 nums[i] 之外其余各元素的乘积。
示例:
输入: [1,2,3,4]
输出: [24,12,8,6]
思路一：
乘积 = 当前数左边的乘积 * 当前数右边的乘积
时间复杂度O(N),空间复杂度O(1)
public int[] productExceptSelf(int[] nums) {
	int len = nums.length;
	int[] res = new int[len];
	// res[i] 表示索引 i 左侧所有元素的乘积
    // 因为索引为 '0' 的元素左侧没有元素， 所以 res[0] = 1
	res[0] = 1;
	for (int i = 1; i < len; i++) {
		res[i] = nums[i - 1] * res[i - 1];
	}
	// right为右侧所有元素的乘积
    // 刚开始右边没有元素，所以 right = 1
	int right = 1;
	for (int i = len - 1; i >= 0; i--) {
		// 对于索引 i，左边的乘积为res[i]，右边的乘积为 right
		res[i] = res[i] * right;
		//  right需要包含右边所有的乘积，所以计算下一个结果时需要将当前值乘到 right上
		right *= nums[i];
	}
	return res;
}
```

```java
day70
239. 滑动窗口最大值
给你一个整数数组 nums，有一个大小为 k 的滑动窗口从数组的最左侧移动到数组的最右侧。你只可以看到在滑动窗口内的 k 个数字。滑动窗口每次只向右移动一位。
返回滑动窗口中的最大值。
示例 1：
输入：nums = [1,3,-1,-3,5,3,6,7], k = 3
输出：[3,3,5,5,6,7]
解释：
滑动窗口的位置                最大值
---------------               -----
[1  3  -1] -3  5  3  6  7       3
 1 [3  -1  -3] 5  3  6  7       3
 1  3 [-1  -3  5] 3  6  7       5
 1  3  -1 [-3  5  3] 6  7       5
 1  3  -1  -3 [5  3  6] 7       6
 1  3  -1  -3  5 [3  6  7]      7
思路一：
双端队列

```



```java
期末考试
一条包含字母 A-Z 的消息通过以下方式进行了编码：
'A' -> 1
'B' -> 2
...
'Z' -> 26
给定一个只包含数字的非空字符串，请计算解码方法的总数。
示例 1:
输入: "12"
输出: 2
解释: 它可以解码为 "AB"（1 2）或者 "L"（12）。
示例 2:
输入: "226"
输出: 3
解释: 它可以解码为 "BZ" (2 26), "VF" (22 6), 或者 "BBF" (2 2 6) 
思路一：
非常类似于斐波那契序列，可以理解为dp[i]=dp[i-1]+dp[i-2]。 但是这个是有条件的：
如果以i为下标的字符串，i-1位与i位组成的两位数不在10-26之间，那么dp[i-2]就不能加。如果以i为下标的元素为0，也不能加dp[i-1].
public int numDecodings(String s) {
    if(s.length() == 0 || s.charAt(0) == '0') {
        return 0;
    }
    if (s.length() == 1) {
        return 1;
    } 
    // dp[i - 2]
    int dp1 = 1;
    // dp[i - 1]
    int dp2 = 1;
    int result = 0;
    for (int i = 1; i < s.length(); i++) {
        int i1 = (s.charAt(i - 1) - 48) * 10 + (s.charAt(i) - 48);
        // 不能连续出现0
        if (i1 == 0) {
            return 0;
        }
        if (i1 >= 10 && i1 <= 26) {
            result = dp1;
        }
        if (s.charAt(i) != '0') {
            result += dp2;
        }
        dp1 = dp2;
        dp2 = result;
        result = 0;
    }
    return dp2;
}
```

```java
127. 单词接龙
给定两个单词（beginWord 和 endWord）和一个字典，找到从 beginWord 到 endWord 的最短转换序列的长度。转换需遵循如下规则：
每次转换只能改变一个字母。
转换过程中的中间单词必须是字典中的单词。
示例 1:
输入:
beginWord = "hit",
endWord = "cog",
wordList = ["hot","dot","dog","lot","log","cog"]
输出: 5
解释: 一个最短转换序列是 "hit" -> "hot" -> "dot" -> "dog" -> "cog",返回它的长度 5。
思路一:
队列
public int ladderLength(String beginWord, String endWord, List<String> wordList) {
	if(!wordList.contains(endWord)) {
		return 0;
	}
	// 初始化序列长度
	int res = 1;
	wordList.remove(beginWord);
	Queue<String> queue = new LinkedList<String>();
	queue.add(beginWord);
	while(!queue.isEmpty()) {
		int size = queue.size();
		while (size > 0) {
			String temp = queue.poll();
			if (temp.equals(endWord)) {
				return res;
			}
			find(queue, temp, wordList);
			size--;
		}
		res++;
	}
	return 0;
}

public void find(Queue<String> queue, String temp, List<String> wordList) {
	for (int i = 0; i < wordList.size(); i++) {
		String s = wordList.get(i);
		int count = 0;
		for (int m = 0; m < s.length(); m++) {
			if (temp.charAt(m) != s.charAt(m)) {
				count++;
			}
		}
		if (count == 1) {
			queue.add(s);
			wordList.remove(s);
			i--;
		}
	}
}
```



