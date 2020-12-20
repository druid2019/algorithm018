学习笔记

```java
day50
547. 朋友圈
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

思路一：
DFS,类似于岛屿问题
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

public void dfs(int[][] m, boolean visited, int i) {
	for (int j = 0; j < M.length; i++) {
		if (m[i][j] == 1 && !visited[j]) {
			visited[j] = true;
			dfs(m, visited, j);
		}
	}
}
```



```java
day51
367. 有效的完全平方数
给定一个正整数 num，编写一个函数，如果 num 是一个完全平方数，则返回 True，否则返回 False。
说明：不要使用任何内置的库函数，如  sqrt。
示例 1：
输入：16
输出：True
示例 2：
输入：14
输出：False
思路一：二分法
public boolean isPerfectSquare(int num) {
	long low = 0;
	long high = num;
	while (low <= high) {
		long middle = low + (high - low) / 2;
		if (middle * middle == num) {
			return true;
		} else if (middle * middle < num) {
			low = middle + 1;
		} else {
			high = middle - 1;
		}
	}
	return false;
}
```

```python
day52
198. 打家劫舍
  你是一个专业的小偷，计划偷窃沿街的房屋。每间房内都藏有一定的现金，影响你偷窃的唯一制约因素就是相邻的房屋装有相互连通的防盗系统，如果两间相邻的房屋在同一晚上被小偷闯入，系统会自动报警。
  给定一个代表每个房屋存放金额的非负整数数组，计算你 不触动警报装置的情况下 ，一夜之内能够偷窃到的最高金额。
示例 1：

输入：[1,2,3,1]
输出：4
解释：偷窃 1 号房屋 (金额 = 1) ，然后偷窃 3 号房屋 (金额 = 3)。
     偷窃到的最高金额 = 1 + 3 = 4 
思路一:动态规划
def rob(self, nums:List[int]) -> int:
    pre = 0
    now = 0
    for i in nums:
        pre, now = now, max(pre+i, now)
    return now
```

```java
// 对应的java代码
public int rob(int[] nums) {
    if (nums == null || nums.length == 0) {
        return 0;
    }
    int len = nums.length;
    int pre = 0;
    int next = 0;
    for (int i = 0; i < len; i++) {
        int temp = pre;
        pre = Math.max(temp,next);
        next = pre + nums[i];
    }
    return Math.max(pre,next);
}
```

```java
day53
190. 颠倒二进制位
颠倒给定的 32 位无符号整数的二进制位。
示例 1：
输入: 00000010100101000001111010011100
输出: 00111001011110000010100101000000
解释: 输入的二进制串 00000010100101000001111010011100 表示无符号整数 43261596，
     因此返回 964176192，其二进制表示形式为 00111001011110000010100101000000。
思路一：使用位运算，简单高效
public int reverseBits(int n) {
	int res = 0;
	for (int i = 0; i < 32; i++) {
		res = (res << 1) + (n & 1);
		n >>= 1;
	}
	return res;
}

```

```java
day54
24. 两两交换链表中的节点
给定一个链表，两两交换其中相邻的节点，并返回交换后的链表。
你不能只是单纯的改变节点内部的值，而是需要实际的进行节点交换。
示例1:
输入：head = [1,2,3,4]
输出：[2,1,4,3]
思路一：
public ListNode swapPairs(ListNode head) {
	ListNode preHead = new ListNode(0);
	preHead.next = head;
	ListNode temp = preHead;
	while (temp.next != null && temp.next.next != null) {
		ListNode node1 = temp.next;
		ListNode node2 = temp.next.next;
		temp.next = node2;
		node1.next = node2.next;
		node2.next = node1;
		temp = node1;
	}
	return preHead.next;
}

```

```java
day55
1122. 数组的相对排序
给你两个数组，arr1 和 arr2，

arr2 中的元素各不相同
arr2 中的每个元素都出现在 arr1 中
对 arr1 中的元素进行排序，使 arr1 中项的相对顺序和 arr2 中的相对顺序相同。未在 arr2 中出现过的元素需要按照升序放在 arr1 的末尾。
示例：
输入：arr1 = [2,3,1,3,2,4,6,7,9,2,19], arr2 = [2,1,4,3,9,6]
输出：[2,2,2,1,4,3,3,9,6,7,19]
思路一：类似于桶排序
1.遍历arr1数组，将arr1中的元素依次作为m数组的下标，对应m数组中的元素存储的是当前下标在arr1中出现的次数，比如m[2]=3就是说在arr1数组中，元素2出现了3次；
2.遍历arr2数组，因为是顺序遍历的，所以同时保证了ref中元素的顺序是和arr2中元素出现的顺序是一致的，m[2],m[1],m[4],m[3]....遍历的同时赋值给ref数组，注意分清楚下标和数组值之间的关系，m[arr2[i]]控制的是赋值的次数，arr2[i]是控制的赋值的值本身
3.最后找到剩余未出现在arr2中但是出现在了arr1中元素，按照顺序赋值给ref数组即可
public int[] relativeSortArray(int[] arr1, int[] arr2) {
	int[] count = new int[1001];
	// 将arr1中元素添加至count数组中
	for (int i: arr1) {
		count[i]++;
	}
	int start = 0;
	//将arr2中元素添加至结果集
	for(int i: arr2) {
		while (count[i] > 0) {
			arr1[start++] = i;
			count[i]--;
		}
	}
	
	// 将不在arr2中的元素顺序添加至结果集
	for (int i = 0; i < 1001; i++) {
		while (count[i] > 0) {
			arr1[start++] = i;
			count[i]--;
		}
	}
	return arr1;
}
```



```java
day56
718. 最长重复子数组
给两个整数数组 A 和 B ，返回两个数组中公共的、长度最长的子数组的长度。
示例：
输入：
A: [1,2,3,2,1]
B: [3,2,1,4,7]
输出：3
解释：
长度最长的公共子数组是 [3, 2, 1] 。
思路一:
动态规划
public int findLength(int[] A, int[] B) {
	if (A.length == 0 || B.length == 0) {
		return 0;
	}
	int res = 0;
	int dp[][]  = new int[A.length + 1][B.length + 1];
	// 最长公共子序列
	for (int i = 1; i < A.length; i++) {
		for (int j = 1; j < B.length; j++) {
			if (A[i - 1] == B[j - 1]) {
				dp[i][j] = dp[i - 1][j - 1] + 1;
			}
			res = Math.max(res, dp[i][j]);
		}
	}
	return res;
 }
```



```java
课后作业
191. 位1的个数
编写一个函数，输入是一个无符号整数（以二进制串的形式），返回其二进制表达式中数字位数为 '1' 的个数（也被称为汉明重量）。
示例 1：
输入：00000000000000000000000000001011
输出：3
解释：输入的二进制串 00000000000000000000000000001011 中，共有三位为 '1'。
思路一：左移
时间复杂度O(1)
public int hammingWeight(int n) {
	int bits = 0;
	int mask = 1;
	for (int i = 0; i < 32; i++) {
		if ((n & mask) != 0) {
			bits++;
		}
		mask <<= 1;
	}
	return bits;
}
思路二：与运算
n & (n-1)清零最低位的1
public int hammingWeight(int n) {
	int sum = 0;
	while (n & (n-1) != 0) {
		sum += 1;
	}
	return sum;
}
```

```python
python
思路三：
直接调库函数
def hammingWeight(self, n):
    return bin(n).count('1');
```



```
231. 2的幂
给定一个整数，编写一个函数来判断它是否是 2 的幂次方。
示例 1:
输入: 1
输出: true
解释: 20 = 1
思路一:
判断是否位2的幂次方，二进制只会有1个1
public boolean isPowerOfTwo(int n) {
	return n > 0 && (n & (n - 1)) == 0;
}
```

```java
493. 翻转对
给定一个数组 nums ，如果 i < j 且 nums[i] > 2*nums[j] 我们就将 (i, j) 称作一个重要翻转对。
你需要返回给定数组中的重要翻转对的数量。
示例 1:
输入: [1,3,2,3,1]
输出: 2
示例 2:
输入: [2,4,3,5,1]
输出: 3
public int reversePairs(int[] nums) {
    if (nums == null || nums.length == 0) return 0;
    return mergeSort(nums, 0, nums.length - 1);
}

private int mergeSort(int[] nums, int l, int r) {
    if (l >= r) return 0;
    int mid = l + ((r - l) >> 1);
    int count = mergeSort(nums, l, mid) + mergeSort(nums, mid + 1, r);
    int[] cache = new int[r - l + 1];
    int i = l, t = l, c = 0;
    for (int j = mid + 1; j <= r; j++, c++) {
        while (i <= mid && nums[i] <= 2 * (long)nums[j]) i++;
        while (t <= mid && nums[t] < nums[j]) cache[c++] = nums[t++];
        cache[c] = nums[j];
        count += mid - i + 1;
    }
    while (t <= mid) cache[c++] = nums[t++];
    System.arraycopy(cache, 0, nums, l, r - l + 1);
    return count;
}
```







