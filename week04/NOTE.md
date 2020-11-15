学习笔记

```python
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
方法一：回溯（不太理解）
时间复杂度：O(n * n!)
def permute(self, nums: List[int]) -> List[List[int]]:
        res = []
        def backtrack(nums, tmp):
            # 判断数组是否为空
            if not nums:
                res.append(tmp)
            for i in range(len(nums)):
                # 去掉i进行回溯
                backtrack(nums[:i] + nums[i+1:], tmp + [nums[i]])
        backtrack(nums, [])
        return res

方法二：
时间复杂度：O(n * n!)
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
        // 所有数都填完了
        if (first == n) {
            res.add(new ArrayList<Integer>(output));
        }
        for (int i = first; i < n; i++) {
            // 动态维护数组
            Collections.swap(output, first, i);
            // 继续递归填下一个数
            backtrack(n, output, res, first + 1);
            // 撤销操作
            Collections.swap(output, first, i);
        }
    }
```

```java
day23:
70. 爬楼梯
假设你正在爬楼梯。需要 n 阶你才能到达楼顶。
每次你可以爬 1 或 2 个台阶。你有多少种不同的方法可以爬到楼顶呢？
注意：给定 n 是一个正整数。
示例 1：
输入： 2
输出： 2
解释： 有两种方法可以爬到楼顶。
1.  1 阶 + 1 阶
2.  2 阶
思路一：递归(不建议，会出现堆栈溢出)
时间复杂度O(N)
每次可以拍1或2，可用数学归纳法的思想f(n) = f(n-1) + f(n-2)
public int climbStairs(int n) {
	if (n == 1 || n == 2) {
        return n;
    }
    return climbStairs(n-1) + climbStairs(n-2)
}
思路二：
手动去存储递归的值，时间复杂度O(N),空间复杂度O(1)
public int climbStairs(int n) {
    // 边界条件
	if (n == 1 || n == 2) {
        return n;
    }
    // 初始值
    int second = 2;
    int result = 3;
    // 处理result
    while (n > 3) {
    	int first = second;
    	second = result;
    	result = first + second;
    	n--;
    }  
}

思路三：简化一下思路二的代码，可以用for循环替换
时间复杂度O(N), 空间复杂度O(1)
public int climbStairs(int n) {
	int first, second = 0, result = 1;
    for (int i = 0; i < n; i++) {
    	first = second;
    	second = result;
    	result = first + second;
    }
    return  result;
}


```

```java
day24
122.买卖股票的最佳时机 II
给定一个数组，它的第 i 个元素是一支给定股票第 i 天的价格。
设计一个算法来计算你所能获取的最大利润。你可以尽可能地完成更多的交易（多次买卖一支股票）。
注意：你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。
示例 2:
输入: [1,2,3,4,5]
输出: 4
解释: 在第 1 天（股票价格 = 1）的时候买入，在第 5 天 （股票价格 = 5）的时候卖出, 这笔交易所能获得利润 = 5-1 = 4 。
     注意你不能在第 1 天和第 2 天接连购买股票，之后再将它们卖出。
     因为这样属于同时参与了多笔交易，你必须在再次购买前出售掉之前的股票。
示例 3:
输入: [7,6,4,3,1]
输出: 0
解释: 在这种情况下, 没有交易完成, 所以最大利润为 0。
思路一：
因为不限交易次数，因此可以简化为只要下一个价格比上一个大，即可完成一次交易，累积可以求得最大利润。
时间复杂度O(N)，空间复杂度O(1)
public int maxProfit(int[] prices) {
    // 初始化结果集
	int result = 0;
    // 遍历，每次取下一次的结果比上一次大的累加
	for (int i=0; i< prices.length - 1; i++) {		
		if (prices[i+1] >= prices[i]) {
			result += prices[i+1] - prices[i];
		}
	}
    // 返回
	return result;
}


```

```java
day 25
860. 柠檬水找零
在柠檬水摊上，每一杯柠檬水的售价为 5 美元。
顾客排队购买你的产品，（按账单 bills 支付的顺序）一次购买一杯。
每位顾客只买一杯柠檬水，然后向你付 5 美元、10 美元或 20 美元。你必须给每个顾客正确找零，也就是说净交易是每位顾客向你支付 5 美元。
注意，一开始你手头没有任何零钱。
如果你能给每位顾客正确找零，返回 true ，否则返回 false 。
示例 1：
输入：[5,5,5,10,20]
输出：true
解释：
前 3 位顾客那里，我们按顺序收取 3 张 5 美元的钞票。
第 4 位顾客那里，我们收取一张 10 美元的钞票，并返还 5 美元。
第 5 位顾客那里，我们找还一张 10 美元的钞票和一张 5 美元的钞票。
由于所有客户都得到了正确的找零，所以我们输出 true。

示例 4：
输入：[5,5,10,10,20]
输出：false
解释：
前 2 位顾客那里，我们按顺序收取 2 张 5 美元的钞票。
对于接下来的 2 位顾客，我们收取一张 10 美元的钞票，然后返还 5 美元。
对于最后一位顾客，我们无法退回 15 美元，因为我们现在只有两张 10 美元的钞票。
由于不是每位顾客都得到了正确的找零，所以答案是 false。
思路一:直译法
统计5美元和10美元的钞票数量，遍历bills数组，如果碰到5美元，bill_five++，如果碰到10美元，bill_ten++，同时bill_five--，最后经过一系列运算，如果bill_five和bill_ten都大于0，则表示该数组可以找零。返回true，否则返回false
public boolean lemonadeChange(int[] bills) {
    // 5美元数量
	int bill_five = 0;
	// 10美元数量
	int bill_ten = 0;
	int len = bills.length;
	for (int i = 0; i < len; i++) {
		// 碰到5美元，bill_five++
		if (bills[i] == 5) {
			bill_five++;
		} 
		// 碰到10美元，bill_ten++,同时bill_five--
		if (bills[i] == 10) {
			if (bill_five < 1) {
				return false;
			}
            bill_five--;
			bill_ten++;
		}
		// 碰到20美元
		if (bills[i] == 20) {
			// 如果至少有一个10美元和5美元，则bill_ten--，bill_five--
			// 如果没有10美元，但至少有3个5美元，则bill_five -= 3
			// 否则，则无法找零，返回false
			if(bill_ten >= 1 && bill_five >= 1) {
				bill_ten--;
				bill_five--;
			} else if (bill_ten < 1 && bill_five >= 3) {
				bill_five -= 3;
			} else {
				return false;
			}
		}
	}
	// 遍历结束，如果
	return true;
}
```

```java
day 26
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
思路：可从深度优先角度考虑
// 定义行列
private int m;
private int n;

public int numIslands(char[][] grid) {
	int count = 0;
    // 行的长度
    m = grid.length;
    if (m == 0) return 0;
    // 列的长度
    n = grid[0].length;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n ; j++) {
            if (grid[i][j] == '1') {
                DFSHandle(grid, i, j);
                count++;
            }
        }
    return count;
    }    
}

private void DFSHandle(char[][] grid, int i, int j) {
        if (i < 0 || j < 0 || i >= n || j>= m || grid[i][j] != '1') {
            return;
        }
        grid[i][j] = '0';
        // 上下左右
        DFSHandle(grid+1, i + 1, j);
        DFSHandle(grid+1, i - 1, j);
        DFSHandle(grid, i, j + 1);
        DFSHandle(grid, i, j - 1);
    }

```

```java
day27
367. 有效的完全平方数
给定一个正整数 num，编写一个函数，如果 num 是一个完全平方数，则返回 True，否则返回 False。
说明：不要使用任何内置的库函数，如  sqrt。
示例 1：
输入：16
输出：True
示例 2：
输入：14
输出：False
思路1：暴力算法，不推荐
从1到num/2遍历，得到则返回true() num = 2000105819时超出时间限制
def isPerfectSquare(self, num: int) -> bool:
        if num == 1:
            return True
        for i in range(int(num/2)+1):
            if i*i == num:
                return True
        return False

思路2：
二分法,每次取一半
public boolean isPerfectSquare(int num) {
        long low = 0;
        long high = num;       
        while (low <= high) {	
            // 二分取一半
            long middle = low + (high - low) / 2;
            long t = middle * middle;
            if (t == num) {
                return true;
            } else if (t < num) {
                low = middle + 1;
            } else {
                high = middle - 1;
            }
        }
        return false;
    }

思路3：转换数学思路
完全平方数都可以表示成连续的奇数和
public boolean isPerfectSquare(int num) {
	int i = 0;
	// 每次减递增的奇数
	while(num > 0) {
		num -= i;
        i += 2;
	}
	returen num == 0;
}
```

```java
day28
169. 多数元素
给定一个大小为 n 的数组，找到其中的多数元素。多数元素是指在数组中出现次数大于 ⌊ n/2 ⌋ 的元素。
你可以假设数组是非空的，并且给定的数组总是存在多数元素。
示例 1:
输入: [3,2,3]
输出: 3
示例 2:
输入: [2,2,1,1,1,2,2]
输出: 2
    
1.暴力算法，不推荐
def majorityElement(self, nums: List[int]) -> int:
		// 去重，减少遍历次数
        nums2 = list(set(nums))
        for i in nums2:
            if (nums.count(i) > len(nums)/2):
                return i
2.先排序，那么下标为n/2的一定是该数
时间复杂度：O(nlogn)，空间复杂度：O(logn)
public int majorityElement(int[] nums) {
        Arrays.sort(nums);
        return nums[nums.length / 2];
    }

                 
```



```java
课后作业
二叉树的层次遍历：
给你一个二叉树，请你返回其按 层序遍历 得到的节点值。 （即逐层地，从左到右访问所有节点）。
示例：
二叉树：[3,9,20,null,null,15,7],
    3
   / \
  9  20
    /  \
   15   7
返回其层次遍历结果：
[
  [3],
  [9,20],
  [15,7]
]
思路一：递归
每一层每一层的遍历，根节点只有一个
private List<List<Integer>> result = new ArrayList<>();
public List<List<Integer>> levelOrder(TreeNode root) {
    if (root != null) {
        getOrder(root, 0);
    }
    return result;
}

public void getOrder(TreeNode root, int level) {
    if (result.size() <= level) {
        result.add(new ArrayList<>());
    }
    result.get(level).add(root.val);
    if(root.left != null) {
        getOrder(root.left,level + 1);
    }
    if (root.right != null) {
        getOrder(root.right, level + 1);
    }
}


455. 分发饼干
假设你是一位很棒的家长，想要给你的孩子们一些小饼干。但是，每个孩子最多只能给一块饼干。
对每个孩子 i，都有一个胃口值 g[i]，这是能让孩子们满足胃口的饼干的最小尺寸；并且每块饼干 j，都有一个尺寸 s[j] 。如果 s[j] >= g[i]，我们可以将这个饼干 j 分配给孩子 i ，这个孩子会得到满足。你的目标是尽可能满足越多数量的孩子，并输出这个最大数值。
示例 1:
输入: g = [1,2,3], s = [1,1]
输出: 1
解释: 
你有三个孩子和两块小饼干，3个孩子的胃口值分别是：1,2,3。
虽然你有两块小饼干，由于他们的尺寸都是1，你只能让胃口值是1的孩子满足。
所以你应该输出1。
示例 2:
输入: g = [1,2], s = [1,2,3]
输出: 2
解释: 
你有两个孩子和三块小饼干，2个孩子的胃口值分别是1,2。
你拥有的饼干数量和尺寸都足以让所有孩子满足。
所以你应该输出2.
方法一：暴力算法，遍历g,s数组
时间复杂度O(N²)，空间复杂度O(1)
public int findContentChildren(int[] g, int[] s) {
    	// 排序
        Arrays.sort(g);
        Arrays.sort(s);
        int count = 0;
        // 记录每次结果的下标
        int num = -1;
        for (int i = 0; i < g.length; i++) {
            for(int j= 0;j< s.length; j++) {
                if (g[i] <= s[j] && num  < j) {
                    count++;
                    num = j;
                    break;
                }
            }
        }
        return count;
    }

方法二：贪心算法，优先满足大口胃的孩子，先排序
时间复杂度O(NlongN),空间复杂度O(1)
public int findContentChildren(int[] g, int[] s) {
        // 排序
        Arrays.sort(g);
        Arrays.sort(s);
        // s的最大下标
        int index = s.length - 1;
        // 初始化结果集
        int count = 0;
        // 每次找到后减一以免重复
        for (int i = g.length - 1; i >= 0; i--) {
            if(index >= 0 && g[i] <= s[index]) {
                index--;
                count++;
            }
        }
        return count;
    }
```

