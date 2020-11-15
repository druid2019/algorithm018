学习笔记

```java
day15:
94. 二叉树的中序遍历
给定一个二叉树，返回它的中序 遍历。
输入: [1,null,2,3]
   1
    \
     2
    /
   3

输出: [1,3,2]
思路1 递归
中序遍历，先左再根在右，然后依次循环，时间复杂度O(N),空间复杂度O(N)
public List<Integer> inOrder(TreeNode root) {
    // 初始化结果集
	List<Integer> alst = new ArrayList<>();
	// 如果根节点为空，则返回空的集合
	if (root == null) return alst;
	// 中序遍历方法
	inOrderHaddle(root, alst);
	// 返回结果集
	return alst;
}
public void inOrderHaddle(TreeNode root, List<Integer> alst) {
    // 先左
	if (root.left != null) {
		inOrderHaddle(root.left, alst);
	}
	// 再中，添加根节点
	alst.add(root.val);
	// 再右
	if (root.right != null) {
		inOrderHaddle(root.right, alst);
	}
}

思路2：
二叉树的中序遍历
栈的思想，入栈和出栈
public List<Integer> inorderTraversal(TreeNode root) {
    List<Integer> list = new ArrayList<Integer>();
	
    Stack<TreeNode> stack = new Stack<TreeNode>();
    TreeNode cur = root;
	
    while(cur!=null || !stack.empty()){
        while(cur!=null){
            // 栈中添加元素
            stack.add(cur);
            // cur的左节点赋值个cur
            cur = cur.left;
        }
        // 栈中元素出
        cur = stack.pop();
        // 结果集中添加元素
        list.add(cur.val);
        // cur的右节点赋值给cur,即null值，下次继续弹出
        cur = cur.right;
    }
    return list;
}

day16:
剑指 Offer 05. 替换空格
请实现一个函数，把字符串 s 中的每个空格替换成"%20"。
示例 1：
输入：s = "We are happy."
输出："We%20are%20happy."
方法一：直接用字符串的替换方法，简单暴力
但是replaceAll消耗内存较大
public String replaceSpace(String s) {
    return s.replaceAll(" ","%20");
}

方法二：正则表达式
内存消耗较第一种要小
public String replaceSpace(String s) {
    return s.replaceAll("\\s","%20");
}

方法三：StringBuilder
public static String replaceSpace(String s) {
	  // 初始化StringBuilder
      StringBuilder str = new StringBuilder();
      // 将s转为字符数组进行判断
      for (char c:s.toCharArray()) {
        // 如果为空字符串，则添加%20
      	if(c == ' ') {
      		str.append("%20");
      	} else {
      		str.append(c);
      		}
      }
      return str.toString();
}

day17:
剑指 Offer 06. 从尾到头打印链表
输入一个链表的头节点，从尾到头反过来返回每个节点的值（用数组返回）。
示例 1：
输入：head = [1,3,2]
输出：[2,3,1]
方法一：
  首先遍历链表，需要从尾到头返回每个节点，与遍历的顺序相反，即第一遍历的节点最后一个输出。这就是典型的“先进后出”，因此可以利用栈的思想来实现。
  时间复杂度O(N)，空间复杂度O(N)
public int[] reversePrint(ListNode head) {
    // 初始化一个栈
	Stack<ListNode> stack = new Stack<>();
	// 往栈中压入元素
	while(head != null) {
		stack.push(head);
		head = head.next;
	}
    int len = stack.size();
	// 初始化结果数组
	int[] result = new int[len];
	// 出栈，并将值赋给结果数组
	for(int i=0;i<len;i++) {
		result[i] = stack.pop().val;
	}
	return result;
}

思路二：
思路与第一种类似，只不过将数组添加改为集合添加
public int[] reversePrint(ListNode head) {
	// 初始化一个栈
	Stack<ListNode> stack = new Stack<>();
	// 往栈中压入元素
    while (head != null) {
    	stack.push(head);
        head = head.next;
    }
    // 定义集合存放结果数据
    List<Integer> alst = new ArrayList<>();
    while(!stack.isEmpty()) {
    	alst.add(stack.pop().val);
    }
    // 将集合转为数组返回
    return alst.stream().mapToInt(Integer::intValue).toArray();
}

思路三：
直译法，先顺序存储链表，然后倒序打印出来
时间复杂度O(N)，空间复杂度O(N)
public int[] reversePrint(ListNode head) {
	List<Integer> alst = new ArrayList<>();
	while (head != null) {
		alst.add(head.val);
		head = head.next;
	}
	int[] result = new int(alst.size());
	for(int i = 0;i < alst.size(); i++) {
		int j = alst.size() - i -1;
		result[i] = alst.get(j);
	}
	return result;
}
```



```java
day18:
剑指 Offer 68 - II. 二叉树的最近公共祖先
给定一个二叉树, 找到该树中两个指定节点的最近公共祖先。

百度百科中最近公共祖先的定义为：“对于有根树 T 的两个结点 p、q，最近公共祖先表示为一个结点 x，满足 x 是 p、q 的祖先且 x 的深度尽可能大（一个节点也可以是它自己的祖先）。”

例如，给定如下二叉树:  root = [3,5,1,6,2,0,8,null,null,7,4]
示例 1:
输入: root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 1
输出: 3
解释: 节点 5 和节点 1 的最近公共祖先是节点 3。
示例 2:
输入: root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 4
输出: 5
解释: 节点 5 和节点 4 的最近公共祖先是节点 5。因为根据定义最近公共祖先节点可以为节点本身。
思路一：递归  时间复杂度O(N) 空间复杂度O(N)
1.p或q有一个根节点，则返回根节点
2.p,q分别在树的两侧，则最近公共祖先肯定是根节点
3.p,q均在左子树或右子树
重复性在于要么分列两边，要么都在一边
public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        // p或q就是根节点，则返回根节点
        if (root == null || root == p || root == q ) {
            return root;
        }
        // 遍历左节点，查找p,q
        TreeNode left = lowestCommonAncestor(root.left, p, q);
        // 遍历右节点，查找p,q
        TreeNode right = lowestCommonAncestor(root.right, p, q);
        // p、q分列左右子树两边，则最近公共祖先为根节点
        if (left != null && right != null) {
            return root;
        }
        // 如果一边为空，则说明p,q都在另一边，返回不为空的结果
        return left == null ? right : left;
    }
```



```java
day19
1. 两数之和
给定一个整数数组 nums 和一个目标值 target，请你在该数组中找出和为目标值的那 两个 整数，并返回他们的数组下标。
你可以假设每种输入只会对应一个答案。但是，数组中同一个元素不能使用两遍。
示例:
给定 nums = [2, 7, 11, 15], target = 9
因为 nums[0] + nums[1] = 2 + 7 = 9
所以返回 [0, 1]
思路1：暴力算法，两次for循环，返回下标，时间复杂度O(N²)，空间复杂度O(1)
思路2：定义一个字典，存储数组下标和值，时间复杂度O(N)，空间复杂度O(N)
public int[] twoSums(int[] nums, int target) {
    // 定义一个字典，存储数组下标和值
	Map<Integer,Integer> map = new HashMap<>();        
    for (int i = 0; i< nums.length; i++) {
        // 如果target - nums[i]也在map中，则返回
        if (map.containsKey(target-nums[i])) {
            return new int[]{map.get(target-nums[i]),i};
        }
        // 将元素添加至HashMap中
        map.put(nums[i],i);
    }
    return new int[]{0};
}
```



```java
day20
15. 三数之和
给你一个包含 n 个整数的数组 nums，判断 nums 中是否存在三个元素 a，b，c ，使得 a + b + c = 0 ？请你找出所有满足条件且不重复的三元组。
注意：答案中不可以包含重复的三元组。
示例：
给定数组 nums = [-1, 0, 1, 2, -1, -4]，
满足要求的三元组集合为：
[
  [-1, 0, 1],
  [-1, -1, 2]
]
思路一：可采用双指针的方法
时间复杂度O(N²)，空间复杂度O(logN)
public List<List<Integer>> threeSum(int[] nums) {
	// 排序
	Arrays.sort(nums);
	List<List<Integer>> result = new ArrayList<>();
	// 数组长度
	int len = nums.length;
	for (int i= 0; i < len; i++) {
		// 当第i个数开始大于0时，跳出循环
		if (nums[i] > 0) break;
		// 当相邻数相等时，跳过该数，避免重复计算
		if (i > 0 && nums[i] == nums[i-1]) {
			continue;
		}
		// 使用双指针，并向中间靠拢
		int left = i+1;
		int right = len -1;
		sum = -nums[i];
		while(left < right) {
			int count = nums[left] + nums[right];
			if (count == sum) {
				result.add(Arrays.asList(nums[i],nums[left],nums[right]));
				// 相邻重复元素
				while(left < right && nums[left] == nums[left+1]) left++;
				while(left < right && nums[right] == nums[right-1]) right--;
				left++;
				right--;
			} else if(count < sum) {
				left++;
			} else {
				right--;
			}
		}
	}
	return result;
}

day21
面试题 17.09. 第 k 个数
有些数的素因子只有 3，5，7，请设计一个算法找出第 k 个数。注意，不是必须有这些素因子，而是必须不包含其他的素因子。例如，前几个数按顺序应该是 1，3，5，7，9，15，21。
示例 1:
输入: k = 5
输出: 9
思路一：
定义三个指针，每次取3，5，7相乘的最小数
public int getKthMagicNumber(int k) {
    // 定义三个指针
	int a = 0;
    int b = 0;
    int c = 0;
    // 初始化结果集
    int[] res = new int[k];
    // 第一个元素为1
    res[0] = 1;
    for (int i=1;i<k;i++) {
    	// 每次取最小值
        int num = Math.min(Math.min(res[a]*3,res[b]*5), res[c]*7);
        if (num % 3 == 0) {
            a++;
        }
        if (num % 5 == 0) {
            b++;
        }
        if (num % 7 == 0) {
            c++;
        }
        res[i] = num;
    }
    return res[k-1];
}
```



```python
习题
78子集
给定一组不含重复元素的整数数组 nums，返回该数组所有可能的子集（幂集）。
说明：解集不能包含重复的子集。
示例:
输入: nums = [1,2,3]
输出:
[
  [3],
  [1],
  [2],
  [1,2,3],
  [1,3],
  [2,3],
  [1,2],
  []
]

思路一：迭代
def subsets(self, nums: List[int]) -> List[List[int]]:
        subsets = [[]]
        for num in nums:
            newsets = []
            for subset in subsets:
                new_subset = subset + [num]
                newsets.append(new_subset)
            subsets.extend(newsets)
        return subsets
思路二：可以对上诉代码做下优化
def subsets(self, nums: List[int]) -> List[List[int]]:
    res = [[]]
    for i in nums:
        res = res + [[i] + num for num in res]
    return res


51. N 皇后
n 皇后问题研究的是如何将 n 个皇后放置在 n×n 的棋盘上，并且使皇后彼此之间不能相互攻击。
给定一个整数 n，返回所有不同的 n 皇后问题的解决方案。
每一种解法包含一个明确的 n 皇后问题的棋子放置方案，该方案中 'Q' 和 '.' 分别代表了皇后和空位。
示例：
输入：4
输出：[
 [".Q..",  // 解法 1
  "...Q",
  "Q...",
  "..Q."],

 ["..Q.",  // 解法 2
  "Q...",
  "...Q",
  ".Q.."]
]
解释: 4 皇后问题存在两个不同的解法。
提示：
皇后彼此不能相互攻击，也就是说：任何两个皇后都不能处于同一条横行、纵行或斜线上。
方法一：（还不太理解，代码比较简洁，先收藏收藏）
def solveNQueue(self,n):
    def DFS(queues, xy_dif, xy_sum):
        p = len(queues)
        if p == n:
            result.append(queues)
            return None
        for q in range(n):
            if q not in queues and p-q not in xy_dif and p+q not in xy_sum:
                DFS(queues + [q], xy_dif + [p-q], xy_sum+[p+q])
    result = []
    DFS([],[],[])
    return [["."*i + "Q" + "."*(n-i-1) for i in sol] for sol in result]

```

```java
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
回溯算法与深度优先遍历
public List<List<Integer>> permute(int[] nums) {
        int len = nums.length;
        // 使用一个动态数组保存所有可能的全排列
        List<List<Integer>> res = new ArrayList<>();
        if (len == 0) {
            return res;
        }

        boolean[] used = new boolean[len];
        List<Integer> path = new ArrayList<>();

        dfs(nums, len, 0, path, used, res);
        return res;
    }

    private void dfs(int[] nums, int len, int depth,
                     List<Integer> path, boolean[] used,
                     List<List<Integer>> res) {
        if (depth == len) {
            res.add(path);
            return;
        }

        // 在非叶子结点处，产生不同的分支，这一操作的语义是：在还未选择的数中依次选择一个元素作为下一个位置的元素，这显然得通过一个循环实现。
        for (int i = 0; i < len; i++) {
            if (!used[i]) {
                path.add(nums[i]);
                used[i] = true;
                dfs(nums, len, depth + 1, path, used, res);
                // 注意：下面这两行代码发生 「回溯」，回溯发生在从 深层结点 回到 浅层结点 的过程，代码在形式上和递归之前是对称的
                used[i] = false;
                path.remove(path.size() - 1);
            }
        }
```

