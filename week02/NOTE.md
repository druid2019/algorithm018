学习笔记

```python
day8
350. 两个数组的交集 II
给定两个数组，编写一个函数来计算它们的交集。
输入：nums1 = [1,2,2,1], nums2 = [2,2]
输出：[2,2]

输入：nums1 = [4,9,5], nums2 = [9,4,9,8,4]
输出：[4,9]

思路一：
求出两个数组共有的不重复元素，放到nums3中
然后算出nums3中每个元素在nums1和nums2中数量，取最小值
将nums3中元素按照最小值的数量添加到结果集
暴力解法
    def intersect(self, nums1: List[int], nums2: List[int]) -> List[int]:
        # 定义一个数组，存放nums1和nums2的共有的不重复元素
        nums3 = []
        # 结果集
        result = []
        for i in nums1:
            if i in nums2 and i not in nums3:
                nums3.append(i)
        for i in nums3:
            # 取nums3得元素，算出该元素在nums1和nums2中的个数，取最小值
            t = min(nums1.count(i), nums2.count(i))
            # 往结果集中添加元素
            for j in range(0,t):
                result.append(i)
        return result
```

```java
思路二：将数组1转换为集合
遍历数组2，如果数组2中元素在集合中，则集合移除该元素
同时将该元素添加到新的集合中，返回结果
public static int[] intersect(int[] nums1, int[] nums2) {
        List<Integer> list = new ArrayList<>();
        List<Integer> lst1 = Arrays.stream(nums1).boxed().collect(Collectors.toList());
        for (int i=0; i< nums2.length;i++) {
            if (lst1.contains(nums2[i])) {
                lst1.remove(lst1.indexOf(nums2[i]));
                list.add(nums2[i]);
            }
        }
        return list.stream().mapToInt(Integer::intValue).toArray();
    }
```



```java
day9滑动窗口:(队列)
方法一：暴力求解,时间复杂度O(n*k)
public static int[] maxSlidingWindow(int[] nums, int k) {
		// 考虑边界情况，传空，返回空
        if (nums == null || nums.length == 0) return new int[0];
        int[] result = new int[nums.length-k+1];
        for(int i=0;i<nums.length-k+1;i++) {
            int t = nums[i];
            // 求nums[i],nums[i+k-1]之间的最大值
            for (int j=i;j< i+k;j++) {
                if (nums[j] > t){
                    t = nums[j];
                }
            }
            result[i] = t;
        }
        return result;
    }

方法二：双端队列，时间复杂度O(N)，空间复杂度O(N)
public static int[] maxSlidingWindow(int[] nums, int k) {
        if (k == 0|| nums.length == 0) return new int[0];
        int[] result = new int[nums.length-k+1];
        // 初始化一个双端队列
        ArrayDeque<Integer> indexDeque = new ArrayDeque<>();
        // 从开始遍历，遇到大的放入，小的移除
        for (int i=0;i<k-1;i++) {
            while (!indexDeque.isEmpty() && nums[i] > nums[indexDeque.getLast()]) {
                indexDeque.removeLast();
            }
            indexDeque.addLast(i);
        }

        // 如果遇到的数字比队列中最大值小，最小值大，那么将比它小数字不可能成为最大值了，删除较小的数字，放入该数字。
        for (int i= k-1;i<nums.length;i++) {
            while (!indexDeque.isEmpty() && nums[i] > nums[indexDeque.getLast()]){
                indexDeque.removeLast();
            }
            // 队列头部的数字如果其下标离滑动窗口末尾的距离大于窗口大小，那么也删除队列头部的数字。
            if (!indexDeque.isEmpty() && (i-indexDeque.getFirst()) >= k) {
                indexDeque.removeFirst();
            }
            indexDeque.addLast(i);
            result[i+1-k] = nums[indexDeque.getFirst()];
        }
        return result;
    }

思路3：堆(优先队列)
超出时间限制
public int[] maxSlidingWindow(int[] nums, int k) {
        if(nums.length == 0 || k == 0) {
            return new int[0];
        }
        int n = nums.length;
        // numbers of windows
        int[] result = new int[n - k + 1];
        // 大顶堆，每次取最大值
        PriorityQueue<Integer> maxPQ = new PriorityQueue<>((o1, o2) -> (o2 - o1));

        for (int i = 0; i < n; i++) {
            int start = i - k;
            // 开始值大于0了，则移除前面元素
            if (start >= 0) {
                maxPQ.remove(nums[start]);
            }
            // 将最大值添加到大顶堆种
            maxPQ.offer(nums[i]);
            // 堆中元素满了，则取最大值给result
            if (maxPQ.size() == k) {
                result[i - k + 1] = maxPQ.peek();
            }
        }
        return result;
    }
```



```java
day10 删除最外层的括号
输入："(()())(())"
输出："()()()"
解释：
输入字符串为 "(()())(())"，原语化分解得到 "(()())" + "(())"，
删除每个部分中的最外层括号后得到 "()()" + "()" = "()()()"。

输入："()()"
输出：""
解释：
输入字符串为 "()()"，原语化分解得到 "()" + "()"，
删除每个部分中的最外层括号后得到 "" + "" = ""。
思路：剥最外层
public static String removeOuterParentheses(String S) {
        char[] chars = S.toCharArray();
        StringBuilder sb = new StringBuilder();
        int count = 0;
        for (char c: chars) {
            if (c == ')') {
                --count;
            }
            if (count > 0) {
                sb.append(c);
            }
            if (c == '(') {
                ++count;
            }
        }
        return sb.toString();
    }
```

```java
day11
写一个程序，输出从 1 到 n 数字的字符串表示。
1. 如果 n 是3的倍数，输出“Fizz”；
2. 如果 n 是5的倍数，输出“Buzz”；
3.如果 n 同时是3和5的倍数，输出 “FizzBuzz”。
n = 15,
返回:
[
    "1",
    "2",
    "Fizz",
    "4",
    "Buzz",
    "Fizz",
    "7",
    "8",
    "Fizz",
    "Buzz",
    "11",
    "Fizz",
    "13",
    "14",
    "FizzBuzz"
]
思路：如果能被15整除，则添加"FizzBuzz"，否则则添加其它字符串
这里需用else if,以免重复判断
public List<String> fizzBuzz(int n) {
        List<String> reList = new ArrayList<>();
        for (int i=1;i<n+1;i++) {
            if (i % 15 == 0) {
                reList.add("FizzBuzz");
            } else if (i % 3 == 0) {
                reList.add("Fizz");
            } else if (i % 5 == 0) {
                reList.add("Buzz");
            } else {
                reList.add(String.valueOf(i));
            }
        }
        return reList;
    }
```



```java
day12
258. 各位相加
给定一个非负整数 num，反复将各个位上的数字相加，直到结果为一位数。
示例:
输入: 38
输出: 2 
解释: 各位相加的过程为：3 + 8 = 11, 1 + 1 = 2。 由于 2 是一位数，所以返回 2。
方法一：
直译法(递归)
public int addDigits(int num) {
        int result = 0;
        while (num / 10 >= 0 && num > 0) {
            result += num % 10;
            num /= 10;
        }
        if (result >= 10) {
            return addDigits(result);
        }
        return result;
    }

方法二(分解)
不太好理解
假设输入的数字是一个5位数字num，则num的各位分别为a、b、c、d、e。
num = a * 10000 + b * 1000 + c * 100 + d * 10 + e
num = (a + b + c + d + e) + (a * 9999 + b * 999 + c * 99 + d * 9)
个位数相加余数等于该数除以9的余数。
同理对a + b + c + d + e再进行分解
(a + b + c + d + e)%9 = a1 + b1 + c1 + d1 + e1
固最后结果num % 9
考虑到被9整除的数，(num-1) % 9 +1 即可
public int addDigits(int num) {
        return (num-1) % 9 + 1;
    }


```

```java
day13
104.给定一个二叉树，找出其最大深度。
二叉树的深度为根节点到最远叶子节点的最长路径上的节点数。
说明: 叶子节点是指没有子节点的节点。
示例：
给定二叉树 [3,9,20,null,null,15,7]，

    3
   / \
  9  20
    /  \
   15   7
返回它的最大深度 3 。
思路：迭代，每次去左右的最大值，然后根据树的深度向下遍历
public int maxDepth(TreeNode root) {
		// 如果没有节点，返回0
        if (root == null) {
            return 0;
        }
        // 取左右节点深度的最大值
        int max = Math.max(maxDepth(root.left),maxDepth(root.left));
        // 加上根节点的深度
        return max + 1;
    }

```



```java
day14
283.移动零
给定一个数组 nums，编写一个函数将所有 0 移动到数组的末尾，同时保持非零元素的相对顺序。
示例:
输入: [0,1,0,3,12]
输出: [1,3,12,0,0]
思路1：穷尽遍历法，非零元素替换之前的0元素，时间复杂度O(n²)
public static void moveZeroes(int[] nums) {
        // 遍历数组
        for (int i = 0; i < nums.length - 1; i++) {
            // 碰到0的元素，将其替换成后面非零的元素
            if (nums[i] == 0){
                for (int j = i + 1; j < nums.length;j++) {
                    if (nums[j] != 0) {
                        nums[i] = nums[j];
                        nums[j] = 0;
                        break;
                    }
                }
            }
        }
    }

思路2：
每次将非零的元素迁移，时间复杂度O(N)
public static void moveZeroes(int[] nums) {
        int j = 0;
        // 遍历数组
        for (int i = 0; i < nums.length; i++) {
            // 如果非零，将nums[i]值赋给num[j]
            if (nums[i] != 0) {
                // 每次将非零的元素赋给num[j]
                nums[j] = nums[i];
                // 替换后面非零的元素
                if (i != j) {
                    nums[i] = 0;
                }
                j++;
            }
        }
    }
```



```java
课后作业：
剑指 Offer 40. 最小的k个数
思路1：排序sort，时间复杂度为NlogN
public int[] getLeastNumbers(int[] arr, int k) {
        // 整型数组转list
        List<Integer> alst = Arrays.stream(arr).boxed().collect(Collectors.toList());
        // 对alst进行排序
        Collections.sort(alst);
        // 定义结果集合
        List<Integer> result = new ArrayList<>();
        // 将前k个数元素添加到结果集
        for(int i=0;i<k;i++) {
            result.add(alst.get(i));
        }
        // list转整型数组
        return result.stream().mapToInt(Integer::intValue).toArray();
    }
思路2：堆heap: NlogK
public static int[] getLeastNumbers(int[] arr, int k) {
        // 利用优先队列定义一个堆
        PriorityQueue<Integer> heap = new PriorityQueue<>();
        // 向堆中添加元素
        for(int i = 0;i < arr.length;i++) {
            heap.add(arr[i]);
        }
        int[] result = new int[k];
        // 取出堆中元素
        for(int i = 0;i < k;i++) {
            result[i] = heap.poll();
        }
        return result;
    }
思路3：快排quick sort


347.前 K 个高频元素
给定一个非空的整数数组，返回其中出现频率前 k 高的元素。
输入: nums = [1,1,1,2,2,3], k = 2
输出: [1,2]
public List<Integer> topKFrequent(int[] nums, int k) {
	Map<Integer, Integer> map = new HashMap<>();
	// 统计元素出现的个数
	for(int n: nums) {
		map.put(n,map.getOrDefault(n,0)+1);
	}
	// 建立一个大顶堆
	PriorityQueue<Map.Entry<Integer, Integer>> maxHeap = new PriorityQueue<>((a,b) -> (b.getValue()-a.getValue()));
	// 把统计的元素放到大顶堆中去，排好序了
	for(Map.Entry<Integer, Integer> entry: map.entrySet()) {
		maxHeap.add(entry);
	}
	int[] resNums = new int[k];
	// 取出前k个元素
    for(int i= 0;i<k;i++){
        Map.Entry<Integer, Integer> entry = maxHeap.poll();
        resNums[i] = entry.getKey();
    }
	return resNums;
}



剑指 Offer 49. 丑数
我们把只包含质因子 2、3 和 5 的数称作丑数（Ugly Number）。求按从小到大的顺序的第 n 个丑数。
示例:
输入: n = 10
输出: 12
解释: 1, 2, 3, 4, 5, 6, 8, 9, 10, 12 是前 10 个丑数。
思路一：直译法n = 1352会报错，并超出时间限制
public int nthUglyNumber(int n) {
        if(n == 0) return 0;
        List<Integer> alist = new ArrayList<>();
        for (int i = 1; ;i++) {
            int k = i;
            while(k % 2 == 0) {
                k /= 2;
            }
            while(k % 3 == 0) {
                k /= 3;
            }
            while(k % 5 == 0) {
                k /= 5;
            }
            if(k == 1) {
                alist.add(i);
            }
            if(alist.size() == n) {
            return alist.get(n-1);
            }
        }
    }

方法2：递推的思想，时间复杂度为O(N),空间复杂度为O(N)
下个丑数是x1*2或x2*3或x3*5的，只需取最小值即可。
xn = min(x1,x2,x3)
public static int nthUglyNumber(int n) {
        // 初始化数组存储丑数
        int[] re = new int[n];
        // 第一个值为1
        re[0] = 1;
        int a = 0,b = 0,c = 0;
        for (int i = 1; i < n; i++) {
            // 丑数为2，3，5的倍数，根据序列取最小值
            // 第a丑数个数需要通过乘2来得到下个丑数，第b丑数个数需要通过乘2来得到下个丑数，同理第c个数
            int t1 = re[a] * 2, t2 = re[b] * 3, t3 = re[c] * 5;
            re[i] = Math.min(Math.min(t1,t2),t3);
            // 如果能被整除，说明取到了该值，该数加1，以免后续重复
            // 第a个数已经通过乘2得到了一个新的丑数，那下个需要通过乘2得到一个新的丑数的数应该是第(a+1)个数
            if (re[i] % 2 == 0) {
                a++;
            }
            // 第 b个数已经通过乘3得到了一个新的丑数，那下个需要通过乘3得到一个新的丑数的数应该是第(b+1)个数
            if (re[i] % 3 == 0) {
                b++;
            }
            // 第 c个数已经通过乘5得到了一个新的丑数，那下个需要通过乘5得到一个新的丑数的数应该是第(c+1)个数
            if (re[i] % 5 == 0) {
                c++;
            }
        }
        return re[n-1];
    }
```



```java
589.N叉树的前序遍历
给定一个 N 叉树，返回其节点值的前序遍历。
例如，给定一个 3叉树 :
返回其前序遍历: [1,3,5,6,2,4]。
方法一：递归
List<Integer> alst = new ArrayList<>();
public List<Integer> preorder(Node root) {
    if(root == null) return alst;
    alst.add(root.val);
    for(Node node:root.children){
        if(node == null) continue;
        preorder(node);
    }
    return alst;
}



94.二叉树的中序遍历：
给定一个二叉树，返回它的中序 遍历。
示例:
输入: [1,null,2,3]
   1
    \
     2
    /
   3

输出: [1,3,2]
方法一：左中右，递归
public List<Integer> inorderTraversal(TreeNode root) {
        List<Integer> res = new ArrayList<>();
        helper(root, res);
        return res;
    }

    public void helper(TreeNode root, List<Integer> res){
        if (root != null){
            if (root.left != null){
                helper(root.left,res);
            }
            res.add(root.val);
            if (root.right != null){
                helper(root.right, res);
            }
        }    
    }


144. 二叉树的前序遍历
思路：同上，先根，再左，再右
public List<Integer> preorderTraversal(TreeNode root) {
        List<Integer> alst = new ArrayList<>();
        if (root == null) {
            return alst;
        }
        help(root,alst);
        return alst;
    }

public void help(TreeNode root, List<Integer> alst) {
        // 先添加根节点
        alst.add(root.val);
        // 再添加左节点
        if (root.left != null) {
            help(root.left, alst);
        }
        // 再添加右节点
        if (root.right != null) {
            help(root.right, alst);
        }
    }


429. N叉树的层序遍历
给定一个 N 叉树，返回其节点值的层序遍历。 (即从左到右，逐层遍历)。
例如，给定一个 3叉树 :
返回其层序遍历:
[
     [1],
     [3,2,4],
     [5,6]
]

方法一：递归
private List<List<Integer>> result = new ArrayList<>();

public List<List<Integer>> levelOrder(Node root) {
    if (root != null) traverseNode(root, 0);
    return result;
}

private void traverseNode(Node node, int level) {
    if (result.size() <= level) {
        result.add(new ArrayList<>());
    }
    result.get(level).add(node.val);
    for (Node child : node.children) {
        traverseNode(child, level + 1);
    }
}

242. 有效的字母异位词
给定两个字符串 s 和 t ，编写一个函数来判断 t 是否是 s 的字母异位词。
示例 1:
输入: s = "anagram", t = "nagaram"
输出: true
示例 2:
输入: s = "rat", t = "car"
输出: false
思路1：直接排序
时间复杂度：O(nlogn),空间复杂度O(1)
public boolean isAnagram(String s, String t) {
    if (s.length() != t.length()) {
        return false;
    }
    // 转换为字符数组
    char[] arr1 = s.toCharArray();
    char[] arr2 = t.toCharArray();
    // 排序
    Arrays.sort(arr1);
    Arrays.sort(arr2);
    // 比较二者是否相等
    return Arrays.equals(arr1, arr2);
}



49. 字母异位词分组
给定一个字符串数组，将字母异位词组合在一起。字母异位词指字母相同，但排列不同的字符串。

示例:

输入: ["eat", "tea", "tan", "ate", "nat", "bat"]
输出:
[
  ["ate","eat","tea"],
  ["nat","tan"],
  ["bat"]
]
思路一:
排序字符串，当且仅当它们的排序字符串相等时，两个字符串是字母异位词。
时间复杂度：O(NKlogK) 空间复杂度：O(NK)
public List<List<String>> groupAnagrams(String[] strs) {
        if (strs.length == 0) return new ArrayList();
        Map<String, List> ans = new HashMap<String, List>();
        for (String s : strs) {
            char[] ca = s.toCharArray();
            Arrays.sort(ca);
            String key = String.valueOf(ca);
            if (!ans.containsKey(key)) ans.put(key, new ArrayList());
            ans.get(key).add(s);
        }
        return new ArrayList(ans.values());
    }



```

