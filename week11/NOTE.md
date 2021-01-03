```java
day72
1046. 最后一块石头的重量
有一堆石头，每块石头的重量都是正整数。
每一回合，从中选出两块 最重的 石头，然后将它们一起粉碎。假设石头的重量分别为 x 和 y，且 x <= y。那么粉碎的可能结果如下：
如果 x == y，那么两块石头都会被完全粉碎；
如果 x != y，那么重量为 x 的石头将会完全粉碎，而重量为 y 的石头新重量为 y-x。
最后，最多只会剩下一块石头。返回此石头的重量。如果没有石头剩下，就返回 0。
示例：
输入：[2,7,4,1,8,1]
输出：1
解释：
先选出 7 和 8，得到 1，所以数组转换为 [2,4,1,1,1]，
再选出 2 和 4，得到 2，所以数组转换为 [2,1,1,1]，
接着是 2 和 1，得到 1，所以数组转换为 [1,1,1]，
最后选出 1 和 1，得到 0，最终数组转换为 [1]，这就是最后剩下那块石头的重量。
思路一：
大顶堆，优先队列实现
时间复杂度O(nlogn),空间复杂度O(n)
public int lastStoneWeight(int[] stones) {
	PriorityQueue<Integer> pq = new PriorityQueue<Integer>((a, b) -> b - a);
    for (int stone: stones) {
        pq.offer(stone);
    }
    while (pq.size() > 1) {
        int a = pq.poll();
        int b = pq.poll();
        if (a > b) {
            pq.offer(a - b);
        }
    }
    return pq.isEmpty() ? 0 : pq.poll();
}
```



```java
day73
605. 种花问题
假设你有一个很长的花坛，一部分地块种植了花，另一部分却没有。可是，花卉不能种植在相邻的地块上，它们会争夺水源，两者都会死去。
给定一个花坛（表示为一个数组包含0和1，其中0表示没种植花，1表示种植了花），和一个数 n 。能否在不打破种植规则的情况下种入 n 朵花？能则返回True，不能则返回False。
示例 1:
输入: flowerbed = [1,0,0,0,1], n = 1
输出: True
示例 2:
输入: flowerbed = [1,0,0,0,1], n = 2
输出: False
思路一:
贪心
难点在于奇偶分析以及分类情况
时间复杂度：O(m),其中 m 是数组，flowerbed 的长度。需要遍历数组一次。
空间复杂度：O(1)。额外使用的空间为常数。
1.下标i,j处都有花，j-i>=2,且在下标 [i+1,j-1]范围内没有种植花,则只有当j−i≥4 时才可以在下标 i和下标j之间种植更多的花，且可以种植花的下标范围是[i+2,j-2]。可以种植花的位置数为p=j-i-3,不管p为奇数还是偶数，该范围内最多可种植(p+1)/2朵花。
2.上述情况是在已有的两朵花之间种植花的情况。
假设花坛的下标l处是最左边的已经种植的花，下标r处是最右边的已经种植的花。
即对于任意 k<l 或 k>r 都有flowerbed[k]=0
a.下标 l左边有 l个位置，当 l<2时无法在下标l左边种植花，当l≥2 时可以在下标范围[0,l−2] 范围内种植花，可以种植花的位置数是 l-1，最多可以种植l/2 朵花。
b.令 m为数组flowerbed 的长度，下标 r右边有m−r−1个位置，可以种植花的位置数是 m-r-2，最多可以种植 (m-r-1)/2朵花。
3.如果花坛上没有任何花朵，则有 m个位置可以种植花，最多可以种植(m+1)/2朵花。
public boolean canPlaceFlowers(int[] flowerbed, int n) {
	int count = 0;
	// prev表示上一朵已经种植的花的下标位置,初始时prev=−1
	int prev = -1;
	int len = flowerbed.length;
	for (int i = 0; i < len; i++) {
		// prev和i的值计算上一个区间内可以种植花的最多数量
		if (flowerbed[i] == 1) {
			if (prev < 0) {
				count += i / 2;
			} else {
				count += (i - prev - 2) / 2;
			}
			// 令prev=i，继续遍历数组flowerbed 剩下的元素
			prev = i;
			if (count >= n) {
				return true;
			}
		}
	}
	if (prev < 0) {
		count += (len + 1) / 2;
	} else {
		count += (len - prev -1) / 2;
	}
	return count >= n;
}
```



```java
day74
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
public int[] maxSlidingWindow(int[] nums, int k) {
    if (k == 0 || nums.length == 0) {
        return new int[0];
    }
    int[] result = new int[nums.length - k + 1];
    // 初始化一个双端队列
    ArrayDeque<Integer> indexDeque = new ArrayDeque<>();
    // 从开始遍历，遇到大的放入，小的移除
    for (int i = 0; i < k - 1; i++) {
        while (!indexDeque.isEmpty() &&  nums[i] > nums[indexDeque.getLast()]) {
            indexDeque.removeLast();
        }
        indexDeque.addLast(i);
    }
    // 如果遇到的数值比队列中最大值小，最小值大，那么比它小的数字不可能成为最大值，删除较小的数字，放入该数字
    for (int i = k - 1; i < nums.length; i++) {
        while (!indexDeque.isEmpty() &&  nums[i] > nums[indexDeque.getLast()]) {
            indexDeque.removeLast();
        }
        // 队列头部的数字如果其下标离滑动末尾距离大于窗口大小，那么也删除队列头部的数字
        if (!indexDeque.isEmpty() && (i - indexDeque.getFirst()) >= k) {
            indexDeque.removeFirst();
        }
        indexDeque.addLast(i);
        result[i + 1 - k] = nums[indexDeque.getFirst()];
    }
    return result;
}

```



```java
day75
86. 分隔链表
给你一个链表和一个特定值 x ，请你对链表进行分隔，使得所有小于 x 的节点都出现在大于或等于 x 的节点之前。
你应当保留两个分区中每个节点的初始相对位置。
示例：
输入：head = 1->4->3->2->5->2, x = 3
输出：1->2->2->4->3->5
思路一：
遍历链表的所有节点，小于x的放到一个小的链表中，大于等于x的放到一个大的链表中，最后把小链表的尾节点指向大链表的头节点即可。
时间复杂度: O(n)，其中n是原链表的长度。我们对该链表进行了一次遍历。
空间复杂度: O(1)。
public ListNode partition(ListNode head, int x) {
	// 小链表的表头
	ListNode smallHead = new ListNode(0);
	// 大链表的表头
	ListNode bigHead = new ListNode(0);
	// 小链表的尾
	ListNode smallTail = smallHead;
	// 大链表的尾
	ListNode bigTail = bigHead;
	// 遍历head链表
	while (head != null) {
		if (head.val < x) {
			// 如果当前节点的值小于x,则把当前节点挂到小链表的后面
			smallTail = smallTail.next = head;
		} else {
			// 否则挂到大链表的后面
			bigTail = bigTail.next = head;
		}
		
		// 循环下一个节点
		head = head.next;
	}
	// 最后再把大小链表组合在一起
	smallTail.next = bigHead.next;
	bigTail.next = null;
	return smallHead.next;
}


```







