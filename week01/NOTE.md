```
用 add first 或 add last 这套新的 API 改写 Deque 的代码
Deque<String> deque = new LinkedList<String>();
        deque.addFirst("a");
        deque.addFirst("b");
        deque.addFirst("c");
        System.out.println(deque);

        String str = deque.getLast();
        System.out.println(str);
        System.out.println(deque);

        while (deque.size() > 0) {
            System.out.println(deque.removeFirst());
        }
        System.out.println(deque);
    }
```



```
分析 Queue 和 Priority Queue 的源码
Queue:
A collection designed for holding elements prior to processing.extends Collection.
method:	add(e) remove() element()
        offer(e) poll() 	peek()
priority Queue:
插入操作：O(1) 取出操作O(logN)
底层具体实现的数据结构较为多样和复杂: heap,bst,treap
```



```java
26.删除数组中重复项
public int removeDuplicates(int[] nums) {
        if (nums.length == 0) return 0;
        int i = 0;
        for (int j = 1;j< nums.length;j++){
            if (nums[j] != nums[i]){
                i++;
                nums[i] = nums[j];
            }
        }
        return i+1;
    }
```

```java
21.合并两个有序链表
将两个升序链表合并为一个新的 升序 链表并返回。
 新链表是通过拼接给定的两个链表的所有节点组成的。
 输入：1->2->4, 1->3->4
 输出：1->1->2->3->4->4
public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
        // 新建一个带头节点的链表
        ListNode a = new ListNode(0);
        ListNode result = a;
        // 比较两个节点值的大小，小的添加到新建链表中
        while (l1 != null && l2 != null) {
            if (l1.val <= l2.val) {
                a.next = l1;
                l1 = l1.next;
                a = a.next;
            } else {
                a.next =l2;
                l2 = l2.next;
                a = a.next;
            }
        }
        // 如果l1遍历完，再去遍历l2
        if (l1 == null) {
            a.next = l2;
        } else {
            a.next = l1;
        }
        // 返回头节点的下一个节点
        return result.next;
    }
```

```java
88.合并两个有序数组
给你两个有序整数数组 nums1 和 nums2，请你将 nums2 合并到 nums1 中，使 nums1 成为一个有序数组。
思路：
把num2添加到num1中，然后排序。
public void merge(int[] nums1, int m, int[] nums2, int n) {
        System.arraycopy(nums2,0,nums1,m,n);
        Arrays.sort(nums1);
    }
```



```java
1.两数之和
给定一个整数数组 nums 和一个目标值 target，请你在该数组中找出和为目标值的那 两个 整数，并返回他们的数组下标。
你可以假设每种输入只会对应一个答案。但是，数组中同一个元素不能使用两遍。
给定 nums = [2, 7, 11, 15], target = 9
因为 nums[0] + nums[1] = 2 + 7 = 9
所以返回 [0, 1]

方法一 ：
循环遍历，暴力解法(简单易懂)
for (int i=0;i<nums.length;i++) {
            for (int j = i+1;j<nums.length;j++) {
                if (nums[i] + nums[j] == target) {
                    return new int[]{i,j};
                }
            }
        }
```

```java
283.移动零
给定一个数组 nums，编写一个函数将所有 0 移动到数组的末尾，同时保持非零元素的相对顺序。
示例:
输入: [0,1,0,3,12]
输出: [1,3,12,0,0]
说明:
必须在原数组上操作，不能拷贝额外的数组。
尽量减少操作次数。
public void moveZeroes(int[] nums) {
        int j = 0;
        for (int i = 0;i< nums.length;++i){
            if (nums[i] != 0){
                nums[j] = nums[i];
                if (i != j){
                nums[i]=0;
                }
                j++;
            }                       
        }
    }
```



```java
66.加一
给定一个由整数组成的非空数组所表示的非负整数，在该数的基础上加一。
最高位数字存放在数组的首位， 数组中每个元素只存储单个数字。
你可以假设除了整数 0 之外，这个整数不会以零开头。
输入: [1,2,3]
输出: [1,2,4]
解释: 输入数组表示数字 123
方法一：
判断尾数+1后是否为10，如果为10，再循环比较上一个，
如果上一位+1后不超过9，最后一位置为0，然后返回，如果仍为10，则循环比较
一直到第一个，如果为10，则开辟比之前长度大1的空间

public static int[] plusOne(int[] digits) {
        int last = digits.length-1;
        digits[last] += 1;

        if (digits[last] == 10) {
            while (last > 0 && digits[last] == 10) {
                digits[last] =0;
                digits[--last] += 1;
            }
            // 所有位都为9的情况
            if (last == 0 && digits[last] == 10) {
                digits = new int[digits.length+1];
                digits[0] = 1;
            }
        }
        return digits;
    }

```

```java
70.爬楼梯
每次你可以爬 1 或 2 个台阶。
思路：
比较常见的递归思想f(n) = f(n-1) + f(n-2)
代码核心实现：
方法一：(超时，会堆栈溢出)
public int climbStairs(int n) {
        if(n <= 2) return n;
        return climbStairs(n-1)+climbStairs(n-2);
    }
时间复杂度：O(n2)
方法二：(先计算，减少递归重复计算)
public int climbStairs(int n) {
        if(n <= 2) return n;
        int second = 1;
        int result = 2;
        while (n >= 3) {
            int first = second;
            second = result;
            result = first + second;
            n--;
        }
        return result;
    }
 时间复杂度：O(n)

```



```java
24.两两交换链表中的节点
给定一个链表，两两交换其中相邻的节点，并返回交换后的链表。
你不能只是单纯的改变节点内部的值，而是需要实际的进行节点交换。
输入：head = [1,2,3,4]
输出：[2,1,4,3]

public ListNode swapPairs(ListNode head) {
        // 初始化一个节点
        ListNode preHead = null;
        // 该节点指向链表的头节点
        preHead.next = head;
        ListNode tmp = preHead;
        // 交换
        while (tmp.next != null && tmp.next.next != null) {
            ListNode node1 = tmp.next;
            ListNode node2 = tmp.next.next;
            // node2的下一个指针赋给node1的指向下一个的指针
            node1.next = node2.next;
            // tmp下一个指向node2
            tmp.next = node2;
            // node1的值赋给tmp
            // tmp,node1,node2整体右移
            tmp = node1;
        }
        return preHead.next;
    }
时间复杂度为O(n),空间复杂度为O(1)。  
```



```java
299.猜数字
你在和朋友一起玩 猜数字（Bulls and Cows）游戏，该游戏规则如下：

你写出一个秘密数字，并请朋友猜这个数字是多少。
朋友每猜测一次，你就会给他一个提示，告诉他的猜测数字中有多少位属于数字和确切位置都猜对了（称为“Bulls”, 公牛），有多少位属于数字猜对了但是位置不对（称为“Cows”, 奶牛）。
朋友根据提示继续猜，直到猜出秘密数字。
请写出一个根据秘密数字和朋友的猜测数返回提示的函数，返回字符串的格式为 xAyB ，x 和 y 都是数字，A 表示公牛，用 B 表示奶牛。

xA 表示有 x 位数字出现在秘密数字中，且位置都与秘密数字一致。
yB 表示有 y 位数字出现在秘密数字中，但位置与秘密数字不一致。
请注意秘密数字和朋友的猜测数都可能含有重复数字，每位数字只能统计一次。

输入: secret = "1807", guess = "7810"
输出: "1A3B"
解释: 1 公牛和 3 奶牛。公牛是 8，奶牛是 0, 1 和 7。

public String getHint(String secret, String guess) {
        int bulls = 0;
        int cows = 0;

        // 存储数组的下标
        int[] a = new int[255];
        int[] b = new int[255];

        for (int i=0;i<secret.length();i++) {
            if (secret.charAt(i) == guess.charAt(i)) {
                bulls += 1;
            } else {
                a[secret.charAt(i)-'0']++;
                b[guess.charAt(i)-'0']++;
            }
        }

        for (int i =0;i< 255;i++) {
            cows += Math.min(a[i],b[i]);
        }

        String result =bulls +"A" + cows + "B";
        return result;
    }
    

```



```java
641.设计循环双端队列
设计实现双端队列。
你的实现需要支持以下操作：

MyCircularDeque(k)：构造函数,双端队列的大小为k。
insertFront()：将一个元素添加到双端队列头部。 如果操作成功返回 true。
insertLast()：将一个元素添加到双端队列尾部。如果操作成功返回 true。
deleteFront()：从双端队列头部删除一个元素。 如果操作成功返回 true。
deleteLast()：从双端队列尾部删除一个元素。如果操作成功返回 true。
getFront()：从双端队列头部获得一个元素。如果双端队列为空，返回 -1。
getRear()：获得双端队列的最后一个元素。 如果双端队列为空，返回 -1。
isEmpty()：检查双端队列是否为空。
isFull()：检查双端队列是否满了。
示例：

MyCircularDeque circularDeque = new MycircularDeque(3); // 设置容量大小为3
circularDeque.insertLast(1);			        // 返回 true
circularDeque.insertLast(2);			        // 返回 true
circularDeque.insertFront(3);			        // 返回 true
circularDeque.insertFront(4);			        // 已经满了，返回 false
circularDeque.getRear();  				// 返回 2
circularDeque.isFull();				        // 返回 true
circularDeque.deleteLast();			        // 返回 true
circularDeque.insertFront(4);			        // 返回 true
circularDeque.getFront();				// 返回 4

class node {
        int val;
        node next = null;
        node pre = null;

        node(int v) {
            this.val = v;
        }
    }

    node firsthead = null;
    node lasthead = null;
    // 链表总节点容量
    int capacity;
    // 链表当前节点容量
    int count;


    /** Initialize your data structure here. Set the size of the deque to be k. */
    public MyCircularDeque(int k) {
        this.capacity = k;
        this.count = 0;
    }

    /** Adds an item at the front of Deque. Return true if the operation is successful. */
    // 头插法，保持队列的先进先出特性
    public boolean insertFront(int value) {
        if (isFull()) {
            return false;
        }
        node nd = new node(value);
        // 只要firsthead为空，那么lasthead必定为空
        if (firsthead == null) {
            firsthead = nd;
            lasthead = nd;
        } else {
            // 只要firsthead不为空 lasthead肯定也不空
            nd.next = firsthead;
            firsthead.pre = nd;
            firsthead = nd;
        }
        count++;
        return true;
    }

    /** Adds an item at the rear of Deque. Return true if the operation is successful. */
    public boolean insertLast(int value) {
        if (isFull()) {
            return false;
        }
        node nd = new node(value);
        if (lasthead == null) {
            lasthead = nd;
            firsthead = nd;
        } else {
            nd.pre = lasthead;
            lasthead.next = nd;
            lasthead = nd;
        }
        count++;
        return true;
    }

    /** Deletes an item from the front of Deque. Return true if the operation is successful. */
    public boolean deleteFront() {
        if (isEmpty()) {
            return false;
        }
        if (count == 1) {
            firsthead = null;
            lasthead = null;
        } else {
            firsthead = firsthead.next;
            firsthead.pre = null;
        }
        count--;
        return true;
    }

    /** Deletes an item from the rear of Deque. Return true if the operation is successful. */
    public boolean deleteLast() {
        if (isEmpty()) {
            return false;
        }
        if (count == 1) {
            firsthead = null;
            lasthead = null;
        } else {
            lasthead = lasthead.pre;
            lasthead.next = null;
        }
        count--;
        return true;
    }

    /** Get the front item from the deque. */
    public int getFront() {
        if (this.firsthead == null) {
            return -1;
        } else {
            return firsthead.val;
        }
    }

    /** Get the last item from the deque. */
    public int getRear() {
        if (this.lasthead == null) {
            return -1;
        } else {
            return this.lasthead.val;
        }
    }

    /** Checks whether the circular deque is empty or not. */
    public boolean isEmpty() {
        if (this.count == 0) {
            return true;
        } else {
            return false;
        }
    }

    /** Checks whether the circular deque is full or not. */
    public boolean isFull() {
        if (this.count == this.capacity) {
            return true;
        } else {
            return false;
        }
    }
```



```java
422.接雨水
给定 n 个非负整数表示每个宽度为 1 的柱子的高度图，计算按此排列的柱子，下雨之后能接多少雨水。
输入：height = [0,1,0,2,1,0,1,3,2,1,2,1]
输出：6
解释：上面是由数组 [0,1,0,2,1,0,1,3,2,1,2,1] 表示的高度图，在这种情况下，可以接 6 个单位的雨水（蓝色部分表示雨水）。 
思路一：暴力解法

public int trap(int[] height) {
        int res = 0;
        int len = height.length;
        for (int i= 0; i<len-1;i++) {
            int max_left = 0;
            int max_right = 0;
            for (int j=i; j>=0;j--){
                max_left = Math.max(max_left, height[j]);
            }
            for (int j = i; j < len; j++) {
                max_right = Math.max(max_right, height[j]);
            }
            res += Math.min(max_left,max_right) - height[i];
        }
        return res;
    }
```

