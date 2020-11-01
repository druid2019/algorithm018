总结：

```
树的面试：
前序遍历: 根左右
中序遍历: 左根右  （升序）
先遍历左子树，然后访问根节点，再遍历右子树。
后序遍历: 左右根

```

```python
前序遍历
def preorder(self,root):
    if root:
        self.traverse_path.append(root.val)
        self.preorder(root.left)
        self.preorder(root.right)
中序遍历
def inorder(self,root):
    if root:
        self.inorder(root.left)
        self.traverse_path.append(root.val)
        self.inorder(root.right)
后续遍历
def postorder(self,root):
    if root:
        self.postorder(self.left)
        self.postorder(self.right)
        self.taverse_path.append(root.val)
    
```

```java
堆 Heap:
可以迅速找到一堆数种的最大或最小值的数据结构。
将根节点最大的堆叫做大顶堆，根节点最小的堆叫小顶堆。

二叉堆性质：
通过完全二叉树来实现(不同于二叉搜索树)
也可以通过二叉搜索树来实现，但是查找元素复杂度为O(logn)
性质1: 是一棵完全树。
性质2: 树种任意节点的值总是>=字节的的值。
 
```

```java
HashMap小结：
  它根据键的hashCode值存储数据，大多数情况下可以直接定位到它的值，因而具有很快的访问速度，但遍历顺序却是不确定的。HashMap最多只允许一条记录的键为null，允许多条记录的值为null。HashMap非线程安全，即任一时刻可以有多个线程同时写HashMap，可能会导致数据的不一致。如果需要满足线程安全，可以用 Collections的synchronizedMap方法使HashMap具有线程安全的能力，或者使用ConcurrentHashMap。
  HashMap就是使用哈希表来存储的。哈希表为解决冲突，可以采用开放地址法和链地址法等来解决问题，Java中HashMap采用了链地址法。链地址法，简单来说，就是数组加链表的结合。在每个数组元素上都一个链表结构，当数据被Hash后，得到数组下标，把数据放在对应下标元素的链表上。
  map.put('a',3)
  系统将调用'a'这个key的hashCode()方法得到其hashCode 值（该方法适用于每个Java对象），然后再通过Hash算法的后两步运算（高位运算和取模运算）来定位该键值对的存储位置，有时两个key会定位到相同的位置，表示发生了Hash碰撞。当然Hash算法计算结果越分散均匀，Hash碰撞的概率就越小，map的存取效率就会越高。
  确定哈希桶数组索引位置方法：
  方法一：
static final int hash(Object key) {   //jdk1.8 & jdk1.7
     int h;
     // h = key.hashCode() 为第一步 取hashCode值
     // h ^ (h >>> 16)  为第二步 高位参与运算
     return (key == null) ? 0 : (h = key.hashCode()) ^ (h >>> 16);
}
  方法二：
static int indexFor(int h, int length) {  
     //jdk1.7的源码，jdk1.8没有这个方法，但是实现原理一样的
     //第三步 取模运算
     return h & (length-1);  
}
  这里的Hash算法本质上就是三步：取key的hashCode值、高位运算、取模运算。
  对于任意给定的对象，只要它的hashCode()返回值相同，那么程序调用方法一所计算得到的Hash码值总是相同的。我们首先想到的就是把hash值对数组长度取模运算，这样一来，元素的分布相对来说是比较均匀的。但是，模运算的消耗还是比较大的，在HashMap中是这样做的：调用方法二来计算该对象应该保存在table数组的哪个索引处。
  这个方法非常巧妙，它通过h & (table.length -1)来得到该对象的保存位，而HashMap底层数组的长度总是2的n次方，这是HashMap在速度上的优化。当length总是2的n次方时，h& (length-1)运算等价于对length取模，也就是h%length，但是&比%具有更高的效率。
  在JDK1.8的实现中，优化了高位运算的算法，通过hashCode()的高16位异或低16位实现的：(h = k.hashCode()) ^ (h >>> 16)，主要是从速度、功效、质量来考虑的，这么做可以在数组table的length比较小的时候，也能保证考虑到高低Bit都参与到Hash的计算中，同时不会有太大的开销。
```

