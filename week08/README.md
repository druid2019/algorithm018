```
位运算
判断奇偶:
x&1 == 1 奇数
x&1 == 0 偶数

x/2  x >> 1 右移1位

x&(x-1) 清零最低位的1

x&-x 得到最低位的1

x&~x  0
```



```
布隆过滤器(Bloom Filter)的原理和实现
https://www.cnblogs.com/cpselvis/p/6265825.html
产生原因：
  判断一个元素是否存在一个集合中，当集合里面的元素数量足够大，如果有500万条记录甚至1亿条记录，数组、链表、树及哈希表等数据结构会存储元素的内容，一旦数据量过大，消耗的内存也会呈现线性增长，最终达到瓶颈。普通计算机是无法提供如此大的内存。这个时候，布隆过滤器（Bloom Filter）就应运而生。
  原理:
  布隆过滤器（Bloom Filter）的核心实现是一个超大的位数组和几个哈希函数。
  查询W元素是否存在集合中的时候，同样的方法将W通过哈希映射到位数组上的点。如果所有点其中有一个点不为1，则可以判断该元素一定不存在集合中。反之，如果所有点都为1，则该元素可能存在集合中。注意：此处不能判断该元素是否一定存在集合中，可能存在一定的误判率。
  布隆过滤器添加元素：
  1.将要添加的元素给k个哈希函数。
  2.得到对应于位数组上的k个位置。
  3.将这k个位置设为1。
  布隆过滤器查询元素:
  1.将要查询的元素给k个哈希函数。
  2.得到对应于位数组上的k个位置
  3.如果k个位置有一个为0，则肯定不在集合中
  4.如果k个位置全部为1，则可能在集合中
```



```python
布隆过滤器代码示例
from bitarray import bitarray
import mmh3

class BloomFilter:
    def __init__(self, size, hash_num):
        self.size = size
        self.hash_num = hash_num
        self.bit_array = bitarray(size)
        self.bit_array.setall(0)
        
    def add(self, s):
        for seed in range(self.hash_num):
            result = mmh3.hash(s, seed) % self.size
            self.bit_array[result] = 1
            
    def lookup(self, s):
        for seed in range(self.hash_num):
            result = mmh3.hash(s, seed) % self.size
            if self.bit_array[result] == 0:
                return "Nope"
        return "Probably"
bf = BloomFilter(500000, 7)
bf.add("dantezhao")
print(bf.lookup("dantezhao"))
print(bf.lookup("yyj"))
```



```python
LRU-cache
OrderedDict实现
class LRUCache(objcet):
    def __init__(self, capacity):
        self.dic = collections.OrderedDict()
        self.remain = capacity
    
    def get(self, key):
        if key not in self.dic:
            return -1
        v = self.dic.pop(key)
        self.dic[key] = v # key as the newest one
        return v
    
    def put(self, ket, value):
        if key in self.dic:
            self.dic.pop(key)
        else:
            if self.remain > 0:
                self.remain -= 1
            else:   # self.dic is full
                self.dic.popitem(last = False)
        self.dic[key] = value
```



```
排序算法
1.比较类排序:
通过比较来决定元素间的相对次序，由于其时间复杂度不能突破O(nlongn),因此也称为非线性时间比较类排序。
比较类排序包括交换，插入，选择，归并排序等。
2.非比较类排序:
不通过比较来决定元素间的相对次序，它可以突破基于比较排序的时间下界，以线性时间运行，因此也成为线性时间非比较类排序。
非比较类排序包括计数，桶，基数排序。
```



```java
选择排序模板
每次找最小元素
public int[] selectSort(int[] arr) {
	int len = arr.length;
	for (int i = 0; i < len; i++) {
		int index = i;
		// 寻找最小的元素，并将其索引保存
		for (int j = i + 1; j < len; j++) {
			if (arr[j] <  arr[index]) {
				index = j;
			}
		}
		int tmp = a[i];
		arr[i] = arr[index];
		arr[index] = tmp;
	}
	return arr;
}
```

```java
插入排序
通过构建有序序列，对于未排序数据，在已排序序列中从后向前扫描，找到相应位置并插入。
public int[] insertSort(int[] arr) {
	int len = arr.length;
	for (int i = 1; i < len; i++) {
		int preIndex = i - 1;
		int current = arr[i];
		while (preIndex >= 0 && arr[preIndex] > current) {
			arr[preIndex + 1] = arr[preIndex];
			preIndex--;
		}
		arr[preIndex + 1] = current;
	}
	return arr;
}
```

```java
冒泡排序
按从小达到顺序
public int[] bubleSort(int[] arr) {
	int len = arr.length;
	for (int i = 0; int i < len; i++) {
		for (int j = i + 1; j < len; j++) {
			if (a[j] < a[j - 1]) {
				int temp = a[j - 1];
				a[j - 1] = a[j];
				a[j] = temp;
			}
		}
	}
	return arr;
}
```



```java
快排
原理：小于某个元素放左边，大于某个元素放右边
public static void quickSort(int[] array, int begin, int end) {
	if (end <= begin) {
		return;
	}
	int pivot = partition(array, begin, end);
	// 左边
	quickSort(array, begin, pivot - 1);
	// 右边
	quickSort(array, pivot + 1, end);
}

public static int partition(int[] a , int begin, int end) {
	// pivot: 标杆位置，counter: 小于pivot的元素的个数
	int pivot = end, counter = begin;
	for (int i = begin; i < end; i++) {
		if (a[i] < a[pivot]) {
			int temp = a[counter];
			a[counter] = a[i];
			a[i] = temp;
			counter++;
		}
	}
	int temp = a[pivot];
	a[pivot] = a[counter];
	a[counter] = temp;
	return counter;
}
```

```java
归并排序
思想：分治
1.把长度为n的输入序列分成两个长度为n/2的子序列;
2.对这两个子序列分别采用归并排序;
3.将两个排序好的子序列合并成一个最终的排序序列。
public static void mergeSort(int[] array, int left, int right) {
	if (right <= left) return;
	int mid = (left + right) >> 1;
	
	mergeSort(array, left, mid);
	mergeSort(array, mid + 1, right);
	merge(array, left, mid, right);
}

punlic static void merge(int[] arr, int left, int mid, int right) {
	// 中间数组
	int[] temp = new int[right - left + 1];
	int i = left, j = mid + 1, k = 0;
	
	while (i <= mid && j <= right) {
		temp[k++] = arr[i] <= arr[j] ? arr[i++] : arr[j++];
	}
	
	while (i <= mid) temp[k++] = arr[i++];
	while (j <= right) temp[k++] = arr[j++];
	
	for (int p = 0; p < temp.length; p++) {
		arr[left + p] = temp[p];
	}
	// 也可以用System.arrarcopy(a, start1, b, start2, length)
}
```



```
归并和快排
归并:先排序左右子数组，然后合并两个有序子数组。
快排:先调配出左右子数组，然后对左右子数组进行排序。
```



```java
堆排序O(N*LogN)
Heap Sort 插入O(logN), 取最大/小值O(1)
1.数组元素依次建立小顶堆
2.依次取堆顶元素，并删除

public static void heapity(int[] array, int length, int i) {
	int left = 2 * i + 1, right = 2 * i + 2;
	int largest = i;
	
	if (left < length && array[left] > array[largest]) {
		largest = left;
	}
	if(right < length && array[right] > array[largest]) {
		largest = right;
	}
	
	if (largest != i) {
		int temp = array[i];
		array[i] = array[largest];
		array[largest] = temp;
	}
	
	if (largest != i) {
		int temp = array[i];
		array[i] = array[largest];
		array[largest] = temp;
		heapify(array, legnth, largest);
	}
}

public static void heapSort(int[] array) {
	if (array.length == 0) return;
	
	int length = array.length;
	for (int i = length >> 1 - 1; i >= 0; i--) {
		heapify(array, length, i);
	}
	
	for (int i = length - 1; i >=0; i--) {
		int temp = array[0];
		array[0] = array[i];
		array[i] = temp;
		heapify(array, i, 0);
	}
}
```



```
计数排序(Counting Sort)
计数排序要求输入的数据必须是有确定范围的整数。将输入的数据值转化为键存储在额外开辟的数组空间中;然后依次把计数大于1的填充回原数组

桶排序(Bucket Sort)
假设输入数组服从均匀分布，将数据分到有限数量的桶里，每个桶再分别排序(有可能再使用别的排序算法或是以递归方式继续使用桶排序进行排)。

基数排序(Radix Sort)
基数排序是按照低位先排序，然后收集;再按照高位排序，然后收集;依次类推，直到最高位。有时候有些属性是有优先级顺序的，先按低优先级排序，再按高优先级排序。
```









