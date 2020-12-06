```
字典树，即Trie树，又称单词查找树或键树，是一种树形结构 。典型应用是用于统计和排序大量的字符串(但不仅限于字符串)，所以经常被搜索引擎系统用于文本词频统计。
Trie树不是二叉树，是多叉树
优点：
最大限度地减少无谓的字符串比较，查询效率比哈希表高。
基本性质:
1.结点本身不存完整单词；
2.从根节点到某一结点，路径上经过的字符连接起来，为该节点对应的字符串。
3.每个结点的所有子结点路径代表的字符都不相同。
```



```python
python的实现方法
class Trie(object):
    
    def __init__(self):
        self.root = {}
        self.end_of_word = "#"
    
    def insert(self, word):
        node = self.root
        for char in word:
            node = node.setdefault(char, {})
        node[self.end_of_word] = self.end_of_word
        
    def search(self, word):
        node = self.root
        for char in word:
            if char not in node:
                return False
            node = node[char]
        return self.end_of_word in node
    
    
    def startsWith(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node:
                return False
            node = node[char]
        return True
```

```java
java的实现方法
class Trie {
	private boolean isEnd;
	private Trie[] next;
	/** Initialize your data structure here*/
	public Trie() {
		isEnd = false;
		next = new Trie[26];
	}
	
	/** Insearts a word into the trie. */
	public void insert(String word) {
		if (word == null || word.length() == 0) {
			return;
		}
		Trie curr = this;
        char[] words = word.toCharryArray();
        for (int i = 0; i < words.length; i++) {
        	int n = words[i] - 'a';
			if (curr.next[n] == null) curr.next[n] = new Trie();
            curr = curr.next[n];
		}
        curr.isEnd = true;
    }
    
	/** Returns if the word is in the trie. */
	public boolean startsWith(String preifx) {
    	Trie node = searchPrefix(prefix);
        return node != null;
	}
		
	private Trie searchPrefix(String word) {
    	Trie node = this;
        char[] words = word.toCharArray();
        for (int i = 0; i < words.length; i++) {
        	node = node.next[word[i] - 'a'];
            if (node == null) return null;
        }
		return node;
	}
}
```



```java
并查集
makeSet(s): 建立一个新的并查集，其中包含s个单元素集合
unionSet(x,y): 把元素x和元素y所在的集合合并，要求x和y所在的集合不相交，如果相交则不合并。
find(x): 找到元素x所在的集合的代表，该操作也可以用于判断两个元素是否位于同一个集合， 只要将它们各自的代表比较一下就可以了。
模板：
class UnionFind {
	private int count = 0;
	private int[] parent;
	public UnionFind(int n) {
		count = n;
		parent = new int[n];
		for (int i = 0; i < n; i++) {
			parent[i] = i;
		}
	}
	public int find(int p) {
		while (p != parent[p]){
			parent[p] = parent[parent[p]];
			p = parent[p];
		}
		return p;
	}
	
	public void union(int p, int q) {
		int rootP = find(p);
		int rootQ = find(q);
		if (rootP == rootQ) return;
		parent[rootP] = rootQ;
		count--;
	}
}
```

```python
def init(p):
	# for i = 0 .. n: p[i] = i
	p = [i for i in range(n)]

def union(self, p, i, j):
	p1 = self.parent(p, i)
	p2 = self.parent(p, j)
	p[p1] = p2
	
def parent(self, p, i):
	root = i
	while p[root] != root:
		root = p[root]
	# 路径压缩
	while p[i] != i:
		x =i
		i = p[i]
		p[x] = root
	return root
```



```python
dfs代码-递归模板

visited = set()

def dfs(node, visited):
    # terminator
    if node in visited:
        # already visited
        return
    
    visited.add(node)
    
    # process current node here
    ...
    for next_node in node.children():
        if not next_node in visited:
            dfs(next_node,visited)

```

```python
bfs模板-队列 python
def BFS(graph, start, end):
	visited = set()
    queue = []
    queue.append([start])
    
    while queue:
        node = queue.pop()
        visited.add(node)
    	
        process(node)
        nodes = generate_related_nodes(node)
        queue.push(nodes)
    # other processing work
    ...  
```

```java
java代码
public class TreeNode {
	int val;
	TreeNode left;
	TreeNode right;
	
	TreeNode (int x) {
		val = x;
	}
}

public List<List<Integer>> leverOrder(TreeNode root) {
	List<List<Integer>> allResults = new ArrayList<>();
	if (root == null) {
		return allResults;
	}
	Queue<TreeNode> nodes = new LinkedList<>();
	nodes.add(root);
	while (!node.isEmpty()) {
		int size = nodes.size();
		List<Integer> results = new ArrayList<>();
		for (int i = 0; i < size; i++) {
			TreeNode node = node.poll();
			resulst.add(node.val);
			if (node.left != null) {
				node.add(node.left);
			}
			if (node.right != null) {
				node.add(node.right);
			}
		}
		allResults.add(results);
	}
	return allResults;
}
```



```
回溯法：
分治+试错
回溯法采用试错的思想，它尝试分布的去解决一个问题。在分布解决问题的过程中，当它通过尝试发现现有的分步答案不能得到有效的正确的解答的时候，它将取消上一步甚至上几步的计算，再通过其它的可能的分步解答再次尝试寻找问题的答案。
回溯法通常用最简单的递归方法来实现，在反复重复上述的步骤后可能出现两种情况：
1.找到一个可能存在的正确的答案
2.在尝试了所有可能的分步方法后宣告该问题没有答案
在最坏的情况下，回溯法会导致一次复杂度为指数时间的计算。
```



```python
A*模板
def AstarSearch(grapth, start, end):
    
    pq = collections.priority_queue()  # 优先级 —> 估价函数
    pq.append([start])
    visited.add(start)
    
    while pq:
        node = pq.pop()
        visited.add(node)
        
        process(node)
        nodes = generate_realated_nodes(node)
   unvisited = [node for node in nodes if node not in visited]
		pq.push(unvisited)
```









