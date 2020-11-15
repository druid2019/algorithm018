DFS模板(深度优先)Deep First Search

```python
递归写法
visited =  set()

def dfs(node, visited):
    # terminator
    if node in visited:
        # already visited
        return
    visited.add(node)
    
    # process current node here
    ...
    for next_node in node.children():
        if next_node not in visited:
            dfs(next_node, visited)

非递归写法(手动维护一个栈)
def DFS(self, tree):
    // terminate
    if tree.root is None:
        return []
    
    visited, stack = [], [tree.root]
    
    # process current node here
    while stack:
        node = stack.pop()
        visited.add(node)
        
        process(node)
        nodes = generate_related_nodes(node)
        stack.push(nodes)
    # other processing work
    ...
```



```java
递归(N叉树或二叉树的层次遍历)
public List<List<Integer>> levelOrder(TreeNode root) {
	List<List<Integer>> allResults = new ArrayList<>();
	if (root == null) {
		return allResults;
	}
	travel(root, 0, allResults);
	return allResults;
}

private void travel(TreeNode root, int level, List<List<Integer>> results) {
	if (results.size() == level) {
		results.add(new ArrayList<>());
	}
	results.get(level).add(root.val);
	if (root.left != null) {
		travel(root.left, level + 1, results)
	}
	if (root.right != null) {
		travel(root.right, level + 1, results)
	}
}
```



```python
BFS(广度优先)Breath First Search
def BFS(graph,  start, end) {
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
}
```



```java
java(链表实现的队列)
public class TreeNode {
    int val;
    TreeNode left;
    TreeNode right;
    
    TreeNode(int x) {
        val = x;
    }
}

public List<List<Integer>> levelOrder(TreeNode root) {
    List<List<Integer>> allResults = new ArrayList<>();
    if (root == null) {
        return allResults;
    }
    Queue<TreeNdoe> nodes = new LinkedList<>();
    nodes.add(root);
    while (!node.isEmpty()) {
        int size = nodes.size();
        List<Integer> results = new ArrayList<>();
        for (int i = 0; i < size; i++) {
            // 出队
            TreeNode node = nodes.poll();
            result.add(node.val);
            if (node.left != null) {
                nodes.add(node.left);
            }
            if (node.right != null) {
                nodes.add(node.right);
            }
        }
        allResults.add(results);
    }
    return allResults;
}
```

