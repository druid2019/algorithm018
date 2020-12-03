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

