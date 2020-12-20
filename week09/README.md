```java
递归
public void recur(int level, int param) {
    
    // terminator
    if (level > MAX_LEVEL) {
    	// process result
    	return;
    }
    
    // process current logic
    process(level, param);
    
    // drill down
    recur(level:level + 1, newParam);
    
    // restore current status
}
```

```python
分治
def devide_conquer(problem, param1, param1, ...):
	# recursion terminator
	if problem is None:
		print_result
		return
		
    # prepare data
    data = prepare_data(problem)
    subproblems = split_problem(problem, data)
    
    # conquer subproblems
    subresult1 = self.divide_conquer(subproblems[0], p1, ...)
    subresult2 = self.divide_conquer(subproblems[1], p1, ...)
    subresult3 = self.divide_conquer(subproblems[2], p1, ...)
    ...
    
    # process and generate the final result
    result = process_result(subresult1, subresult2, subresult3, ...)
    
    # revert the current level states
```



```
动态规划Dynamic Programming
1."Simplifying a complicated problem by breaking it down into simpler sub-problems"(in a recursive manner)
2.Divide & Conquer + Optimal substructure
分治 + 最优子结构
3.顺推形式：动态递推
```

```python
DP顺推模板
function DP():
	# 二维情况
	dp = [][]
	
	for i = 0.. M {
		for j = 0.. N {
			dp[i][j] = _Function(dp[i'][j']...)
		}
	}
```



```java
字符串匹配算法
KMP 算法,字符串a里面是否包含另一个字符b。
许多算法可以完成这个任务，Knuth-Morris-Pratt算法（简称KMP）是最常用的之一。它以三个发明者命名，起头的那个K就是著名科学家Donald Knuth。

暴力法代码:
public static int forceSearch(String txt, String pat) {
	int M = txt.length();
	int N = pat.length();
	for (int i = 0; i <= M - N; i++) {
        int j;
        for (j = 0; j < N; j++) {
            if (txt.charAt(i + j) != pat.charAt(j))
                break;
        }
        if (j == N) {
            return i;
        }
    }
    return -1;
}
```











