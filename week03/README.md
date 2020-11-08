```java
递归-循环
通过函数体来进行的循环
public void recur(int level, int param) {
	// terminator
	if (level > MAX_LEVEL) {
		// process result
		return;
	}
	
	// process current logic
	process(level, param);
	
	// drill down
	recur(level: level + 1, newParam);
	
	// restore current status
}
```



```python
分治
# python
def devide_conquer(problem,param1,param2, ...):
    # recursion terminator
    if problem is None:
        print_result
        return
    
    #prepare data
    data = prepare_data(problem)
    subproblems = split_problem(problem, data)
    
    # conquer subproblems
    subresult1 = self.divide_conquer(subproblems[0], p1, ...)
    subresult2 = self.divide_conquer(subproblems[1], p1, ...)
    subresult3 = self.divide_conquer(subproblems[2], p1, ...)
    ...
    
    # process and generate the final result
    result = process_result(subresult1,subresult2,subresult3,...)
    
    # revert the current level states
```



```java
分治
// java
private static int divide_conquer(Problem problem, ) {
	// recursion terminator
	if (problem == Null) {
		int res = process_last_result();
		return res;
	}
	// process data
	subProblems = split_problem(problem);
	
	// conquer subproblems
	res0 = divide_conquer(subProblems[0]);
	res1 = divide_conquer(subProblems[1]);
	res2 = divide_conquer(subProblems[2]);
	...
	
	// process and generate the final result
	result = process_result(res0,res1,res2,...)
	return result;
}
```



```
回溯
  回溯采用试错的思想，尝试分步去解决问题。在分步解决问题的过程中，当它通过尝试发现现有的分步答案不能得到有效正确的解答时，它将取消上一步甚至上几步的计算，再通过其它可能的分步解答再次尝试寻找问题答案。简单来讲，就是不断地试错，然后找到一条解法。
  回溯法通常用最简单的递归方法来实现，在反复重复上诉步骤后可能出现两种情况：
  找到一个可能存在的正确答案;
  在尝试了所有可能的分步方法后宣告该问题没有答案。
  在最坏的情况下，回溯法会导致一次复杂度为指数时间的计算。
```

