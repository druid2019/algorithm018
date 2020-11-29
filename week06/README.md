``` 
递归
public void recur(int level, int param) {
	// terminator
	if (level >  Max_LEVEL) {
		// process result
		return;
	}
	
	// prcess current logic
	process(level, param);
	
	// drill down
	recur(level: level+1, newParam);
	
	// restore current status
	
}
```

```java
分治
private static int divide_conquer(Problem problem, ) {
	if (problem == null) {
		int res = process_last_result();
		return res;
	}
	
	subProblems = split_problem(problem);
	
	res0 = devide_conquer(subProblems[0]);
	res1 = devide_conquer(subProblems[1]);
	
	result = process_result(res0, res1);
	
	return result;
}
```

```
动态规划
Dynamic programming
	Dynamic programming is both a mathematical optimization method and a computer programming method. 
	In both contexts it refers to simplifying a complicated problem by breaking it down into simpler sub-problems in a recursive manner.While some decision problems cannot be taken apart this way, decision that span several points in time do often break apart recursively.
```



```java
斐波拉契数
int fib(int n) {
	return n <= 1?n:fib(n-1) + fib(n-2);
}

缓存，减少重复值的计算
int fib (int n, int[] memo) {
	if (n <= 1) {
		return n;
	}
	
	// 如果不等于0，说明已经计算过了，就不用走该方法了，直接返回，避免重复计算
	if (memo[n] == 0) {
		memo[n] = fib(n-1) + fib(n-2);
	}
	
	return memo[n];
}
```

















