学习笔记

```java
day29
127. 单词接龙
给定两个单词（beginWord 和 endWord）和一个字典，找到从 beginWord 到 endWord 的最短转换序列的长度。转换需遵循如下规则：
每次转换只能改变一个字母。
转换过程中的中间单词必须是字典中的单词。
说明:
如果不存在这样的转换序列，返回 0。
所有单词具有相同的长度。
所有单词只由小写字母组成。
字典中不存在重复的单词。
你可以假设 beginWord 和 endWord 是非空的，且二者不相同。
示例 1:
输入:
beginWord = "hit",
endWord = "cog",
wordList = ["hot","dot","dog","lot","log","cog"]
输出: 5
解释: 一个最短转换序列是 "hit" -> "hot" -> "dot" -> "dog" -> "cog",
     返回它的长度 5。
示例 2:
输入:
beginWord = "hit"
endWord = "cog"
wordList = ["hot","dot","dog","lot","log"]
输出: 0
解释: endWord "cog" 不在字典中，所以无法进行转换。
思路一：求最短长度，用bfs,逐层寻找与上一层只差一个字符的单词，当单词与目标单词一致时返回当前层数。
可采取队列来实现广度优先
代码逻辑比较清晰，时间复杂度较高，空间复杂度较低。
执行用时：820 ms, 在所有 Java 提交中击败了22.72%的用户
内存消耗：38.5 MB, 在所有 Java 提交中击败了96.62%的用户
public int ladderLength(String beginWord, String endWord, List<String> wordList) {
	if (!wordList.contains(endWord)) {
		return 0;
	}
	// 初始化层数
	int res = 1;
	// 在原字典中去掉beginWord
	wordList.remove(beginWord);
	// 用队列去存储每层的单词
	Queue<String> queue = new LiknedList<String>();
	queue.add(beginWord);
	while(!queue.isEmpty()) {
		// 读取当前层有多少单词需要遍历
		int size = queue.size();
		while(size > 0) {
			String temp = queue.poll();
			// 如果和目标单词一致则返回层数
			if(temp.equals(endWord)) {
				return res;
			}
			// 找到wordList中和当前单词只差一位的单词加入队列
			find(queue, temp, wordList);
			// 每遍历一个单词减一
			size--;
		}
		// 当前层遍历完，层数加一
		res++;
	}
	return 0;
}

public void find(Queue<String> queue, String temp, List<String> wordList) {
	for (int i = 0; i < wordList.size(); i++) {
		// 获取wordList的元素
		String s = wordList.get(i);
		// 初始化相同元素数量
		int count = 0;
		// 比较字典中的单词元素和所给的单词匹配程度
		for (int m = 0; m < s.length();m++) {
			if (temp.charAt(m) != s.charAt(m)) {
				count++;
			}
        }	
        // 如果当前单词和temp只差一位
        if (count == 1) {
        // 将其加入队列
        queue.add(s);
        // 从wordList中移除这个单词
        wordList.remove(s);
        // 移除后wordList中所有的会向前移一位，故i--保证从下一个开始遍历
        i--;
		}
	}
}
```

```java
day30
1. 两数之和
给定一个整数数组 nums 和一个目标值 target，请你在该数组中找出和为目标值的那 两个 整数，并返回他们的数组下标。
你可以假设每种输入只会对应一个答案。但是，数组中同一个元素不能使用两遍。
示例:
给定 nums = [2, 7, 11, 15], target = 9
因为 nums[0] + nums[1] = 2 + 7 = 9
所以返回 [0, 1]
思路一：哈希表存储值
定义一个map，记录下标和值，避免重复，找到即返回。
时间复杂度O(N),空间复杂度O(N)。
public int[] twoSum(int[] nums, int target) {
    // 初始化一个map
	Map<Integer, Integer> map = new HashMap<>();
	// 往map中添加元素，如果找到即返回
    for (int i = 0;i < nums.length; i++) {
    	if (!map.containsKey(target-nums[i])) {
    		map.put(nums[i], i);
    	} else {
    		return new int[]{map.get(target-nums[i]), i};
    	}		
    }       
    return new int[]{};
}

```

```java
day31
874. 模拟行走机器人
机器人在一个无限大小的网格上行走，从点 (0, 0) 处开始出发，面向北方。该机器人可以接收以下三种类型的命令：
-2：向左转 90 度
-1：向右转 90 度
1 <= x <= 9：向前移动 x 个单位长度
在网格上有一些格子被视为障碍物。
第 i 个障碍物位于网格点  (obstacles[i][0], obstacles[i][1])
机器人无法走到障碍物上，它将会停留在障碍物的前一个网格方块上，但仍然可以继续该路线的其余部分。
返回从原点到机器人所有经过的路径点（坐标为整数）的最大欧式距离的平方。
示例 1：
输入: commands = [4,-1,3], obstacles = []
输出: 25
解释: 机器人将会到达 (3, 4)
示例 2：
输入: commands = [4,-1,4,-2,4], obstacles = [[2,4]]
输出: 65
解释: 机器人在左转走到 (1, 8) 之前将被困在 (1, 4) 处
思路一：obstacleSet 存储一个序列，记录障碍物坐标，因为每一步都是一格一格走的
官方题解，不太好理解
时间复杂度：O(N+K),空间复杂度：O(K)，用于存储 obstacleSet 而使用的空间。
public int robotSim(int[] commands, int[][] obstacles) {
	int[] dx = new int[]{0, 1, 0, -1};
    int[] dy = new int[]{1, 0, -1, 0};
    int x = 0, y = 0, di = 0;

	//之所以将坐标(x+30000) * (2^16),是将网格中每个坐标降维成一维，所有的点均在一个数轴上的值
    Set<Long> obstacleSet = new HashSet();
    for (int[] obstacle: obstacles) {
    	long ox = (long) obstacle[0] + 30000;
        long oy = (long) obstacle[1] + 30000;
        obstacleSet.add((ox << 16) + oy);
    }

	int ans = 0;
    for (int command : commands) {
    	// 向左旋转
        if (command == -2) {
        	di = (di + 3) % 4;
        } else if (command == -1) { // 向右旋转
        	di = (di + 1) % 4;
        } else {
        	for (int k = 0; k < command; k++) {
            	int nx = x + dx[di];
                int ny = y + dy[di];
                long code = (((long) nx + 30000) << 16) + ((long) ny + 30000);
                if (!obstacleSet.contains(code)) {
                	x = nx;
                    y = ny;
                    ans = Math.max(ans, x * x + y * y);
                }
            }
        }
    }
  return ans;
}

方法二：
public int robotSim(int[] commands, int[][] obstacles) {
        int[] direx={0,1,0,-1};
        int[] direy={1,0,-1,0};
        int curx=0,cury=0;
        int curdire=0;
        int comLen=commands.length;
        int ans=0;
        Set<Pair<Integer,Integer>> obstacleSet=new HashSet<>();
        for (int i = 0; i <obstacles.length ; i++) {
            obstacleSet.add(new Pair<>(obstacles[i][0],obstacles[i][1]));
        }
        for (int i = 0; i <comLen ; i++) {
            if(commands[i]==-1){
                curdire=(curdire+1)%4;
            }
            else if(commands[i]==-2){
                curdire=(curdire+3)%4;
            }
            else {
                for (int j = 0; j <commands[i] ; j++) {
                    int nx=curx+direx[curdire];
                    int ny=cury+direy[curdire];
                    if(!obstacleSet.contains(new Pair<>(nx,ny))){
                        curx=nx;
                        cury=ny;
                        ans=Math.max(ans,curx*curx+cury*cury);
                    }
                    else {
                        break;
                    }
                }
            }
        }
        return ans;
    }


```

```java
day32 
53. 最大子序和
给定一个整数数组 nums ，找到一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。
示例:
输入: [-2,1,-3,4,-1,2,1,-5,4]
输出: 6
解释: 连续子数组 [4,-1,2,1] 的和最大，为 6。
思路一：暴力计算，不推荐，时间复杂度O(N²)
每种计算都遍历一遍，取最大值,比较好理解
public int maxSubArray(int[] nums) {
	int res = nums[0];
	for (int i = 0; i < nums.length; i++) {
        int sum = 0;
		for(int j = i; j < nums.length; j++) {
            sum += nums[j];
            if (sum > res) {
                res = sum;
            }
        }
	}
	return res;
}

思路二：扫描法，时间复杂度O(N)
代码简洁，但不太好理解
public static int maxSubArray(int[] nums) {
	int init = 0;
    int res = nums[0];
    // 如果全为负数，则返回最大的负数
    for (int i = 0; i < nums.length - 1; i++) {
    	// 小于0，舍去(否则影响接下来的数)
    	if (init < 0 ) {
        	init = nums[i];
        } else {
        	init += nums[i]; // 不为负，则累加
        }
        // 保留最大值
        if (init > res) {
        	res = init;
        }
    }
    return res;
}

思路三：动态规划,时间复杂度O(N)
假设对于元素i，所有以它前面的元素结尾的子数组的长度都已经求得，那么以第i个元素结尾且和最大的连续子数组实际上，要么是以第i-1个元素结尾且和最大的连续子数组加上这个元素，要么是只包含第i个元素，即sum[i]
= max(sum[i-1] + a[i], a[i])。可以通过判断sum[i-1] + a[i]是否大于a[i]来做选择，而这实际上等价于判断sum[i-1]是否大于0。
public int maxSubArray(int[] nums) {
    int sum = nums[0];
    int init = nums[0];
    for (int i = 1; i < nums.length; i++) {
        if (init > 0) {
            init += nums[i];
        } else {
            init = nums[i];
        }
        if (sum < init) {
            sum = init;
        }       
    }
    return sum;
}

```

```java
day33
1143. 最长公共子序列
给定两个字符串 text1 和 text2，返回这两个字符串的最长公共子序列的长度。
一个字符串的 子序列 是指这样一个新的字符串：它是由原字符串在不改变字符的相对顺序的情况下删除某些字符（也可以不删除任何字符）后组成的新字符串。
例如，"ace" 是 "abcde" 的子序列，但 "aec" 不是 "abcde" 的子序列。两个字符串的「公共子序列」是这两个字符串所共同拥有的子序列。
若这两个字符串没有公共子序列，则返回 0。

思路一：动态规划
text1,text2作为二维数组,用d[i][j]来统计数量,
如果text1[i] = text2[j],则d[i][j] = d[i-1][j-1] + 1
如果text1[i] != text2[j],则d[i][j] = max(d[i-1][j],d[i][j-1])
定义为a串第i位置b串第j位置以前的两个序列的最大的LCS,显然第一个d[0][0] = 0;

public int longestCommonSubsequence(String text1, String text2) {
    // 字符串长度要增加一个，错开一位
    text1 = "1" + text1;
    text2 = "2" + text2;
    int[][] d = new int[text1.length() + 1][text2.length() + 1];   
    for (int i = 0; i < text1.length(); i++) {
        for (int j = 0; j < text2.length(); j++) {
            if (i == 0 || j == 0) {
                d[i][j] = 0;
            } else if (text1.charAt(i)  == text2.charAt(j)) {
                d[i][j] = d[i-1][j-1] + 1;
            } else {
                d[i][j] = Math.max(d[i-1][j],d[i][j-1]);
            }

        }
    }
    // 返回最后一个元素
    return d[text1.length()-1][text2.length()-1];
}
```

```java
day34
74. 搜索二维矩阵
编写一个高效的算法来判断 m x n 矩阵中，是否存在一个目标值。该矩阵具有如下特性：

每行中的整数从左到右按升序排列。
每行的第一个整数大于前一行的最后一个整数。
示例 1：
输入：matrix = [[1,3,5,7],[10,11,16,20],[23,30,34,50]], target = 3
输出：true
示例 2：
输入：matrix = [[1,3,5,7],[10,11,16,20],[23,30,34,50]], target = 13
输出：false
示例 3：
输入：matrix = [], target = 0
输出：false
思路一：二分查找
时间复杂度O(log(mn))，空间复杂度O(1),m,n分别为行列的长度
public boolean searchMatrix(int[][] matrix, int target) {
	// 从左到右递增，从上到下递增
	// if (matrix == null&& matrix.length == 0){
	//     return false;
	// }
	// 行
	int r = matrix.length;
	if(r == 0) return false;
	// 列
	int c = matrix[0].length-1;
	if(c == -1) return false;
	for (int i=0;i<r;i++) {
		int low = 0;
		int high= c;
		// 参照点的下标
		int mid = 0;
		while (low <= high) {
			mid = (low+high)/2;
			if (target == matrix[i][mid]) {
				return true;
			}
			if (target < matrix[i][mid]) {
				high = mid-1;
			} else {
				low = mid +1;
			}
		}
	}
	return false;
}
```

```java
day35
剑指 Offer 05. 替换空格
请实现一个函数，把字符串 s 中的每个空格替换成"%20"。
示例 1：

输入：s = "We are happy."
输出："We%20are%20happy."

思路一：直接替换
public String replaceSpace(String s) {
	return s.replace(" ","%20");
}
// 不建议，时间复杂度比第一个要高
public String replaceSpace(String s) {
	return s.replaceAll(" ","%20");
}

public String replaceSpace(String s) {
	return s.replaceAll("\\s","%20");
}

思路二：使用StringBuilder
public String replaceSpace(String s) {
	char[] c = s.toCharArray();
	StringBuilder sb = new StringBuilder();
	for (char i: c) {
		if (i != ' ') {
			sb.append(i);
		} else {
			sb.append("%20");
		}
	}
	return sb.toString();
}
```
