```java
期中考试
四数之和
思路一：可将四数之和转换为三数之和，只是多了一个循环
时间复杂度O(N³)，空间复杂度O(logN)
public List<List<Integer>> fourSum(int[] nums,int target) {
        List<List<Integer>> result = new ArrayList<>();
        // 数组长度
        int len = nums.length;
        if (len < 4) {
            return result;
        }
        // 排序
        Arrays.sort(nums);
        for (int i = 0;i< len-3;i++) {
            // 当相邻数相等时，跳过该数，避免重复计算
            if (i > 0 && nums[i] == nums[i-1]) {
                continue;
            }
            for (int j = i+1;j < len-2;j++) {
                if (j > i+1 && nums[j] == nums[j-1]) {
                    continue;
                }
                int left = j+1,right = len -1, sum = target - nums[i] - nums[j];
                while (left < right) {
                    if (nums[left] + nums[right] == sum) {
                        result.add(Arrays.asList(nums[i], nums[j], nums[left], nums[right]));
                        // 相邻重复元素
                        while (left < right && nums[left] == nums[left+1]) left++;
                        while (left < right && nums[right-1] == nums[right]) right--;
                        left++;
                        right--;
                    } else if (nums[left] + nums[right] < sum) {
                        left++;
                    } else {
                        right--;
                    }
                }
            }
        }
        return result;
    }

最小跳跃次数
输入: [2,3,1,1,4]
输出: 2
解释: 跳到最后一个位置的最小跳跃数是 2。
从下标为 0 跳到下标为 1 的位置，跳 1 步，然后跳 3 步到达数组的最后一个位置
思路一：贪心，curMax当前能走的最大长度，nextMax最终能走的最大长度，step为次数
时间复杂度：O(N)
public static int jump(int[] nums) {
	int len  = nums.length;
	// 跳跃次数
    int step = 0;
    // 当前能走的最大长度
    int curMax = 0;
    // 最终能走的最大长度
    int nextMax = 0;
    for (int i = 0;i < len - 1;i++) {
    	nextMax = Math.max(nextMax, nums[i] + i);
        if (i == curMax) {
        	step++;
            curMax = nextMax;
        }
    }
        return step;
}
```

