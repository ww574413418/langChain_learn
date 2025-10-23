# 1、编写一个函数来查找字符串数组中的最长公共前缀。
# 示例 1：
# 输入：strs = ["flower","flow","flight"]
# 输出："fl"
# 示例 2：
# 输入：strs = ["dog","racecar","car"]
# 输出：""
# 解释：输入不存在公共前缀。

def longCommonoPrefix(str):
    if not str:
        return ""

    #假设第一个字符为公共前缀
    prefix = str[0]

    #便利剩余字符
    for s in str[1:]:
        while not s.startswith(prefix):
            prefix = prefix[:-1]
            if not prefix:
                return ""

    return prefix

str1 = ["flower","flow","flight"]
str2 = ["dog","racecar","car"]

print(longCommonoPrefix(str1))