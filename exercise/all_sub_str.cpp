题目：求【无重复字母】字符串的所有子串
分析：一个字符串没有重复字母，这里子串指的是字符串中字母的所有组合可能。但是组合中，字母的先后顺序不能变。

void func(string s){
    int num_sub = pow(2, s.size())-1;
    vector<string> ans(num_sub);
    int i;
    for(i = 0; i < s.size(); i++){
        ans[i] = string(1, s[i]);
    }
    int index = 0;
    while(i < num_sub){
        char t = ans[index][ans[index].size()-1];
        for(int j = string.find(t)+1; j < s.size(); j++){
            ans[i++] = ans[index] + string(1, s[j]);
        }
        index++;
    }
}