---
title: 2020 年「计算机科学与工程学院」新生赛总结
urlname: scutpc2020-summary
date: 2020-11-23 17:54:01
updated: 2020-11-23 17:54:01
toc: true
tags:
- ACM
- 算法
categories:
- 计算机
- 算法竞赛
---

<style type="text/css">
.content .tabs ul { margin: 0; }
.tab-content { display: none; }
</style>

第一次作为出题人参与一场算法竞赛，感受还是很不同的。相比与参赛者，少了一些紧张刺激新鲜感，当然也少了一些自闭。

比赛在[SCUT CODE](https://scut.online)上举行，总体而言这个系统做的还是挺不错的，响应迅速，功能齐全。唯一要吐槽的就是题目竟然只能添加不能删除！添加比赛需要一些玄之又玄的操作。还有Special Judge也是非常难配置，还缺少了交互功能。第一场由于没有放特别简单的签到题导致大量选手爆0，导致第二场人数锐减。。。不过第二场比赛的题目最后经过调整还是简单了许多的。下面按难度总结一下这次比赛的题目，目前题目已经全部开放了，可以在题库中找到提交。

[第一场比赛链接](https://scut.online/contest/106)
[第二场比赛链接](https://scut.online/contest/99)

<!-- more -->

所有相关题目的题库链接：

[658. 垃圾邮件](https://scut.online/p/658)
[661. 一个小游戏](https://scut.online/p/661)
[662. Overload](https://scut.online/p/662)
[663. 乒乓球](https://scut.online/p/663)
[664. 超人高中生的基建计划](https://scut.online/p/664)
[665. 灰之魔女的旅行](https://scut.online/p/665)
[666. 艾尔奇亚的国王](https://scut.online/p/666)
[667. umi炒饭](https://scut.online/p/667)
[668. Dirichlet卷积](https://scut.online/p/668)
[669. 1-2 心与心之间的距离，永不点亮的音乐会](https://scut.online/p/669)
[670. 序列构造](https://scut.online/p/670)
[671. 爆炸就是艺术](https://scut.online/p/671)
[672. Tree](https://scut.online/p/672)
[673. 猪灵的胜利之舞](https://scut.online/p/673)
[674. Merge Stone](https://scut.online/p/674)
[675. 学园都市的神秘代码](https://scut.online/p/675)
[676. 世界棋盘](https://scut.online/p/676)
[677. 谜拟Q plus](https://scut.online/p/677)
[678. 射击训练](https://scut.online/p/678)
[679. 小斯巴达们的历练](https://scut.online/p/679)
[680. 主人和雷姆拉姆的秘密](https://scut.online/p/680)
[682. 爆炸就是艺术2](https://scut.online/p/682)
[683. 乒乓球2](https://scut.online/p/683)


## [射击训练](https://scut.online/p/678)

题意：统计有多少个点在圆内。

利用距离公式计算即可，浮点数可以直接输出。


{% tabs p678 %}
<!-- tab C++ -->
```cpp
#include <bits/stdc++.h>
using namespace std;
double sqr(double x) {
    return x * x;
}
int main() {
    int a, b, r, n, cnt = 0;
    cin >> a >> b >> r >> n;
    for (int i = 0; i < n; i++) {
        double x, y;
        cin >> x >> y;
        if (sqr(x-a)+sqr(y-b) <= sqr(r)) cnt++;
    }
    cout << double(cnt)/n << '\n';
}
```
<!-- endtab -->
<!-- tab Python -->
```py
a, b, r = map(int, input().split())
n = int(input())
cnt = 0
for _ in range(n):
    x, y = map(float, input().split())
    if (x-a)**2+(y-b)**2<=r*r: cnt += 1
print(cnt/n)
```
<!-- endtab -->
{% endtabs %}

## [Overload](https://scut.online/p/662)

题意：一颗以1为根的树，给定每个节点到父亲节点的距离，求到根节点的最大距离。

用dfs或bfs将树遍历一遍即可，也可以用带权并查集计算。

{% tabs p662 %}
<!-- tab C++ (DFS) -->
```cpp
#include <bits/stdc++.h>
using namespace std;
int main() {
    int n, fa;
    cin >> n;
    vector<vector<int>> G(n+1);
    vector<int> d(n+1);
    for (int i = 2; i <= n; i++) {
        cin >> fa >> d[i];
        G[fa].push_back(i);
    }
    function<void(int)> dfs=[&](int u) {
        for (auto v : G[u]) {
            d[v] += d[u];
            dfs(v);
        }
    };
    dfs(1);
    cout << *max_element(d.begin(), d.end()) << '\n';
    return 0;
}
```
<!-- endtab -->
<!-- tab C++ (BFS) -->
```cpp
#include <bits/stdc++.h>
using namespace std;
int main() {
    int n, fa;
    cin >> n;
    vector<vector<int>> G(n+1);
    vector<int> d(n+1);
    for (int i = 2; i <= n; i++) {
        cin >> fa >> d[i];
        G[fa].push_back(i);
    }
    queue<int> Q;
    Q.push(1);
    while (!Q.empty()) {
        int u = Q.front();
        Q.pop();
        for (auto v : G[u]) {
            d[v] += d[u];
            Q.push(v);
        }
    }
    cout << *max_element(d.begin(), d.end()) << '\n';
    return 0;
}
```
<!-- endtab -->
{% endtabs %}

## 乒乓球
### [乒乓球](https://scut.online/p/663)

{% msg warning %}
此题未加入比赛，[乒乓球2](https://scut.online/p/683)为本题的简化版本。
{% endmsg %}

题意：求$\prod\frac{p_i}{100} \mod 1000000007$。

在模意义下求值，利用[费马小定理](https://oi-wiki.org/math/fermat/#_1)与[快速幂算法](https://oi-wiki.org/math/quick-pow/)计算。

{% tabs p663 %}
<!-- tab C++ -->

```cpp
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
const ll M = 1e9 + 7;
ll qpow(ll x, ll y) {
    ll ans = 1;
    while (y) {
        if (y & 1) ans = ans * x % M;
        x = x * x % M;
        y >>= 1;
    }
    return ans;
}
int main() {
    int n;
    cin >> n;
    ll ans = 1;
    for (int i = 1; i <= n; i++) {
        ll x;
        cin >> x;
        ans = ans * x % M;
    }
    cout << ans*qpow(100, n*(M-2))%M << '\n';
    return 0;
}
```
<!-- endtab -->
<!-- tab Python -->

```py
from functools import reduce
n, m = int(input()), 10**9+7
a = list(map(int, input().split(' ')))
x = reduce((lambda x,y:x*y%m), a)
print(x*pow(100,n*(m-2),m)%m)
```

python自带的pow（注意不是math.pow）第三个参数可以传mod，复杂度为快速幂复杂度。
<!-- endtab -->
{% endtabs %}

### [乒乓球2](https://scut.online/p/683)
题意就是输出一列数的最小值和最大值。

这应该是整场比赛中最简单的题目了，只要能熟练掌握一种编程语言（C/C++/Java/Python）都应该能写出这题。

具体实现细节可以参考代码。

{% tabs p683 %}
<!-- tab C++ -->

```cpp
#include <bits/stdc++.h>
using namespace std;
int main() {
    int n, Min = 100001, Max = 0;
    cin >> n;
    for (int i = 0; i < n; i++) {
        int x;
        cin >> x;
        if (x > Max) Max = x;
        if (x < Min) Min = x;
    }
    cout << Min << ' ' << Max << '\n';
    return 0;
}
```
<!-- endtab -->
<!-- tab Python -->

```py
input()
a = list(map(int, input().split()))
print(min(a), max(a))
```
<!-- endtab -->
<!-- tab Java -->
注：在算法竞赛中Java通常要使用优化的读入方法，但为了简单起见，这里直接用Scanner类读入。

```java
import java.util.Scanner;
public class Main {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        int n = sc.nextInt(), min = 100001, max = 0;
        for (int i = 0; i < n; i++) {
            int x = sc.nextInt();
            if (x > max) max = x;
            if (x < min) min = x;
        }
        System.out.printf("%d %d\n", min, max);
        sc.close();
    }
}
```
<!-- endtab -->
{% endtabs %}

从比赛中提交的代码来看，没有通过此题大多数情况是由于**没有初始化变量**。


## [学园都市的神秘代码](https://scut.online/p/675)
题意：简化下面的代码。

```cpp 点击展开代码 >folded
#include <iostream>
using namespace std;
const int mod = 998244353;
int main() {
    int ans = 0;
    long long n;
    cin >> n;
    for (long long i = 1; i <= n; ++i) {
        for (long long j = 1; j <= n / i; ++j) {
            for (long long k = 1; k <= j; ++k) {
                long long temp = k;
                while (!(j % temp == 0 && k % temp == 0)) {
                    temp--;
                }
                if (temp == 1) {
                    ans = (ans + 1) % mod;
                }
            }
        }
    }
    cout << ans << endl;
}
```

做出这题不难，只需要把题目中的代码输入到电脑中，运行**找找规律**就能发现答案是n×(n+1)/2。

小技巧：在模意义下除2不用求逆元，可以转换成$(mod+1)/2$。

常见出错原因：`n%mod*(n+1)%mod`导致溢出！

{% tabs p675 %}
<!-- tab C++ -->

```cpp
#include <bits/stdc++.h>
using namespace std;
using ll=long long;
const ll mod = 998244353;
int main() {
    int T;
    cin >> T;
    while (T--) {
        ll n;
        cin >> n;
        n %= mod;
        cout << n*(n+1)%mod*((mod+1)/2)%mod << endl;
    }
    return 0;
}
```
<!-- endtab -->
<!-- tab Python -->

```py
for _ in range(int(input())):
    n = int(input())
    print(n*(n+1)//2%998244353)
```
<!-- endtab -->
{% endtabs %}

## [一个小游戏](https://scut.online/p/661)

题意：一个n×m的01矩阵，每次可以选择以（1，1）为左上角，（x，y）为右下角的矩形翻转，（x，y）必须为1。L先手M后手，无法翻转的人输。

可以发现，每一步无论如何选择，（1，1）是必定要被翻转的。最终局面全0为必败态，反推可知（1，1）处为1是必胜态，为0是必败态。

{% tabs p661 %}
<!-- tab C++ -->
```cpp
#include <bits/stdc++.h>
using namespace std;
int main() {
    int x;
    cin >> x >> x >> x;
    cout << (x ? "Laurent" : "Makoto") << endl;
}
```
<!-- endtab -->
<!-- tab Python -->
```py
input()
print('Laurent' if input().split(' ')[0] == '1' else 'Makoto')
```
<!-- endtab -->
{% endtabs %}

## [垃圾邮件](https://scut.online/p/658)

题意：给一个长度为n的字符串s，有q次询问。每次临时将下标p的字符改为c，比较区间$[L_1, R_1]$和$[L_2, R_2]$的子串是否相同。

{% msg warning %}
此题数据较弱，导致有一些假做法。
{% endmsg %}

{% tabs p658 %}
<!-- tab C++（字符串哈希） -->
期望的做法是使用[字符串哈希](https://oi-wiki.org/string/hash/)，每次询问可以做到`O(1)`。

$hs[n]$表示字符串$s[1..n]$的哈希值，$bn[n]$表示$base^n$，均采用`unsigned long long`自然溢出。

子串$s[l..r]$的哈希值为$hs[r]-hs[l-1]\times bn[r-l+1]$。

```cpp
#include<bits/stdc++.h>
using namespace std;
const int base = 131;
typedef unsigned long long ull;
ull tr(char c) { return c-'a'+1; }
int main(){
    ios::sync_with_stdio(false), cin.tie(0);
    int n, q;
    cin >> n >> q;
    string s;
    cin >> s;
    vector<ull> hs(n+1), bn(n+1);
    bn[0] = 1;
    for (int i = 1; i <= n; i++) {
        bn[i] = bn[i-1]*base;
        hs[i] = hs[i-1]*base+tr(s[i-1]);
    }
    while (q--) {
        int l1, r1, l2, r2, p;
        char c;
        cin >> l1 >> r1 >> l2 >> r2 >> p >> c;
        ull h1 = hs[r1]-hs[l1-1]*bn[r1-l1+1];
        ull h2 = hs[r2]-hs[l2-1]*bn[r2-l2+1];
        if (l1 <= p && p <= r1) h1 += (tr(c)-tr(s[p-1])) * bn[r1-p];
        if (l2 <= p && p <= r2) h2 += (tr(c)-tr(s[p-1])) * bn[r2-p];
        cout << (h1 == h2 ? "yes" : "no") << '\n';
    }
    return 0;
}
```
<!-- endtab -->
<!-- tab C++（string） -->

```cpp
#include<bits/stdc++.h>
using namespace std;
int main(){
    ios::sync_with_stdio(false), cin.tie(0);
    int n, q;
    cin >> n >> q;
    string s;
    cin >> s;
    while (q--) {
        int l1, r1, l2, r2, p;
        char c;
        cin >> l1 >> r1 >> l2 >> r2 >> p >> c;
        swap(c, s[p-1]);
        if (s.substr(l1-1, r1-l1+1)==s.substr(l2-1, r2-l2+1)) 
            cout << "yes\n";
        else cout << "no\n";
        swap(c, s[p-1]);
    }
    return 0;
}
```
<!-- endtab -->
{% endtabs %}

## [超人高中生的基建计划](https://scut.online/p/664)

题意：n个点，i、j边权为i|j(i位或j），求最小生成树。
分析：显然边权不可能小于本身点的编号。对于奇数点，与1连最划算，边权为本身。剩下的点中除了$(10000)_2$这种的，都可以做到边权为本身。用log2统计这类点的数量。

答案就是$\frac{n\times (n+1)}{2}-1+\lfloor\log_2n\rfloor$

{% tabs p664 %}
<!-- tab C++ -->

```cpp
#include<bits/stdc++.h>
using namespace std;
int main(){
    long long n;
	cin >> n;
    cout << n*(n+1)/2-1+(int)log2(n) << '\n';
    return 0;
}
```
<!-- endtab -->
<!-- tab Python -->

```py
from math import log2
n = int(input())
print(n*(n+1)//2-1+int(log2(n)))
```
<!-- endtab -->
{% endtabs %}


## [灰之魔女的旅行](https://scut.online/p/665)

题意：一颗以1为根的树，每次可以选则一个未访问的点，访问该点及其子节点。先选择1节点，问最多能选择几次。

{% msg warning %}
本题输入数据量较大，可以使用`ios::sync_with_stdio(false), cin.tie(0);`关闭流同步以加速cin输入。或者使用scanf/getchar/fread等方法。
{% endmsg %}

{% tabs p665 %}
<!-- tab 树形DP -->

树形DP入门题，类似题目：[P1352 没有上司的舞会](https://www.luogu.com.cn/problem/P1352)。

设$f[i][0]$表示不选择节点$i$的最大答案，$f[i][1]$表示选择节点$i$的最大答案。
转移：$f[i][0]=\sum max(f[i.son][1], f[i.son][0])$
$f[i][1]=\sum f[i.son][0]$

初始值$f[i][1]=1$，答案为$f[1][1]$。


```cpp
#include <bits/stdc++.h>
using namespace std;
int main() {
    ios::sync_with_stdio(false), cin.tie(0);
    int n; cin >> n;
    vector<vector<int>> G(n+1);
    for (int i = 1; i < n; i++) {
        int u, v; cin >> u >> v;
        G[u].push_back(v), G[v].push_back(u);
    }
    vector<array<int, 2>> f(n+1);
    function<void(int, int)> dfs=[&](int u, int p) {
        f[u][1] = 1;
        for (auto v : G[u]) {
            if (v == p) continue;
            dfs(v, u);
            f[u][0] += max(f[v][0], f[v][1]);
            f[u][1] += f[v][0];
        }
    };
    dfs(1, 0);
    cout << f[1][1] << '\n';
    return 0;
}
```
<!-- endtab -->
<!-- tab 自底向上贪心 -->

由于输入保证为一棵树，即有子节点数目大于等于父节点数目，因此从叶子节点往上贪心选择可以保证最优。

```cpp
#include<bits/stdc++.h>
using namespace std;
int main(){
    ios::sync_with_stdio(false), cin.tie(0);
    int n;
    cin >> n;
    vector<vector<int>> G(n+1);
    for (int i = 1; i < n; i++) {
        int u, v;
        cin >> u >> v;
        G[u].push_back(v);
        G[v].push_back(u);
    }
    vector<bool> vis(n+1);
    int ans = 1;
    vis[1] = true;
    for (auto v : G[1]) vis[v] = true;
    function<void(int, int)> dfs=[&](int u, int fa) {
        for (auto v : G[u]) if (v != fa) dfs(v, u);
        if (!vis[u]) vis[u] = vis[fa] = true, ans++;
    };
    dfs(1, 0);
    cout << ans << '\n';
    return 0;
}
```
<!-- endtab -->
{% endtabs %}

## [merge stone](https://scut.online/p/674)

分析：【多叉哈夫曼树】【贪心】

参考代码：

```cpp
#include <cstdio>
#include <cstring>
#include <iostream>
#include <algorithm>
#include <queue>
#include <ext/pb_ds/priority_queue.hpp> //pb_ds库
#define LL long long 
using namespace std;
struct node{
    LL w,h;
    node(LL W, LL H){
        w=W,h=H;
    }
};
bool operator<(node a, node b){
    if(a.w!=b.w) return a.w>b.w; 
    return a.h>b.h;  //如果长度相等，高度小的优先
} //构造小根堆的操作。
__gnu_pbds::priority_queue <node, std::less<node>, __gnu_pbds::pairing_heap_tag> q; //优先队列
int n,k,cnt;
LL temp,maxh,ans;

int main()
{
    scanf("%d",&n);
    k = 3;
    for(int i=1; i<=n; i++){
        scanf("%lld",&temp);
        q.push(node(temp,1));
    }
    if((n-1)%(k-1) != 0) cnt=k-1-(n-1)%(k-1);  //判断是否要补空节点
    for (int i=1; i<=cnt; i++)
        q.push(node(0,1)); //补空节点
    cnt+=n;     //cnt为根节点个数(最初每个根节点都为其本身）
    while(cnt>1){
        temp=maxh=0;
        for(int i=1; i<=k; i++){
            temp+=q.top().w;
            maxh=max(maxh,q.top().h);
            q.pop();
        }
        ans+=temp; //维护带权路径长度之和
        q.push(node(temp, maxh+1)); //合并，高度为最高子树高度+1
        cnt-=k-1; //减少根节点
    }
    printf("%lld\n\n",ans);
    return 0;
```

## [心与心之间的距离，永不点亮的音乐会](https://scut.online/p/669)


{% msg warning %}
此题因为数据原因未加入正赛，放在了热身赛。
{% endmsg %}

题意：问两颗二叉搜索树上是否存在两个和为x的值。（分别位于两棵树上）

分析：【双指针】期望复杂度O(n)，然而用set就可以过。。

参考代码：
```cpp
#pragma comment(linker, “/STACK:1024000000,1024000000”)

#include <bits/stdc++.h>
using namespace std;
#define ll long long
#define S 100000
char bf[S], *p1 = bf, *p2 = bf;
#define nc() (p1==p2&&(p2=(p1=bf)+fread(bf,1,S,stdin),p2==p1)?-1:*p1++)

inline ll read() {
    ll x = 0, f = 1;
    char ch = nc();
    for (; (ch < '0' || ch > '9') && (ch != '-'); ch = nc());
    if (ch == '-')ch = nc(), f = -1;
    for (; ch <= '9' && ch >= '0'; x = x * 10 + ch - 48, ch = nc());
    return f * x;
}

#define N 500050
int t[N][2], n, m, tot;
ll val[N], a[N], b[N];

void dfs(int x) {
    if (t[x][0] > 0)dfs(t[x][0]);
    a[++tot] = x;
    if (t[x][1] > 0)dfs(t[x][1]);
}

void dfs2(int x) {
    if (t[x][0] > 0)dfs2(t[x][0]);
    b[++tot] = x;
    if (t[x][1] > 0)dfs2(t[x][1]);
}

inline void solve() {
    int flag = 0;
    n = read(), m = read();
    ll x = read();
    for (int i = 1; i <= n; ++i) {
        val[i] = read();
        t[i][0] = read(), t[i][1] = read();
    }
    tot = 0, dfs(1);
    for (int i = 1; i <= n; ++i)a[i] = val[a[i]];
    for (int i = 1; i <= m; ++i) {
        val[i] = read();
        t[i][0] = read(), t[i][1] = read();
    }
    tot = 0, dfs2(1);
    for (int i = 1; i <= m; ++i)b[i] = val[b[i]];
    int l = 1, r = m;
    while (l <= n && r >= 1) {
        while (a[l] + b[r] > x && r >= 1)--r;
        if (a[l] + b[r] == x && r >= 1) {
            flag = 1;
            break;
        }
        ++l;
    }
    puts(flag ? "yes" : "no");
}

int main() {
    // freopen("./data/10.in", "r", stdin);
    int T = 1;
    scanf("%d", &T);
    while (T--)solve();
}
```



## [convolution](https://scut.online/p/668)
【Dirichlet卷积】

参考代码：

```cpp
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
const int M = 1e9 + 7;
ll qpow(ll x, ll y) {
    ll ans = 1;
    while (y) {
        if (y & 1) ans = ans * x % M;
        x = x * x % M;
        y >>= 1;
    }
    return ans;
}
int main() {
    ios::sync_with_stdio(false), cin.tie(0);
    // freopen("data.in", "r", stdin);
    // freopen("data.out", "w", stdout);
    int t;
    cin >> t;
    while (t--) {
        int m;
        cin >> m;
        ll ans = 1;
        for (int i = 1; i <= m; i++) {
            ll p, q;
            cin >> p >> q;
            ans = ans * ((p + p * q % M - q + M) % M * qpow(p, q - 1) % M) % M;
        }
        cout << ans << '\n';
    }
    return 0;
}
```


## [小斯巴达们的历练](https://scut.online/p/679)
分析：
【最短路】【Dijkstra算法】

参考代码

```cpp
#include<cstdio>
#include<cstdlib>
#include<cstring>
#include<cmath>
#include<queue>
#include<iostream>
#include<algorithm>
using namespace std;
typedef long long LL;
#define maxn 201000 

struct qnode
{
	int x;LL c;
	qnode(int _x=0,int _c=0):x(_x),c(_c){}
	bool operator < (const qnode &y) const {
		return c>y.c;
	}
};
priority_queue<qnode> q;
struct node
{
	int y,nt;LL c;
}a[maxn*2];int len,first[maxn];
void ins(int x,int y,LL c)
{
	len++;a[len].y=y;a[len].c=c;
	a[len].nt=first[x];first[x]=len;
}
LL d[4][maxn];int c[maxn];bool vis[maxn];
LL mymin(LL x,LL y){return (x<y)?x:y;}
void dfs(int S,int tt)
{
	memset(d[tt],-1,sizeof(d[tt]));
	memset(vis,false,sizeof(vis)); 
	d[tt][S]=0;
	while (!q.empty()) q.pop();
	q.push(qnode(S,0));
	vis[S]=true;
	while (!q.empty())
	{
		qnode now = q.top();
		q.pop();
		int x=now.x;
		for (int k=first[x];k!=-1;k=a[k].nt)
		{
			int y=a[k].y;
			if (d[tt][y]==-1 || d[tt][y]>=d[tt][x]+a[k].c)
			{
				d[tt][y]=d[tt][x]+a[k].c;
				if (tt>0 && c[y]<3) continue;
				if (!vis[y])
				{
					q.push(qnode(y,d[tt][y]));
					vis[y]=true;
				}
			}
		}
		vis[x]=false;
	}
}
queue<int> qans;
int main()
{
	//freopen("01.in","r",stdin);
	//freopen("01.out","w",stdout);
	int n,m,i,t,x,y,s[4];LL ans,z;
	scanf("%d%d",&n,&m);
	t=-1;
	for (i=1;i<=n;i++) 
	{
		scanf("%d",&c[i]);
		if (c[i]==1) t=i;
	}
	len=0;memset(first,-1,sizeof(first));
	for (i=1;i<=m;i++)
	{
		scanf("%d%d%lld",&x,&y,&z);
		ins(x,y,z);ins(y,x,z);
	}
	for (i=1;i<=3;i++) scanf("%d",&s[i]);
	for (i=1;i<=3;i++) 
	{
		dfs(s[i],i);
		/*for (int j=1;j<=n;j++)
		 printf("%d ",d[i][j]);
		printf("\n");*/ 
	}
	dfs(t,0);
	/*for (int j=1;j<=n;j++)
	  printf("%d ",d[0][j]);
	printf("\n");*/
	ans=-1;
	while (!qans.empty()) qans.pop();
	for (i=1;i<=n;i++)
	{
		if (d[0][i]==-1 || d[1][i]==-1 || d[2][i]==-1 || d[3][i]==-1) continue; 
		if (ans==-1 || ans>d[0][i]+d[1][i]+d[2][i]+d[3][i]) 
		{
			ans=d[0][i]+d[1][i]+d[2][i]+d[3][i];
			while (!qans.empty()) qans.pop();
			qans.push(i);
		}
		else if (ans==d[0][i]+d[1][i]+d[2][i]+d[3][i]) qans.push(i);
		//printf("%d %d\n",i,ans); 
	}
	printf("%lld\n",ans);
	//while (!qans.empty()) {printf("%d ",qans.front());qans.pop();}
    return 0;
}
/*
5 6
3 3 3 3 1
1 4 2
1 5 1
2 4 4
2 5 3
3 4 1
4 5 2
1 2 3

5 5
3 3 3 2 1
1 5 1
2 4 4
2 5 3
3 4 1
4 5 2
1 2 3

6 11
3 1 3 2 3 3
1 2 3
1 3 1
1 4 2
1 5 4
2 3 4
2 4 1
2 6 4
3 4 2
4 5 3
4 6 3
5 6 1
3 5 6
*/ 
```



## [序列构造](https://scut.online/p/670)

题意：构造一个长度为 n 的整数序列，使得该序列的和与积相等，且恰好等于 n。

### 构造法

参考证明：[每日一题【1050】和积相等](http://lanqi.org/everyday/25568/)

**情形一**　 n=4k+1 时，可以分成 2k 个 −1，2k 个 1，1 个 4k+1．  
**情形二**　n=8k 时，可以分成 2k 个 −1，6k−2 个 1，1 个 2，1 个 4k．  
**情形三**　 n=8k+12 时，可以分成 2k+1 个 −1，6k+9 个 1，1 个 −2，1 个 4k+6．  
**情形四**　 n=4 时，|ai|∈{1,2,4}，其中 i=1,2,3,4，且只有可能取 1 个 4 或 2 个 2，容易验证都不可行．  
**情形五**　n=4k+2 时，由于a1⋅a2⋯an≡2(mod4),于是 a1,a2,⋯,an 为 1 个偶数和 4k+1 个奇数，它们的和为奇数，矛盾．  
**情形六**　 n=4k+3 时，a1,a2,⋯,an 为 4k+3 个奇数，设其中模 4 余 1 的有 m 个，模 4 余 3 的有 4k+3−m 个，因此a1+a2+⋯+an≡m−(4k+3−m)≡2m+1≡3(mod4),于是 m 为奇数，进而a1⋅a2⋯an≡$1^m⋅(−1)^{4k+3−m}$≡1(mod4),矛盾．  
综上所述，所有具有性质 Q 的正整数构成的集合为{x∣x≡0,1,4,5(mod8),x≠4,x∈N∗}.

{% tabs p670构造 %}
<!-- tab C++ -->

```cpp
#include <bits/stdc++.h>
using namespace std;
void print(int t, int x) {
    while (t--) cout << x << ' ';
}
int main() {
    int n;
    cin >> n;
    int nm = n % 8;
    if (n == 4||nm==2||nm==3||nm==6||nm==7) {
        cout << "no\n";
        return 0;
    }
    cout << "yes\n";
    if (n % 4 == 1) {
        int k = n / 4;
        print(2*k, -1);
        print(2*k, 1);
        print(1, 4*k+1);
    } else if (n % 8 == 0) {
        int k = n / 8;
        print(2*k, -1);
        print(6*k-2, 1);
        print(1, 2);
        print(1, 4*k);
    } else if (n % 8 == 4) {
        int k = (n-12)/8;
        print(2*k+1, -1);
        print(6*k+9, 1);
        print(1, -2);
        print(1, 4*k+6);
    }
    cout << '\n';
    return 0;
}
```
<!-- endtab -->
<!-- tab Python -->

```py
n = int(input())
if n == 4 or n%8 in [2,3,6,7]:
    print('no')
else:
    print('yes')
    if n % 4 == 1:
        k = n // 4
        print('1 -1 '*(2*k), end='')
        print(4*k+1)
    elif n % 8 == 0:
        k = n // 8
        print('1 '*(6*k-2), end='')
        print('-1 '*(2*k), end='')
        print(2, 4*k)
    elif n % 8 == 4:
        k = (n-12)//8
        print('1 '*(6*k+9), end='')
        print('-1 '*(2*k+1), end='')
        print(-2, 4*k+6)
```

<!-- endtab -->
{% endtabs %}

### 搜索法
由$\lfloor\log_210^5\rfloor=16$可知，序列中最多有16个绝对值大于1的数，剩下的数用1或-1填充。

于是可以考虑搜索，`dfs(num, target)`尝试用target个大于1的数构造一个乘积为num的序列，满足乘积后`check`检查是否能通过填充1或-1来满足和也为n。

```cpp
#include <bits/stdc++.h>
using namespace std;
const int maxn = 1e5 + 5;
int n, cnt, yinshu[maxn];
bool found;
void check() {
    int sum = 0;
    for (int i = 1; i <= cnt; i++) sum += yinshu[i];
    int need = n - sum;
    int remain = n - cnt;
    if (remain >= need && ((remain - need) % 4) == 0) {
        found = 1;
        cout << "yes\n";
        int xx = (remain - need) / 4;
        for (int i = 1; i <= xx; i++) cout << "-1 -1 1 1 ";
        for (int i = 1; i <= need; i++) cout << "1 ";
        for (int i = 1; i < cnt; i++) cout << yinshu[i] << ' ';
        cout << yinshu[cnt] << '\n';
    }
}
void dfs(int num, int target) {
    if (found) return;
    if (cnt == target - 1) {
        yinshu[++cnt] = num;
        check();
        if (!found) cnt--;
        return;
    }
    for (int x = 2; x * x <= num; x++) {
        if (num % x) continue;
        int y = num / x;
        yinshu[++cnt] = x;
        dfs(y, target);
        if (found) return; else cnt--;
    }
}
int main() {
    cin >> n;
    found = 0;
    for (int i = 1; i <= 16; i++) {
        cnt = 0;
        dfs(n, i);
        if (found) break;
    }
    if (found == 0) cout << "no\n";
}
```


## 爆炸就是艺术
### [爆炸就是艺术2](https://scut.online/p/682)
题意：给n个TNT的坐标，一个TNT可以引爆上下左右**以及本身位置**的TNT。点燃代价为到原点距离，求最小代价。

判断点是否存在可以用map/set/hash，找连通块可以用bfs/dfs/dsu，做法非常多。注意实现细节，**有重复的点**。

{% tabs p682 %}
<!-- tab map+dsu -->

```cpp
#include <bits/stdc++.h>
using namespace std;
void solve() {
    int n; cin >> n;
    vector<pair<int, int>> p(n);
    map<pair<int, int>, int> st;
    vector<long long> w(n);
    for (int i = 0; i < n; i++) {
        cin >> p[i].first >> p[i].second;
        st[p[i]] = i;
        w[i] = 1ll*p[i].first*p[i].first+1ll*p[i].second*p[i].second;
    }
    vector<int> f(n);
    for (int i = 0; i < n; i++) f[i] = i;
    function<int(int)> Find = [&](int x) { return x == f[x] ? x : f[x] = Find(f[x]); };
    for (int i = 0; i < n; i++) {
        int dx[]{0, -1, 1, 0, 0}, dy[]{0, 0, 0, -1, 1};
        for (int j = 0; j < 5; j++) {
            auto pos = st.find({p[i].first+dx[j], p[i].second+dy[j]});
            if (pos == st.end()) continue;
            int a = Find(pos->second), b = Find(i);
            if (a != b) {
                f[a] = b;
                w[b] = min(w[b], w[a]);
            }
        }
    }
    long long ans = 0;
    for (int i = 0; i < n; i++)
        if (i == Find(i)) ans += (long long)sqrt(w[i]);
    cout << ans << endl;
}
int main() {
    int T; cin >> T;
    while (T--) solve();
    return 0;
}
```
<!-- endtab -->
<!-- tab 数据生成器 -->

```cpp
#include "testlib.h"
#include <iostream>
using namespace std;
int testIndex = 0;
void nextTest() {
    testIndex++;
    freopen(("data/"+to_string(testIndex)+".in").c_str(), "w", stdout);
}
void randTest(int mint=1, int maxt=10, 
              int minn=1, int maxn=1e5, 
              int minx=-1e9, int maxx=1e9, 
              int miny=-1e9, int maxy=1e9) {
    nextTest();
    int T = rnd.next(mint, maxt);
    cout << T << endl;
    while (T--) {
        int n = rnd.next(minn, maxn);
        cout << n << endl;
        while (n--) {
            cout << rnd.next(minx, maxx) << ' ' << rnd.next(miny, maxy) << endl;
        }
    }
}
int main(int argc, char* argv[]) {
    registerGen(argc, argv, 1);
    nextTest();
    cout << 1 << endl;
    cout << 3 << endl;
    cout << 0 << ' ' << 0 << endl;
    cout << 1 << ' ' << 0 << endl;
    cout << 0 << ' ' << 8 << endl;
    nextTest();
    cout << 2 << endl;
    cout << 1 << endl;
    cout << 0 << ' ' << 0 <<  endl;
    cout << 1 << endl;
    cout << (int)1e9 << ' ' << (int)1e9 <<  endl;
    randTest(1, 1, 1, 1e2, 1, 1e4, 1, 1e4);
    randTest(1, 10, 1, 1e4, -1e5, 1e5, -1e5, 1e5);
    randTest(10, 10, 1, 1e4, -1e5, 1e5, -1e5, 1e5);
    randTest();
    randTest();
    randTest(10, 10, 1e5, 1e5, -10, 10, -1e9, 1e9);
    randTest(10, 10, 1e5, 1e5, -1e9, 1e9, -10, 10);
    randTest(10, 10, 1e5, 1e5, -10, 10, -10, 10);
    randTest(10, 10, 1e5, 1e5, 1e9, 1e9, -1e9, -1e9);
    return 0;
}
```
<!-- endtab -->
{% endtabs %}

### [爆炸就是艺术](https://scut.online/p/671)

{% msg warning %}
此题未加入比赛，[爆炸就是艺术2](https://scut.online/p/682)为本题的简化版本。
{% endmsg %}


题意：给出平面上n个点的坐标，点i爆炸的代价为i到原点的距离（向下取整），点i爆炸会同时让与点i距离小于等于7的点爆炸。问爆炸所有点的最小代价。
分析：类似平面最近点对的方法建图。然后用并查集缩点。

对每个点扫描半径7以内的其他点的做法容易被卡。（虽然已经开到3s了）
建图具体做法参考：[平面最近点对——非分治算法](https://oi-wiki.org/geometry/nearest-points/#_6)

{% tabs p671 %}
<!-- tab 最近点对建图+dsu -->

```cpp
#include <bits/stdc++.h>
using namespace std;

const int K = 7;
typedef long long ll;
struct Point {
    int x, y, i;
    Point(int x = 0, int y = 0, int i = 0) : x(x), y(y), i(i) {}
    bool operator<(const Point &o) const {
        return x < o.x || (x == o.x && y < o.y);
    }
    bool operator==(const Point &o) const {
        return x == o.x && y == o.y && i == o.i;
    }
};
struct cmpy {
    bool operator()(const Point &a, const Point &b) const { return a.y < b.y; }
};
ll dis2(const Point &a, const Point &b) {
    return (ll)(a.x - b.x) * (a.x - b.x) + (ll)(a.y - b.y) * (a.y - b.y);
}
void solve() {
    int n;
    cin >> n;
    vector<Point> p(n);
    for (int i = 0; i < n; i++) {
        cin >> p[i].x >> p[i].y;
    }
    sort(p.begin(), p.end());
    p.erase(unique(p.begin(), p.end()), p.end());
    n = (int)p.size();
    for (int i = 0; i < n; i++) p[i].i = i;
    multiset<Point, cmpy> s;
    vector<int> par(n);
    vector<ll> w(n);
    for (int i = 0; i < n; i++)
        w[i] = (ll)p[i].x * p[i].x + (ll)p[i].y * p[i].y;
    for (int i = 0; i < n; i++) par[i] = i;
    function<int(int)> Find = [&](int x) {
        return x == par[x] ? x : par[x] = Find(par[x]);
    };
    for (int i = 0, l = 0; i < n; i++) {
        while (l < i && p[i].x - p[l].x > K) s.erase(s.find(p[l++]));
        auto lowy = s.lower_bound(Point(p[i].x, p[i].y - K));
        for (auto it = lowy; it != s.end() && it->y - p[i].y <= K; it++) {
            if (dis2(*it, p[i]) <= K * K) {
                int u = p[i].i, v = it->i;
                int fu = Find(u), fv = Find(v);
                if (fu != fv) {
                    w[fv] = min(w[fv], w[fu]);
                    par[fu] = fv;
                }
            }
        }
        s.insert(p[i]);
    }
    ll ans = 0;
    for (int i = 0; i < n; i++)
        if (Find(i) == i) ans += (ll)sqrt(w[i]);
    cout << ans << '\n';
}

int main() {
    ios::sync_with_stdio(false), cin.tie(0);
    int T;
    cin >> T;
    while (T--) solve();
    return 0;
}
```
<!-- endtab -->
<!-- tab fread+hash+dsu -->

```cpp
#include<bits/stdc++.h>
using namespace std;
typedef long long ll;
typedef unsigned long long ull;
struct In {
    static const int N = 5e7;
    char s[N], *p;
    size_t c;
    In() { c = fread(s, 1, N, stdin), p = s; }
} in;
template <typename T>
In& operator>>(In &i, T &x) {
    char *&p = i.p;
    while (*p != '-' && (*p < '0' || *p > '9')) p++;
    if (p == i.s + i.c) return i;
    bool sgn = false;
    if (*p == '-') sgn = true, p++;
    for (x = 0; *p >= '0' && *p <= '9'; p++) x = x * 10 + *p - '0';
    if (sgn) x = -x;
    return i;
}
struct hash_tables {
    static const int sz = 1000037;
    int head[sz], nxt[sz], mark, tmmark[sz], key2[sz], len;
    ll key[sz];
    void clear() {
        len = 0;
        mark++;
    }
    void add(ll s, int v) {
        int val = s % sz;
        if (val < 0) val += sz;
        if (tmmark[val] != mark) tmmark[val] = mark, head[val] = -1;
        nxt[len] = head[val], head[val] = len, key[len] = s, key2[len] = v;
        len++;
    }
    int count(ll s) {
        int val = s % sz;
        if(val < 0) val += sz;
        if(tmmark[val] != mark) return 0;
        for(int j = head[val]; j != -1; j = nxt[j]) if(key[j] == s) return key2[j];
        return 0;
    }
}g;

const int maxn = 100005;
int par[maxn], x[maxn], y[maxn]; ll w[maxn];
int find(int x) {return x==par[x]?x:par[x]=find(par[x]);}
vector<pair<int,int>> v;
int main() {
    g.clear();
    for(int dx=-7;dx<=7;++dx) {
        for(int dy=0;dy<=7;++dy) {
            if(dx*dx+dy*dy<=49) v.push_back({dx,dy});
        }
    }
    int t;
    in >> t;
    while(t--) {
        int n;
        in >> n;
        for(int i=1;i<=n;++i) par[i]=i;
        for(int i=1;i<=n;++i) {
            in >> x[i] >> y[i];
            g.add(y[i]^((ll)x[i]<<32), i);
            w[i]=1ll*x[i]*x[i]+1ll*y[i]*y[i];
        }
        for(int i=1;i<=n;++i) {
            for(auto &p:v) {
                int xx=x[i]+p.first,yy=y[i]+p.second;
                ull id = yy^((ll)xx<<32);
                int tmp = 0;
                if((tmp=g.count(id))) {
                    int a=find(tmp),b=find(i);
                    if(a!=b) par[a]=b, w[b]=min(w[b],w[a]);
                }
            }
        }
        ll ans=0;
        for(int i=1;i<=n;++i) if(i==find(i)) ans+=(ll)sqrt(w[i]);
        cout << ans << '\n';
        g.clear();
    }
}
```
<!-- endtab -->
{% endtabs %}

## [tree](https://scut.online/p/672)

【树形dp】

参考代码

```cpp
#include<bits/stdc++.h>
using namespace std;
const int maxn = 1e4 + 1;
typedef long long ll;
int ca[maxn], dis[2][3][maxn], son[2][2][maxn]; 
// ca[i] = 0 代表i节点属于A类，ca[i] = 1代表i节点属于B类 
// dis[i][j][k] 
// i: 0, 1 分别代表A和B 
// j: 0, 1, 2 分别代表最远距离和次远距离，以及不经过以k为根节点的子树，但经过其父节点的最长路径
// k: 代表当前节点的编号
// dis[0][0][k] 代表节点k到其子树中最远的A类节点的距离
// dis[1][1][k] 代表节点k到其子树中次远的B类节点的距离
// son[1][0][k] 代表节点k到其子树中最远的B类节点的路上经过的儿子

struct node{
    int v, w;
};
vector<node> e[maxn];

void init(int n){
    for(int i = 0; i <= n; i++){
        ca[i] = 0;
        e[i].clear();
        dis[0][0][i] = dis[0][1][i] = dis[1][0][i] = dis[1][1][i] = 0;
        dis[0][2][i] = dis[1][2][i] = 0; 
        son[0][0][i] = son[0][1][i] = son[1][0][i] = son[1][1][i] = 0;
    }
}

void get_ca(int val, int num){ 
    int index; 
    for(int i = 1; i <= num; i++) {
        cin >> index;
        ca[index] = val;
    }
}

void add_edge(int u, int v, int w){
    e[u].push_back({v, w});
    e[v].push_back({u, w});
}

void dfs1(int u, int fa){
    for(auto i: e[u]){
        int v = i.v, w = i.w, tmp_son, d;
        if(v == fa) continue;
        dfs1(v, u);
        //A类 
        tmp_son = v;
        if(dis[0][0][v] == 0) 
            d = (ca[v] == 0) * w;
        else d = dis[0][0][v] + w;
        if(d > dis[0][0][u]){
            swap(d, dis[0][0][u]);
            swap(tmp_son, son[0][0][u]);
        }
        if(d > dis[0][1][u]){
            swap(d, dis[0][1][u]);
            swap(tmp_son, son[0][1][u]);
        }
        //B类
       tmp_son = v;
        if(dis[1][0][v] == 0) 
            d = (ca[v] == 1) * w;
        else d = dis[1][0][v] + w;
        if(d > dis[1][0][u]){
            swap(d, dis[1][0][u]);
            swap(tmp_son, son[1][0][u]);
        }
        if(d > dis[1][1][u]){
            swap(d, dis[1][1][u]);
            swap(tmp_son, son[1][1][u]);
        }

    }
}

void dfs2(int u, int fa){
    for(auto i: e[u]){
        int v = i.v, w = i.w;
        if(v == fa) continue;
        //A类
        if(v == son[0][0][u])
            dis[0][2][v] = max(dis[0][1][u], dis[0][2][u]) + w;
        else 
            dis[0][2][v] = max(dis[0][0][u], dis[0][2][u]) + w;
        //B类
         if(v == son[1][0][u])
            dis[1][2][v] = max(dis[1][1][u], dis[1][2][u]) + w;
        else 
            dis[1][2][v] = max(dis[1][0][u], dis[1][2][u]) + w;
        dfs2(v, u);
    }
}

inline bool read(int &num) {
    char in;bool IsN=false;
    in=getchar();
    if(in==EOF) return false;
    while(in!='-'&&(in<'0'||in>'9')) in=getchar();
    if(in=='-'){ IsN=true;num=0;}
    else num=in-'0';
    while(in=getchar(),in>='0'&&in<='9'){
        num*=10,num+=in-'0';
    }
    if(IsN) num=-num;
    return true;
}

int main(){
    // freopen("game.in", "r", stdin);
    // freopen("game.out", "w", stdout);
    int t;
    read(t);
    while (t--) {
        int n, m, u, v, w;
        read(n), read(m);
        init(n + m);
        get_ca(0, n);
        get_ca(1, m);
        for(int i = 1; i <= n + m - 1; i++){
            read(u), read(v), read(w);
            add_edge(u, v, w);
        }
        dfs1(1, 0);
        dfs2(1, 0);

        ll da = 0, db = 0;
        for(int i = 1; i <= n + m; i++){
            if(ca[i] == 0) 
                da += max(dis[0][0][i], dis[0][2][i]);
            else
                db += max(dis[1][0][i], dis[1][2][i]);
        }
        if(da > db) printf("A\n");
        else if (da == db) printf("T\n");
        else printf("B\n");
    }
}
```

```cpp
#include <bits/stdc++.h>

using namespace std;
#define N 20200
#define ll long long
#define SS 100000
char bf[SS], *p1 = bf, *p2 = bf;
#define nc() (p1==p2&&(p2=(p1=bf)+fread(bf,1,SS,stdin),p2==p1)?-1:*p1++)

inline int read() {
    int x = 0, f = 1;
    char ch = nc();
    for (; (ch < '0' || ch > '9') && (ch != '-'); ch = nc());
    if (ch == '-')ch = nc(), f = -1;
    for (; ch <= '9' && ch >= '0'; x = x * 10 + ch - 48, ch = nc());
    return f * x;
}

int n, m, cnt, last[N], sz[N], fa[N], son[N], top[N], dep[N], Dep, who, x[N], y[N], id[N], a1, b1, a2, b2;
struct edge {
    int to, next, w;
} e[N << 1];

inline void add(int u, int v, int w) {
    e[++cnt] = {v, last[u], w}, last[u] = cnt;
    e[++cnt] = {u, last[v], w}, last[v] = cnt;
}

void dfs(int x, int fa, int d, int f) {
    dep[x] = d;
    for (int i = last[x], y; i; i = e[i].next)
        if ((y = e[i].to) != fa)dfs(y, x, d + e[i].w, f);
    if (id[x] == f)if (Dep < dep[x])Dep = dep[x], who = x;
}

void dfs1(int x, int d) {
    sz[x] = 1, dep[x] = d;
    for (int i = last[x], y; i; i = e[i].next)
        if ((y = e[i].to) != fa[x]) {
            fa[y] = x, dfs1(y, d + e[i].w), sz[x] += sz[y];
            if (sz[son[x]] < sz[y])son[x] = y;
        }
}

void dfs2(int x, int d) {
    top[x] = d;
    if (son[x])dfs2(son[x], d);
    for (int i = last[x], y; i; i = e[i].next)if ((y = e[i].to) != fa[x] && y != son[x])dfs2(y, y);
}

inline int lca(int l, int r) {
    if (l == r)return l;
    for (; top[l] != top[r]; dep[top[l]] < dep[top[r]] ? r = fa[top[r]] : l = fa[top[l]]);
    return dep[l] < dep[r] ? l : r;
}

inline int dis(int l, int r) { return dep[l] + dep[r] - (dep[lca(l, r)] << 1); }

inline void solve() {
    n = read(), m = read();
    for (int i = 1; i <= n + m; ++i)last[i] = sz[i] = fa[i] = son[i] = top[i] = dep[i] = cnt = 0;
    for (int i = 1; i <= n; ++i)x[i] = read(), id[x[i]] = 1;
    for (int i = 1; i <= m; ++i)y[i] = read(), id[y[i]] = 2;
    for (int i = 1; i < n + m; ++i) {
        int u = read(), v = read(), w = read();
        add(u, v, w);
    }
    Dep = 0, dep[0] = 0, dfs(1, 0, 0, 1), b1 = who;
    Dep = 0, dep[0] = 0, dfs(b1, 0, 0, 1), a1 = who;
    Dep = 0, dep[0] = 0, dfs(1, 0, 0, 2), b2 = who;
    Dep = 0, dep[0] = 0, dfs(b2, 0, 0, 2), a2 = who;
    dep[0] = 0, dfs1(1, 0), dfs2(1, 1);
    ll da = 0, db = 0;
    for (int i = 1; i <= n; ++i) da += max(dis(a1, x[i]), dis(b1, x[i]));
    for (int i = 1; i <= m; ++i) db += max(dis(a2, y[i]), dis(b2, y[i]));
    if (da > db) puts("A");
    else if (da < db) puts("B");
    else puts("T");
}

int main() {
    int T = 1;
    scanf("%d", &T);
    while (T--) {
        solve();
    }
}
```
## [猪灵的胜利之舞](https://scut.online/p/673)
题意：$f[3]=4, f[4]=7, f[n]=f[n-1]+f[n-2]$，求$\frac{f[n]}{2^n}$

### 十进制快速幂
二进制快速幂与十进制快速幂的对比（伪代码）

{% tabs fpow-cmp %}
<!-- tab 十进制 -->
```cpp
ans = 1
while (y != 0) {
    ans = ans * pow(x, y%10);
    x = pow(x, 10);
    y = y / 10;
}
```
<!-- endtab -->
<!-- tab 二进制 -->
```cpp
ans = 1
while (y != 0) {
    ans = ans * pow(x, y%2);
    x = pow(x, 2);
    y = y / 2;
}
```
<!-- endtab -->
{% endtabs %}

十进制快速幂和二进制快速幂在原理上是一样的，不过二进制快速幂通常实现为位运算，而十进制则用字符串来处理。

### 循环节

> 斐波那契数列在1e9+7下循环节为2e9+16，这题答案就是斐波那契数列的第n项加上n-2项。

循环节可以通过如下程序暴力找出：

```cpp
#include <bits/stdc++.h>
using namespace std;
int main() {
    int a = 1, b = 1, m = 1e9 + 7;
    long long cnt = 0;
    while(true) {
        cnt++;
        a = (a + b) % m;
        swap(a, b);
        if (a == 1 && b== 1) break;
    }
    cout << cnt << '\n';
    cerr << "Time: " << (double)clock()/CLOCKS_PER_SEC << '\n';
}
```

运行结果：

```
2000000016
Time: 6.851
```

### 参考代码


{% tabs p673 %}
<!-- tab 十进制快速幂 -->

```cpp
#include <bits/stdc++.h>
using namespace std;
using ll = long long;
const int N = 1e6 + 5;
const ll M = 1e9 + 7;
using Mat = array<array<ll, 2>, 2>;
#define FOR(i, n) for(int i = 0; i < n; i++)
string n;
int len;
Mat operator*(const Mat &a, const Mat &b) {
    Mat c{};
    FOR(i, 2) FOR(j, 2) FOR(k, 2)
        c[i][j] = (c[i][j] + a[i][k] * b[k][j]) % M;
    return c;
}
Mat Pow(Mat x, int y) {
    Mat ans{{{1, 0}, {0, 1}}};
    FOR(t, y) ans = ans * x;
    return ans;
}
ll Pow(ll x, int y) {
    ll ans = 1;
    FOR(t, y) ans = ans * x % M;
    return ans;
}
ll Pow(ll x, ll y) {
    ll ans = 1;
    while (y) {
        if (y & 1) ans = ans * x % M;
        x = x * x % M;
        y >>= 1;
    }
    return ans;
}
Mat Pow(Mat x) {
    Mat ans{{{1, 0}, {0, 1}}};
    FOR(i, len) {
        ans = ans*Pow(x, n[i]-'0');
        x = Pow(x, 10);
    }
    return ans;
}
ll Pow(ll x) {
    ll ans = 1;
    FOR(i, len) {
        ans = ans*Pow(x, n[i]-'0')%M;
        x = Pow(x, 10);
    }
    return ans;
}
int main() {
    cin >> n;
    len = (int)n.length();
    reverse(n.begin(), n.end());
    Mat A = Pow(Mat{{{1, 1}, {1, 0}}});
    A = A * Mat{{{1, 0}, {2, 0}}};
    cout << A[1][0] * Pow(Pow(2ll), M-2) % M;
    return 0;
}
```
<!-- endtab -->
<!-- tab 循环节 -->

```cpp
#include<bits/stdc++.h>
typedef long long ll;
using namespace std;
const int mod = 1e9+7;
const int T = 2e9+16;
inline pair<ll,ll> read() {
    char c=getchar();ll x=0,y=0;
    while(c<'0'||c>'9') c=getchar();
    while(c>='0'&&c<='9')x=(x*10+c-'0')%T,y=(y*10+c-'0')%(mod-1),c=getchar();
    return {x,y};
}
ll fpow(ll a,ll b){ll r=1;for(a%=mod;b;b>>=1){if(b&1)r=r*a%mod;a=a*a%mod;}return r;}
ll fib(ll n) {
    function<ll(ll,ll,ll,ll,ll)> f = [&](ll a,ll b,ll p,ll q,ll n) -> ll {
        if(!n) return b;
        if(n&1) return f((b*q+a*q+a*p)%mod,(b*p+a*q)%mod,p,q,n-1);
        return f(a,b,(p*p+q*q)%mod,(q*q+2*q*p)%mod,n/2);
    };
    return f(1,0,0,1,n);
}
int main() {
    pair<ll,ll> p=read();
    ll n=p.first+T,n2=p.second;
    cout<<(fib(n+1)+fib(n-1))%mod*fpow((mod+1)/2,n2)%mod<<endl;
}
```
<!-- endtab -->
<!-- tab 数据生成器 -->

```cpp
#include "testlib.h"
#include <iostream>
using namespace std;
int testIndex = 0;
void nextTest() {
    testIndex++;
    freopen(("data/"+to_string(testIndex)+".in").c_str(), "w", stdout);
}
int main(int argc, char* argv[]) {
    registerGen(argc, argv, 1);
    for (int i = 3; i <= 5; i++) {
        nextTest();
        cout << i << endl;
    }
    nextTest();
    cout << rnd.next("[1-9][0-9]{1,10}") << endl;
    nextTest();
    cout << rnd.next("[1-9][0-9]{10,100}") << endl;
    nextTest();
    cout << rnd.next("[1-9][0-9]{100,1000}") << endl;
    nextTest();
    cout << rnd.next("[1-9][0-9]{1000,10000}") << endl;
    nextTest();
    cout << rnd.next("[1-9][0-9]{10000,100000}") << endl;
    nextTest();
    cout << rnd.next("[1-9][0-9]{100000,999999}") << endl;
    nextTest();
    cout << rnd.next("[1-9][0-9]{999999}") << endl;
    nextTest();
    cout << rnd.next("1[0]{1000000}") << endl;
    return 0;
}
```

<!-- endtab -->
{% endtabs %}

## [艾尔奇亚的国王](https://scut.online/p/666)
题意：A有 n 个筹码，可以从$[l_1, r_1]$区间随机选整数。
B有 m 个筹码，可以从$[l_2, r_2]$区间随机选整数。

数字大的人拿走数字小的人的筹码，（数字相同算平局），没有筹码的人算输。

求A、B赢的概率。

```cpp
#include "bits/stdc++.h"
using namespace std;
typedef long long ll;
const int maxn=1e6+7;
double p,k,q,sum;
double f[maxn];
int n,l1,r1,m,l2,r2;
void getf(int pos,double x)
{
    if(pos==n+m) f[pos]=1;
    else
    {
        double tx=p/(1-k-q*x);
        getf(pos+1,tx);
        f[pos]=tx*f[pos+1];
    }
}
int main()
{
    while(scanf("%d%d%d%d%d%d",&n,&l1,&r1,&m,&l2,&r2)==6)
    {
        p=k=q=sum=0;
        for(int i=l1;i<=r1;i++)
        {
            for(int j=l2;j<=r2;j++)
            {
                if(i>j) p+=1;
                else if(i==j) k+=1;
                else q+=1;
                sum+=1;
            }
        }
        p/=sum;k/=sum;q/=sum;
        getf(1,0);
        printf("%.3f %.3f\n",f[n],1.0-f[n]);
    }
}
```


## [umi炒饭](https://scut.online/p/667)

题意：给n个点的坐标以及权值，问一个长w宽h的矩形内的点权值之和的最大值。

前置知识：线段树、扫描线


```cpp
#include "bits/stdc++.h"
using namespace std;
typedef long long ll;
const int maxn=1e4+7;
struct Node
{
    int x,y,val;
    bool operator < (const Node &tmp) const
    {
        if(y==tmp.y) return x<tmp.x;
        else return y<tmp.y;
    }
};
Node node[maxn<<1];
int n,w,h,cnt;
int tx[maxn<<1],ty[maxn<<1],tree[maxn<<2],lazy[maxn<<2];
void build(int root,int l,int r)
{
    if(l>=r)
    {
        tree[root]=0;
        lazy[root]=0;
        return;
    }
    int mid=(l+r)>>1;
    int lrt=root<<1;
    int rrt=lrt+1;
    build(lrt,l,mid);
    build(rrt,mid+1,r);
    tree[root]=0;
    lazy[root]=0;
}
void pushdown(int root)
{
    if(lazy[root])
    {
        int lrt=root<<1;
        int rrt=lrt+1;
        lazy[lrt]+=lazy[root];
        lazy[rrt]+=lazy[root];
        tree[lrt]+=lazy[root];
        tree[rrt]+=lazy[root];
        lazy[root]=0;
    }
}
void add(int ql,int qr,int val,int root,int l,int r)
{
    if(l>=ql&&r<=qr)
    {
        tree[root]+=val;
        lazy[root]+=val;
        return;
    }
    pushdown(root);
    int mid=(l+r)>>1;
    int lrt=root<<1;
    int rrt=lrt+1;
    if(qr<=mid) add(ql,qr,val,lrt,l,mid);
    else if(ql>mid) add(ql,qr,val,rrt,mid+1,r);
    else add(ql,qr,val,lrt,l,mid),add(ql,qr,val,rrt,mid+1,r);
    tree[root]=max(tree[lrt],tree[rrt]);
}
int query(int ql,int qr,int root,int l,int r)
{
    if(l>=ql&&r<=qr)
    {
        return tree[root];
    }
    pushdown(root);
    int mid=(l+r)>>1;
    int lrt=root<<1;
    int rrt=lrt+1;
    if(qr<=mid) return query(ql,qr,lrt,l,mid);
    else if(ql>mid) return query(ql,qr,rrt,mid+1,r);
    else return query(ql,qr,lrt,l,mid)+query(ql,qr,rrt,mid+1,r);
}
int main()
{
    while(scanf("%d%d%d",&n,&w,&h)==3)
    {
        cnt=0;
        for(int i=1;i<=n;i++)
        {
            scanf("%d%d%d",&node[i].x,&node[i].y,&node[i].val);
            tx[++cnt]=node[i].x;
            ty[cnt]=node[i].y;

            node[i+n].x=node[i].x;
            node[i+n].y=node[i].y+h+1;
            node[i+n].val=-node[i].val;
            tx[++cnt]=node[i+n].x;
            ty[cnt]=node[i+n].y;
        }
        sort(node+1,node+1+n+n);
        sort(tx+1,tx+1+cnt);
        sort(ty+1,ty+1+cnt);
        int cx=unique(tx+1,tx+1+cnt)-tx-1;
        int cy=unique(ty+1,ty+1+cnt)-ty-1;
        build(1,1,cx);
        int ans=0;
        for(int i=1,j=1;i<=cy;i++)
        {
            while(j<=2*n&&node[j].y<=ty[i])
            {
                int ql=lower_bound(tx+1,tx+1+cx,node[j].x)-tx;
                int qr=upper_bound(tx+1,tx+1+cx,node[j].x+w)-tx-1;
                add(ql,qr,node[j].val,1,1,cx);
                j++;
            }
            ans=max(ans,tree[1]);
        }
        printf("%d\n",ans);
    }
}
```

```cpp
//#pragma GCC optimize(2)


#include<bits/stdc++.h>
using namespace std;
//#include<ext/pb_ds/assoc_container.hpp>
//#include<ext/pb_ds/tree_policy.hpp>
//#include<ext/pb_ds/hash_policy.hpp>
//#include<ext/pb_ds/trie_policy.hpp>
//#include<ext/pb_ds/priority_queue.hpp>
//#include<ext/rope>
//using namespace __gnu_cxx;
//using namespace __gnu_pbds;
//void err(istream_iterator<string> it){cerr<<endl;}
//template<typename T, typename... Args>void err(istream_iterator<string> it, T a, Args... args){cerr << *it << " = " << a << " , ";err(++it, args...);}
//#define error(args...) { string _s = #args; replace(_s.begin(), _s.end(), ',', ' '); stringstream _ss(_s); istream_iterator<string> _it(_ss); err(_it, args); }
#define mem(a,b) memset((a),b,sizeof((a)))
#define fpre(x) cout<<fixed<<setprecision(x)
#define clr(v) (v).clear()
#define pii pair<int,int>
#define pdd pair<double,double>
#define pli pair<ll,int>
#define pll pair<ll,ll>
#define mp make_pair
#define eb emplace_back
#define pb emplace_back
#define ll long long
#define ld long double
#define ull unsigned long long
#define uint unsigned int
#define ushort unsigned short
#define IOS ios::sync_with_stdio(0);cin.tie(0);cout.tie(0)
#define lowbit(i) (i&(-i))
#define lson (rt<<1)
#define rson lson|1
#define fi first
#define se second
const ld eps=1e-10;
const ld pi=acos(-1);
inline int dcmp(ld x)
{
    if(x<-eps)
        return -1;
    if(x>eps)
        return 1;
    return 0;
}
//-----------------------------------------------head

const int maxn=1e5+5;
int t[maxn<<2];
int lzy[maxn<<2];
void build(int l,int r,int rt)
{
    t[rt]=lzy[rt]=0;
    if(l==r)
        return;
    int m=(l+r)>>1;
    build(l,m,lson);
    build(m+1,r,rson);
}
void pd(int rt)
{
    if(lzy[rt])
    {
        int v=lzy[rt];
        t[lson]+=v;
        t[rson]+=v;
        lzy[lson]+=v;
        lzy[rson]+=v;
        lzy[rt]=0;
    }
}
void upd(int L,int R,int v,int l,int r,int rt)
{
    if(L<=l&&r<=R)
    {
        t[rt]+=v;
        lzy[rt]+=v;
        return;
    }
    pd(rt);
    int m=(l+r)>>1;
    if(m>=L)
        upd(L,R,v,l,m,lson);
    if(m<R)
        upd(L,R,v,m+1,r,rson);
    t[rt]=max(t[lson],t[rson]);
}
pair<pii,ll>a[maxn];
int main()
{
    int n,w,h;
    while(cin>>n>>w>>h)
    {
        assert(n>=0&&n<=10000);
        assert(w>=1&&w<=20000);
        assert(h>=1&&w<=20000);
        build(1,1e5,1);
        for(int i=0;i<n;++i)
        {
            cin>>a[i].fi.fi>>a[i].fi.se>>a[i].se;
            assert(a[i].fi.fi>=-1e4&&a[i].fi.fi<=1e4);
            assert(a[i].fi.se>=-1e4&&a[i].fi.se<=1e4);
            assert(a[i].se>=1&&a[i].se<=1e5);
            a[i].fi.fi+=3e4+1;
            a[i].fi.se+=3e4+1;
        }
        sort(a,a+n);
        int mx=0;
        for(int i=0,j=0;i<n;++i)
        {
            upd(a[i].fi.se-h,a[i].fi.se,a[i].se,1,1e5,1);
            while(a[i].fi.fi-w>a[j].fi.fi)
            {
                upd(a[j].fi.se-h,a[j].fi.se,-a[j].se,1,1e5,1);
                ++j;
            }
            mx=max(mx,t[1]);
        }
        cout<<mx<<endl;

    }
}
```


## [谜拟Q plus](https://scut.online/p/677)
分析：【贪心】【dp】【二分】

参考代码

```cpp
#include<bits/stdc++.h>
using namespace std;
const int N=1e3+10;
int n,Q,num1,num2;
long long dp[N],rem[N];
struct Tp1{
	long long a,b;
	bool operator <(const Tp1 &op)const{ return (a==op.a)?b>op.b:a<op.a; }
}tp1[N];
struct Tp2{
	long long a,b;
	bool operator <(const Tp2 &op)const{ return (b==op.b)?a<op.a:b>op.b; }
}tp2[N];
struct hh
{
	long long x,id,ans;
}q[N];
bool cmp(hh a,hh b)
{
	return a.x<b.x;
}
bool cmp2(hh a,hh b)
{
	return a.id<b.id;
}
int find(long long x)
{
	int l,r,mid,ret=0;
	l=1;r=num2;
	while(l<=r)
	{
		mid=l+r>>1;
		if(dp[mid]<=x) ret=max(ret,mid),l=mid+1;
		else r=mid-1;
	}

	return ret;
}
int main(){
	scanf("%d",&n);
	for(int i=1;i<=n;i++){
		int x,y; scanf("%d%d",&x,&y);
		if(x<=y) tp1[++num1]=(Tp1){x,y};
		else tp2[++num2]=(Tp2){x,y}; 
	}
	sort(tp1+1,tp1+num1+1); sort(tp2+1,tp2+num2+1);
	
	memset(dp,127/3,sizeof(dp)); dp[0]=0;
	for(int i=num2;i>=1;i--){
		for(int j=num2-i+1;j>=1;j--){
			if(dp[j-1]>tp2[i].b) dp[j]=min(dp[j],dp[j-1]-tp2[i].b+tp2[i].a);
			else dp[j]=min(dp[j],1+tp2[i].a);
		}
	}
	scanf("%d",&Q);
	for(int i=1;i<=Q;i++)
		cin>>q[i].x,q[i].id=i,q[i].ans=-1;
	sort(q+1,q+1+Q,cmp);
	int pos,x,tans;
	pos=x=tans=0;
	pos=1;
	for(int i=1;i<=num1;i++)
		rem[i]=rem[i-1]-tp1[i].a+tp1[i].b;
	for(int i=1;i<=num1;i++)
	{
		//cout<<"___"<<q[pos].x<<"  "<<tp1[i].a-x<<endl;
		while(pos<=Q&&q[pos].x<=tp1[i].a-x) q[pos++].ans=tans;
		x+=-tp1[i].a+tp1[i].b;tans++;
		//cout<<i<<"*"<<x<<" "<<pos<<"    "<<tp1[i].a<<" "<<tp1[i].b<<endl;
		if(pos>Q) break;
	}
	for(int i=1;i<=Q;i++)
		if(q[i].ans==-1) q[i].ans=num1;
	sort(q+1,q+1+Q,cmp2);
	/*for(int i=1;i<=Q;i++)
		cout<<q[i].ans<<" ";cout<<endl; 
	for(int i=1;i<=num2;i++)
		cout<<dp[i]<<" ";cout<<endl; */
	for(int i=1;i<=Q;i++){
		printf("%lld\n",q[i].ans+find(rem[q[i].ans]+q[i].x));
	}
}
```


## [世界棋盘](https://scut.online/p/676)
分析【min25筛法】【杜教筛】

参考代码：

```cpp
#include<cstdio>
#include<cmath>
using namespace std;
typedef long long ll;
const int maxn = 1e5 + 7;
const int mod = 998244353;
ll n, k, sqr, nsqr;
int p[maxn], cntp;
ll prep[maxn];
bool np[maxn];
void sieve(int x){
    for(int i = 2; i <= x; ++i){
        if(!np[i]){
            p[++cntp] = i;
            prep[cntp] = prep[cntp - 1] + i;
        }
        for(int j = 1; j <= cntp && i * p[j] <= x; ++j){
            np[i * p[j]] = true;
            if(i % p[j] == 0){
                break;
            }
        }
    }
}
ll w[maxn], cntw, h[maxn], s[maxn];
int ID(ll x){
    return x <= nsqr ? cntw - x + 1 : n / x;
}
void calw(){
    cntw = sqr + nsqr - 1;
    for(int i = 1; i <= sqr; ++i){
        w[i] = n / i;
    }
    for(int i = sqr + 1; i <= cntw; ++i){
        w[i] = w[i - 1] - 1;
    }
    for(int i = 1; i <= cntw; ++i){
        h[i] = w[i] * (w[i] + 1) / 2 - 1;
    }
    for(int i = 1; i <= cntp; ++i){
        for(int j = 1; j <= cntw && 1ll * p[i] * p[i] <= w[j]; ++j){
            h[j] -= p[i] * (h[ID(w[j] / p[i])] - prep[i - 1]);
        }
    }
    for (int i = cntp; i >= 1; --i) {
        for (int j = 1; j <= cntw && 1ll * p[i] * p[i] <= w[j]; ++j) {
            for (ll k = p[i]; 1ll * k * p[i] <= w[j]; k *= p[i]) {
                s[j] += s[ID(w[j] / k)] + h[ID(w[j] / k)] - prep[i - 1];
            }
        }
    }
}
ll S(ll x, ll k){
    if(x <= 2 || p[k] > x){
        return 0;
    }
    ll ans = 0;
    for(int i = k; i <= cntp && 1ll * p[i] * p[i] <= x; ++i){
        for(ll j = p[i];1ll * j * p[i] <= x; j *= p[i]){
            ans = (ans + h[ID(x / j)] - prep[i - 1] + S(x / j, i + 1)) % mod;
        }
    }
    return ans;
}
ll g[maxn];
ll G(ll x){
    if (x <= 1) {
        return x;
    }
    int id = ID(x);
    if(g[id]){
        return g[id];
    }
    ll ans = (1 + s[id] + h[id]) % mod;
    for(ll l = 2, r; l <= x; l = r + 1){
        r = x / (x / l);
        ans = (ans - (r - l + 1) * G(x / l) % mod + mod) % mod;
    }
    return g[id] = ans;
}
int main(){
    scanf("%lld", &n);
    sqr = sqrt(n);
    nsqr = n / sqr;
    sieve(sqr);
    calw();
    ll ans = 0;
    for(ll l = 1, r; l <= n; l = r + 1){
        r = n / (n / l);
        ans = (ans + (n / l) * (n / l) % mod * (G(r) - G(l - 1) + mod) % mod) % mod;
    }
    printf("%lld\n", ans);
}
```


```cpp
#include<bits/stdc++.h>
using namespace std;
typedef double db;
typedef long long ll;
#define fi first
#define se second
#define all(x) (x).begin(), (x).end()
const int mod = 998244353;

int power_mod(int a, int b) {
	int r = !!a;
	for(; b; b >>= 1, a = (ll) a * a % mod)
		if(b & 1) r = (ll) r * a % mod;
	return r;
}

const int N = 1e5 + 1000;

bitset<N> np;
int p[N>>2], pn;
ll pg[N>>2], ph[N>>2]; // pg(n) = \sum_{i=1}^n g[p[i]]
void sieve(int sz) {
	for(int i = 2; i <= sz; i++) {
		if(!np[i]) {
			++pn;
			p[pn] = i;
			pg[pn] = (pg[pn - 1] + 1) % mod;
			ph[pn] = (ph[pn - 1] + i) % mod;
		}
		for(int j = 1; j <= pn && i * p[j] <= sz; j++) {
			np[i * p[j]] = 1;
			if(i % p[j] == 0) {
				break;
			}
		}
	}
}

int _id[N*2], m;
ll n;
ll w[N * 2];
ll Pg[N * 2], Ph[N * 2]; // Pg(n) = \sum_{i=1}^{w[n]} g[i] [i in Prime ]


int id(ll x) {
	return x < N ? x : n / x + N;
}
int Id(ll x) {
	return _id[id(x)];
}


typedef __int128 i16;

ll G[N * 2], H[N * 2];
/***
*** G(n) = \sum_{i = 1}^{w[n]} -mu[i]
*** H(n) = \sum_{i = 1}^{w[n]} -mu[i] * minp[i]
        G(n, j) = \sum_{i = 1}^n [minp[i] >= p[j] and i not in Prime] -mu[i] = G(n, j + 1) + mu[p[j]] * (G(n / p[j], j + 1) + Pg(n / p[j]) - pg(j - 1))
        H(n, j) = \sum_{i = 1}^n [minp[i] >= p[j] and i not in Prime] -mu[i]*minp[i] = H(n, j + 1) - p[j] * (G(n / p[j], j + 1) + Pg(n / p[j]) - pg(j - 1))
*** ans = \sum_{i = 1}^n (n/i)^2 * mu[i] * (1 - minp[i])
                = \sum_{i = 1}^n (n/i)^2 * (H(n) - G(n))
***/
void init(int sz) {
	sieve(sz);
	m = 0;
	for(ll l = 1, r; l <= n; l = r + 1) {
		r = n / (n / l);
		w[++m] = n / l;
		_id[id(w[m])] = m;
		Pg[m] = (w[m] - 1) % mod;
		Ph[m] = ((i16) w[m] * (w[m] + 1) / 2 - 1) % mod;
	}
	for(int j = 1; j <= pn; j++) {
		for(int i = 1; i <= m && (ll) p[j] * p[j] <= w[i]; i++) {
			int k = Id(w[i] / p[j]);
			Pg[i] = (Pg[i] - (Pg[k] - pg[j - 1])) % mod;
			Ph[i] = (Ph[i] - p[j] * (Ph[k] - ph[j - 1])) % mod;
		}
	}
	for(int j = pn; j >= 1; j--) {
		for(int i = 1; i <= m && (ll) p[j] * p[j] <= w[i]; i++) {
			int k = Id(w[i] / p[j]);
			G[i] = (G[i] - (G[k] + Pg[k] - pg[j])) % mod;
			H[i] = (H[i] - p[j] * (G[k] + Pg[k] - pg[j])) % mod;
//			cout << w[i] << ' ' << p[j] << ' ' << G[i] << " ha " << endl;
		}
	}
	for(int i = 1; i <= m; i++) {
		G[i] = (G[i] + Pg[i] - 1) % mod;
		H[i] = (H[i] + Ph[i]) % mod;
		if(G[i] < 0) G[i] += mod;
		if(H[i] < 0) H[i] += mod;
	}
	/*
	cout << m << " JDH  " << endl;
	for(int i = 1; i <= m; i++) {
	  cout << w[i] << ' ' << G[i] << ' ' << H[i] << '\n';
	}*/
}
/*** \sum_{i=1}^n \sum_{d|i} mu[d] * maxp[i / d] = sum_{i=1}^n mu[i] * (1 - minp[i]) ***/

ll Muf(ll n) {
	if(n == 0) return 0;
	int k = Id(n);
	ll ret = H[k] - G[k];
	if(ret < 0) ret += mod;
	return ret;
}

int main() {
#ifdef local
	freopen("in.txt", "r", stdin);
#endif
	ios::sync_with_stdio(false);
	cin.tie(0), cout.tie(0);

	cin >> n;
	ll B = sqrt(n) + 10;
	init(B);
	ll ret = 0, last = 0;
	for(ll l = 1, r; l <= n; l = r + 1) {
		r = n / (n / l);
		ll cur = Muf(r);
		ret = (ret + (i16) (n / l) * (n / l) % mod * (cur - last)) % mod;
		last = cur;
//		cout << l << ' ' << r << ' ' << ret << endl;
	}
/*
	for(int i = 1; i <= m; i++) {
	  cout << w[i] << ' ' << Muf(w[i]) << ' ' << Pg[i] << ' ' << Ph[i] << endl;;
	}*/
	if(ret < 0) ret += mod;
	cout << ret << '\n';
	return 0;
}
```


