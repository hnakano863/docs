+++
title = "Kaplan-Meier Method"
author = ["Hiroshi Nakano"]
draft = false
+++

## カプランマイヤー法 {#カプランマイヤー法}

カプランマイヤー法は、生存関数を実データから推測するためのノンパラメトリックな手法のひとつ。生存時間解析では最も基本的な手法である。  

本稿では、カプランマイヤー法の説明のために、以下のデータセットを用いる。  

```text
    start  group  z  stop  id  event
0       0    1.0  0   3.0   1   True
1       0    1.0  0   5.0   2  False
2       0    1.0  1   5.0   3   True
3       0    1.0  0   6.0   4   True
4       0    1.0  0   6.0   5  False
5       6    1.0  1   8.0   5  False
6       0    0.0  1   4.0   6  False
7       0    0.0  0   5.0   7  False
8       5    0.0  1   7.0   7   True
9       0    0.0  0   8.0   8  False
10      0    0.0  0   5.0   9  False
11      5    0.0  1   9.0   9   True
12      0    0.0  0   3.0  10  False
13      3    0.0  1  10.0  10   True
```

このデータセットは、カリフォルニア サンディエゴ大学の数学科が公開している講義資料の中に出て来た例示用のデータセットである。  
<http://www.math.ucsd.edu/~rxu/math284/slect7.pdf> から、もとのデータセットを見ることができる。  

`event` 列が死亡の有無、 `start` は観察開始時期、 `stop` は観察終了時期を表している。他の列については本稿では用いない。  

それぞれのサンプルの観察期間と、死亡の有無を確認すると、  

{{< figure src="/learn-docs/ox-hugo/359e8499d0ce81af783a94fb361ad4c89e8faf50.png" >}}  

半分程度が打切りであり、観測期間もサンプルによってまちまちであることがわかる。  


### 観測データの前処理 {#観測データの前処理}

カプランマイヤー法の適用のために、まずは観測データを前処理する。  

まずは、サンプルの観察開始点をそろえる。  

{{< figure src="/learn-docs/ox-hugo/5361b5ff0a07c9c27cea1d9d695a0a207ae73324.png" >}}  

次に、これを生存時間順に並べ替える  

{{< figure src="/learn-docs/ox-hugo/1bfafb38790e5b3e4d57474eff3d34a1ff78e3e1.png" >}}  


### 生命表 {#生命表}

上のグラフをみると、観察期間が 2 年までは全 14 サンプルのデータが存在しているが、その後 3 年、4年と観察期間が伸びていくほどに、サンプル数が減っていくことがわかる。  

そこで、観察期間とサンプル数、死亡数の推移をまとめた表を作成する。この表を **生命表** という。  

```text
          removed  observed  censored  entrance  at_risk
event_at                                                
0.0             0         0         0        14       14
2.0             2         1         1         0       14
3.0             2         1         1         0       12
4.0             2         1         1         0       10
5.0             4         1         3         0        8
6.0             2         1         1         0        4
7.0             1         1         0         0        2
8.0             1         0         1         0        1
```

生命表のインデックスは観察期間を示す。これはグラフの x 軸に対応する。  
`at_risk` 列が期間中のサンプルの数、 `observed` は死亡数、 `censored` が右打切りの数である。  

例えば、2年目まではサンプルは 14 個体、そのうち、右打切りが 1 個体、死亡が 1 個体である。  


### 期別生存率 {#期別生存率}

サンプル \\(n\\) 個体中、 \\(m\\) 個体が死んだ場合、死亡率は \\(\frac{m}{n}\\) で計算できる。生存率は、死亡率を 1 から引けば求められるので、 \\(1 - \frac{m}{n}\\) である。  

ここで、生命表を見ればわかるように、サンプルの数は観察期間によって異なる。そこで、観察期間ごとに生存率を算出する。この生存率を、期別生存率という。  

```text
          removed  observed  censored  entrance  at_risk  death_rate  \
event_at                                                               
0.0             0         0         0        14       14    0.000000   
2.0             2         1         1         0       14    0.071429   
3.0             2         1         1         0       12    0.083333   
4.0             2         1         1         0       10    0.100000   
5.0             4         1         3         0        8    0.125000   
6.0             2         1         1         0        4    0.250000   
7.0             1         1         0         0        2    0.500000   
8.0             1         0         1         0        1    0.000000   

          survival_rate  
event_at                 
0.0            1.000000  
2.0            0.928571  
3.0            0.916667  
4.0            0.900000  
5.0            0.875000  
6.0            0.750000  
7.0            0.500000  
8.0            1.000000  
```


### 累積生存率 {#累積生存率}

期別生存率を互いに掛け合せたもの(累積積)を **累積生存率** という。  

カプランマイヤー法とは、この累積生存率を生存関数の推定値とする統計手法である。  

```text
          removed  observed  censored  entrance  at_risk  death_rate  \
event_at                                                               
0.0             0         0         0        14       14    0.000000   
2.0             2         1         1         0       14    0.071429   
3.0             2         1         1         0       12    0.083333   
4.0             2         1         1         0       10    0.100000   
5.0             4         1         3         0        8    0.125000   
6.0             2         1         1         0        4    0.250000   
7.0             1         1         0         0        2    0.500000   
8.0             1         0         1         0        1    0.000000   

          survival_rate  cumulative_survival_rate  
event_at                                           
0.0            1.000000                  1.000000  
2.0            0.928571                  0.928571  
3.0            0.916667                  0.851190  
4.0            0.900000                  0.766071  
5.0            0.875000                  0.670312  
6.0            0.750000                  0.502734  
7.0            0.500000                  0.251367  
8.0            1.000000                  0.251367  
```

この累積生存率をグラフにすると以下のようになる。  

{{< figure src="/learn-docs/ox-hugo/7787d4ef99c5d23a4013bcadf328747eb215168a.png" >}}  


## Python での実装 {#python-での実装}

Python の生存時間解析用ライブラリである `lifelines` を使えば、カプランマイヤー法がすぐに使える。  

```python
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.datasets import load_dfcv
from lifelines.plotting import plot_lifetimes

# データセットのダウンロード
# 前の章までのデータと同じものを使っている。
dfcv_data = load_dfcv()
# 観察期間の長さを計算
dfcv_data['duration'] = dfcv_data['stop'] - dfcv_data['start']

# インスタンス化
kmf = KaplanMeierFitter()

# データに Fit
kmf.fit(
    dfcv_data['duration'],
    event_observed=dfcv_data['event']
)

# プロット
kmf.plot()
plt.show()
```

{{< figure src="/learn-docs/ox-hugo/5dd6dcccd713ad0518943a71e1f15634df915a83.png" >}}  

薄色は 95%信頼区間であり、 `plot()` のキーワード引数 `ci_show` に `False` を渡すことで表示を無くせる。また、系列名はデフォルトで `KM_estimate` となるが、これも同じくキーワード引数 `label` に渡す値で変えられる。他にも、 `lifelines.plotting.add_at_risk_counts` を使えば、サンプル数の変化を表示させることができる。  

```python
from lifelines.plotting import add_at_risk_counts
from matplotlib.ticker import PercentFormatter

fig, ax = plt.subplots()

kmf.plot(
    ax=ax,  # 既に存在する subplot にプロットする
    ci_show=False,  # 信頼区間を非表示
    label="Fantastic Result",  # 系列名の変更
    iloc=slice(0,7),  # プロットする範囲を 0~7 年に制限
    linestyle="--", linewidth=2, color="red",  # matplotlib の plot と同じキーワード引数が使える。
)


# サンプル数の変化を表示
add_at_risk_counts(kmf, ax=ax, labels=["Fantastic Result"])

# 普通の matplotlib のオブジェクトとして操作できる。
ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
ax.set_ylim(0, None)

plt.show()
```

{{< figure src="/learn-docs/ox-hugo/7c141419641be6771f4245076d5925e454a67e6f.png" >}}
