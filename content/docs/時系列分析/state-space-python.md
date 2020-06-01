+++
title = "Python による状態空間モデルの実装法"
author = ["Hiroshi Nakano"]
draft = false
+++

Python で線形ガウス状態空間モデルを実現するには、統計分析向けライブラリの [ `statsmodels` ](https://statsmodels.org) を用いるとよい。  


## statsmodels での線形ガウス状態空間モデルの式 {#statsmodels-での線形ガウス状態空間モデルの式}

`statsmodels` での線形ガウス状態空間モデルの式は、前の記事で導入した式と若干異なるので、注意が必要。実際に `statsmodels` で採用されている形式での線形ガウス状態空間モデルの式を示す。  

\\[ \begin{align}
y\_t &= Z\_t \alpha\_t + d\_t + \varepsilon\_t, & \varepsilon\_t \sim \mathcal{N}(0, H\_t) \\\\\\
\alpha\_{t+1} &= T\_t \alpha\_t + c\_t + R\_t \eta\_t, & \eta\_t \sim \mathcal{N}(0, Q\_t) \\\\\\
\end{align} \\]  

第 1 式の、\\(y\_t\\) についての式が観測方程式である。  

いくつかのベクトルについては、 `statsmodels` 固有の名前が与えられている。  

| 記号       | 名称              | 説明     |
|----------|-----------------|--------|
| \\(Z\_t\\) | `design`          |          |
| \\(d\_t\\) | `obs_intercept`   |          |
| \\(H\_t\\) | `obs_cov`         | 観測誤差の共分散 |
| \\(T\_t\\) | `transition`      |          |
| \\(c\_t\\) | `state_intercept` |          |
| \\(R\_t\\) | `selection`       |          |
| \\(Q\_t\\) | `state_cov`       | 過程誤差の共分散 |


## ローカル線形トレンドモデルの実装 {#ローカル線形トレンドモデルの実装}

ローカル線形トレンドモデルの式を下記に再掲する。  

\\[ \begin{align}
y\_t & = \mu\_t + \varepsilon\_t \qquad & \varepsilon\_t \sim N(0, \sigma\_\varepsilon^2) \\\\\\
\mu\_{t+1} & = \mu\_t + \nu\_t + \xi\_t & \xi\_t \sim N(0, \sigma\_\xi^2) \\\\\\
\nu\_{t+1} & = \nu\_t + \zeta\_t & \zeta\_t \sim N(0, \sigma\_\zeta^2) \\\\\\
\end{align} \\]  

ただし、第 1 式が観測方程式、第 3 式がトレンド変化に関する式である。  

`statsmodels` では行列を用いた表現を前提とするので、上記ローカル線形トレンドモデルの式を行列を用いて書き直す。  

\\[ \begin{align}
y\_t & = \begin{bmatrix} 1 & 0 \end{bmatrix} \begin{bmatrix} \mu\_t \\\\\\ \nu\_t \end{bmatrix} + \varepsilon\_t
\\\\\\ \begin{bmatrix} \mu\_{t+1} \\\\\\ \nu\_{t+1} \end{bmatrix} &= \begin{bmatrix} 1 & 1 \\\\\\ 0 & 1 \end{bmatrix} \begin{bmatrix} \mu\_t \\\\\\ \nu\_t \end{bmatrix} + \begin{bmatrix} \xi\_t \\\\\\ \zeta\_t \end{bmatrix} \\\\\\
\end{align} \\]  

このとき、推定すべきパラメタは、\\(\sigma^2\_\varepsilon\\) , \\(\sigma^2\_\xi\\) , \\(\sigma^2\_\zeta\\) である。  

ただし、 \\(\xi\_t\\) と \\(\zeta\_t\\) はひとつのベクトルにまとまっているので、 `state_cov` \\(Q\_t\\) は \\((2 \times 2)\\) の行列になることに注意。  

\\[ \begin{align}
H\_t & = \begin{bmatrix} \sigma\_\varepsilon^2 \end{bmatrix} \\\\\\
Q\_t & = \begin{bmatrix} \sigma\_\xi^2 & 0 \\\\\\ 0 & \sigma\_\zeta^2 \end{bmatrix}
\end{align} \\]  


### モデリングフレームワーク {#モデリングフレームワーク}

観測データから、カルマンフィルタと最尤法を用いて状態とパラメタを推定したい。それを実現するには、 `statsmodels.tsa.statespace.MLEModel` を継承したクラスを作成すればよい。ちなみに MLE は最尤推定(Maximum Likelyhood Estimator)の略。  

```python
import statsmodels.api as sm

class LocalLinearTrend(sm.tsa.statespace.MLEModel):

    def __init__(self, endog):
```

この新しく作成するモデルには、必ず以下のメソッドを定義する。  


#### `__init__()` {#init}

観測データを `endog` として引数にとる。  
`__init__()` では、モデルの方程式と状態の初期化方法を定義しなければならない。  

<!--list-separator-->

-  モデル式の定義

    モデル式の定義方法から説明する。  
    
    まず、状態ベクトルの次元数 `k_states` と、過程誤差ベクトルの次元数 `k_posdef` を指定する。ローカル線形トレンドモデルの場合、どちらも 2 である。  
    
    ```python
    def __init__(self, endog):
        k_states = k_posdef = 2
    ```
    
    次に、状態空間行列を定義する。状態空間行列とは、 \\(Z\_t\\) や \\(d\_t\\) などの `statsmodels` で名前のついている行列のことである。デフォルトではすべて零行列となっているので、変更しておきたい部分だけ定義すればよい。  
    
    ローカル線形トレンドモデルの場合に定義するのは、以下の表の通り。  
    
    | 行列       | 名前         | 値                                                    |
    |----------|------------|------------------------------------------------------|
    | \\(Z\_t\\) | `design`     | \\(\begin{bmatrix} 1 & 0 \\ \end{bmatrix}\\)          |
    | \\(T\_t\\) | `transition` | \\(\begin{bmatrix} 1 & 1 \\\\\\ 0 & 1 \\ \end{bmatrix}\\) |
    | \\(R\_t\\) | `selection`  | 単位行列                                              |
    
    とくに、 `selection` は単位行列であることが多いのだが、うっかりで定義し忘れやすいので注意。  
    
    ```python
    # 状態空間行列は self.ssm に格納する。
    # ssm は多分 state space matrix の略。
    import numpy as np
    
    self.ssm['design'] = np.array([1, 0])
    self.ssm['transition'] = np.array([[1, 1],
                                       [0, 1]])
    self.ssm['selection'] = np.eye(k_states)  # 2×2 の単位行列
    ```

<!--list-separator-->

-  状態の初期化

    カルマンフィルタを実現するためには、\\(t=0\\) のときの状態ベクトルとの平均と分散を定める必要がある。  
    
    `MLEModel` には、初期化を助けるための関数がそなわっているので、どの関数を選ぶかを指定するだけでよい。  
    
    -   `initialize_known(initial_state, initial_state_cov)` : 初期値の確率分布がわかっているときに使う。
    -   `initialize_stationary` : ARMA などのように、定常状態であることがわかっているときに使う。
    -   `initialize_approximate_diffuse` : 散漫初期化をする。上記 2 つが適切でないときに使うと考えてよい。
    
    散漫初期化や定常状態の場合  
    
    ```python
    # 初期化は親クラスの機能を使う
    super(LocalLinearTrend, self).__init__(
        endog,
        k_states=k_states,
        k_posdef=k_posdef,
        initialization="approximate_diffuse"
        # initialization="stationary"
    )
    ```
    
    初期値がわかっているとき  
    
    ```python
    super(LocalLinearTrend, self).__init__(
        endog,
        k_states=k_states,
        k_posdef=k_posdef,
        initialization="known",
        initial_state=np.array([0,0])  # 初期値の平均ベクトル。
        initial_state_cov=np.array([[50, 0],
                                    [0, 30]])  # 初期値の共分散行列。
    )
    ```


#### `update(params)` {#update--params}

最尤法を用いてパラメタを推定するときに、 `MLEModel` がパラメタの更新に用いるメソッド。新たに計算されたパラメタをそれぞれどこに代入すればいいかを設定する。  

```python
@property
def param_names(self):
    """
    推定するパラメタの名前。
    分析結果を表示するときにもこの名前が使われる。
    """
    # ローカル線形トレンドモデルで推定するパラメタは、
    # 観測誤差の分散/状態変化の誤差の分散/トレンド変化の誤差の分散の三つ
    return ['var.measurement', 'var.state', 'var.trend']

@property
def start_params(self):
    """
    3 つのパラメタの初期値。
    わりと適当でいいけど、絶対に取り得ない値だけは避ける。
    例えば、分散は 0 以下にはならないので、必ず正の値にする。
    """
    # 今回は観測値の分散を 3 つのパラメタの初期値とする。
    return [np.std(self.endog)] * 3

def update(self, params, *args, **kwargs):
    # まずは親クラスの結果を受け取る。
    params = super(LocalLinearTrend, self).update(params, *args, **kwargs)
    # パラメタは array-like
    # 格納順序は`param-names`で決めたとおり。
    # 観測誤差の分散
    self.ssm['obs_cov',0,0] = params[0]
    # 過程誤差の分散
    # `state_cov` は 2×2 の行列で、更新するのはその対角成分だけ。
    self.ssm['state_cov',0,0] = params[1]
    self.ssm['state_cov',1,1] = params[2]
```


#### transform/untransform {#transform-untransform}

パラメタの推定を行うときに、 `statsmodels` の最適化アルゴリズムでは、パラメタの値の範囲を制限できない。しかし、推定したいパラメタが分散のように負の値をとらないだとか、なんらかの制約条件をもつことは多い。  

この問題の解決のために、パラメタをそのまま最適化アルゴリズムで求めるのではなく、何らかの値の変換を介することができるように設計されている。  

ローカル線形トレンドモデルの場合、パラメタは全て分散なので、値は負にならない。従って、最適化アルゴリズムで求めた値(=負もとりうる)を 2 乗する。  

```python
def transform(self, unconstrained):
    """
    最適化アルゴリズムの出力値を尤度評価の時に変換する。
    unconstrained は np.ndarray
    """
    # ローカル線形トレンドモデルのパラメタは分散なので負にならない。
    # 従って、`unconstrained` を 2 乗する
    return unconstrained ** 2

def untransform(self, constrained):
    """
    パラメタを最適化アルゴリズムに渡すときの変換。
    constrained は np.ndarray
    基本的に transform の逆の演算を行えばよい。
    """
    return constrained ** 0.5
```


#### まとめ {#まとめ}

これまでのコードをひとまとめにする。以下がローカル線形トレンドモデルの実装になる。  

```python
import numpy as np
import pandas as pd
from scipy.stats import norm
import statsmodels.api as sm

class LocalLinearTrend(sm.tsa.statespace.MLEModel):
    """ローカル線形トレンドモデル"""

    def __init__(self, endog):
        k_states = k_posdef = 2

        # 状態空間の初期化
        super(LocalLinearTrend, self).__init__(
            endog,
            k_states=k_states,
            k_posdef=k_posdef,
            initialization="approximate_diffuse",
            loglikelihood_burn=k_states  # 尤度を求めるときに使う
        )

        # 状態空間行列の指定
        self.ssm['design'] = np.array([1,0])
        self.ssm['transition'] = np.array([[1,1], [0,1]])
        self.ssm['selection'] = np.eye(k_states)

        # 過程誤差のパラメタの更新に使う
        # np.diag_indices は行列の対角成分を返す関数
        self._state_cov_idx = ('state_cov',) + np.diag_indices(k_posdef)

    @property
    def param_names(self):
        """推定するパラメタの名前"""
        return ['var.measurement', 'var.level', 'var.trend']

    @property
    def start_params(self):
        """パラメタの初期値"""
        return [np.std(self.endog)] * 3

    def transform_params(self, unconstrained):
        """最適化 -> 尤度評価の際の変換"""
        return unconstrained ** 2

    def untrandform_params(self, constrained):
        """尤度評価->最適化の際の変換"""
        returnconstrained ** 0.5

    def update(self, params, *args, **kwargs):
        """パラメタの更新"""
        params = super(LocalLinearTrend, self).update(params, *args, **kwargs)
        # 観測誤差の分散
        self.ssm['obs_cov',0,0] = params[0]
        # 過程誤差の分散
        self.ssm[self._state_cov_idx] = params[1:]
```


## サンプルデータ分析 {#サンプルデータ分析}

分析の感覚をつかむために、サンプルデータセットに作成したモデルを適用する。  

サンプルには、 [ `Rdatasets` ](https://vincentarelbundock.github.io/Rdatasets/)の `BJsales` を使う。  

```python
bjsales = sm.datasets.get_rdataset("BJsales", "datasets")
# メタデータ
print(bjsales.__doc__)
print('='*80)
# データシェマ
bjsales.data.info()
print('='*80)
# データの先頭 5 行
bjsales.data.head()
```

```text
+---------+-----------------+
| BJsales | R Documentation |
+---------+-----------------+

Sales Data with Leading Indicator
---------------------------------

Description
~~~~~~~~~~~

The sales time series ``BJsales`` and leading indicator ``BJsales.lead``
each contain 150 observations. The objects are of class ``"ts"``.

Usage
~~~~~

::

   BJsales
   BJsales.lead

Source
~~~~~~

The data are given in Box & Jenkins (1976). Obtained from the Time
Series Data Library at
http://www-personal.buseco.monash.edu.au/~hyndman/TSDL/

References
~~~~~~~~~~

G. E. P. Box and G. M. Jenkins (1976): *Time Series Analysis,
Forecasting and Control*, Holden-Day, San Francisco, p. 537.

P. J. Brockwell and R. A. Davis (1991): *Time Series: Theory and
Methods*, Second edition, Springer Verlag, NY, pp. 414.

================================================================================
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 150 entries, 0 to 149
Data columns (total 2 columns):
 #   Column  Non-Null Count  Dtype  
---  ------  --------------  -----  
 0   time    150 non-null    int64  
 1   value   150 non-null    float64
dtypes: float64(1), int64(1)
memory usage: 2.5 KB
================================================================================
   time  value
0     1  200.1
1     2  199.5
2     3  199.4
3     4  198.9
4     5  199.0
```

```python
import matplotlib.pyplot as plt

# 必要なのは value のみ
ts = bjsales.data['value']

# どんなデータしているか、グラフで見る。
fig = plt.figure(constrained_layout=True, figsize=(8, 5))
gs = fig.add_gridspec(2,2)
ax1 = fig.add_subplot(gs[0,:])
ts.plot(ax=ax1)
ax1.set_title('BJ Sales Data')

ax2 = fig.add_subplot(gs[1,0])
sm.graphics.tsa.plot_acf(ts, ax=ax2)
ax2.set_title("Autocorrelation")

ax3 = fig.add_subplot(gs[1,1])
sm.graphics.tsa.plot_pacf(ts,ax=ax3)
ax3.set_title("Partial Autocorrelation")
plt.show()
```

{{< figure src="/learn-docs/ox-hugo/a653d09dbab7af2b519b23cbd13289378dcd244d.png" >}}  

これにモデルを適用する  

```python
model = LocalLinearTrend(np.log(ts))
result = model.fit()
result.summary()
```

```text
<class 'statsmodels.iolib.summary.Summary'>
"""
                           Statespace Model Results                           
==============================================================================
Dep. Variable:                  value   No. Observations:                  150
Model:               LocalLinearTrend   Log Likelihood                 544.866
Date:                Mon, 01 Jun 2020   AIC                          -1083.732
Time:                        17:02:47   BIC                          -1074.740
Sample:                             0   HQIC                         -1080.079
                                - 150                                         
Covariance Type:                  opg                                         
===================================================================================
                      coef    std err          z      P>|z|      [0.025      0.975]
-----------------------------------------------------------------------------------
var.measurement  2.971e-11   2.98e-06   9.99e-06      1.000   -5.83e-06    5.83e-06
var.level        2.778e-05   7.44e-06      3.735      0.000    1.32e-05    4.24e-05
var.trend        2.296e-06   1.01e-06      2.264      0.024    3.09e-07    4.28e-06
===================================================================================
Ljung-Box (Q):                       43.58   Jarque-Bera (JB):                 0.54
Prob(Q):                              0.32   Prob(JB):                         0.76
Heteroskedasticity (H):               0.35   Skew:                            -0.00
Prob(H) (two-sided):                  0.00   Kurtosis:                         3.30
===================================================================================

Warnings:
[1] Covariance matrix calculated using the outer product of gradients (complex-step).
"""
```

予測残差がホワイトノイズに近いほど、よいモデルといえる。  

ホワイトノイズは正規分布で、自己相関がゼロで、分散が一定だった。  

Ljung-Box 検定(自己相関の検定)と Jarque-Bera 検定(正規性の検定)の結果から、残差は正規分布に従い、かつ、自己相関もないといえる。  

ただし、Heteroskedasticity test(分散不均一性検定)の結果を見ると、分散が一定であるとは言えなさそうである。このことから、時間変化する分散をモデルに組込めば更に精度を上げられると考えられる。  

予測残差の分布は `plot_diagnostics()` で見ることもできる。  

```python
fig = plt.figure(figsize=(8, 6))
result.plot_diagnostics(fig=fig)
```

![](/learn-docs/ox-hugo/3ccdd66ff7b9f77db377376ff8ac7ee7bb4cd46c.png)  


### 予測 {#予測}

```python
predict = result.get_prediction()
forecast = result.get_forecast(30)

fig, ax = plt.subplots(figsize=(9,6))
# 観測データを黒色×マークで表す
ts.plot(style='.:k', ax=ax, label='Observations')
np.exp(predict.predicted_mean).plot(ax=ax, label='One-step-ahead Prediction')
predict_ci = predict.conf_int(alpha=0.8)
predict_index = np.arange(len(predict_ci))
ax.fill_between(predict_index[2:], np.exp(predict_ci).iloc[2:, 0], np.exp(predict_ci).iloc[2:, 1], alpha=0.1)

np.exp(forecast.predicted_mean).plot(ax=ax, style='r', label='Forecast')
forecast_ci = forecast.conf_int()
forecast_index = np.arange(len(predict_ci), len(predict_ci) + len(forecast_ci))
ax.fill_between(forecast_index, np.exp(forecast_ci).iloc[:, 0], np.exp(forecast_ci).iloc[:, 1], alpha=0.1)

# Cleanup the image
ax.set_ylim(150, None)
legend = ax.legend(loc='upper left', bbox_to_anchor=(0, 1));
```

{{< figure src="/learn-docs/ox-hugo/5570483b45fdb00a6d3c43d0395d6c8bde7974d2.png" >}}  


### ナイーブ予測との比較 {#ナイーブ予測との比較}

このモデルを使うことに価値があるかどうかを考えてみる。最も単純なランダムウォークと比較する。  

```python
model_rwalk = sm.tsa.UnobservedComponents(np.log(ts), level='rwalk')
res_rwalk = model_rwalk.fit()
res_rwalk.summary()
```

```text
<class 'statsmodels.iolib.summary.Summary'>
"""
                        Unobserved Components Results                         
==============================================================================
Dep. Variable:                  value   No. Observations:                  150
Model:                    random walk   Log Likelihood                 535.509
Date:                Mon, 01 Jun 2020   AIC                          -1069.017
Time:                        17:02:50   BIC                          -1066.013
Sample:                             0   HQIC                         -1067.797
                                - 150                                         
Covariance Type:                  opg                                         
================================================================================
                   coef    std err          z      P>|z|      [0.025      0.975]
--------------------------------------------------------------------------------
sigma2.level  4.418e-05   4.31e-06     10.241      0.000    3.57e-05    5.26e-05
===================================================================================
Ljung-Box (Q):                      117.00   Jarque-Bera (JB):                 6.26
Prob(Q):                              0.00   Prob(JB):                         0.04
Heteroskedasticity (H):               0.34   Skew:                             0.45
Prob(H) (two-sided):                  0.00   Kurtosis:                         3.44
===================================================================================

Warnings:
[1] Covariance matrix calculated using the outer product of gradients (complex-step).
"""
```

まず、残差に自己相関が残っており、分布も正規分布でないことがすぐにわかる。また、AIC()を比較すると  

```python
print("LocalLinearTrend: {:.6}".format(result.aic))
print("RandomWalk: {:.6}".format(res_rwalk.aic))
```

```text
LocalLinearTrend: -1083.73
RandomWalk: -1069.02
```

となり、ローカル線形トレンドモデルのほうが低い。すなわち、少なくともランダムウォークよりはローカル線形トレンドモデルのほうがよいモデルであるとわかる。
