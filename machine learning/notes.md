# Marchine Learning

## 1. k-NN regression

```python
import numpy as np
import matplotlib.pyplot as plt

# create data
np.random.seed(20)
x = np.random.uniform(0, 10, 18)
y = np.random.uniform(0, 10, 18)

# create scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(x, y, c='blue', label='$Random\; Points$')
plt.title('$Scatter\; Plot\; with\; 18\; Random\; Points$')
plt.xlabel('$X$')
plt.ylabel('$Y$')
plt.legend()
plt.show()
```
![scatter](https://github.com/weinter0101/python-practice/blob/main/machine%20learning/figure/Figure1.1.png)


### k=n
- k=1
```python
k = 1
knnRegressor = KNeighborsRegressor(k)
knnRegressor.fit(x, y)

T = np.linspace(0, 10, 500)[:, np.newaxis] 
knnLine = knnRegressor.predict(T)

plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='blue', label='$Data\; Points$')
plt.plot(T, knnLine, color='red', label=f'$k={k}\; Regression$')
plt.title('$k-NN\; Regression\; with\; k=3$')
plt.xlabel('$X$')
plt.ylabel('$Y$')
plt.legend()
plt.show()
```
![scatter](https://github.com/weinter0101/python-practice/blob/main/machine%20learning/figure/Figure1.2.png)

- k=2, 3, 5, 18
     [code](https://gist.github.com/7e0deab4e3c6ca9323e3c195fc77b71f.git)
![scatter](https://github.com/weinter0101/python-practice/blob/main/machine%20learning/figure/Figure1.3.png)

1. k值過小：overfitting, 無法對未見過的數據進行準確預測。
2. k值過大：underfitting, 由於過度平滑所以對訓練數據和新數據都不能進行有效的預測。

## 2. Linear Regression Model
- find a linear function to fit oberseve data
- Least Squares (LS) estimation:
     - Fitting criterion: sum of square distance between actual y and predicted y in the traning set
     - Find a function that minimizes this critersoin.
- general form of linear function:

$$
\begin{align*}
\mathbf{f}(\mathbf{x}; \boldsymbol{\beta}) &= \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \ldots + \beta_k x_k \quad (x_0 = 1) \\
&= \sum_{j=0}^{k} \beta_j x_j \\
&= \mathbf{x}^{\top} \boldsymbol{\beta},
\end{align*}
$$



- define a loss function(fitting criterion):

$$
square \quad loss: \( l(y, \hat{y}) = (y - \hat{y})^2 \)
$$

- example: advertising data
```python
import pandas as pd
import os
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf #OLS
import seaborn as sns

path = r'C:\Users\chewei\Documents\python-practice\machine learning\data'
name = 'Advertising.csv'

advertising = pd.read_csv(os.path.join(path, name))
advertising.info()
advertising.head(3)

dependentVariable = ['TV', 'radio', 'newspaper']

for variables in dependentVariable:
    est = smf.ols(f"sales ~ {variables}", advertising).fit()
    print(est.summary())
    
est = smf.ols('sales ~ TV + radio + newspaper', advertising).fit()
print(est.summary())
print(advertising.corr())

est = smf.ols('sales ~ TV + radio + TV * radio', advertising).fit()
print(est.summary())

#%% figure
plt.figure(figsize=(18, 6))

for i, variable in enumerate(dependentVariable, 1):
    plt.subplot(1, 3, i) 
    sns.regplot(x=variable, y='sales', data=advertising)
    plt.title(f'Sales to {variable}')
    plt.xlabel(variable)
    plt.ylabel('Sales')

plt.tight_layout()
plt.show()
```
![scatter](https://github.com/weinter0101/python-practice/blob/main/machine%20learning/figure/Figure2.1.png)




## 3. Gradient Descent

- 無法使用 Least Square Method 求解 $\beta$
     - the data is too large to compute the inverse for X'X
     - the colsed form solution is not exist
- $\beta^{(1)} = \beta^{(0)} + \eta d$
     - $\beta^{(0)} \text{ is the initial guess.}$
     - $\beta^{(1)} \text{ is a guess after the first correction.}$
     - $\eta \text{ is the learning rate.}$
     - d is the descent direction.
          - gradient descent: find maximun, d=-1
          - gradient ascent: find minimun, d=1
- $L(\beta^{(0)} + \eta d) = L(\beta^{(0)}) + \eta \nabla L(\beta^{(0)})^\mathrm{T} d + R$
```python
import numpy as np
from numpy.linalg import inv

# create data
np.random.seed(20)
N = 100     # sample size
k = 5        
beta = np.array([[0.9, 0.5, 0.2, -0.2, -0.7]]).T       # true beta
X = np.random.randn(N, k)
e = np.random.randn(N,1)
y = X@beta+e

# OLS
bhat1 = inv(X.T@X)@X.T@y

# gradient descent
bhat2 = np.zeros([k, 1])
cost1 = (y-X@bhat2).T @ (y-X@bhat2) / N
eta = 0.05
criterion = 0.0000001   # 停止條件  e.g. (bhat-b)/b
maxIters = 500      # 最多尋找次數
count = 0
delta = 999

while delta > criterion and count < maxIters:
     gradient = -X.T @ (y-X@bhat2) / N    # form
   # d = (g>0)*-1 + (g<0)*1   # 將梯度轉換成 1 或 -1 的方向
     bhat2 = bhat2 - eta*gradient
   # bhat2 = bhat2 + eta*d    # 以方向去迭代初 bhat2
     cost2 = (y-X@bhat2).T @ (y-X@bhat2) / N
     delta = abs((cost2-cost1) / cost1)
     cost1 = cost2
     count += 1
```
![scatter](https://github.com/weinter0101/python-practice/blob/main/machine%20learning/figure/Figure3.1.png)
code

- step size
     - 以 eta*gradient 的方式進行迭代，會較快求出最適解。因為當離最適解遠時，梯度會較大而步長會更大。
     - 以 eta*d 的方式進行迭代，會較慢求出最適解。因為僅依靠學習率作為更新距離，若學習率較大，可能會在最適解附近來回跳動，無法更趨近於最適解。

- tweak learning rate
     - eta設定過大，會造成無法收斂；eta設定過小，會造成收斂過慢。
- common criteria
     - Max number of iterations
     - Min change(delta) in objective function
     - Min change(delta) in model parameters 

## 4. Model Complexity Theory
- 模型複雜度的定義
     - model complexity = number of models in the class, $\left|\mathcal{F}\right|.$
- empirical loss and risk：
     - empirical loss： 
          - $L(f) = \frac{1}{N} \sum_{i} l(f, x_i, y_i)$
          - 於訓練集中計算出來，面對已知data的預測能力。
     - risk： 
          - $R(f) = E_{x_0, y_0} \left[l(f; x_0, y_0)\right]$
          - 於新的數據計算出來，面對未知data的預測能力。
     - 當發生overfiting時，模型中的empirical loss趨近於0，但risk會非常高。
     - statistical learning theory(SLT) tries to bound the different between L( f ) and R( f ), $\forall \ f \ \in F$
- Hoeffding's inequality：
     - $P\left(\sup_{f \in \mathcal{F}} \left|R(f) - L(f)\right| \leq \varepsilon \right) > 1 - 2|\mathcal{F}| e^{-2N\varepsilon^2} = 1 - \delta$
          - $\delta \ = \ 2|\mathcal{F}| e^{-2N\varepsilon^2}$
          - $M \ = \ dim(\mathcal{F}) \ = \ \left|\mathcal{F}\right|$
          - $\varepsilon = \sqrt{\frac{\log 2M - \log \delta}{2N}} $ 
     - 提供一個bound，用來描述 |R( f )-L( f )| 小於某個概率 $\varepsilon$，涉及到模型複雜度 $\left|\mathcal{F}\right|$ 及樣本數N。
     - training error $\varepsilon$ 會隨著模型複雜度的增加而增加。( traning error就是empirical loss的具體實現 )
     - 模型複雜度 $\left|\mathcal{F}\right|$ 大，樣本數N小，此時若empirical loss很小可能只是偶然的，因為在大量的模型中進行選擇，較容易找出適合traning data的模型，但此模型面對new data時的預測能力就會比較差。
     - 模型複雜度 $\left|\mathcal{F}\right|$ 小，樣本數N大，此時若empirical loss很小是可靠的，因為複雜度較低的模型減少了overfitting的風險，而大樣本提供足夠的訊息來驗證模型的效能。
- polynomial regression models：
![scatter](https://github.com/weinter0101/python-practice/blob/main/machine%20learning/figure/Figure4.1.jpg)
     - 當樣本數N的增加，estimation variance隨之下降，代表樣本越多可以減少模型的不確定性。
     - 在正常情況下，當模型複雜度越低時，所有的損失都較高，也隨著複雜度的提升而下降。
     - 當模型過度複雜，出現overfitting時，會使的LOOCV與estimation variance飆升。
     - **選擇模型時選擇適當複雜度的模型，特別是在training data較少時。**

## 5. Penalizing Model / Complexity Model / Model Selection
- traing error increases with $M \ = \ \left|\mathcal{F}\right|$, for model m=1, 2, ..., M, we have the corresponding empirical loss,
     $L_m = \frac{1}{N} \sum_i l(f_{m}; x_i, y_i) \quad (f_m \in \mathcal{F})$
- penalize by the number of parametes 
     1. $AIC_M \ = \ L_M + \frac{2d_m}{N}$
     2. $BIC_M \ = \ L_m + \frac{log(N)}{2N}d_m$
     3. $adjusted \ R^2_m$
     **$d_m : 模型參數量$**
- model seletion as AIC and BIC
     - $Treu \ model：y_i = x_{i1} \beta_1 + x_{i2} \beta_2 + x_{i3} \beta_3$ 
          - $beta_1 \ = \ 0.5, beta_2 \ = \ 0.3 \ and \ \beta3 \ = \ 0$
     - AIC choose a model including $x_{i1}, \ x_{i2}\ and \ x_{i3}$，較傾向選擇複雜的模型。
     - BIC choose a model including $x_{i1} \ and \ x_{i2}$，較傾向選擇簡潔的模型，懲罰力度更強。
- example of The Creit Data form ISLR'
     - predictors of the credit data set:
          Income, Limit, Rating, Card, Age, Education, Gender, Student, Married, Ethnicity, Balace
     - the best models of each size for the Credit data set
     ![scatter](https://github.com/weinter0101/python-practice/blob/main/machine%20learning/figure/Figure5.1.jpg)
     **$Mallows \ C_p \ = \ \frac{SSE_p}{S^2}-(n-2p), \quad p=模型參數量$**
- Best Model Selection: fit a model by each possible combination of the $\mathcal{p}$ predictors
     - 總共有 $2^\mathcal{p}$ 種組合，每個變數都可以選擇要或不要。
     - 此模型選擇方法可以找出最優模型，但在實際操作中會因計算成本和時間成本過高而難以應用，特別是 $\mathcal{p}$ 越大的時候。
- Forward Stepwise Model Selection:
     - 尋找最佳模型的方法
          - step1：$\mathcal{M_0}$ denotes the null model, include intercept.
          - step2.1：增加一個變數至模型中，$\mathcal{M_1}$，在 $\mathcal{M_0}$ 的基礎下新增一個變數，使得目前模型的 $R^2$ 最大。
          - step2.2：增加一個變數至模型中，$\mathcal{M_2}$，在 $\mathcal{M_1}$ 的基礎下再新增一個變數，使得目前模型的 $R^2$ 最大。
          - step2.3：增加一個變數至模型中，$\mathcal{M_3}$，添加第三個變數。
          - step2.k：增加一個變數至模型中，$\mathcal{M_k}$，添加第k個變數。
          - step3：從上述k個模型中可得到 $R^2$最大者
          - step4：
     - 優點：可有效減少數據維度，步驟明確且易於執行與解釋。
     - 缺點：最優模型依賴於變數加入順序，可能無法找到 best model，因為一旦變數加入模型後就無法被移除。
     - code

```python
     import numpy as np
     import statsmodels.api as sm
     from numpy.linalg import inv

     #%% True model

     np.random.seed(20)

     N = 200
     X = np.random.randn(N, 3)
     X = sm.add_constant(X)
     beta = np.array([[1, 0.5, 0.3, 0]])
     Y = X@beta.T + np.random.randn(N, 1)

     sm.OLS(Y, X).fit().summary()
     result  = sm.OLS(Y, X).fit()
     
     # aic bic selection in true model
     for kk in range(1, 5):          #kk=1, null model
          result = sm.OLS(Y, X[:, :kk]).fit()
     print('number of regressors=', kk, 'AIC=', result.aic)

     for kk in range(1, 5):
          result = sm.OLS(Y, X[:, :kk]).fit()
     print('number of regressors=', kk, 'BIC=', result.bic)
     
     #%% OLS model

     np.random.seed(20)
     N = 500
     k = 5
     beta0 = np.array([[0.9,0,0.5,0,0.2]]).T
     X = np.random.randn(N,k)
     e = np.random.randn(N,1)
     y = X@beta0+e

     bhat = inv(X.T@X)@X.T@y

     #%% forward stepwise model selection
     
     SST = y.T@y
     remainingIndices = [0, 1, 2, 3, 4]
     selectedIndices = []
     Xk = X[:, selectedIndices]
     AIC = np.zeros((k+1,1))
     AIC[0,0] = y.T@y/N      # aic = SST/N + 2*0/N
     minAIC = np.inf
     minAIC_indices = []

     for i in range(k):
          Rsquared = np.zeros([k]) - 999      # 設定一個極小的 R squared
          for m in remainingIndices:
          '''把剩餘的解釋變數計算 R squared'''
               X1 = np.concatenate((Xk, X[:, m:m+1]), axis=1)
               bhats = inv(X1.T@X1)@X1.T@y
               SSR = (y-X1@bhats).T @ (y-X1@bhats)
               Rsquared[m] = (1-SSR/SST).item()
          '''求出使 R squared 最大的 X並加入selectedIndices中'''
          selectedIndices.append(np.argmax(Rsquared))         
          remainingIndices.remove(selectedIndices[-1])        
          Xk = X[:, selectedIndices]
          AIC[i+1, 0] = (y - Xk @ inv(Xk.T@Xk) @ Xk.T@y).T@(y - Xk @ inv(Xk.T@Xk) @ Xk.T@y) / N + 2*(i+1)/N
          print(selectedIndices)      # 得知每次選擇的 X 順序為何
     
          '''select the minimum AIC'''
          if AIC[i+1, 0] < minAIC:
               minAIC = AIC[i+1, 0]
               minAIC_indices = selectedIndices.copy()
```