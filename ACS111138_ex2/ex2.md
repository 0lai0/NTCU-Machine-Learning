ex2說明文件

實驗過程中我加入了很多方法，例如

1.PCA：降維處理，去除冗餘特徵，保留重要元件

2.LOF (Local Outlier Factor)：透過鄰近點密度判斷、偵測潛在異常樣本，為非監督式偵測異常的方法

3.AutoEncoder：以重建誤差作為額外異常判斷依據，提供額外的異常特徵

4.SMOTE：針對不平衡資料進行採樣合成

5.CNN（Convolutional Neural Network）：嘗試以卷積神經網路從特徵空間學習

其中，SMOTE 雖然理論上可以改善資料不平衡的問題，CNN 也具備強大的特徵學習能力，但在本資料集中卻使 F1-score 降到約 0.74。推測原因可能是：CNN 較適合處理具有空間結構或複雜特徵的資料，例如圖片，對於本資料集這類數值型特徵不具優勢；而 SMOTE 雖可用來合成詐欺樣本，但容易造成 overfitting，導致模型在測試資料上表現不佳。

一開始先使用isolation+XGBoost，f1-score只有0.78，後來透過調整參數並且加入PCA、LOF、AutoEncoder三個方法慢慢調整到0.85
我先使用 RandomizedSearchCV 自動搜尋出最佳 XGBoost 組合
後面固定最佳組合後加入PCA，降到20維，有效減少雜訊並保留重要元件，有助於整體預測
再加入LOF，偵測淺在異常點，將異常點分數與標籤作為額外特徵分數提供給XGBoost
最後透過CNN的失敗我想到可以新增AutoEncoder，寫了一個簡單四維的AutoEncoder，將資料壓縮再還原的神經網路模型

AutoEncoder壓縮再還原後
如果資料為正常的，模型可以很接近的還原回原始資料的樣子
但如果資料是不正常的(詐騙)，他跟正常交易不同就會還原得比較差，可以理解為"重建誤差"，最後我再將其當作異常的分數加到最終特徵裡

最終模型整合了PCA、Isolation Forest、LOF、AutoEncoder再搭配先前最佳化的 XGBoost 參數組合成功訓練出比一開始0.78更高的0.85的f1-score

![image](https://github.com/user-attachments/assets/2710bea8-204b-4a4e-a547-8d07deabab8b)
