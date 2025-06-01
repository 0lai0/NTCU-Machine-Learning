# 監督式學習
**使用XGBoost**<br>
![image](https://github.com/piHD/NTCU-Machine-Learning/blob/main/ACS110141_ex1/result_Pic/XGBoost.png)<br>
- 透過迴圈測試不同learning rate和threshold對應其他參數調整後，以目前數據得到最佳結果。

# 非監督式學習
**使用IsolationForest**<br>
![image](https://github.com/piHD/NTCU-Machine-Learning/blob/main/ACS110141_ex1/result_Pic/IsolationForest.png)<br>
- n_estimators(樹的數量)增加，會降低整體結果，故選擇較小的數字5。
- contamination=len(fraud)/(len(fraud)+len(nonfraud))，以原始資料異常的比例，提高模型預測異常的精準度。
- max_samples=0.825(每棵樹隨機抽樣的樣本數)，是在固定其他數值後，測試出來較佳的數值。
- 在多次調整後發現，max_features=1(每棵樹使用最大特徵數)會有較佳情況，故在後續調整時維持不更動。<br>
<p>測試過程中有加上PCA降低資料維度(29->10)，以及改用Autoencoder，但未取得較佳結果數據。</p>
