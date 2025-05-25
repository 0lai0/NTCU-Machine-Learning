挑戰1.
改善random forest模型的輸出數據結果:

1.Random Forest 調參數 (Tuned Random Forest) 
使用 Random Forest 分類器來偵測信用卡詐欺。

2.調整模型參數以改善分類效果，主要參數如下：

n_estimators=100：森林中樹的數量，提升模型穩定性。

max_depth=10：限制樹的最大深度，避免過度擬合。

min_samples_split=10：節點分裂所需的最少樣本數，控制模型複雜度。

class_weight='balanced'：解決資料類別不平衡問題，對少數類別（詐欺）加權。


使用分層抽樣將資料切分為訓練集和測試集，確保正負樣本比例一致。

評估指標包括 Accuracy、Precision、Recall 與 F1-score。

調參後模型相較於預設參數模型，在召回率（Recall）及整體 F1-score 有明顯提升，更能有效辨識詐欺樣本。

3.結果
Tuned Random Forest Evaluation:
=============================================
         Accuracy: 0.9994850368081645
  Precision Score: 0.8333333333333334
     Recall Score: 0.8455882352941176
         F1 Score: 0.8394160583941606

Classification Report:
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     85307
           1       0.83      0.85      0.84       136

    accuracy                           1.00     85443
   macro avg       0.92      0.92      0.92     85443
weighted avg       1.00      1.00      1.00     85443

4.比較:
Random Forest Evaluation:
=============================================
         Accuracy: 0.9996371850239341
  Precision Score: 0.9411764705882353
     Recall Score: 0.8235294117647058
         F1 Score: 0.8784313725490196

Classification Report:
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     85307
           1       0.94      0.82      0.88       136

    accuracy                           1.00     85443
   macro avg       0.97      0.91      0.94     85443
weighted avg       1.00      1.00      1.00     85443


挑戰1.
改善kmeans模型的輸出數據結果:

1.KMeans + PCA (監督式學習捨棄)
使用 KMeans 無監督分群方法來嘗試偵測詐欺交易。

由於資料維度高，先透過 PCA（主成分分析） 進行降維，保留 90% 以上的資訊，降低運算成本與噪音影響。

只用正常交易（非詐欺）子集作為 KMeans 的訓練資料，因無標籤資料且異常樣本少，訓練時不考慮詐欺樣本。

2.透過 Silhouette Score 找出最佳的叢集數 (k 值)，範圍在 2 到 4 之間。

叢集結果與真實標籤對齊，推估各叢集所代表的類別。

最後在測試集上評估，計算 Accuracy、Precision、Recall、F1-score，並輸出 Silhouette Score 以評估叢集品質。

3.分析:
優點是不用依賴標註資料，缺點是偵測性能較監督式模型低，且易受參數設定影響。對於原先的結果來說，PCA的recall score、f1 score過低

Best K value: 4

KMeans + PCA Evaluation:
=============================================
         Accuracy: 0.9987242957293166
  Precision Score: 0.782608695652174
     Recall Score: 0.36486486486486486
         F1 Score: 0.4976958525345622

Classification Report:
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     85295
           1       0.78      0.36      0.50       148

    accuracy                           1.00     85443
   macro avg       0.89      0.68      0.75     85443
weighted avg       1.00      1.00      1.00     85443


Silhouette Score on test set: 0.1075

4.考慮:
在選擇資料前的結果過於理想化了，導致在調整參數者的過程中我認為脫離了非監督式學習的範疇，故改用PCA取樣