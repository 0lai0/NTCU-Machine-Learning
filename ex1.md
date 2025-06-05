#挑戰1:

🧪 有監督式學習（Supervised Learning）

📌 模型：Random Forest 分類器

🔧 執行步驟簡述：
使用 creditcard.csv 資料集

資料標準化與特徵選擇（去除 Time、標準化 Amount）

使用 train_test_split() 切分訓練與測試集

應對資料不平衡問題，使用 class_weight='balanced'

建立 RandomForestClassifier 模型訓練與測試

✅ 評估結果（實際執行）：

Random Forest Evaluation:

Accuracy:         0.9995318516437859

Precision Score:  0.9576271186440678

Recall Score:     0.7635135135135135

F1 Score:         0.849624060150376

Classification Report:

    Class 0 - Precision: 1.00, Recall: 1.00
    Class 1 - Precision: 0.96, Recall: 0.76

🔍 解讀：
Accuracy 高達 99.95%，但更值得注意的是：

召回率（Recall）為 76.3%，能成功抓出大多數詐欺交易

F1 Score 達 0.85，整體模型表現穩定，對少數類別仍具備辨識能力
