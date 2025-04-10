# COMP3027J-ASSIGNMENT-1
### this are the source codes of assignment in COMP3027J: Data Mining and Machine Learning.
### These achieves plant seedlings classification task by improved DeiT model.
## Author: Haoyan Shi
## Dataset: https://vision.eng.au.dk/plant-seedlings-dataset
## 📊 Result:
---

| Class                        | Precision | Recall | F1-Score | Support |
|-----------------------------|-----------|--------|----------|---------|
| Black-grass                 | 0.85      | 0.93   | 0.89     | 56      |
| Charlock                    | 1.00      | 0.98   | 0.99     | 66      |
| Cleavers                    | 0.97      | 0.99   | 0.98     | 73      |
| Common Chickweed            | 0.95      | 0.98   | 0.97     | 57      |
| Common wheat                | 0.95      | 0.98   | 0.97     | 57      |
| Fat Hen                     | 1.00      | 0.96   | 0.98     | 56      |
| Loose Silky-bent            | 0.92      | 0.88   | 0.90     | 69      |
| Maize                       | 0.98      | 0.97   | 0.98     | 63      |
| Scentless Mayweed           | 1.00      | 0.97   | 0.98     | 59      |
| Shepherds Purse             | 1.00      | 0.98   | 0.99     | 48      |
| Small-flowered Cranesbill   | 0.98      | 0.97   | 0.97     | 60      |
| Sugar beet                  | 0.98      | 1.00   | 0.99     | 58      |

---
## Overall Metrics

- **Accuracy**: 0.97  
- **Macro Average**: Precision: 0.97, Recall: 0.97, F1-Score: 0.97  
- **Weighted Average**: Precision: 0.97, Recall: 0.97, F1-Score: 0.97

---

## 🧠 Model Info

> _Details about the model architecture, training strategy, dataset, and preprocessing steps can be added here if needed._

---

## 📌 Notes
- pre-train.py used to train teacher models.
- train.py used to train DeiT models.
- val.py used to achieve evaluate method.
- test.py used to test trained-DeiT
- options.py used to give parameters to other files
