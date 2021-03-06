import numpy as np
from other_metrics import accuracy_score_, precision_score_, recall_score_, f1_score_
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# Example 2:

y_hat = np.array(['norminet', 'dog', 'norminet', 'norminet', 'dog', 'dog', 'dog', 'dog']).reshape(-1, 1)
y = np.array(['dog', 'dog', 'norminet', 'norminet', 'dog', 'norminet', 'dog', 'norminet']).reshape(-1, 1)
# Accuracy
## your implementation
print(accuracy_score_(y, y_hat))
## Output:
# 0.625
## sklearn implementation
print(accuracy_score(y, y_hat))
## Output:
# 0.625
# Precision
## your implementation
print(precision_score_(y, y_hat, pos_label='dog'))
## Output:
# 0.6
## sklearn implementation
print(precision_score(y, y_hat, pos_label='dog'))
## Output:
# 0.6
# Recall
## your implementation
print(recall_score_(y, y_hat, pos_label='dog'))
## Output:
# 0.75
## sklearn implementation
print(recall_score(y, y_hat, pos_label='dog'))
## Output:
# 0.75
# F1-score
## your implementation
print(f1_score_(y, y_hat, pos_label='dog'))
## Output:
# 0.6666666666666665
## sklearn implementation
print(f1_score(y, y_hat, pos_label='dog'))
## Output:
# 0.6666666666666665
