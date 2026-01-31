from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import numpy as np

class PerformanceMetrics:
    def __init__(self, y_true, y_pred, labels):
        self.y_true = y_true
        self.y_pred = y_pred
        self.labels = labels
      
    def Class_PerformanceMetrics(self, label):
        true_ = np.array(self.y_true == label).astype(int)
        pred_ = np.array(self.y_pred == label).astype(int)
        cm_ = confusion_matrix(true_, pred_)
        tn_, fp_, fn_, tp_ = cm_.ravel()
        class_metrics = {
            'true_negative': tn_,
            'false_positive': fp_,
            'false_negative': fn_,
            'true_positive': tp_,
            'confusion_matrix': cm_,
            'sensitivity': recall_score(true_, pred_),
            'specificity': tn_ / (tn_ + fp_),
            'precision': precision_score(true_, pred_),
            'f1_score': f1_score(true_, pred_),
            'accuracy': accuracy_score(true_, pred_),
            'fpr': fp_ / (tn_ + fp_), # false positive rate
            'fnr': fn_ / (tn_ + fn_) # false negative rate
        }
        return class_metrics
    
    def Overall_PerformanceMetrics(self):
        tn = fp = fn = tp = 0
        for l in self.labels:
            class_metrics = self.Class_PerformanceMetrics(label=l)
            tn += class_metrics['true_negative']
            fp += class_metrics['false_positive']
            fn += class_metrics['false_negative']
            tp += class_metrics['true_positive']
        overall_metrics = {
            'true_negative': tn,
            'false_positive': fp,
            'false_negative': fn,
            'true_positive': tp,
            'confusion_matrix': confusion_matrix(self.y_true, self.y_pred),
            'sensitivity': recall_score(self.y_true, self.y_pred, average='micro'),
            'specificity': tn / (tn + fp),
            'precision': precision_score(self.y_true, self.y_pred, average='micro'),
            'f1_score': f1_score(self.y_true, self.y_pred, average='micro'),
            'accuracy': accuracy_score(self.y_true, self.y_pred),
            'fpr': fp / (tn + fp),
            'fnr': fn / (tn + fn)
        }
        return overall_metrics
    
    def Print_PerformanceMetrics(self):
        m = self.Overall_PerformanceMetrics()
        for i in m:
            if i == 'confusion_matrix':  
                print("Confusion Matrix:")
                for r in m[i]:  
                    print(" ".join(f"{val:5d}" for val in r))  
            else:
                print(f"{i}: {m[i]}")