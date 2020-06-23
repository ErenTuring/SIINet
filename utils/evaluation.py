'''
Evaluation.

'''
import numpy as np
from skimage import morphology


def fast_hist(label_true, label_pred, n_class):
    '''Computational confusion matrix.
    -------------------------------------------
    |          | p_cls_1 | p_cls_2 |   ....   |
    -------------------------------------------
    | gt_cls_1 |         |         |          |
    -------------------------------------------
    | gt_cls_2 |         |         |          |
    -------------------------------------------
    |   ....   |         |         |          |
    -------------------------------------------
    '''
    # mask = (label_true >= 0) & (label_true < n_class)
    if len(label_true.shape) > 1:
        label_true = label_true.flatten()
        label_pred = label_pred.flatten()
    hist = np.bincount(
        n_class * label_true.astype(int) + label_pred,
        minlength=n_class ** 2,
    ).reshape(n_class, n_class)
    return hist


class runingScore(object):
    ''' Evaluation class '''
    def __init__(self, n_classes=2):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes), dtype=np.int64)

    def reset(self):
        ''' Reset confusion_matrix. '''
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes), dtype=np.int64)

    def update_all(self, label_trues, label_preds):
        ''' Add new pairs of predicted label and GT label to update the confusion_matrix.
        Note: Only suitable for segmentation
        '''
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += fast_hist(lt, lp, self.n_classes)

    def print_score(self, score, mode=0):
        ''' Print the score dict.
        mode-0: print the final total scores
        mode-1: print per pair of data's scores
        '''
        str_score = ''
        for key in score:
            if 'Class' in key:
                value_str = ','.join('%.4f' % i for i in score[key])
            else:
                value_str = '%.4f' % score[key]
            str_score += '%s,' % value_str if mode else key+': %s\n' % value_str
        str_score = str_score.strip(',').strip()  # discard the last suffix
        if mode == 0:
            print(str_score)

        return str_score


class RoadExtractionScore(runingScore):
    '''Accuracy evaluation for road extraction.
    Only two class: 0-bg, 1-road.
    '''

    def update(self, label_true, label_pred):
        '''Evaluate a new pair of predicted label and GT label,
        and update the confusion_matrix. '''
        hist = fast_hist(label_true, label_pred, self.n_classes)
        self.confusion_matrix += hist
        return self.get_scores(hist)

    def add(self, label_true, label_pred):
        '''Add a new pair of predicted label and GT label,
        update the confusion_matrix. '''
        hist = fast_hist(label_true, label_pred, self.n_classes)
        self.confusion_matrix += hist

    def get_scores(self, hist=None):
        """Returns accuracy score evaluation result.
            - 1. Precision{ TP / (TP+FP) }
            - 2. Recall{ TP / (TP+FN) }
            - 3. F1score
            - 4. Class IoU
            - 5. Mean IoU
            - 6. FreqW Acc
        """
        hist = self.confusion_matrix if hist is None else hist

        # Take class 1-road as postive class:
        TP = hist[1, 1]  # Ture Positive(road pixels are classified into road class)
        FN = hist[1, 0]  # False Negative(road pixels are classified into bg class)
        FP = hist[0, 1]  # False Positive(bg pixels are classified into road class)
        # TN = hist[0, 0]  # Ture Negative(bg pixels are classified into bg class)

        prec = TP / (TP + FP + 1e-8)  # Precision
        rec = TP / (TP + FN + 1e-8)  # Recall
        F1 = 2*TP / (2*TP + FP + FN + 1e-8)  # F1 Score

        # IoU (tested)
        cls_iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(cls_iu)
        # Frequency Weighted IoU(FWIoU) 根据每个类出现的频率为其设置权重
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * cls_iu[freq > 0]).sum()
        # cls_iu = dict(zip(range(self.n_classes), iu))

        return (
            {
                'Precision': prec,
                'Recall': rec,
                'F1score': F1,
                'Class IoU': cls_iu,
                'Mean IoU': mean_iu,
                'FreqW Acc': fwavacc,
            }  # Return as a dictionary
        )

    def keys(self):
        score_keys = [
            'Precision,Recall,F1score,Class IoU,Class IoU,Mean IoU,FreqW Acc'
        ]  # note 'Class IoU'
        return score_keys


class RelaxedRoadExtractionScore(runingScore):
    '''Relax Accuracy evaluation for road extraction.
    Only two class: 0-bg, 1-road.
    '''
    def __init__(self, rho=1):
        self.rho = rho*2 + 1
        self.confusion_matrix_p = np.zeros((2, 2), np.int64)  # For relaxed precision
        self.confusion_matrix_r = np.zeros((2, 2), np.int64)  # For relaxed recall

    def update(self, label_true, label_pred):
        '''Evaluate a new pair of predicted label and GT label,
        and update the confusion_matrix.'''
        if self.rho > 1:
            selem = morphology.square(self.rho, dtype=label_true.dtype)
            tp_label_true = morphology.dilation(label_true, selem)
            tp_label_pred = morphology.binary_dilation(label_pred, selem)
            hist1 = fast_hist(tp_label_true, label_pred, 2)
            hist2 = fast_hist(label_true, tp_label_pred, 2)
        else:
            hist = fast_hist(label_true, label_pred, 2)
            hist1, hist2 = hist, hist

        self.confusion_matrix_p += hist1
        self.confusion_matrix_r += hist2
        return self.get_scores(hist1, hist2)

    def add(self, label_true, label_pred):
        ''' Add new pairs of predicted label and GT label to update the confusion_matrix. '''
        if self.rho > 0:
            selem = morphology.square(self.rho, dtype=np.int64)
            tp_lt = morphology.binary_dilation(label_true, selem)
            tp_lp = morphology.binary_dilation(label_pred, selem)
            self.confusion_matrix_p += fast_hist(tp_lt, label_pred, 2)
            self.confusion_matrix_r += fast_hist(label_true, tp_lp, 2)
        else:
            hist = fast_hist(label_true, label_pred, 2)
            self.confusion_matrix_p += hist
            self.confusion_matrix_r += hist

    def get_scores(self, hist_p=None, hist_r=None):
        hist_p = self.confusion_matrix_p if hist_p is None else hist_p
        hist_r = self.confusion_matrix_r if hist_r is None else hist_r

        prec = hist_p[1, 1] / (hist_p[1, 1] + hist_p[0, 1] + 1e-8)  # Precision
        rec = hist_r[1, 1] / (hist_r[1, 1] + hist_r[1, 0] + 1e-8)  # Recall
        f1 = 2 * prec * rec / (prec + rec)
        return (
            {
                "Precision": prec,
                "Recall": rec,
                "F1score": f1
            }  # Return as a dictionary
        )

    def reset(self):
        ''' Reset confusion_matrixs. '''
        self.confusion_matrix_p = np.zeros((2, 2), dtype=np.int64)
        self.confusion_matrix_r = np.zeros((2, 2), dtype=np.int64)
