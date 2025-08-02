# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Created by: jasonseu
# Created on: 2021-9-24
# Email: zhuxuelin23@gmail.com
#
# Copyright © 2021 - CPSS Group
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import os
import abc
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score, recall_score, auc
from sklearn.metrics import hamming_loss,label_ranking_loss,\
    coverage_error,label_ranking_average_precision_score,f1_score


class AverageLoss(object):
    def __init__(self):
        super(AverageLoss, self).__init__()

    def reset(self):
        self._sum = 0
        self._counter = 0

    def update(self, loss, n=0):
        self._sum += loss * n
        self._counter += n

    def compute(self):
        return self._sum / self._counter


class Meter(object):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes


    def reset(self):
        self.output = np.array([], dtype=np.float64).reshape(0, self.num_classes)
        self.label = np.array([], dtype=np.float64).reshape(0, self.num_classes)

    def update(self, scores, targets):
        self.output = np.vstack((self.output, scores))
        self.label = np.vstack((self.label, targets))



class Metric(Meter):
    def __init__(self, num_classes):
        super().__init__(num_classes)


    def accuracy_subset(self, threash=0.5):
        y_pred = self.output
        y_true = self.label
        y_pred = np.where(y_pred > threash, 1, 0)
        accuracy = accuracy_score(y_true, y_pred)
        return accuracy


    def Accuracy(self,threash=0.5):
        y_true = self.label
        y_pred = self.output
        y_pred = np.where(y_pred >= threash, 1, 0)
        count = 0
        k = 0
        for i in range(y_true.shape[0]):
            p = sum(np.logical_and(y_true[i], y_pred[i]))
            q = sum(np.logical_or(y_true[i], y_pred[i]))
            if q == 0:
                k += 1
                continue
            count += p / q
        return count / (y_true.shape[0]-k)

    def accuracy_multiclass(self):
        y_pred = self.output
        y_true = self.label
        accuracy = accuracy_score(np.argmax(y_pred, 1), np.argmax(y_true, 1))
        return accuracy

    def micfscore(self, threash=0.5, type='micro'):
        y_pred = self.output
        y_true = self.label
        y_pred = np.where(y_pred > threash, 1, 0)
        return f1_score(y_pred, y_true, average=type)

    def macfscore(self, threash=0.5, type='macro'):
        y_pred = self.output
        y_true = self.label
        y_pred = np.where(y_pred > threash, 1, 0)
        return f1_score(y_pred, y_true, average=type)

    def hamming_distance(self, threash=0.5):
        y_pred = self.output
        y_true = self.label
        y_pred = np.where(y_pred >= threash, 1, 0)
        return hamming_loss(y_true, y_pred)

    def fscore_class(self, type='micro'):
        y_pred = self.output
        y_true = self.label
        return f1_score(np.argmax(y_pred, 1), np.argmax(y_true, 1), average=type)

    def auROC(self):
        y_pred = self.output
        y_true = self.label
        row, col = y_true.shape
        temp = []
        ROC = 0
        for i in range(col):
            sigle_ROC = roc_auc_score(y_true[:, i], y_pred[:, i], average='macro', sample_weight=None)
            # print("%d th AUROC: %f"%(i,ROC))
            temp.append(sigle_ROC)

            ROC += sigle_ROC
        return ROC / (col)

    def MacroAUC(self):
        y_pred = self.output  # num_instance*num_label
        y_true = self.label  # num_instance*num_label
        num_instance, num_class = y_pred.shape
        count = np.zeros((num_class, 1))  # store the number of postive instance'score>negative instance'score
        num_P_instance = np.zeros((num_class, 1))  # number of positive instance for every label
        num_N_instance = np.zeros((num_class, 1))
        AUC = np.zeros((num_class, 1))  # for each label
        count_valid_label = 0
        for i in range(num_class):  # 第i类
            num_P_instance[i, 0] = sum(y_true[:, i] == 1)  # label,,test_target
            num_N_instance[i, 0] = num_instance - num_P_instance[i, 0]
            # exclude the label on which all instances are positive or negative,
            # leading to num_P_instance(i,1) or num_N_instance(i,1) is zero
            if num_P_instance[i, 0] == 0 or num_N_instance[i, 0] == 0:
                AUC[i, 0] = 0
                count_valid_label = count_valid_label + 1
            else:

                temp_P_Outputs = np.zeros((int(num_P_instance[i, 0]), num_class))
                temp_N_Outputs = np.zeros((int(num_N_instance[i, 0]), num_class))
                #
                temp_P_Outputs[:, i] = y_pred[y_true[:, i] == 1, i]
                temp_N_Outputs[:, i] = y_pred[y_true[:, i] == 0, i]
                for m in range(int(num_P_instance[i, 0])):
                    for n in range(int(num_N_instance[i, 0])):
                        if (temp_P_Outputs[m, i] > temp_N_Outputs[n, i]):
                            count[i, 0] = count[i, 0] + 1
                        elif (temp_P_Outputs[m, i] == temp_N_Outputs[n, i]):
                            count[i, 0] = count[i, 0] + 0.5

                AUC[i, 0] = count[i, 0] / (num_P_instance[i, 0] * num_N_instance[i, 0])
        macroAUC1 = sum(AUC) / (num_class - count_valid_label)
        return float(macroAUC1), AUC

    def avgPrecision(self):
        y_pred = self.output
        y_true = self.label
        num_instance, num_class = y_pred.shape
        precision_value = 0
        precisions = []
        for i in range(num_instance):
            p = precision_score(y_true[i, :], y_pred[i, :])
            precisions.append(p)
            precision_value += p
            # print(precision_value)
        pre_list = np.array([1.0] + precisions + [0.0])  # for get AUPRC
        # print(pre_list)
        return float(precision_value / num_instance), pre_list

    def avgRecall(self):
        y_pred = self.output
        y_true = self.label
        num_instance, num_class = y_pred.shape
        recall_value = 0
        recalls = []
        for i in range(num_instance):
            p = recall_score(y_true[i, :], y_pred[i, :])
            recalls.append(p)
            recall_value += p
        rec_list = np.array([0.0] + recalls + [1.0])  # for get AUPRC
        sorting_indices = np.argsort(rec_list)
        # print(rec_list)
        return float(recall_value / num_instance), rec_list, sorting_indices

    def getAUPRC(self):
        avgPrecision, precisions = self.avgPrecision()
        avfRecall, recalls, sorting_indices = self.avgRecall()
        # x is either increasing or decreasing
        # such as recalls[sorting_indices]
        auprc = auc(recalls[sorting_indices], precisions[sorting_indices])
        return auprc

    def cal_single_label_micro_auc(self, x, y):
        idx = np.argsort(x)  # 升序排列
        y = y[idx]
        m = 0
        n = 0
        auc = 0
        for i in range(x.shape[0]):
            if y[i] == 1:
                m += 1
                auc += n
            if y[i] == 0:
                n += 1
        auc /= (m * n)
        return auc

    def get_micro_auc(self):
        """
        :param x: the predicted outputs of the classifier, the output of the ith instance for the jth class is stored in x(i,j)
        :param y: the actual labels of the instances, if the ith instance belong to the jth class, y(i,j)=1, otherwise y(i,j)=0
        :return: the micro auc
        """
        x = self.output
        y = self.label
        n, d = x.shape
        if x.shape[0] != y.shape[0]:
            print("num of  instances for output and ground truth is different!!")
        if x.shape[1] != y.shape[1]:
            print("dim of  output and ground truth is different!!")
        x = x.reshape(n * d)
        y = y.reshape(n * d)
        auc = self.cal_single_label_micro_auc(x, y)
        return auc

    def cal_single_instance_coverage(self, x, y):
        idx = np.argsort(x)  # 升序排列
        y = y[idx]
        loc = x.shape[0]
        for i in range(x.shape[0]):
            if y[i] == 1:
                loc -= i
                break
        return loc

    def get_coverage(self,threash=0.5):
        """
        :param x: the predicted outputs of the classifier, the output of the ith instance for the jth class is stored in x(i,j)
        :param y: the actual labels of the test instances, if the ith instance belong to the jth class, y(i,j)=1, otherwise y(i,j)=0
        :return: the coverage
        """
        x = self.output
        y = self.label
        x = np.where(x > threash, 1, 0)
        n, d = x.shape
        if x.shape[0] != y.shape[0]:
            print("num of  instances for output and ground truth is different!!")
        if x.shape[1] != y.shape[1]:
            print("dim of  output and ground truth is different!!")
        cover = 0
        for i in range(n):
            cover += self.cal_single_instance_coverage(x[i], y[i])
        cover = cover / n - 1
        return cover

    def cal_single_instance_ranking_loss(self, x, y):
        idx = np.argsort(x)  # 升序排列
        y = y[idx]
        m = 0
        n = 0
        rl = 0
        for i in range(x.shape[0]):
            if y[i] == 1:
                m += 1
            if y[i] == 0:
                rl += m
                n += 1
        rl /= (m * n)
        return rl

    def get_ranking_loss(self):
        """
        :param x: the predicted outputs of the classifier, the output of the ith instance for the jth class is stored in x(i,j)
        :param y: the actual labels of the test instances, if the ith instance belong to the jth class, y(i,j)=1, otherwise x(i,j)=0
        :return: the ranking loss
        """
        x = self.output
        y = self.label
        n, d = x.shape
        if x.shape[0] != y.shape[0]:
            print("num of  instances for output and ground truth is different!!")
        if x.shape[1] != y.shape[1]:
            print("dim of  output and ground truth is different!!")
        m = 0
        rank_loss = 0
        for i in range(n):
            s = np.sum(y[i])
            if s in range(1, d):
                rank_loss += self.cal_single_instance_ranking_loss(x[i], y[i])
                m += 1
        rank_loss /= m
        return rank_loss

    def One_Error(self):
        '''
        Compute the One Error

        :return: The One Error
        '''
        output = self.output
        test_target = self.label
        instanceNum = output.shape[0]

        errorList = []
        # "One Error" needs to judge each row of the label matrix.
        # If we judge matrix by column, the information may be too little.
        for i in range(instanceNum):
            # The target label consists only of 0 and 1.
            # If the sample has no positive label, the sample is not predicted.
            if np.mean(test_target[i]) == 0.0:
                continue
            index = np.argmax(output[i])
            if test_target[i][index] == 0:
                errorList.append(1)
            else:
                errorList.append(0)
        return np.mean(np.array(errorList))

    def f1(self,threash=0.5):
        y_pred = self.output
        y_true = self.label
        y_pred = np.where(y_pred > threash, 1, 0)
        f1_e = f1_score(y_true,y_pred,average='samples')
        f1_macro = f1_score(y_true, y_pred, average='macro')
        f1_micro = f1_score(y_true, y_pred, average='micro')
        return f1_e,f1_micro,f1_macro

    def comp_pvalue(self,y_pred1,y_pred2,alpha=0.05): # [numsaples, numlabels]
        y_true = self.label
        correct_probabilities1 = [y_pred1[i, y_true[i]] for i in range(len(y_true))]
        correct_probabilities2 = [y_pred2[i, y_true[i]] for i in range(len(y_true))]
        from scipy import stats

        # 假设 `prob_model1` 和 `prob_model2` 是模型1和模型2对正确标签的预测概率
        statistic, p_value = stats.ttest_ind(correct_probabilities1, correct_probabilities2)
        print(f"T-statistic: {statistic}, P-value: {p_value}")
        if p_value < alpha:
            print(f"Reject the null hypothesis - significant difference")
        else:
            print(f"Fail to reject the null hypothesis - no significant difference")

    def save_pred(self):
        path1 = 'label.npy'
        path2 = 'RNALoc_Att.npy'
        np.save(path2, self.output)
        np.save(path1, self.label)
        print("save done")


    def main(self):
        x = self.output
        y = self.label
        AP = label_ranking_average_precision_score(y,x)
        rloss = label_ranking_loss(y,x)
        cover = coverage_error(y,x)-1
        hamm_loss = self.hamming_distance()
        acc = self.Accuracy()

        f1_e,f1_micro,f1_macro = self.f1()

        return AP,f1_e,f1_micro,f1_macro,hamm_loss,acc,rloss,cover

    def siglelabel_matix(self,p):
        TP = p[1][1]
        TN = p[0][0]
        FP = p[0][1]
        FN = p[1][0]
        # Accuracy
        acc = (TP + TN) / (TP + TN + FP + FN)

        # Sensitivity (Recall)
        sensitivity = TP / (TP + FN) if (TP + FN) != 0 else 0

        # Specificity
        specificity = TN / (TN + FP) if (TN + FP) != 0 else 0

        # F1-score (Macro)
        precision = TP / (TP + FP) if (TP + FP) != 0 else 0
        recall = sensitivity
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

        # MCC (Matthews Correlation Coefficient)
        mcc = (TP * TN - FP * FN) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) if (TP + FP) * (TP + FN) * (
                    TN + FP) * (TN + FN) != 0 else 0
        return acc, sensitivity, specificity, f1, mcc

    def label_matrix(self):
        from sklearn.metrics import multilabel_confusion_matrix
        from sklearn.metrics import classification_report
        y_pred = self.output
        y_true = self.label

        y_pred = np.where(y_pred >= .5, 1, 0)
        #LABEL_COLUMNS = ['Nucleus','Cytoplasm']
        LABEL_COLUMNS = ['Exosome','Nucleus','Nucleoplasm','Chromatin','Nucleolus','Cytosol','Membrane','Ribosome','Cytoplasm']
        results = []
        print(classification_report(
            y_true,
            y_pred,
            digits=4,
            target_names=LABEL_COLUMNS,
            zero_division=0
        ))
        for i in range(len(LABEL_COLUMNS)):
            acc, sensitivity, specificity, f1, mcc = self.siglelabel_matix(
                multilabel_confusion_matrix(y_true, y_pred)[i])
            results.append([LABEL_COLUMNS[i], acc, sensitivity, specificity, f1, mcc])

            # Create a DataFrame from the results list
        df = pd.DataFrame(results, columns=["Label", "Accuracy", "Sensitivity", "Specificity", "F1-score", "MCC"])

        # Save to Excel file
        with pd.ExcelWriter("res.xlsx", engine="xlsxwriter") as writer:
            df.to_excel(writer, sheet_name="Evaluation Metrics", index=False)

    def plot_matrix(self):
        from sklearn.metrics import multilabel_confusion_matrix
        import matplotlib.pyplot as plt
        import numpy
        from sklearn import metrics
        y_pred = self.output
        y_true = self.label

        y_pred = np.where(y_pred >= .5, 1, 0)
        conf_mat = multilabel_confusion_matrix(y_true, y_pred)

        Exosome = conf_mat[0]
        Nucleus = conf_mat[1]
        Nucleoplasm = conf_mat[2]
        Chromatin = conf_mat[3]
        Cytoplasm = conf_mat[4]
        Nucleolus = conf_mat[5]
        Cytosol = conf_mat[6]
        Membrane = conf_mat[7]
        Ribosome = conf_mat[8]

        cm_Exosome = metrics.ConfusionMatrixDisplay(confusion_matrix=Exosome, display_labels=['Not Exosome', 'Exosome'])
        cm_Nucleus = metrics.ConfusionMatrixDisplay(confusion_matrix=Nucleus, display_labels=['Not Nucleus', 'Nucleus'])
        cm_Nucleoplasm = metrics.ConfusionMatrixDisplay(confusion_matrix=Nucleoplasm, display_labels=['Not Nucleoplasm', 'Nucleoplasm'])
        cm_Chromatin = metrics.ConfusionMatrixDisplay(confusion_matrix=Chromatin, display_labels=['Not Chromatin', 'Chromatin'])
        cm_Cytoplasm = metrics.ConfusionMatrixDisplay(confusion_matrix=Cytoplasm, display_labels=['Not Cytoplasm', 'Cytoplasm'])
        cm_Nucleolus = metrics.ConfusionMatrixDisplay(confusion_matrix=Nucleolus, display_labels=['Not Nucleolus', 'Nucleolus'])
        cm_Cytosol = metrics.ConfusionMatrixDisplay(confusion_matrix=Cytosol, display_labels=['Not Cytosol', 'Cytosol'])
        cm_Membrane = metrics.ConfusionMatrixDisplay(confusion_matrix=Membrane, display_labels=['Not Membrane', 'Membrane'])
        cm_Ribosome = metrics.ConfusionMatrixDisplay(confusion_matrix=Ribosome, display_labels=['Not Ribosome', 'Ribosome'])


        cm_Exosome.plot()
        cm_Nucleus.plot()
        cm_Nucleoplasm.plot()
        cm_Chromatin.plot()
        cm_Cytoplasm.plot()
        cm_Nucleolus.plot()
        cm_Cytosol.plot()
        cm_Membrane.plot()
        cm_Ribosome.plot()

        plt.show()

    def polt_prob(self):
        import pandas as pd
        import matplotlib.pyplot as plt
        LABEL_COLUMNS = ['Exosome', 'Nucleus', 'Nucleoplasm', 'Chromatin', 'Nucleolus', 'Cytosol', 'Membrane',
                         'Ribosome', 'Cytoplasm']
        y_pred = self.output
        df = pd.DataFrame(y_pred, columns=LABEL_COLUMNS)
        plt.figure(figsize=(10, 6))

        # Plot a density plot for each label's probabilities
        for label in LABEL_COLUMNS:
            df[label].plot(kind='density', label=label)

        plt.title('Label-wise Probability Distribution (Density Plot)')
        plt.xlabel('Predicted Probability')
        plt.ylabel('Density')
        plt.legend(title='Labels')
        plt.grid(False)
        plt.show()