#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Athul
# @Date:   2015-10-29 12:12:09
# @Last Modified by:   Athul
# @Last Modified time: 2015-10-29 12:29:35
import sklearn.metrics as metrics

accuracy = metrics.accuracy_score(trueLabels, predLabels)
avg_precision = metrics.average_precision_score(trueLabels, predLabels)
f1_score = metrics.f1_score(trueLabels, predLabels)
recall_score = metrics.f1_score(trueLabels, predLabels)
conf_mat = metrics.confusion_matrix(trueLabels, predLabels)


def plotROC(trueLabels, predLabels):
    