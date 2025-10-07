#!/usr/bin/env python
# -*- coding: utf-8 -*-
from collections import namedtuple
import numpy as np
from shapely.geometry import Polygon
"""
cite from:
PaddleOCR, github: https://github.com/PaddlePaddle/PaddleOCR
PaddleOCR reference from :
https://github.com/MhLiao/DB/blob/3c32b808d4412680310d3d28eeb6a2d5bf1566c5/concern/icdar2015_eval/detection/iou.py#L8
"""


class DetectionIoUEvaluator(object):
    def __init__(self, iou_constraint=0.5, area_precision_constraint=0.5):
        self.iou_constraint = iou_constraint
        self.area_precision_constraint = area_precision_constraint

    def evaluate_image(self, gt, pred):
        def get_union(pD, pG):
            return Polygon(pD).union(Polygon(pG)).area

        def get_intersection_over_union(pD, pG):
            return get_intersection(pD, pG) / get_union(pD, pG)

        def get_intersection(pD, pG):
            return Polygon(pD).intersection(Polygon(pG)).area

        detMatched = 0
        gtPols = []
        detPols = []
        gtPolPoints = []
        detPolPoints = []
        gtDontCarePolsNum = set()
        detDontCarePolsNum = set()
        pairs = []
        evaluationLog = ""

        for n in range(len(gt)):
            points = gt[n]['points']
            dontCare = gt[n]['ignore']
            try:
                poly = Polygon(points)
                if not poly.is_valid or not poly.is_simple:
                    continue
            except:
                continue

            gtPols.append(points)
            gtPolPoints.append(points)
            if dontCare:
                gtDontCarePolsNum.add(len(gtPols) - 1)

        evaluationLog += f"GT polygons: {len(gtPols)}"
        if gtDontCarePolsNum:
            evaluationLog += f" ({len(gtDontCarePolsNum)} don't care)\n"
        else:
            evaluationLog += "\n"

        for n in range(len(pred)):
            points = pred[n]['points']
            
            try:
                poly = Polygon(points)
                if not poly.is_valid or not poly.is_simple:
                    continue
            except:
                continue

            detPols.append(points)  # Keep original points for compatibility
            detPolPoints.append(points)
            
            # Check against don't care GT polygons (optimized)
            if gtDontCarePolsNum:
                for dontCareIdx in gtDontCarePolsNum:
                    dontCarePol = gtPols[dontCareIdx]
                    intersected_area = get_intersection(dontCarePol, points)
                    pdDimensions = Polygon(points).area
                    precision_score = intersected_area / pdDimensions if pdDimensions > 0 else 0
                    if precision_score > self.area_precision_constraint:
                        detDontCarePolsNum.add(len(detPols) - 1)
                        break

        evaluationLog += f"DET polygons: {len(detPols)}"
        if detDontCarePolsNum:
            evaluationLog += f" ({len(detDontCarePolsNum)} don't care)\n"
        else:
            evaluationLog += "\n"

        # Calculate IoU matrix and find matches (optimized)
        iouMat = np.empty([1, 1])  # Default empty matrix
        
        if len(gtPols) > 0 and len(detPols) > 0:
            # Pre-allocate arrays
            outputShape = [len(gtPols), len(detPols)]
            iouMat = np.zeros(outputShape)
            gtRectMat = np.zeros(len(gtPols), dtype=bool)
            detRectMat = np.zeros(len(detPols), dtype=bool)
            
            # Calculate IoU matrix
            for gtNum in range(len(gtPols)):
                if gtNum in gtDontCarePolsNum:
                    continue
                for detNum in range(len(detPols)):
                    if detNum in detDontCarePolsNum:
                        continue
                    pG = gtPols[gtNum]
                    pD = detPols[detNum]
                    iouMat[gtNum, detNum] = get_intersection_over_union(pD, pG)

            # Find matches using optimized algorithm
            for gtNum in range(len(gtPols)):
                if gtRectMat[gtNum] or gtNum in gtDontCarePolsNum:
                    continue
                    
                for detNum in range(len(detPols)):
                    if (detRectMat[detNum] or 
                        detNum in detDontCarePolsNum or
                        iouMat[gtNum, detNum] <= self.iou_constraint):
                        continue
                    
                    gtRectMat[gtNum] = True
                    detRectMat[detNum] = True
                    detMatched += 1
                    pairs.append({'gt': gtNum, 'det': detNum})
                    evaluationLog += f"Match GT #{gtNum} with Det #{detNum}\n"
                    break

        numGtCare = len(gtPols) - len(gtDontCarePolsNum)
        numDetCare = len(detPols) - len(detDontCarePolsNum)
        if numGtCare == 0:
            recall = float(1)
            precision = float(0) if numDetCare > 0 else float(1)
        else:
            recall = float(detMatched) / numGtCare
            precision = 0 if numDetCare == 0 else float(detMatched) / numDetCare

        hmean = 0 if (precision + recall) == 0 else 2.0 * precision * recall / (precision + recall)

        return {
            'precision': precision,
            'recall': recall,
            'hmean': hmean,
            'pairs': pairs,
            'iouMat': [] if len(detPols) > 100 else iouMat.tolist(),
            'gtPolPoints': gtPolPoints,
            'detPolPoints': detPolPoints,
            'gtCare': numGtCare,
            'detCare': numDetCare,
            'gtDontCare': list(gtDontCarePolsNum),
            'detDontCare': list(detDontCarePolsNum),
            'detMatched': detMatched,
            'evaluationLog': evaluationLog
        }

    def combine_results(self, results):
        numGlobalCareGt = 0
        numGlobalCareDet = 0
        matchedSum = 0
        for result in results:
            numGlobalCareGt += result['gtCare']
            numGlobalCareDet += result['detCare']
            matchedSum += result['detMatched']

        methodRecall = 0 if numGlobalCareGt == 0 else float(
            matchedSum) / numGlobalCareGt
        methodPrecision = 0 if numGlobalCareDet == 0 else float(
            matchedSum) / numGlobalCareDet
        methodHmean = 0 if methodRecall + methodPrecision == 0 else 2 * \
                                                                    methodRecall * methodPrecision / (
                                                                            methodRecall + methodPrecision)
        # print(methodRecall, methodPrecision, methodHmean)
        # sys.exit(-1)
        methodMetrics = {
            'precision': methodPrecision,
            'recall': methodRecall,
            'hmean': methodHmean
        }

        return methodMetrics


if __name__ == '__main__':
    evaluator = DetectionIoUEvaluator()
    gts = [[{
        'points': [(0, 0), (1, 0), (1, 1), (0, 1)],
        'text': 1234,
        'ignore': False,
    }, {
        'points': [(2, 2), (3, 2), (3, 3), (2, 3)],
        'text': 5678,
        'ignore': False,
    }]]
    preds = [[{
        'points': [(0.1, 0.1), (1, 0), (1, 1), (0, 1)],
        'text': 123,
        'ignore': False,
    }]]
    results = []
    for gt, pred in zip(gts, preds):
        results.append(evaluator.evaluate_image(gt, pred))
    metrics = evaluator.combine_results(results)
    print(metrics)
