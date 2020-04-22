import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as ptch

def compute_iou(box_1, box_2):
    '''
    This function takes a pair of bounding boxes and returns intersection-over-
    union (IoU) of two bounding boxes.
    '''
    hi = min([box_1[2], box_2[2]])-max([box_1[0], box_2[0]])
    hi = 0 if hi<0 else hi
    wi = min([box_1[3], box_2[3]])-max([box_1[1], box_2[1]])
    wi = 0 if wi<0 else wi
    ai = hi*wi
    a1 = (box_1[2]-box_1[0])*(box_1[3]-box_1[1])
    a2 = (box_2[2]-box_2[0])*(box_2[3]-box_2[1])
    iou = ai/(a1+a2-ai)
    assert (iou >= 0) and (iou <= 1.0)

    return iou

def compute_iou_conf(preds, gts):
    '''
    This function takes a pair of dictionaries (with our JSON format; see ex.) 
    corresponding to predicted and ground truth bounding boxes for a collection
    of images and returns the iou matrix list and confidence score matrix list 
    <preds> is a dictionary containing predicted bounding boxes and confidence
    scores for a collection of images.
    <gts> is a dictionary containing ground truth bounding boxes for a
    collection of images.
    <ious> is a list of matrices of the IoU between the predictions and ground
    truth.
    <confs> is a list of matrices of the confidence score of the predicitons
    '''
    ious = []
    confs = []
    for pred_file, pred in preds.items():
        gt = gts[pred_file]
        n_gt = len(gt)
        n_pred = len(pred)
        iou = np.zeros((n_gt, n_pred))
        conf = np.zeros((1, n_pred))
        for i in range(n_gt):
            for j in range(n_pred):
                iou[i, j] = compute_iou(pred[j][:4], gt[i])
                conf[0, j] = pred[j][4]
        ious.append(iou)
        confs.append(conf)
    
    return ious, confs
    

def compute_counts(ious, confs, iou_thr=0.5, conf_thr=0.5):
    '''
    This function takes a pair of lists corresponding to computed IoU and 
    confidence scores of predicted and ground truth bounding boxes for a 
    collection of images and returns the number of true positives, false 
    positives, and false negatives. 
    <ious> is a list of matrices of the IoU between the predictions and ground
    truth.
    <confs> is a list of matrices of the confidence score of the predicitons
    '''
    TP = 0
    FP = 0
    FN = 0
    
    for iou, conf in zip(ious, confs):
        n_gt = iou.shape[0]
        n_pred = iou.shape[1]
        if n_gt==0 and n_pred==0:
            continue
        elif n_gt==0:
            FP += (conf>=conf_thr).sum()
            continue
        elif n_pred==0:
            FN += n_gt
        else:
            iou = iou*(iou>iou_thr)*(conf>=conf_thr)
            T = iou*(iou==np.amax(iou, 0))*(iou==np.amax(iou, 1).reshape((n_gt, 1)))
            tp = (T>0).sum()
            TP += tp
            FP += (conf>=conf_thr).sum()-tp
            FN += n_gt-tp

    return TP, FP, FN

def compute_pr_curve(preds, gts, iou_thr):
    '''
    This function takes a pair of dictionaries (with our JSON format; see ex.) 
    corresponding to predicted and ground truth bounding boxes for a collection
    of images as well as an array of confidence thresholds, and returns the PR 
    curve
    '''
    iou_thr = np.array(iou_thr)
    confidence_thrs = np.sort(np.array([x[4] for fname, pred in preds.items() for x in pred]))
    n_iou = len(iou_thr)
    n_conf = len(confidence_thrs)
    tp = np.zeros((n_conf, n_iou))
    fp = np.zeros((n_conf, n_iou))
    fn = np.zeros((n_conf, n_iou))
    ious, confs = compute_iou_conf(preds, gts)
    for i, conf_thr in enumerate(confidence_thrs):
        for j in range(n_iou):
            tp[i, j], fp[i, j], fn[i, j] = compute_counts(ious, confs, iou_thr[j], conf_thr)
    p = tp/(tp+fp)
    r = tp/(tp+fn)
    
    return p, r, tp, fp, fn, confidence_thrs

if __name__=='__main__':

    # set a path for predictions and annotations:
    preds_path = '../data/hw02_preds'
    gts_path = '../data/hw02_annotations'

    # load splits:
    split_path = '../data/hw02_splits'
    file_names_train = np.load(os.path.join(split_path,'file_names_train.npy'))
    file_names_test = np.load(os.path.join(split_path,'file_names_test.npy'))

    # Set this parameter to True when you're done with algorithm development:
    done_tweaking = True

    '''
    Load training data. 
    '''
    with open(os.path.join(preds_path,'preds_train_2.json'),'r') as f:
        preds_train = json.load(f)

    with open(os.path.join(gts_path, 'annotations_train.json'),'r') as f:
        gts_train = json.load(f)

    if done_tweaking:
        '''
        Load test data.
        '''
        with open(os.path.join(preds_path,'preds_test_2.json'),'r') as f:
            preds_test = json.load(f)

        with open(os.path.join(gts_path, 'annotations_test.json'),'r') as f:
            gts_test = json.load(f)

    iou_thr = np.array([0.25, 0.5, 0.75])
    p_train, r_train, tp_train, fp_train, fn_train, confidence_thr_train = compute_pr_curve(preds_train, gts_train, iou_thr)
    if done_tweaking:
        p_test, r_test, tp_test, fp_test, fn_test, confidence_thr_test = compute_pr_curve(preds_test, gts_test, iou_thr)

    plt.figure(figsize=(12, 8), tight_layout=True)
    colors = ['r', 'g', 'b']
    for i, thr in enumerate(iou_thr):
        plt.plot(r_train[:, i], p_train[:, i], colors[i]+'-', label='train, IoU threshold '+str(thr))
        plt.plot(r_test[:, i], p_test[:, i], colors[i]+':', label='test, IoU threshold '+str(thr))
    plt.gca().legend(fontsize=12)
    plt.xlabel('Recall', fontsize=14)
    plt.ylabel('Precision', fontsize=14)
    plt.xlim(0, 0.5)
    plt.ylim(0, 0.3)
    plt.savefig(os.path.join(gts_path, 'PR_curves_2.pdf'), pad_inches=0, bbox_inches='tight')
