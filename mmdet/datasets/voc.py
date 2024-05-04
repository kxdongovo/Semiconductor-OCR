from collections import OrderedDict

from mmcv.utils import print_log

from mmdet.core import eval_map, eval_recalls
from .builder import DATASETS
from .xml_style import XMLDataset


@DATASETS.register_module()
class VOCDataset(XMLDataset):

    CLASSES = ('0-0','0-90','0-180','0-270','1-0','1-90','1-180','1-270','2-0','2-90','2-180','2-270','3-0','3-90','3-180','3-270',
'4-0','4-90','4-180','4-270','5-0','5-90','5-180','5-270', '6-0','6-90','6-180','6-270','7-0','7-90','7-180','7-270','8-0','8-90','8-180','8-270',
'9-0','9-90','9-180','9-270','A-0','A-90','A-180','A-270','B-0','B-90','B-180','B-270','C-0','C-90','C-180','C-270','D-0','D-90','D-180','D-270',
'E-0','E-90','E-180','E-270','F-0','F-90','F-180','F-270','G-0','G-90','G-180','G-270','H-0','H-90','H-180','H-270','I-0','I-90','I-180','I-270','J-0',
'J-90','J-180','J-270','K-0','K-90','K-180','K-270','L-0','L-90','L-180','L-270','M-0','M-90','M-180','M-270','N-0','N-90','N-180','N-270',
'O-0','O-90','O-180','O-270','P-0','P-90','P-180','P-270','Q-0','Q-90','Q-180','Q-270','R-0','R-90','R-180','R-270','S-0','S-90','S-180','S-270',
'T-0','T-90','T-180','T-270','U-0','U-90','U-180','U-270','V-0','V-90','V-180','V-270','W-0','W-90','W-180','W-270','X-0','X-90','X-180','X-270',
'Y-0','Y-90','Y-180','Y-270','Z-0','Z-90','Z-180','Z-270')

    def __init__(self, **kwargs):
        super(VOCDataset, self).__init__(**kwargs)
        if 'VOC2007' in self.img_prefix:
            self.year = 2007
        elif 'VOC2012' in self.img_prefix:
            self.year = 2012
        else:
            raise ValueError('Cannot infer dataset year from img_prefix')

    def evaluate(self,
                 results,
                 metric='mAP',
                 logger=None,
                 proposal_nums=(100, 300, 1000),
                 iou_thr=0.5,
                 scale_ranges=None):
        """Evaluate in VOC protocol.

        Args:
            results (list[list | tuple]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. Options are
                'mAP', 'recall'.
            logger (logging.Logger | str, optional): Logger used for printing
                related information during evaluation. Default: None.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thr (float | list[float]): IoU threshold. Default: 0.5.
            scale_ranges (list[tuple], optional): Scale ranges for evaluating
                mAP. If not specified, all bounding boxes would be included in
                evaluation. Default: None.

        Returns:
            dict[str, float]: AP/recall metrics.
        """

        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        allowed_metrics = ['mAP', 'recall']
        if metric not in allowed_metrics:
            raise KeyError(f'metric {metric} is not supported')
        annotations = [self.get_ann_info(i) for i in range(len(self))]
        eval_results = OrderedDict()
        iou_thrs = [iou_thr] if isinstance(iou_thr, float) else iou_thr
        if metric == 'mAP':
            assert isinstance(iou_thrs, list)
            if self.year == 2007:
                ds_name = 'voc07'
            else:
                ds_name = self.CLASSES
            mean_aps = []
            for iou_thr in iou_thrs:
                print_log(f'\n{"-" * 15}iou_thr: {iou_thr}{"-" * 15}')
                mean_ap, _ = eval_map(
                    results,
                    annotations,
                    scale_ranges=None,
                    iou_thr=iou_thr,
                    dataset=ds_name,
                    logger=logger)
                mean_aps.append(mean_ap)
                eval_results[f'AP{int(iou_thr * 100):02d}'] = round(mean_ap, 3)
            eval_results['mAP'] = sum(mean_aps) / len(mean_aps)
        elif metric == 'recall':
            gt_bboxes = [ann['bboxes'] for ann in annotations]
            recalls = eval_recalls(
                gt_bboxes, results, proposal_nums, iou_thr, logger=logger)
            for i, num in enumerate(proposal_nums):
                for j, iou in enumerate(iou_thr):
                    eval_results[f'recall@{num}@{iou}'] = recalls[i, j]
            if recalls.shape[1] > 1:
                ar = recalls.mean(axis=1)
                for i, num in enumerate(proposal_nums):
                    eval_results[f'AR@{num}'] = ar[i]
        return eval_results
