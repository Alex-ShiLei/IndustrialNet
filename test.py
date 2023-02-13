r""" IndustrialNetwork testing code """
import argparse
import torch
from model.indus import IndustrialNetwork
from common.logger import Logger, AverageMeter
from common.vis import Visualizer
from common.evaluation import Evaluator
from common import utils
from data.dataset import FSSDataset
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def test(model, dataloader, nshot):
    r""" Test IndustrialNetwork """

    # Freeze randomness during testing for reproducibility
    utils.fix_randseed(0)
    average_meter = AverageMeter(dataloader.dataset)

    for idx, batch in enumerate(dataloader):
        # 1. Hypercorrelation Squeeze Networks forward pass
        batch = utils.to_cuda(batch)
        pred_mask = model.predict_mask_nshot(batch, nshot=nshot)

        assert pred_mask.size() == batch['query_mask'].size()

        # 2. Evaluate prediction
        area_inter, area_union = Evaluator.classify_prediction(pred_mask.clone(), batch)
        average_meter.update(area_inter, area_union, batch['class_id'], loss=None)
        average_meter.write_process(idx, len(dataloader), epoch=-1, write_batch_idx=1)
        # Visualize predictions
        if Visualizer.visualize:
            Visualizer.visualize_prediction_batch(batch['support_imgs'],
                                                  batch['query_img'], batch['query_mask'],
                                                  pred_mask, batch['class_id'], idx,
                                                  area_inter[1].float() / area_union[1].float())

    # Write evaluation results
    average_meter.write_result('Test', 0)
    miou, fb_iou = average_meter.compute_iou()

    return miou, fb_iou


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='IndustrialNetwork Pytorch Implementat1ion')
    parser.add_argument('--datapath', type=str, default='..')
    parser.add_argument('--save_path', type=str, default='./resume')
    parser.add_argument('--benchmark', type=str, default='industrial', choices=['pascal', 'coco', 'industrial'])
    parser.add_argument('--logpath', type=str, default='')
    parser.add_argument('--method', type=str, default='industrial', choices=['industrial', 'fss', 'msnet'])
    parser.add_argument('--bsz', type=int, default=1)
    parser.add_argument('--shot', type=int, default=5)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0.00005)
    parser.add_argument('--lr', type=float, default=0.025)
    parser.add_argument('--niter', type=int, default=300)
    parser.add_argument('--nworker', type=int, default=1)
    parser.add_argument('--load', type=str, default='./l2-5-1.pth')
    parser.add_argument('--fold', type=int, default=1, choices=[0, 1, 2, 3])
    parser.add_argument('--backbone', type=str, default='resnet50', choices=['vgg16', 'resnet50', 'resnet101','net'])
    args = parser.parse_args()
    Logger.initialize(args, training=True)

    # Model initialization
    model = IndustrialNetwork(args.backbone, False,shot=args.shot)
    model.eval()
    Logger.log_params(model)

    # Device setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    Logger.info('# available GPUs: %d' % torch.cuda.device_count())
    model = model.cuda()

    # Load trained model
    if args.load == '': raise Exception('Pretrained model not specified.')
    model.load_state_dict(torch.load(args.load))

    # Helper classes (for testing) initialization
    Evaluator.initialize()
    Visualizer.initialize(True)

    # Dataset initialization
    FSSDataset.initialize(img_size=440, datapath=args.datapath, use_original_imgsize=False)
    dataloader_test = FSSDataset.build_dataloader(args.benchmark, args.bsz, args.nworker, args.fold, 'test', args.shot)

    # Test MSHNet
    with torch.no_grad():
        test_miou, test_fb_iou = test(model, dataloader_test, args.shot)
    Logger.info('Fold %d mIoU: %5.2f \t FB-IoU: %5.2f' % (args.fold, test_miou.item(), test_fb_iou.item()))
    Logger.info('==================== Finished Testing ====================')
