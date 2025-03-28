import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import *
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FasterRCNN, FastRCNNPredictor
from torchvision.models.detection.roi_heads import fastrcnn_loss
from typing import List, Tuple
from collections import OrderedDict

class RCNN(nn.Module):
    def __init__(self, num_classes, base_model = fasterrcnn_resnet50_fpn(weights = "DEFAULT"), device = "cuda"):
        super().__init__()
        self.transform = base_model.transform
        self.backbone = base_model.backbone
        self.rpn = base_model.rpn
        self.roi_heads = base_model.roi_heads
        self.box_roi_pool = base_model.roi_heads.box_roi_pool
        self.box_head = base_model.roi_heads.box_head
        
        # Modify base model to predict the correct number of classes. +1 is for background class
        in_features = list(self.box_head.children())[0].out_features
        self.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)

        self.device = torch.device(device)

    def forward(self, images, targets = None, mode = "full"):
        assert (mode == "full") or (mode == "partial"), "mode must be either 'full' or 'partial'!"
        # Do all the things from the object detection demo notebook to get to the point where box_predictor is called
        # Right before box_predictor is called is when we need to do VOS synthesis step
        original_image_sizes: List[Tuple[int, int]] = []
        for img in images:
            val = img.shape[-2:]
            torch._assert(
                len(val) == 2,
                f"expecting the last two dimensions of the Tensor to be H and W instead got {img.shape[-2:]}",
            )
            original_image_sizes.append((val[0], val[1]))
        # Step 1: Transform data
        if self.training:
            images, targets = self.transform(images, targets)
            targets = [{k:v.to(self.device) for k, v in d.items()} for d in targets]
        else:
            images, targets = self.transform(images)

        # Step 2: Pass data through backbone and get features
        features = self.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])

        # Step 3: Make object proposals
        proposals, proposal_losses = self.rpn(images, features, targets)
        
        if self.training:
            loss_objectness = proposal_losses["loss_objectness"]
            loss_rpn_box_reg = proposal_losses["loss_rpn_box_reg"]
            proposals, matched_idxs, labels, regression_targets = self.roi_heads.select_training_samples(proposals,
                                                                                                            targets)

        # Step 4: Semi-pass through roi_heads
        box_features = self.box_roi_pool(features, proposals, images.image_sizes)
        box_features = self.box_head(box_features)

        # Step 5: Process features to remove spurious (background, low area, etc.) detections
        # class_logits, box_regression = self.box_predictor(box_features)
        if mode == "partial":
            return box_features
        elif mode == "full":
            class_logits, box_regression = self.box_predictor(box_features)
            step_boxes, step_scores, step_labels = self.roi_heads.postprocess_detections(class_logits,
                                                                                        box_regression,
                                                                                        proposals,
                                                                                        images.image_sizes)
            result = []
            num_images = len(step_boxes)
            for i in range(num_images):
                result.append(
                    {
                        "boxes": step_boxes[i],
                        "labels": step_labels[i],
                        "scores": step_scores[i],
                    }
                )
            if self.training:
                loss_classifier, loss_box_reg = fastrcnn_loss(class_logits, box_regression, labels, regression_targets)
                return loss_classifier, loss_box_reg, loss_objectness, loss_rpn_box_reg
            else:
                return self.transform.postprocess(result, images.image_sizes, original_image_sizes)