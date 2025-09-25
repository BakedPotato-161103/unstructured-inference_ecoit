# Copyright (c) Megvii, Inc. and its affiliates.
# Unstructured modified the original source code found at:
# https://github.com/Megvii-BaseDetection/YOLOX/blob/237e943ac64aa32eb32f875faa93ebb18512d41d/yolox/data/data_augment.py
# https://github.com/Megvii-BaseDetection/YOLOX/blob/ac379df3c97d1835ebd319afad0c031c36d03f36/yolox/utils/demo_utils.py

import os
import cv2
import numpy as np
import onnxruntime
from PIL import Image as PILImage

from unstructured_inference.constants import ElementType, Source
from unstructured_inference.inference.layoutelement import LayoutElements
from unstructured_inference.models.unstructuredmodel import (
    UnstructuredObjectDetectionModel,
)
from unstructured_inference.utils import (
    LazyDict,
    LazyEvaluateInfo,
    download_if_needed_and_get_local_path,
)

PADDLE_LABELS = {'paragraph_title': 0,
                'image': 1,
                'text': 2,
                'number': 3,
                'abstract': 4,
                'content': 5,
                'figure_title': 6,
                'formula': 7,
                'table': 8,
                'table_title': 9,
                'reference': 10,
                'doc_title': 11,
                'footnote': 12,
                'header': 13,
                'algorithm': 14,
                'footer': 15,
                'seal': 16,
                'chart_title': 17,
                'chart': 18,
                'formula_number': 19,
                'header_image': 20,
                'footer_image': 21,
                'aside_text': 22}
PADDLE_UNSTRUCTURED_MAP = {
    'abstract': ElementType.PARAGRAPH,
    'doc_title': ElementType.TITLE,
    'paragraph_title': ElementType.TITLE,
    'chart_title': ElementType.TITLE,
    'figure_title': ElementType.CAPTION,
    'table_title': ElementType.CAPTION,
    'footnote': ElementType.FOOTNOTE,
    'formula': ElementType.FORMULA,
    'footer': ElementType.PAGE_FOOTER,
    'header': ElementType.PAGE_HEADER,
    'image': ElementType.PICTURE,
    'table': ElementType.TABLE,
    'text': ElementType.TEXT,
    'content': ElementType.TEXT,
    'page_number': ElementType.PAGE_NUMBER,
    'aside_text': ElementType.PAGE_FOOTER, 
    'Block': ElementType.UNCATEGORIZED_TEXT,
}
PADDLE_MAP = dict([(PADDLE_LABELS[key], PADDLE_UNSTRUCTURED_MAP[key] 
                        if key in PADDLE_UNSTRUCTURED_MAP.keys() 
                            else ElementType.UNCATEGORIZED_TEXT) 
                    for key in PADDLE_LABELS.keys()])

MODEL_TYPES = {
    'paddle': LazyDict(model_path = os.environ.get("PADDLE_LAYOUT_MODEL_PATH", ""),
                        label_map=PADDLE_MAP)
}
# Let's say we don't have the option to use ONNX models
class UnstructuredPPLayoutModel(UnstructuredObjectDetectionModel):
    def predict(self, x: PILImage.Image):
        """Predict using YoloX model."""
        super().predict(x)
        return self.image_processing(x)

    def initialize(self, model_path: str, label_map: dict = PADDLE_MAP):
        """Start inference session for YoloX model."""
        print("Using Paddle for Layout Detection")
        self.model_path = model_path
        available_providers = onnxruntime.get_available_providers()
        ordered_providers = [
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]
        providers = [provider for provider in ordered_providers if provider in available_providers]
        try: 
            self.model = onnxruntime.InferenceSession(
                model_path,
                providers=providers,
            )
        except Exception as e: 
            print(f"Failed at model init: {e}")
        
        self.scale = np.array([[1., 1.]], dtype=np.float32)
        self.layout_classes = label_map

    def image_processing(
        self,
        image: PILImage.Image,
    ) -> LayoutElements:
        """Method runing YoloX for layout detection, returns a PageLayout
        parameters
        ----------
        page
            Path for image file with the image to process
        origin_img
            If specified, an Image object for process with YoloX model
        page_number
            Number asigned to the PageLayout returned
        output_directory
            Boolean indicating if result will be stored
        """
        # The model was trained and exported with this shape
        # TODO (benjamin): check other shapes for inference
        # No, they cannot, thats why it failed :) 
        input_shape = (640, 640)
        origin_img = np.array(image)
        im_shape = np.array([origin_img.shape[:2]], dtype=np.float32)
        img = preprocess(origin_img, input_shape)
        session = self.model

        ort_inputs = {'image': img[None, :], 'scale_factor': self.scale, 'im_shape': im_shape}
        # Just get first since its not batch processed
        output = session.run(None, ort_inputs)[0]
        

        boxes = output[:, 2:].copy()
        scores = output[:, 1].copy()
        labels = output[:, 0].copy().astype(int)
        boxes_xyxy = boxes[:, [1, 0, 3, 2]]
        # Note (Benjamin): Distinct models (quantized and original) requires distincts
        # levels of thresholds
        # I don't know why it is fixed ? 
        if "quantized" in self.model_path:
            dets = multiclass_nms(boxes_xyxy, labels, scores, nms_thr=0.0, score_thr=0.07)
        else:
            dets = multiclass_nms(boxes_xyxy, labels, scores, nms_thr=0.1, score_thr=0.2)
        # Sorted along Y-axis ?
        order = np.argsort(dets[:, 1])
        sorted_dets = dets[order]

        return LayoutElements(
            element_coords=sorted_dets[:, :4].astype(float),
            element_probs=sorted_dets[:, 4].astype(float),
            element_class_ids=sorted_dets[:, 5].astype(int),
            element_class_id_map=self.layout_classes,
            sources=np.array([Source.PADDLE] * sorted_dets.shape[0]),
        )
# Note: preprocess function was named preproc on original source

def preprocess(img, input_size, swap=(2, 0, 1)):
    """Preprocess image data before Paddle Layout Detection inference."""
    if len(img.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
    else:
        padded_img = np.ones(input_size, dtype=np.uint8) * 114

    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.uint8)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32) / 255
    return padded_img

def multiclass_nms(boxes, cls_inds, scores, nms_thr, score_thr, class_agnostic=True):
    """Multiclass NMS implemented in Numpy"""
    # TODO(benjamin): check for non-class agnostic
    # if class_agnostic:
    nms_method = multiclass_nms_class_agnostic
    # else:
    #    nms_method = multiclass_nms_class_aware
    return nms_method(boxes, cls_inds, scores, nms_thr, score_thr)


def multiclass_nms_class_agnostic(boxes, cls_inds, cls_scores, nms_thr, score_thr):
    """Multiclass NMS implemented in Numpy. Class-agnostic version."""
    valid_score_mask = cls_scores > score_thr
    valid_scores = cls_scores[valid_score_mask]
    valid_boxes = boxes[valid_score_mask]
    valid_cls_inds = cls_inds[valid_score_mask]
    keep = nms(valid_boxes, valid_scores, nms_thr)
    dets = np.concatenate(
        [valid_boxes[keep], valid_scores[keep, None], valid_cls_inds[keep, None]],
        1,
    )
    return dets


def nms(boxes, scores, nms_thr):
    """Single class NMS implemented in Numpy."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= nms_thr)[0]
        order = order[inds + 1]

    return keep
