#!/usr/bin/env python3
"""
YOLO tiny em TensorFlow/Keras para detectar placas de veículos (1 classe),
com exportação para TFLite int8 + Vela (Ethos-U55) para o Grove Vision AI V2.

Arquitetura (pensada para a NPU Ethos-U55):
  - backbone MobileNetV2 (alpha 0.35, pré-treinado no ImageNet), só ReLU6
  - neck estilo FPN com UpSampling nearest
  - 2 escalas de detecção (stride 8 e 16) com 3 âncoras cada (k-means no dataset)
  - parametrização do YOLOv5 (só sigmoid, sem exp), atribuição com células vizinhas
  - saída exportada no formato YOLOv5: [1, N, 6] = cx, cy, w, h, obj, cls (normalizados 0..1)

Uso com o dataset do Kaggle (PASCAL VOC):
    kaggle datasets download -d andrewmvd/car-plate-detection --unzip -p data/car-plate-detection
    python train_tf.py --voc data/car-plate-detection --imgsz 192 --epochs 150

Uso com um dataset já convertido (datasets/placas/images|labels/train|val):
    python train_tf.py --data-dir datasets/placas

Só exportar a partir dos melhores pesos já treinados:
    python train_tf.py --data-dir datasets/placas --skip-train

Saídas em --out (padrão runs/tf_placas):
    best.weights.h5, anchors.json, log.csv
    placas_float32.tflite                 -> referência em ponto flutuante
    placas_full_integer_quant.tflite      -> testar no PC (detect_tflite.py)
    vela_out/placas_full_integer_quant_vela.tflite -> enviar para o Grove Vision AI V2
"""
import argparse
import csv
import json
import math
import os
import random
import shutil
import subprocess
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import numpy as np
import tensorflow as tf
import keras
from keras import layers, ops

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
PAD = 114
STRIDES = (8, 16)
NUM_ANCHORS = 3
NC = 1                      # uma classe: placa
NO = 5 + NC                 # cx, cy, w, h, obj, cls


# =========================================================================== #
# Dados
# =========================================================================== #
def load_split(root: Path, split: str, cache_side: int):
    """Lê imagens + rótulos YOLO (cx cy w h normalizados) e guarda em memória."""
    img_dir, lbl_dir = root / "images" / split, root / "labels" / split
    if not img_dir.is_dir():
        raise SystemExit(f"Pasta não encontrada: {img_dir}")
    items = []
    for p in sorted(img_dir.iterdir()):
        if p.suffix.lower() not in IMG_EXTS:
            continue
        img = cv2.imread(str(p))
        if img is None:
            print(f"[aviso] não consegui abrir {p}")
            continue
        h, w = img.shape[:2]
        r = cache_side / max(h, w)
        if r < 1:   # reduz imagens grandes para economizar memória (rótulos são normalizados)
            img = cv2.resize(img, (int(round(w * r)), int(round(h * r))), interpolation=cv2.INTER_AREA)
        boxes = []
        lp = lbl_dir / f"{p.stem}.txt"
        if lp.exists():
            for ln in lp.read_text().splitlines():
                v = ln.split()
                if len(v) == 5:
                    boxes.append([float(x) for x in v[1:]])
        items.append((img, np.array(boxes, np.float32).reshape(-1, 4), p.name))
    return items


def norm_to_xyxy(b, w, h):
    """cx,cy,w,h normalizados -> x1,y1,x2,y2 em pixels."""
    if len(b) == 0:
        return np.zeros((0, 4), np.float32)
    return np.stack([(b[:, 0] - b[:, 2] / 2) * w, (b[:, 1] - b[:, 3] / 2) * h,
                     (b[:, 0] + b[:, 2] / 2) * w, (b[:, 1] + b[:, 3] / 2) * h], 1).astype(np.float32)


def letterbox(img, boxes_n, size):
    h0, w0 = img.shape[:2]
    r = min(size / h0, size / w0)
    nh, nw = int(round(h0 * r)), int(round(w0 * r))
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas = np.full((size, size, 3), PAD, np.uint8)
    canvas[top:top + nh, left:left + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    b = norm_to_xyxy(boxes_n, w0, h0) * r + np.array([left, top, left, top], np.float32)
    return canvas, b


def random_affine(img, boxes_n, size, rng, scale=(0.35, 1.3), translate=0.2):
    """Escala + translação aleatórias (simula placas mais longe/perto e fora do centro)."""
    h0, w0 = img.shape[:2]
    r = min(size / h0, size / w0) * rng.uniform(*scale)
    tx = (size - w0 * r) / 2 + rng.uniform(-translate, translate) * size
    ty = (size - h0 * r) / 2 + rng.uniform(-translate, translate) * size
    M = np.array([[r, 0, tx], [0, r, ty]], np.float32)
    out = cv2.warpAffine(img, M, (size, size), flags=cv2.INTER_LINEAR, borderValue=(PAD, PAD, PAD))
    b = norm_to_xyxy(boxes_n, w0, h0) * r + np.array([tx, ty, tx, ty], np.float32)
    if len(b):
        area0 = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        b = np.clip(b, 0, size)
        area1 = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        keep = ((b[:, 2] - b[:, 0]) >= 3) & ((b[:, 3] - b[:, 1]) >= 3) & (area1 >= 0.4 * area0)
        b = b[keep]
    return out, b


def hsv_jitter(img, rng, gh=0.015, gs=0.6, gv=0.5):
    r = rng.uniform(-1, 1, 3) * [gh, gs, gv] + 1
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    x = np.arange(256, dtype=np.float32)
    lut_h = ((x * r[0]) % 180).astype(np.uint8)
    lut_s = np.clip(x * r[1], 0, 255).astype(np.uint8)
    lut_v = np.clip(x * r[2], 0, 255).astype(np.uint8)
    hsv = cv2.merge((cv2.LUT(hue, lut_h), cv2.LUT(sat, lut_s), cv2.LUT(val, lut_v)))
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def to_input(bgr):
    """BGR uint8 -> RGB float 0..1 (mesmo pré-processamento do detect_tflite.py)."""
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def kmeans_anchors(items, size, n=2 * NUM_ANCHORS, iters=300, seed=0):
    """Âncoras (em pixels na entrada do modelo) por k-means com distância 1-IoU."""
    wh = []
    for img, b, _ in items:
        h0, w0 = img.shape[:2]
        r = min(size / h0, size / w0)
        if len(b):
            wh.append(b[:, 2:4] * np.array([w0, h0]) * r)
    wh = np.concatenate(wh) if wh else np.zeros((0, 2))
    if len(wh) < n:
        print("[âncoras] poucas caixas, usando âncoras padrão")
        base = np.array([[12, 6], [20, 9], [30, 13], [44, 18], [64, 26], [96, 40]], np.float32)
        return (base * size / 192).reshape(len(STRIDES), NUM_ANCHORS, 2)

    def wh_iou(a, c):
        inter = np.minimum(a[:, None, 0], c[None, :, 0]) * np.minimum(a[:, None, 1], c[None, :, 1])
        return inter / (a[:, None].prod(2) + c[None].prod(2) - inter)

    rng = np.random.default_rng(seed)
    centers = wh[rng.choice(len(wh), n, replace=False)].astype(np.float64)
    for _ in range(iters):
        assign = wh_iou(wh, centers).argmax(1)
        new = np.array([np.median(wh[assign == k], 0) if np.any(assign == k) else centers[k] for k in range(n)])
        if np.allclose(new, centers):
            break
        centers = new
    centers = centers[np.argsort(centers.prod(1))].astype(np.float32)
    return centers.reshape(len(STRIDES), NUM_ANCHORS, 2)


def build_targets(boxes, size, anchors):
    """Alvos estilo YOLOv5 por escala: [G*G*A, 5] = gx, gy, gw, gh (unid. de grade), máscara.
    Cada caixa vai para as âncoras com razão de tamanho < 4 e para até 2 células vizinhas."""
    outs = [np.zeros(((size // s) ** 2 * NUM_ANCHORS, 5), np.float32) for s in STRIDES]

    def put(si, ci, cj, a, gx, gy, w, h):
        s, G = STRIDES[si], size // STRIDES[si]
        outs[si][(cj * G + ci) * NUM_ANCHORS + a] = [gx, gy, w / s, h / s, 1.0]

    for x1, y1, x2, y2 in boxes:
        w, h = x2 - x1, y2 - y1
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        matched = False
        best = (1e9, 0, 0)
        for si, s in enumerate(STRIDES):
            G = size // s
            gx, gy = cx / s, cy / s
            gi, gj = min(int(gx), G - 1), min(int(gy), G - 1)
            fx, fy = gx - int(gx), gy - int(gy)
            cells = [(gi, gj)]
            if fx < 0.5 and gi > 0: cells.append((gi - 1, gj))
            if fx > 0.5 and gi < G - 1: cells.append((gi + 1, gj))
            if fy < 0.5 and gj > 0: cells.append((gi, gj - 1))
            if fy > 0.5 and gj < G - 1: cells.append((gi, gj + 1))
            for a in range(NUM_ANCHORS):
                aw, ah = anchors[si, a]
                ratio = max(w / aw, aw / w, h / ah, ah / h)
                if ratio < best[0]:
                    best = (ratio, si, a)
                if ratio < 4.0:
                    matched = True
                    for ci, cj in cells:
                        put(si, ci, cj, a, gx, gy, w, h)
        if not matched:   # caixa muito diferente das âncoras: usa a mais parecida
            _, si, a = best
            s, G = STRIDES[si], size // STRIDES[si]
            gx, gy = cx / s, cy / s
            put(si, min(int(gx), G - 1), min(int(gy), G - 1), a, gx, gy, w, h)
    return outs


def batch_generator(items, size, anchors, batch, augment, seed):
    rng = np.random.default_rng(seed)

    def gen():
        idx = np.arange(len(items))
        if augment:
            rng.shuffle(idx)
        for k in range(0, len(idx), batch):
            xs, t3s, t4s = [], [], []
            for i in idx[k:k + batch]:
                img, bn, _ = items[i]
                if augment:
                    im, b = random_affine(img, bn, size, rng)
                    im = hsv_jitter(im, rng)
                    if rng.random() < 0.1:
                        im = cv2.cvtColor(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
                    if rng.random() < 0.1:
                        im = cv2.GaussianBlur(im, (3, 3), 0)
                else:
                    im, b = letterbox(img, bn, size)
                t3, t4 = build_targets(b, size, anchors)
                xs.append(to_input(im)); t3s.append(t3); t4s.append(t4)
            yield np.stack(xs), np.stack(t3s), np.stack(t4s)

    n3 = (size // STRIDES[0]) ** 2 * NUM_ANCHORS
    n4 = (size // STRIDES[1]) ** 2 * NUM_ANCHORS
    sig = (tf.TensorSpec((None, size, size, 3), tf.float32),
           tf.TensorSpec((None, n3, 5), tf.float32),
           tf.TensorSpec((None, n4, 5), tf.float32))
    return tf.data.Dataset.from_generator(gen, output_signature=sig).prefetch(2)


# =========================================================================== #
# Modelo
# =========================================================================== #
def conv_bn(x, f, k=1, s=1):
    x = layers.Conv2D(f, k, strides=s, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    return layers.ReLU(6.0)(x)


def dw_block(x, f):
    """Convolução separável (depthwise 3x3 + pointwise 1x1), barata e bem suportada na NPU."""
    x = layers.DepthwiseConv2D(3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(6.0)(x)
    return conv_bn(x, f, 1)


def build_train_model(size, alpha=0.35, pretrained=True):
    inp = keras.Input((size, size, 3), name="image")          # RGB 0..1
    x = layers.Rescaling(2.0, offset=-1.0)(inp)               # MobileNetV2 espera -1..1
    weights = "imagenet" if pretrained else None
    bb = keras.applications.MobileNetV2(input_shape=(size, size, 3), include_top=False,
                                        weights=weights, alpha=alpha)
    feats = keras.Model(bb.input, [bb.get_layer(n).output for n in ("block_5_add", "block_12_add", "out_relu")],
                        name="backbone")
    c3, c4, c5 = feats(x)                                     # strides 8, 16, 32

    p5 = conv_bn(c5, 96, 1)
    p4 = layers.Concatenate()([conv_bn(c4, 64, 1), layers.UpSampling2D(2, interpolation="nearest")(p5)])
    p4 = dw_block(dw_block(p4, 96), 96)
    p3 = layers.Concatenate()([conv_bn(c3, 32, 1),
                               layers.UpSampling2D(2, interpolation="nearest")(conv_bn(p4, 48, 1))])
    p3 = dw_block(dw_block(p3, 64), 64)

    h3 = layers.Conv2D(NUM_ANCHORS * NO, 1, name="head_p3")(p3)
    h4 = layers.Conv2D(NUM_ANCHORS * NO, 1, name="head_p4")(p4)
    r3 = layers.Reshape((-1, NO), name="raw_p3")(h3)          # [B, G*G*A, 6]
    r4 = layers.Reshape((-1, NO), name="raw_p4")(h4)
    model = keras.Model(inp, [r3, r4], name="yolo_tiny_placas")

    # Viés inicial do YOLOv5: poucas células com objeto no começo do treino
    for name, s in (("head_p3", STRIDES[0]), ("head_p4", STRIDES[1])):
        l = model.get_layer(name)
        w, b = l.get_weights()
        b = b.reshape(NUM_ANCHORS, NO)
        b[:, 4] += math.log(8 / (size / s) ** 2)
        b[:, 5:] += math.log(0.6 / (NC - 0.99))
        l.set_weights([w, b.reshape(-1)])
    return model


class YoloDecode(layers.Layer):
    """Decodifica a saída crua de uma escala para cx,cy,w,h normalizados + obj + cls.
    Usa só sigmoid, multiplicação e soma (tudo int8 e rodando na NPU)."""

    def __init__(self, stride, anchors, size, **kw):
        super().__init__(**kw)
        self.stride, self.size = int(stride), int(size)
        self.anchors = np.asarray(anchors, np.float32)
        G = self.size // self.stride
        gy, gx = np.meshgrid(np.arange(G), np.arange(G), indexing="ij")
        grid = np.stack([gx, gy], -1).reshape(G * G, 1, 2).repeat(NUM_ANCHORS, 1).reshape(-1, 2)
        self.grid = grid.astype(np.float32)
        self.anc = np.tile(self.anchors, (G * G, 1)).astype(np.float32)

    def call(self, raw):
        p = ops.sigmoid(raw)
        xy = (p[..., 0:2] * 2.0 - 0.5 + self.grid) * (self.stride / self.size)
        wh = ops.square(p[..., 2:4] * 2.0) * (self.anc / self.size)
        return ops.concatenate([xy, wh, p[..., 4:]], axis=-1)

    def get_config(self):
        return {**super().get_config(), "stride": self.stride, "anchors": self.anchors.tolist(), "size": self.size}


def build_export_model(train_model, anchors, size):
    r3, r4 = train_model.outputs
    d3 = YoloDecode(STRIDES[0], anchors[0], size, name="decode_p3")(r3)
    d4 = YoloDecode(STRIDES[1], anchors[1], size, name="decode_p4")(r4)
    out = layers.Concatenate(axis=1, name="detections")([d3, d4])   # [B, N, 6]
    return keras.Model(train_model.input, out, name="yolo_tiny_placas_export")


# =========================================================================== #
# Loss (YOLOv5: GIoU para a caixa, BCE para objeto e classe)
# =========================================================================== #
def giou(b1, b2, eps=1e-7):
    b1x1, b1x2 = b1[..., 0] - b1[..., 2] / 2, b1[..., 0] + b1[..., 2] / 2
    b1y1, b1y2 = b1[..., 1] - b1[..., 3] / 2, b1[..., 1] + b1[..., 3] / 2
    b2x1, b2x2 = b2[..., 0] - b2[..., 2] / 2, b2[..., 0] + b2[..., 2] / 2
    b2y1, b2y2 = b2[..., 1] - b2[..., 3] / 2, b2[..., 1] + b2[..., 3] / 2
    inter = tf.maximum(tf.minimum(b1x2, b2x2) - tf.maximum(b1x1, b2x1), 0.0) * \
            tf.maximum(tf.minimum(b1y2, b2y2) - tf.maximum(b1y1, b2y1), 0.0)
    union = b1[..., 2] * b1[..., 3] + b2[..., 2] * b2[..., 3] - inter + eps
    iou = inter / union
    cw = tf.maximum(b1x2, b2x2) - tf.minimum(b1x1, b2x1)
    ch = tf.maximum(b1y2, b2y2) - tf.minimum(b1y1, b2y1)
    c_area = cw * ch + eps
    return iou - (c_area - union) / c_area


class YoloLoss:
    def __init__(self, anchors, size):
        nl = len(STRIDES)
        self.levels = []
        for si, s in enumerate(STRIDES):
            G = size // s
            gy, gx = np.meshgrid(np.arange(G), np.arange(G), indexing="ij")
            grid = np.stack([gx, gy], -1).reshape(G * G, 1, 2).repeat(NUM_ANCHORS, 1).reshape(-1, 2)
            anc = np.tile(anchors[si] / s, (G * G, 1))
            self.levels.append((tf.constant(grid, tf.float32), tf.constant(anc, tf.float32)))
        self.balance = [4.0, 1.0]
        self.box_gain = 0.05 * 3 / nl
        self.obj_gain = 1.0 * (size / 640) ** 2 * 3 / nl
        self.cls_gain = 0.5 * NC / 80 * 3 / nl

    def __call__(self, preds, targets):
        lbox = lobj = lcls = 0.0
        for raw, t, (grid, anc), bal in zip(preds, targets, self.levels, self.balance):
            p = tf.sigmoid(raw)
            pxy = p[..., 0:2] * 2.0 - 0.5 + grid
            pwh = tf.square(p[..., 2:4] * 2.0) * anc
            mask = t[..., 4]
            npos = tf.maximum(tf.reduce_sum(mask), 1.0)
            g = giou(tf.concat([pxy, pwh], -1), t[..., 0:4])
            lbox += tf.reduce_sum((1.0 - g) * mask) / npos
            tobj = mask * tf.stop_gradient(tf.clip_by_value(g, 0.0, 1.0))
            lobj += bal * tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tobj, logits=raw[..., 4]))
            lcls += tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.ones_like(raw[..., 5]), logits=raw[..., 5]) * mask) / npos
        bs = tf.cast(tf.shape(preds[0])[0], tf.float32)
        total = (lbox * self.box_gain + lobj * self.obj_gain + lcls * self.cls_gain) * bs
        return total, tf.stack([lbox, lobj, lcls])


# =========================================================================== #
# Avaliação (mAP@0.5 de uma classe)
# =========================================================================== #
def nms(boxes, scores, iou_thr):
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0]); yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2]); yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        a = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        iou = inter / (a[i] + a[order[1:]] - inter + 1e-9)
        order = order[1:][iou < iou_thr]
    return np.array(keep, int)


def box_iou(a, b):
    inter = (np.maximum(0, np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0])) *
             np.maximum(0, np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1])))
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + ab[None, :] - inter + 1e-9)


def decode_detections(pred, size, conf, iou_thr=0.5):
    score = pred[:, 4] * pred[:, 5]
    k = score >= conf
    p, score = pred[k], score[k]
    if not len(p):
        return np.zeros((0, 4)), np.zeros(0)
    b = np.stack([p[:, 0] - p[:, 2] / 2, p[:, 1] - p[:, 3] / 2, p[:, 0] + p[:, 2] / 2, p[:, 1] + p[:, 3] / 2], 1) * size
    keep = nms(b, score, iou_thr)
    return b[keep], score[keep]


def evaluate(predict_fn, items, size, conf=0.001):
    """Retorna mAP50, precisão e recall (com score >= 0.5)."""
    records, n_gt, tp50, fp50 = [], 0, 0, 0
    for img, bn, _ in items:
        canvas, gt = letterbox(img, bn, size)
        pred = predict_fn(to_input(canvas)[None])[0]
        boxes, scores = decode_detections(pred, size, conf)
        n_gt += len(gt)
        used = np.zeros(len(gt), bool)
        ious = box_iou(boxes, gt) if len(gt) and len(boxes) else np.zeros((len(boxes), len(gt)))
        for j in np.argsort(-scores):
            tp = False
            if len(gt):
                cand = np.where(~used & (ious[j] >= 0.5))[0]
                if len(cand):
                    used[cand[ious[j, cand].argmax()]] = True
                    tp = True
            records.append((scores[j], tp))
            if scores[j] >= 0.5:
                tp50 += tp; fp50 += (not tp)
    if n_gt == 0:
        return 0.0, 0.0, 0.0
    records.sort(key=lambda r: -r[0])
    tps = np.array([r[1] for r in records], np.float32)
    ctp, cfp = np.cumsum(tps), np.cumsum(1 - tps)
    recall = ctp / n_gt if len(tps) else np.zeros(0)
    precision = ctp / np.maximum(ctp + cfp, 1e-9) if len(tps) else np.zeros(0)
    mrec = np.concatenate([[0.0], recall, [1.0]])
    mpre = np.concatenate([[1.0], precision, [0.0]])
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = float(np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1]))
    p = tp50 / max(tp50 + fp50, 1)
    r = tp50 / n_gt
    return ap, p, r


def keras_predict_fn(model):
    @tf.function(reduce_retracing=True)
    def f(x):
        return model(x, training=False)
    return lambda x: f(tf.constant(x)).numpy()


def tflite_predict_fn(path):
    interp = tf.lite.Interpreter(model_path=str(path))
    interp.allocate_tensors()
    inp, out = interp.get_input_details()[0], interp.get_output_details()[0]

    def f(x):
        if inp["dtype"] in (np.int8, np.uint8):
            s, z = inp["quantization"]
            info = np.iinfo(inp["dtype"])
            x = np.clip(np.round(x / s + z), info.min, info.max).astype(inp["dtype"])
        interp.set_tensor(inp["index"], x)
        interp.invoke()
        y = interp.get_tensor(out["index"])
        if out["dtype"] in (np.int8, np.uint8):
            s, z = out["quantization"]
            y = (y.astype(np.float32) - z) * s
        return y
    return f


# =========================================================================== #
# Exportação
# =========================================================================== #
def export_tflite(export_model, rep_items, size, out_dir: Path, n_rep=200):
    @tf.function(input_signature=[tf.TensorSpec([1, size, size, 3], tf.float32, name="image")])
    def serve(x):
        return export_model(x, training=False)

    concrete = serve.get_concrete_function()

    conv = tf.lite.TFLiteConverter.from_concrete_functions([concrete], export_model)
    f32 = out_dir / "placas_float32.tflite"
    f32.write_bytes(conv.convert())

    rep = list(rep_items)
    random.Random(0).shuffle(rep)
    rep = rep[:n_rep]

    def representative():
        for img, bn, _ in rep:
            canvas, _ = letterbox(img, bn, size)
            yield [to_input(canvas)[None]]

    conv = tf.lite.TFLiteConverter.from_concrete_functions([concrete], export_model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = representative
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    i8 = out_dir / "placas_full_integer_quant.tflite"
    i8.write_bytes(conv.convert())
    print(f"[export] {f32} ({f32.stat().st_size / 1024:.0f} KB)")
    print(f"[export] {i8} ({i8.stat().st_size / 1024:.0f} KB)")
    return f32, i8


def run_vela(tflite: Path, out_dir: Path, accelerator: str):
    if shutil.which("vela") is None:
        print("[vela] 'vela' não encontrado. Instale com: pip install ethos-u-vela")
        return None
    vout = out_dir / "vela_out"
    vout.mkdir(parents=True, exist_ok=True)
    cmd = ["vela", "--accelerator-config", accelerator, "--output-dir", str(vout), str(tflite)]
    print("[vela]", " ".join(cmd))
    subprocess.run(cmd, check=True)
    vf = vout / f"{tflite.stem}_vela.tflite"
    print(f"[vela] Modelo para o Grove Vision AI V2: {vf}")
    return vf


# =========================================================================== #
# Principal
# =========================================================================== #
def parse_args():
    p = argparse.ArgumentParser(description="YOLO tiny em TensorFlow -> TFLite int8 -> Vela")
    p.add_argument("--voc", default=None, help="dataset PASCAL VOC (images/ + annotations/); é convertido antes")
    p.add_argument("--data-dir", default="datasets/placas", help="dataset no formato YOLO (images|labels/train|val)")
    p.add_argument("--val", type=float, default=0.2, help="fração de validação na conversão VOC")
    p.add_argument("--imgsz", type=int, default=192, help="entrada (múltiplo de 32: 160, 192, 224, 256)")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--warmup", type=int, default=3, help="épocas de aquecimento do learning rate")
    p.add_argument("--alpha", type=float, default=0.35, help="largura do MobileNetV2 (0.35, 0.5, 0.75, 1.0)")
    p.add_argument("--no-pretrained", action="store_true", help="não baixar os pesos do ImageNet")
    p.add_argument("--eval-every", type=int, default=5, help="avaliar mAP50 a cada N épocas")
    p.add_argument("--out", default="runs/tf_placas")
    p.add_argument("--skip-train", action="store_true", help="só exportar usando best.weights.h5 de --out")
    p.add_argument("--rep-samples", type=int, default=200, help="imagens para calibrar a quantização int8")
    p.add_argument("--accelerator", default="ethos-u55-64", help="configuração da NPU do Grove Vision AI V2")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def setup_device():
    gpus = tf.config.list_physical_devices("GPU")
    for g in gpus:
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except Exception:
            pass
    if gpus:
        print(f"[gpu] TensorFlow {tf.__version__} usando: {[g.name for g in gpus]}")
    else:
        print(f"[gpu] TensorFlow {tf.__version__} sem GPU visível, treinando na CPU.")


def main():
    args = parse_args()
    if args.imgsz % 32:
        raise SystemExit("--imgsz precisa ser múltiplo de 32 (ex.: 160, 192, 224, 256)")
    random.seed(args.seed); np.random.seed(args.seed); tf.random.set_seed(args.seed)
    setup_device()

    if args.voc:
        from voc_to_yolo import convert
        convert(args.voc, args.data_dir, val=args.val, seed=args.seed)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data = Path(args.data_dir)
    cache_side = max(2 * args.imgsz, 640)
    train_items = load_split(data, "train", cache_side)
    val_items = load_split(data, "val", cache_side)
    print(f"[dados] {len(train_items)} treino | {len(val_items)} validação")

    anchors_path = out / "anchors.json"
    if args.skip_train and anchors_path.exists():
        anchors = np.array(json.loads(anchors_path.read_text())["anchors"], np.float32)
    else:
        anchors = kmeans_anchors(train_items, args.imgsz)
        anchors_path.write_text(json.dumps({"imgsz": args.imgsz, "strides": STRIDES,
                                            "anchors": anchors.round(2).tolist()}, indent=2))
    print(f"[âncoras] stride 8: {anchors[0].round(1).tolist()} | stride 16: {anchors[1].round(1).tolist()}")

    model = build_train_model(args.imgsz, args.alpha, pretrained=not args.no_pretrained)
    export_model = build_export_model(model, anchors, args.imgsz)
    print(f"[modelo] {model.count_params() / 1e6:.2f} M parâmetros")
    best_path = out / "best.weights.h5"

    if not args.skip_train:
        loss_fn = YoloLoss(anchors, args.imgsz)
        opt = keras.optimizers.AdamW(learning_rate=args.lr, weight_decay=5e-4, clipnorm=10.0)
        ds = batch_generator(train_items, args.imgsz, anchors, args.batch, augment=True, seed=args.seed)
        steps = math.ceil(len(train_items) / args.batch)

        @tf.function(reduce_retracing=True)
        def train_step(x, t3, t4):
            with tf.GradientTape() as tape:
                r3, r4 = model(x, training=True)
                loss, parts = loss_fn([r3, r4], [t3, t4])
            grads = tape.gradient(loss, model.trainable_variables)
            opt.apply_gradients(zip(grads, model.trainable_variables))
            return loss, parts

        predict = keras_predict_fn(export_model)
        best_ap = -1.0
        with open(out / "log.csv", "w", newline="") as fh:
            log = csv.writer(fh)
            log.writerow(["epoch", "lr", "loss", "box", "obj", "cls", "mAP50", "precision", "recall"])
            for ep in range(1, args.epochs + 1):
                if ep <= args.warmup:
                    lr = args.lr * ep / args.warmup
                else:
                    t = (ep - args.warmup) / max(1, args.epochs - args.warmup)
                    lr = args.lr * (0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * t)))
                opt.learning_rate = lr

                t0 = time.time()
                tot, parts_sum, n = 0.0, np.zeros(3), 0
                for x, t3, t4 in ds:
                    loss, parts = train_step(x, t3, t4)
                    tot += float(loss); parts_sum += parts.numpy(); n += 1
                tot /= max(n, 1); parts_sum /= max(n, 1)

                row = [ep, f"{lr:.6f}", f"{tot:.4f}", *(f"{v:.4f}" for v in parts_sum), "", "", ""]
                msg = (f"época {ep:3d}/{args.epochs} | lr {lr:.5f} | loss {tot:.3f} "
                       f"(box {parts_sum[0]:.3f} obj {parts_sum[1]:.3f} cls {parts_sum[2]:.3f}) | {time.time() - t0:.1f}s")
                if ep % args.eval_every == 0 or ep == args.epochs:
                    ap, p, r = evaluate(predict, val_items, args.imgsz)
                    row[6:] = [f"{ap:.4f}", f"{p:.4f}", f"{r:.4f}"]
                    msg += f" | mAP50 {ap:.3f} P {p:.3f} R {r:.3f}"
                    if ap > best_ap:
                        best_ap = ap
                        model.save_weights(str(best_path))
                        msg += "  <- melhor"
                print(msg, flush=True)
                log.writerow(row); fh.flush()
        print(f"[treino] melhor mAP50 na validação: {best_ap:.3f}")

    if not best_path.exists():
        raise SystemExit(f"Pesos não encontrados: {best_path}")
    model.load_weights(str(best_path))

    f32, i8 = export_tflite(export_model, train_items + val_items, args.imgsz, out, args.rep_samples)
    for name, fn in (("Keras float", keras_predict_fn(export_model)),
                     ("TFLite float32", tflite_predict_fn(f32)),
                     ("TFLite int8", tflite_predict_fn(i8))):
        ap, p, r = evaluate(fn, val_items, args.imgsz)
        print(f"[val] {name:15s} mAP50 {ap:.3f} | P {p:.3f} | R {r:.3f}")
    run_vela(i8, out, args.accelerator)


if __name__ == "__main__":
    main()
