#!/usr/bin/env python3
"""
YOLO tiny em TensorFlow/Keras para detectar placas de veículos (1 classe),
com exportação para TFLite int8 + Vela (Ethos-U55) para o Grove Vision AI V2.

Arquiteturas (--arch):
  - swift (padrão): Swift-YOLO tiny da Seeed (SSCMA) = YOLOv5 com backbone
    CSPDarknet (widen 0.15, deepen 0.33), neck PAFPN, SPPF e ReLU; treinada do zero,
    com mosaico
  - mobilenet: backbone MobileNetV2 (alpha 0.35, pré-treinado no ImageNet) + FPN leve
Comum às duas (pensado para a NPU Ethos-U55):
  - 3 escalas de detecção (stride 8, 16 e 32) com 3 âncoras cada (k-means no dataset),
    como o firmware SSCMA do Grove Vision AI V2 exige para modelos YOLOv5
  - parametrização do YOLOv5 (só sigmoid, sem exp), atribuição com células vizinhas
  - saída exportada no formato YOLOv5: [1, N, 6] = cx, cy, w, h, obj, cls (normalizados 0..1)

Uso com o dataset do Kaggle (PASCAL VOC):
    kaggle datasets download -d andrewmvd/car-plate-detection --unzip -p data/car-plate-detection
    python train_tf.py --voc data/car-plate-detection --imgsz 192 --epochs 300

Uso com um dataset já convertido (datasets/placas/images|labels/train|val):
    python train_tf.py --data-dir datasets/placas --out runs/swift_yolo --vela .venv-vela/bin/vela

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
STRIDES = (8, 16, 32)      # o firmware da Seeed (SSCMA) exige exatamente estas 3 escalas
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
    return out, clip_boxes(b, size)


def clip_boxes(b, size, min_side=3, min_area_ratio=0.4):
    """Corta as caixas na borda da imagem e descarta as que ficaram muito pequenas ou cortadas."""
    if not len(b):
        return np.zeros((0, 4), np.float32)
    area0 = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    b = np.clip(b, 0, size)
    area1 = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    keep = ((b[:, 2] - b[:, 0]) >= min_side) & ((b[:, 3] - b[:, 1]) >= min_side) & (area1 >= min_area_ratio * area0)
    return b[keep].astype(np.float32)


def mosaic4(items, idxs, size, rng, scale=(0.5, 1.5), translate=0.1):
    """Mosaico do YOLOv5: junta 4 imagens numa tela 2x maior e recorta com escala/translação aleatórias.
    Multiplica a variedade de fundos e de posições das placas, essencial treinando sem pré-treino."""
    s = size
    canvas = np.full((2 * s, 2 * s, 3), PAD, np.uint8)
    xc, yc = (int(rng.uniform(0.5 * s, 1.5 * s)) for _ in range(2))
    boxes = []
    for k, i in enumerate(idxs):
        img, bn, _ = items[i]
        h0, w0 = img.shape[:2]
        r = s / max(h0, w0)
        img = cv2.resize(img, (max(1, int(round(w0 * r))), max(1, int(round(h0 * r)))), interpolation=cv2.INTER_LINEAR)
        h, w = img.shape[:2]
        if k == 0:    # cima-esquerda
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
        elif k == 1:  # cima-direita
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, 2 * s), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif k == 2:  # baixo-esquerda
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(2 * s, yc + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
        else:         # baixo-direita
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, 2 * s), min(2 * s, yc + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
        canvas[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
        b = norm_to_xyxy(bn, w0, h0) * r
        if len(b):
            b += np.array([x1a - x1b, y1a - y1b] * 2, np.float32)
            # só a parte da caixa que caiu dentro do pedaço colado
            b[:, [0, 2]] = np.clip(b[:, [0, 2]], x1a, x2a)
            b[:, [1, 3]] = np.clip(b[:, [1, 3]], y1a, y2a)
            boxes.append(b)
    b = np.concatenate(boxes) if boxes else np.zeros((0, 4), np.float32)
    b = b[((b[:, 2] - b[:, 0]) > 2) & ((b[:, 3] - b[:, 1]) > 2)]

    sc = rng.uniform(*scale)
    tx = s / 2 - sc * s + rng.uniform(-translate, translate) * s   # centro da tela -> centro da saída
    ty = s / 2 - sc * s + rng.uniform(-translate, translate) * s
    M = np.array([[sc, 0, tx], [0, sc, ty]], np.float32)
    out = cv2.warpAffine(canvas, M, (s, s), flags=cv2.INTER_LINEAR, borderValue=(PAD, PAD, PAD))
    return out, clip_boxes(b * sc + np.array([tx, ty, tx, ty], np.float32), s)


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


def kmeans_anchors(items, size, n=len(STRIDES) * NUM_ANCHORS, iters=300, seed=0):
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
        base = np.array([[10, 5], [16, 7], [22, 10], [30, 13], [40, 17], [54, 22],
                         [72, 30], [100, 42], [140, 60]], np.float32)
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


def batch_generator(items, size, anchors, batch, augment, seed, state=None):
    """state["mosaic"] = probabilidade de usar mosaico (o laço de treino muda no fim do treino)."""
    rng = np.random.default_rng(seed)
    state = state if state is not None else {"mosaic": 0.0}

    def gen():
        idx = np.arange(len(items))
        if augment:
            rng.shuffle(idx)
        for k in range(0, len(idx), batch):
            xs, ts = [], [[] for _ in STRIDES]
            for i in idx[k:k + batch]:
                img, bn, _ = items[i]
                if augment and rng.random() < state["mosaic"]:
                    others = rng.integers(0, len(items), 3)
                    im, b = mosaic4(items, [i, *others], size, rng)
                elif augment:
                    im, b = random_affine(img, bn, size, rng)
                if augment:
                    im = hsv_jitter(im, rng)
                    if rng.random() < 0.1:
                        im = cv2.cvtColor(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
                    if rng.random() < 0.1:
                        im = cv2.GaussianBlur(im, (3, 3), 0)
                else:
                    im, b = letterbox(img, bn, size)
                xs.append(to_input(im))
                for lst, t in zip(ts, build_targets(b, size, anchors)):
                    lst.append(t)
            yield (np.stack(xs), *(np.stack(t) for t in ts))

    sig = (tf.TensorSpec((None, size, size, 3), tf.float32),
           *(tf.TensorSpec((None, (size // s) ** 2 * NUM_ANCHORS, 5), tf.float32) for s in STRIDES))
    return tf.data.Dataset.from_generator(gen, output_signature=sig).prefetch(2)


# =========================================================================== #
# Modelo
# =========================================================================== #
class Upsample2x(layers.Layer):
    """Upsampling 2x 'nearest' que vira um único RESIZE_NEAREST_NEIGHBOR no TFLite.
    O layers.UpSampling2D do Keras 3 vira ExpandDims + TILE + Reshape 5D: o TILE
    não existe no firmware do Grove Vision AI V2 e tensores 5D não rodam na NPU."""

    def call(self, x):
        h, w = x.shape[1], x.shape[2]
        return tf.compat.v1.image.resize_nearest_neighbor(
            x, (2 * h, 2 * w), align_corners=False, half_pixel_centers=False)

    def compute_output_shape(self, s):
        return (s[0], 2 * s[1], 2 * s[2], s[3])


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


# --------------------------------------------------------------------------- #
# Swift-YOLO (Seeed SSCMA): YOLOv5 com CSPDarknet + PAFPN, ReLU e largura reduzida
# Referência: ModelAssistant/configs/swift_yolo/swift_yolo_tiny_1xb16_300e_coco.py
# --------------------------------------------------------------------------- #
def make_div(c, widen, divisor=8):
    return int(math.ceil(c * widen / divisor) * divisor)


def make_round(n, deepen):
    return max(round(n * deepen), 1) if n > 1 else n


def cbr(x, f, k=1, s=1):
    """ConvModule do SSCMA: Conv (sem bias) + BatchNorm (momentum 0.03 do PyTorch) + ReLU."""
    x = layers.Conv2D(f, k, strides=s, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization(momentum=0.97, epsilon=1e-3)(x)
    return layers.ReLU()(x)


def darknet_bottleneck(x, c, add):
    y = cbr(cbr(x, c, 1), c, 3)
    return layers.Add()([x, y]) if add else y


def csp_layer(x, out, n, add):
    """CSPLayer (C3 do YOLOv5): dois ramos 1x1, gargalos no ramo principal, concat e 1x1."""
    mid = int(out * 0.5)
    short = cbr(x, mid, 1)
    main = cbr(x, mid, 1)
    for _ in range(n):
        main = darknet_bottleneck(main, mid, add)
    return cbr(layers.Concatenate()([main, short]), out, 1)


def sppf(x, out):
    mid = x.shape[-1] // 2
    x = cbr(x, mid, 1)
    y1 = layers.MaxPool2D(5, strides=1, padding="same")(x)
    y2 = layers.MaxPool2D(5, strides=1, padding="same")(y1)
    y3 = layers.MaxPool2D(5, strides=1, padding="same")(y2)
    return cbr(layers.Concatenate()([x, y1, y2, y3]), out, 1)


def swift_yolo_features(x, widen=0.15, deepen=0.33):
    """Backbone YOLOv5CSPDarknet (P5) + neck YOLOv5PAFPN. Retorna (p3, p4, p5)."""
    # backbone
    x = cbr(x, make_div(64, widen), 6, 2)                          # stem 6x6/2
    feats = []
    for c_out, n, spp in ((128, 3, False), (256, 6, False), (512, 9, False), (1024, 3, True)):
        c = make_div(c_out, widen)
        x = cbr(x, c, 3, 2)
        x = csp_layer(x, c, make_round(n, deepen), add=True)
        if spp:
            x = sppf(x, c)
        feats.append(x)
    c3, c4, c5 = feats[1:]                                          # strides 8, 16, 32
    ch = [make_div(c, widen) for c in (256, 512, 1024)]
    n = make_round(3, deepen)

    # top-down
    r5 = cbr(c5, ch[1], 1)
    t4 = csp_layer(layers.Concatenate()([Upsample2x()(r5), c4]), ch[1], n, add=False)
    t4 = cbr(t4, ch[0], 1)
    p3 = csp_layer(layers.Concatenate()([Upsample2x()(t4), c3]), ch[0], n, add=False)
    # bottom-up
    p4 = csp_layer(layers.Concatenate()([cbr(p3, ch[0], 3, 2), t4]), ch[1], n, add=False)
    p5 = csp_layer(layers.Concatenate()([cbr(p4, ch[1], 3, 2), r5]), ch[2], n, add=False)
    return p3, p4, p5


def build_train_model(size, alpha=0.35, pretrained=True, arch="swift", widen=0.15, deepen=0.33):
    inp = keras.Input((size, size, 3), name="image")          # RGB 0..1
    if arch == "swift":
        p3, p4, p5 = swift_yolo_features(inp, widen, deepen)
        return _add_heads(inp, (p3, p4, p5), size, "swift_yolo_placas")
    x = layers.Rescaling(2.0, offset=-1.0)(inp)               # MobileNetV2 espera -1..1
    weights = "imagenet" if pretrained else None
    bb = keras.applications.MobileNetV2(input_shape=(size, size, 3), include_top=False,
                                        weights=weights, alpha=alpha)
    feats = keras.Model(bb.input, [bb.get_layer(n).output for n in ("block_5_add", "block_12_add", "out_relu")],
                        name="backbone")
    c3, c4, c5 = feats(x)                                     # strides 8, 16, 32

    p5 = conv_bn(c5, 96, 1)
    # nomes iguais aos do UpSampling2D antigo, para os pesos já treinados continuarem carregando
    p4 = layers.Concatenate()([conv_bn(c4, 64, 1), Upsample2x(name="up_sampling2d")(p5)])
    p4 = dw_block(dw_block(p4, 96), 96)
    p3 = layers.Concatenate()([conv_bn(c3, 32, 1),
                               Upsample2x(name="up_sampling2d_1")(conv_bn(p4, 48, 1))])
    p3 = dw_block(dw_block(p3, 64), 64)
    p5 = dw_block(p5, 96)
    return _add_heads(inp, (p3, p4, p5), size, "yolo_tiny_placas")


def _add_heads(inp, feats, size, name):
    """Cabeça YOLOv5 (DetHead): uma conv 1x1 por escala com 3 âncoras x (5 + classes)."""
    raws = []
    for lvl, feat in zip((3, 4, 5), feats):
        h = layers.Conv2D(NUM_ANCHORS * NO, 1, name=f"head_p{lvl}")(feat)
        raws.append(layers.Reshape((-1, NO), name=f"raw_p{lvl}")(h))   # [B, G*G*A, 6]
    model = keras.Model(inp, raws, name=name)

    # Viés inicial do YOLOv5: poucas células com objeto no começo do treino
    for name, s in zip(("head_p3", "head_p4", "head_p5"), STRIDES):
        l = model.get_layer(name)
        w, b = l.get_weights()
        b = b.reshape(NUM_ANCHORS, NO)
        b[:, 4] += math.log(8 / (size / s) ** 2)
        b[:, 5:] += math.log(0.6 / (NC - 0.99))
        l.set_weights([w, b.reshape(-1)])
    return model


class YoloDecode(layers.Layer):
    """Decodifica a saída crua de uma escala para cx,cy,w,h normalizados + obj + cls.
    Usa só sigmoid, multiplicação, soma e concatenação (tudo int8)."""

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
        t = p[..., 2:4] * 2.0
        wh = (t * t) * (self.anc / self.size)   # MUL em vez de SQUARE: SQUARE não tem versão int8
        return ops.concatenate([xy, wh, p[..., 4:]], axis=-1)

    def get_config(self):
        return {**super().get_config(), "stride": self.stride, "anchors": self.anchors.tolist(), "size": self.size}


def build_export_model(train_model, anchors, size):
    decoded = [YoloDecode(s, anchors[i], size, name=f"decode_p{i + 3}")(r)
               for i, (s, r) in enumerate(zip(STRIDES, train_model.outputs))]
    # [B, N, 6] com N = ((S/8)^2 + (S/16)^2 + (S/32)^2) * 3, o formato YOLOv5 do firmware
    out = layers.Concatenate(axis=1, name="detections")(decoded)
    return keras.Model(train_model.input, out, name=f"{train_model.name}_export")


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
        self.balance = [4.0, 1.0, 0.4]
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

    # Congela os pesos como constantes. Sem isso o Keras 3 deixa os pesos como
    # variáveis (READ_VARIABLE) e a calibração int8 falha.
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
    concrete = convert_variables_to_constants_v2(serve.get_concrete_function())

    def new_converter():
        return tf.lite.TFLiteConverter.from_concrete_functions([concrete])

    conv = new_converter()
    f32 = out_dir / "placas_float32.tflite"
    f32.write_bytes(conv.convert())

    rep = list(rep_items)
    random.Random(0).shuffle(rep)
    rep = rep[:n_rep]

    def representative():
        for img, bn, _ in rep:
            canvas, _ = letterbox(img, bn, size)
            yield [to_input(canvas)[None]]

    conv = new_converter()
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


# Configuração de memória do Grove Vision AI V2 publicada pela Seeed:
# pesos na flash externa (Axi1) e ativações na SRAM (Axi0).
VELA_CONFIG = """\
[System_Config.My_Sys_Cfg]
core_clock=400e6
axi0_port=Sram
axi1_port=OffChipFlash
Sram_clock_scale=1.0
Sram_burst_length=32
Sram_read_latency=16
Sram_write_latency=16
Dram_clock_scale=0.75
Dram_burst_length=128
Dram_read_latency=500
Dram_write_latency=250
OnChipFlash_clock_scale=0.25
OffChipFlash_clock_scale=0.015625
OffChipFlash_burst_length=32
OffChipFlash_read_latency=64
OffChipFlash_write_latency=64

[Memory_Mode.My_Mem_Mode_Parent]
const_mem_area=Axi1
arena_mem_area=Axi0
cache_mem_area=Axi0
"""


def run_vela(tflite: Path, out_dir: Path, accelerator: str, vela_bin: str = "vela"):
    if shutil.which(vela_bin) is None:
        print(f"[vela] '{vela_bin}' não encontrado. Rode o Vela à mão (veja vela_config.ini).")
        return None
    vout = out_dir / "vela_out"
    vout.mkdir(parents=True, exist_ok=True)
    cfg = out_dir / "vela_config.ini"
    cfg.write_text(VELA_CONFIG)
    cmd = [vela_bin, "--accelerator-config", accelerator,
           "--config", str(cfg), "--system-config", "My_Sys_Cfg", "--memory-mode", "My_Mem_Mode_Parent",
           "--output-dir", str(vout), str(tflite)]
    print("[vela]", " ".join(cmd))
    if subprocess.run(cmd).returncode != 0:
        print("[vela] O Vela falhou. O .tflite int8 foi salvo; rode o Vela 3.12 num ambiente com Python 3.12 e numpy<2:")
        print("       python train_tf.py ... --skip-train --vela .venv-vela/bin/vela")
        return None
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
    p.add_argument("--arch", choices=("swift", "mobilenet"), default="swift",
                   help="swift = Swift-YOLO da Seeed (CSPDarknet+PAFPN); mobilenet = MobileNetV2 pré-treinado")
    p.add_argument("--width", type=float, default=0.15, help="widen_factor da Swift-YOLO (tiny = 0.15)")
    p.add_argument("--depth", type=float, default=0.33, help="deepen_factor da Swift-YOLO (tiny = 0.33)")
    p.add_argument("--mosaic", type=float, default=0.8, help="probabilidade de mosaico no treino (0 desliga)")
    p.add_argument("--close-mosaic", type=int, default=15, help="desliga o mosaico nas últimas N épocas")
    p.add_argument("--epochs", type=int, default=300)
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
    p.add_argument("--vela", default="vela",
                   help="executável do Vela (ex.: .venv-vela/bin/vela, com ethos-u-vela==3.12.0 e numpy<2)")
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
        meta = json.loads(anchors_path.read_text())
        anchors = np.array(meta["anchors"], np.float32)
        # usa a mesma arquitetura com que os pesos foram treinados
        # (pastas antigas, sem "arch", são do MobileNetV2)
        args.arch = meta.get("arch", "mobilenet")
        args.width, args.depth = meta.get("width", args.width), meta.get("depth", args.depth)
        args.alpha = meta.get("alpha", args.alpha)
    else:
        anchors = kmeans_anchors(train_items, args.imgsz)
        anchors_path.write_text(json.dumps({"imgsz": args.imgsz, "strides": STRIDES,
                                            "anchors": anchors.round(2).tolist(),
                                            "arch": args.arch, "width": args.width, "depth": args.depth,
                                            "alpha": args.alpha}, indent=2))
    print(f"[âncoras] stride 8: {anchors[0].round(1).tolist()} | stride 16: {anchors[1].round(1).tolist()}")

    model = build_train_model(args.imgsz, args.alpha, pretrained=not args.no_pretrained,
                              arch=args.arch, widen=args.width, deepen=args.depth)
    print(f"[modelo] arquitetura: {args.arch}" +
          (f" (widen {args.width}, deepen {args.depth})" if args.arch == "swift" else f" (alpha {args.alpha})"))
    export_model = build_export_model(model, anchors, args.imgsz)
    print(f"[modelo] {model.count_params() / 1e6:.2f} M parâmetros")
    best_path = out / "best.weights.h5"

    if not args.skip_train:
        loss_fn = YoloLoss(anchors, args.imgsz)
        opt = keras.optimizers.AdamW(learning_rate=args.lr, weight_decay=5e-4, clipnorm=10.0)
        aug_state = {"mosaic": args.mosaic}
        ds = batch_generator(train_items, args.imgsz, anchors, args.batch, augment=True, seed=args.seed,
                             state=aug_state)
        steps = math.ceil(len(train_items) / args.batch)

        @tf.function(reduce_retracing=True)
        def train_step(x, *targets):
            with tf.GradientTape() as tape:
                preds = model(x, training=True)
                loss, parts = loss_fn(list(preds), list(targets))
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
                if ep == args.epochs - args.close_mosaic + 1 and aug_state["mosaic"] > 0:
                    aug_state["mosaic"] = 0.0
                    print("[treino] mosaico desligado nas últimas épocas (ajuste fino com imagens inteiras)")

                t0 = time.time()
                tot, parts_sum, n = 0.0, np.zeros(3), 0
                for x, *targets in ds:
                    loss, parts = train_step(x, *targets)
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
    run_vela(i8, out, args.accelerator, args.vela)


if __name__ == "__main__":
    main()