#!/usr/bin/env python3
"""
Detecção de placas de veículos com YOLO nano em TensorFlow Lite (int8).

Roda no PC o mesmo modelo que vai para o Grove Vision AI V2 e reproduz a
lógica de disparo: quando uma placa é detectada com confiança >= limiar em
N quadros seguidos, a foto é salva (no PC, numa pasta; na placa, no cartão SD).

Aceita o modelo do train_tf.py (saída YOLOv5: cx,cy,w,h,obj,cls) e também
modelos YOLOv8/11 exportados pelo Ultralytics (cx,cy,w,h,cls); o formato é detectado sozinho.

IMPORTANTE: use o arquivo *_full_integer_quant.tflite (sem Vela).
O arquivo *_vela.tflite só roda na NPU Ethos-U, não no interpretador do PC.

Exemplos:
    python detect_tflite.py --model best_full_integer_quant.tflite --source carro.jpg
    python detect_tflite.py --model best_full_integer_quant.tflite --source pasta_de_fotos/
    python detect_tflite.py --model best_full_integer_quant.tflite --source 0 --show      # webcam
    python detect_tflite.py --model best_full_integer_quant.tflite --source video.mp4 --conf 0.65
"""
import argparse
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# --------------------------------------------------------------------------- #
# Interpretador TFLite (aceita LiteRT, tflite-runtime ou TensorFlow completo)
# --------------------------------------------------------------------------- #
def load_interpreter(model_path: str, threads: int):
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        try:
            from tflite_runtime.interpreter import Interpreter
        except ImportError:
            import tensorflow as tf
            Interpreter = tf.lite.Interpreter
    interp = Interpreter(model_path=model_path, num_threads=threads)
    interp.allocate_tensors()
    return interp


# --------------------------------------------------------------------------- #
# Detector
# --------------------------------------------------------------------------- #
class PlateDetector:
    def __init__(self, model_path, conf=0.5, iou=0.45, threads=4, class_names=("placa",)):
        self.interp = load_interpreter(model_path, threads)
        self.inp = self.interp.get_input_details()[0]
        self.out = self.interp.get_output_details()[0]
        _, self.h, self.w, _ = self.inp["shape"]
        self.conf = conf
        self.iou = iou
        self.names = class_names
        print(f"[modelo] entrada {self.inp['shape']} {np.dtype(self.inp['dtype']).name} | "
              f"saída {self.out['shape']} {np.dtype(self.out['dtype']).name}")

    # Redimensiona mantendo a proporção, com bordas cinza (igual ao treino do YOLO)
    def letterbox(self, img):
        h0, w0 = img.shape[:2]
        r = min(self.h / h0, self.w / w0)
        nh, nw = int(round(h0 * r)), int(round(w0 * r))
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        top, left = (self.h - nh) // 2, (self.w - nw) // 2
        canvas = np.full((self.h, self.w, 3), 114, dtype=np.uint8)
        canvas[top:top + nh, left:left + nw] = resized
        return canvas, r, left, top

    def preprocess(self, bgr):
        img, r, dx, dy = self.letterbox(bgr)
        x = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        dtype = self.inp["dtype"]
        if dtype in (np.int8, np.uint8):
            scale, zp = self.inp["quantization"]
            info = np.iinfo(dtype)
            x = np.clip(np.round(x / scale + zp), info.min, info.max).astype(dtype)
        return x[None, ...], r, dx, dy

    def postprocess(self, raw, r, dx, dy, shape0):
        # Desquantiza a saída int8 -> float
        if self.out["dtype"] in (np.int8, np.uint8):
            scale, zp = self.out["quantization"]
            raw = (raw.astype(np.float32) - zp) * scale

        pred = raw[0]
        if pred.shape[0] < pred.shape[1]:      # (4+nc, N) -> (N, 4+nc)
            pred = pred.T

        boxes = pred[:, :4].astype(np.float32).copy()   # cx, cy, w, h
        nc = len(self.names)
        if pred.shape[1] == 5 + nc:
            # Formato YOLOv5 (modelo do train_tf.py): cx,cy,w,h,obj,classes -> score = obj * classe
            cls_scores = pred[:, 5:] * pred[:, 4:5]
        else:
            # Formato YOLOv8/11 (Ultralytics): cx,cy,w,h,classes
            cls_scores = pred[:, 4:]
        cls_ids = cls_scores.argmax(axis=1)
        scores = cls_scores.max(axis=1)

        keep = scores >= self.conf
        boxes, scores, cls_ids = boxes[keep], scores[keep], cls_ids[keep]
        if len(scores) == 0:
            return []

        # O export TFLite do Ultralytics devolve coordenadas normalizadas (0..1)
        if boxes.max() <= 2.0:
            boxes[:, [0, 2]] *= self.w
            boxes[:, [1, 3]] *= self.h

        # cx,cy,w,h -> x1,y1,x2,y2 e desfaz o letterbox
        h0, w0 = shape0
        x1 = np.clip((boxes[:, 0] - boxes[:, 2] / 2 - dx) / r, 0, w0 - 1)
        y1 = np.clip((boxes[:, 1] - boxes[:, 3] / 2 - dy) / r, 0, h0 - 1)
        x2 = np.clip((boxes[:, 0] + boxes[:, 2] / 2 - dx) / r, 0, w0 - 1)
        y2 = np.clip((boxes[:, 1] + boxes[:, 3] / 2 - dy) / r, 0, h0 - 1)

        rects = [[float(a), float(b), float(c - a), float(d - b)] for a, b, c, d in zip(x1, y1, x2, y2)]
        idx = cv2.dnn.NMSBoxes(rects, scores.astype(float).tolist(), self.conf, self.iou)
        idx = np.array(idx).flatten() if len(idx) else []

        dets = []
        for i in idx:
            dets.append({
                "box": (int(x1[i]), int(y1[i]), int(x2[i]), int(y2[i])),
                "score": float(scores[i]),
                "class": self.names[int(cls_ids[i])] if int(cls_ids[i]) < len(self.names) else str(int(cls_ids[i])),
            })
        return dets

    def __call__(self, bgr):
        x, r, dx, dy = self.preprocess(bgr)
        self.interp.set_tensor(self.inp["index"], x)
        t0 = time.perf_counter()
        self.interp.invoke()
        ms = (time.perf_counter() - t0) * 1000
        raw = self.interp.get_tensor(self.out["index"])
        return self.postprocess(raw, r, dx, dy, bgr.shape[:2]), ms


# --------------------------------------------------------------------------- #
# Lógica de disparo (o que a placa faz antes de gravar no cartão SD)
# --------------------------------------------------------------------------- #
class SaveTrigger:
    """Salva a foto quando há placa com score >= limiar em N quadros seguidos,
    respeitando um intervalo mínimo entre fotos (evita repetir o mesmo carro)."""

    def __init__(self, threshold, frames_needed, cooldown_s, outdir):
        self.threshold = threshold
        self.frames_needed = max(1, frames_needed)
        self.cooldown_s = cooldown_s
        self.outdir = Path(outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.streak = 0
        self.last_save = -1e9

    def update(self, frame, dets):
        best = max((d["score"] for d in dets), default=0.0)
        self.streak = self.streak + 1 if best >= self.threshold else 0

        now = time.monotonic()
        if self.streak >= self.frames_needed and now - self.last_save >= self.cooldown_s:
            self.last_save = now
            self.streak = 0
            name = datetime.now().strftime("placa_%Y%m%d_%H%M%S_%f")[:-3] + f"_{best:.2f}.jpg"
            path = self.outdir / name
            cv2.imwrite(str(path), frame)   # salva a foto original, sem as caixas
            return path
        return None


def draw(frame, dets, ms):
    out = frame.copy()
    for d in dets:
        x1, y1, x2, y2 = d["box"]
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 220, 0), 2)
        label = f"{d['class']} {d['score']:.2f}"
        cv2.putText(out, label, (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 2)
    cv2.putText(out, f"{ms:.1f} ms", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
    return out


# --------------------------------------------------------------------------- #
# Fontes de imagem
# --------------------------------------------------------------------------- #
def iter_images(source):
    p = Path(source)
    files = sorted(f for f in p.iterdir() if f.suffix.lower() in IMG_EXTS) if p.is_dir() else [p]
    for f in files:
        img = cv2.imread(str(f))
        if img is None:
            print(f"[aviso] não consegui abrir {f}")
            continue
        yield f.name, img


def iter_stream(source):
    cap = cv2.VideoCapture(int(source) if str(source).isdigit() else str(source))
    if not cap.isOpened():
        raise SystemExit(f"Não foi possível abrir a fonte de vídeo: {source}")
    i = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield f"frame_{i:06d}", frame
            i += 1
    finally:
        cap.release()


def main():
    ap = argparse.ArgumentParser(description="Detecção de placas com YOLO nano (TFLite int8)")
    ap.add_argument("--model", required=True, help="arquivo *_full_integer_quant.tflite")
    ap.add_argument("--source", required=True, help="imagem, pasta, vídeo ou índice da webcam (0)")
    ap.add_argument("--conf", type=float, default=0.25, help="confiança mínima para mostrar a caixa")
    ap.add_argument("--iou", type=float, default=0.45, help="limiar de IoU do NMS")
    ap.add_argument("--save-threshold", type=float, default=0.65, help="confiança mínima para salvar a foto")
    ap.add_argument("--frames", type=int, default=3, help="quadros seguidos acima do limiar (vídeo)")
    ap.add_argument("--cooldown", type=float, default=2.0, help="segundos mínimos entre fotos salvas")
    ap.add_argument("--outdir", default="sd_card", help="pasta onde as fotos são salvas")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--show", action="store_true", help="mostrar janela com as detecções")
    args = ap.parse_args()

    det = PlateDetector(args.model, conf=min(args.conf, args.save_threshold), iou=args.iou, threads=args.threads)

    src = Path(args.source)
    is_images = src.is_dir() or src.suffix.lower() in IMG_EXTS
    frames = iter_images(args.source) if is_images else iter_stream(args.source)

    # Em imagens soltas não existe "quadros seguidos" nem intervalo de tempo
    trigger = SaveTrigger(
        threshold=args.save_threshold,
        frames_needed=1 if is_images else args.frames,
        cooldown_s=0.0 if is_images else args.cooldown,
        outdir=args.outdir,
    )

    saved = 0
    for name, frame in frames:
        dets, ms = det(frame)
        shown = [d for d in dets if d["score"] >= args.conf]
        path = trigger.update(frame, dets)
        if path:
            saved += 1
            print(f"[salvo] {path}")
        if is_images:
            scores = ", ".join(f"{d['score']:.2f}" for d in shown) or "nenhuma"
            print(f"{name}: {len(shown)} placa(s) [{scores}] em {ms:.1f} ms")

        if args.show:
            cv2.imshow("Placas - YOLO TFLite", draw(frame, shown, ms))
            key = cv2.waitKey(0 if is_images else 1) & 0xFF
            if key in (ord("q"), 27):
                break

    if args.show:
        cv2.destroyAllWindows()
    print(f"\nFotos salvas: {saved} em '{args.outdir}/'")


if __name__ == "__main__":
    main()
