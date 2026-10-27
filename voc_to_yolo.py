#!/usr/bin/env python3
"""
Converte um dataset PASCAL VOC (ex.: Kaggle "andrewmvd/car-plate-detection")
para o formato YOLO, com divisão treino/validação.

Estrutura de entrada esperada (a do Kaggle):
    car-plate-detection/
        images/       Cars0.png, Cars1.png, ...
        annotations/  Cars0.xml, Cars1.xml, ...

Estrutura de saída:
    datasets/placas/
        images/train/  images/val/
        labels/train/  labels/val/      (uma linha por placa: 0 cx cy w h, normalizados)

Todas as classes do XML (no Kaggle: "licence") viram a classe única 0 = "placa".

Uso:
    kaggle datasets download -d andrewmvd/car-plate-detection --unzip -p data/car-plate-detection
    python voc_to_yolo.py --src data/car-plate-detection --dst datasets/placas --val 0.2
"""
import argparse
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


def find_dir(root: Path, names):
    for n in names:
        d = root / n
        if d.is_dir():
            return d
    # procura um nível abaixo (alguns zips vêm com uma pasta extra)
    for sub in root.iterdir():
        if sub.is_dir():
            for n in names:
                d = sub / n
                if d.is_dir():
                    return d
    raise SystemExit(f"Não encontrei nenhuma das pastas {names} em {root}")


def find_image(img_dir: Path, xml_root, xml_path: Path):
    fname = xml_root.findtext("filename")
    if fname:
        p = img_dir / Path(fname).name
        if p.exists():
            return p
    for ext in IMG_EXTS:
        p = img_dir / (xml_path.stem + ext)
        if p.exists():
            return p
    return None


def image_size(xml_root, img_path: Path):
    size = xml_root.find("size")
    if size is not None:
        w = int(float(size.findtext("width", "0")))
        h = int(float(size.findtext("height", "0")))
        if w > 0 and h > 0:
            return w, h
    import cv2  # só se o XML não trouxer o tamanho
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    return img.shape[1], img.shape[0]


def parse_voc(xml_path: Path, img_dir: Path):
    root = ET.parse(xml_path).getroot()
    img_path = find_image(img_dir, root, xml_path)
    if img_path is None:
        return None, [], set()
    wh = image_size(root, img_path)
    if wh is None:
        return None, [], set()
    W, H = wh

    lines, classes = [], set()
    for obj in root.iter("object"):
        classes.add((obj.findtext("name") or "").strip())
        bb = obj.find("bndbox")
        if bb is None:
            continue
        x1 = float(bb.findtext("xmin")); y1 = float(bb.findtext("ymin"))
        x2 = float(bb.findtext("xmax")); y2 = float(bb.findtext("ymax"))
        # corrige caixas invertidas ou fora da imagem
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))
        x1, x2 = max(0.0, x1), min(float(W), x2)
        y1, y2 = max(0.0, y1), min(float(H), y2)
        bw, bh = x2 - x1, y2 - y1
        if bw < 1 or bh < 1:
            continue
        cx, cy = (x1 + bw / 2) / W, (y1 + bh / 2) / H
        lines.append(f"0 {cx:.6f} {cy:.6f} {bw / W:.6f} {bh / H:.6f}")
    return img_path, lines, classes


def convert(src, dst, val=0.2, seed=42, overwrite=True):
    src, dst = Path(src), Path(dst)
    img_dir = find_dir(src, ["images", "JPEGImages", "imgs"])
    ann_dir = find_dir(src, ["annotations", "Annotations", "labels_xml"])

    if overwrite and dst.exists():
        shutil.rmtree(dst)
    for split in ("train", "val"):
        (dst / "images" / split).mkdir(parents=True, exist_ok=True)
        (dst / "labels" / split).mkdir(parents=True, exist_ok=True)

    samples, all_classes, skipped = [], set(), 0
    for xml_path in sorted(ann_dir.glob("*.xml")):
        img_path, lines, classes = parse_voc(xml_path, img_dir)
        if img_path is None:
            skipped += 1
            continue
        all_classes |= classes
        samples.append((img_path, lines))

    if not samples:
        raise SystemExit("Nenhuma anotação válida encontrada.")

    random.Random(seed).shuffle(samples)
    n_val = max(1, int(round(len(samples) * val)))
    splits = {"val": samples[:n_val], "train": samples[n_val:]}

    rel_sizes = []
    for split, items in splits.items():
        for img_path, lines in items:
            shutil.copy2(img_path, dst / "images" / split / img_path.name)
            (dst / "labels" / split / f"{img_path.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
            for ln in lines:
                _, _, _, w, h = map(float, ln.split())
                rel_sizes.append((w, h))

    n_boxes = len(rel_sizes)
    print(f"[voc->yolo] classes no XML: {sorted(all_classes)} -> todas viram 'placa'")
    print(f"[voc->yolo] imagens: {len(splits['train'])} treino | {len(splits['val'])} validação "
          f"| {skipped} ignoradas | {n_boxes} placas")
    if rel_sizes:
        import statistics as st
        mw = st.median(w for w, _ in rel_sizes)
        mh = st.median(h for _, h in rel_sizes)
        print(f"[voc->yolo] tamanho mediano da placa: {mw * 100:.1f}% x {mh * 100:.1f}% da imagem")
        for s in (192, 224, 256):
            print(f"            em {s}x{s} isso dá ~{mw * s:.0f} x {mh * s:.0f} pixels")
    return dst


def main():
    ap = argparse.ArgumentParser(description="PASCAL VOC -> YOLO")
    ap.add_argument("--src", required=True, help="pasta do dataset VOC (com images/ e annotations/)")
    ap.add_argument("--dst", default="datasets/placas", help="pasta de saída no formato YOLO")
    ap.add_argument("--val", type=float, default=0.2, help="fração para validação")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    convert(args.src, args.dst, args.val, args.seed)


if __name__ == "__main__":
    main()
