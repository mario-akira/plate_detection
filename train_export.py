#!/usr/bin/env python3
"""
Treina um YOLO nano (YOLOv8n) para detectar placas de veículos (1 classe)
e exporta para TensorFlow Lite int8 + Vela (Ethos-U55) para o Grove Vision AI V2.

Uso com o dataset do Kaggle (PASCAL VOC), que é convertido automaticamente:
    kaggle datasets download -d andrewmvd/car-plate-detection --unzip -p data/car-plate-detection
    python train_export.py --voc data/car-plate-detection --imgsz 192 --epochs 150

Uso com um dataset já no formato YOLO:
    python train_export.py --data data.yaml --imgsz 192 --epochs 150

Só exportar um modelo já treinado:
    python train_export.py --data data.yaml --weights runs/placas/train/weights/best.pt --skip-train

Saídas:
    <pasta>_saved_model/best_full_integer_quant.tflite   -> testar no PC (detect_tflite.py)
    vela_out/best_full_integer_quant_vela.tflite         -> enviar para o Grove Vision AI V2
"""
import argparse
import shutil
import subprocess
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser(description="Treino e exportação YOLO nano -> TFLite int8 -> Vela")
    p.add_argument("--data", default="data.yaml", help="arquivo YAML do dataset (formato YOLO)")
    p.add_argument("--voc", default=None,
                   help="pasta de um dataset PASCAL VOC (images/ + annotations/); converte e gera --data")
    p.add_argument("--yolo-dir", default="datasets/placas", help="onde salvar o dataset convertido")
    p.add_argument("--val", type=float, default=0.2, help="fração de validação na conversão VOC")
    p.add_argument("--model", default="yolov8n.pt", help="modelo base (yolov8n.pt, yolo11n.pt...)")
    p.add_argument("--weights", default=None, help="pesos já treinados (.pt) para pular o treino")
    p.add_argument("--skip-train", action="store_true", help="apenas exportar --weights")
    p.add_argument("--imgsz", type=int, default=192, help="tamanho de entrada (192, 224 ou 256)")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--project", default="runs/placas")
    p.add_argument("--name", default="train")
    p.add_argument("--vela-out", default="vela_out", help="pasta de saída do Vela")
    p.add_argument("--accelerator", default="ethos-u55-64", help="configuração da NPU do Grove Vision AI V2")
    return p.parse_args()


def train(args) -> Path:
    model = YOLO(args.model)
    model.train(
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        project=args.project,
        name=args.name,
        patience=30,        # para cedo se não melhorar
        # Aumentos de dados úteis para placas (câmera fixa, luz variável).
        # Sem flip horizontal: placas espelhadas não existem no mundo real.
        fliplr=0.0,
        hsv_v=0.5,
        degrees=5.0,
        scale=0.5,
        mosaic=1.0,
    )
    return Path(model.trainer.best)


def export_tflite(weights: Path, args) -> Path:
    model = YOLO(str(weights))

    metrics = model.val(data=args.data, imgsz=args.imgsz)
    print(f"\n[val] mAP50 = {metrics.box.map50:.3f} | mAP50-95 = {metrics.box.map:.3f}\n")

    # int8=True faz quantização pós-treino usando imagens do dataset para calibração.
    exported = Path(model.export(format="tflite", imgsz=args.imgsz, int8=True, data=args.data))

    # O arquivo "full_integer_quant" tem entrada e saída int8, que é o que a NPU espera.
    saved_model_dir = exported.parent if exported.suffix == ".tflite" else exported
    candidates = sorted(saved_model_dir.glob("*full_integer_quant.tflite"))
    tflite = candidates[0] if candidates else exported
    print(f"[export] TFLite int8: {tflite}")
    return tflite


def run_vela(tflite: Path, args):
    if shutil.which("vela") is None:
        print("[vela] 'vela' não encontrado. Instale com: pip install ethos-u-vela")
        return None
    out = Path(args.vela_out)
    out.mkdir(parents=True, exist_ok=True)
    cmd = [
        "vela",
        "--accelerator-config", args.accelerator,
        "--output-dir", str(out),
        str(tflite),
    ]
    print("[vela]", " ".join(cmd))
    subprocess.run(cmd, check=True)
    vela_file = out / f"{tflite.stem}_vela.tflite"
    print(f"[vela] Modelo para o Grove Vision AI V2: {vela_file}")
    print("[vela] Confira no relatório acima quantas operações ficaram fora da NPU (CPU ops).")
    return vela_file


def prepare_voc(args):
    """Converte o dataset VOC para YOLO e escreve um data.yaml com caminho absoluto
    (caminho relativo faz o Ultralytics procurar dentro da pasta 'datasets' dele)."""
    from voc_to_yolo import convert

    out = convert(args.voc, args.yolo_dir, val=args.val).resolve()
    yaml_text = (
        f"path: {out.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: placa\n"
    )
    Path(args.data).write_text(yaml_text)
    print(f"[voc] data.yaml gerado em {Path(args.data).resolve()}")


def main():
    args = parse_args()

    if args.voc:
        prepare_voc(args)

    if args.skip_train:
        if not args.weights:
            raise SystemExit("--skip-train exige --weights caminho/para/best.pt")
        weights = Path(args.weights)
    else:
        weights = train(args)

    tflite = export_tflite(weights, args)
    run_vela(tflite, args)


if __name__ == "__main__":
    main()
