#!/usr/bin/env python3
"""
Confere se um .tflite atende às regras do firmware SSCMA do Grove Vision AI V2
para detectores YOLOv5 (as mesmas checagens de ma_model_yolov5.cpp).

Lê o arquivo diretamente (formato flatbuffer), então funciona também com o
arquivo do Vela, que tem o operador "ethos-u" que só existe na placa.

Uso:
    python check_model.py runs/tf_placas_v2/placas_full_integer_quant.tflite
    python check_model.py runs/tf_placas_v2/vela_out/placas_full_integer_quant_vela.tflite
"""
import os
import sys
from collections import Counter
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np

INT8 = 9  # TensorType.INT8 no schema do TFLite

# Operadores registrados no firmware SSCMA (ma_engine_tflite.cpp). Qualquer outro
# faz o modelo falhar ao carregar na placa ("model failed" / "invoke failed").
FIRMWARE_OPS = {
    "ABS", "ADD", "ADD_N", "ARG_MAX", "ARG_MIN", "ASSIGN_VARIABLE", "AVERAGE_POOL_2D", "BATCH_TO_SPACE_ND",
    "BROADCAST_ARGS", "BROADCAST_TO", "CALL_ONCE", "CAST", "CEIL", "CIRCULAR_BUFFER", "CONCATENATION",
    "CONV_2D", "COS", "CUMSUM", "DEPTH_TO_SPACE", "DEPTHWISE_CONV_2D", "DEQUANTIZE",
    "TFLite_Detection_PostProcess", "DIV", "ELU", "EQUAL", "ethos-u", "EXP", "EXPAND_DIMS", "FILL", "FLOOR",
    "FLOOR_DIV", "FLOOR_MOD", "FULLY_CONNECTED", "GATHER", "GATHER_ND", "GREATER", "GREATER_EQUAL",
    "HARD_SWISH", "IF", "L2_NORMALIZATION", "L2_POOL_2D", "LEAKY_RELU", "LESS", "LESS_EQUAL", "LOG",
    "LOG_SOFTMAX", "LOGICAL_AND", "LOGICAL_NOT", "LOGICAL_OR", "LOGISTIC", "MAX_POOL_2D", "MAXIMUM", "MEAN",
    "MINIMUM", "MIRROR_PAD", "MUL", "NEG", "NOT_EQUAL", "PACK", "PAD", "PADV2", "PRELU", "QUANTIZE",
    "READ_VARIABLE", "REDUCE_MAX", "RELU", "RELU6", "RESHAPE", "RESIZE_BILINEAR", "RESIZE_NEAREST_NEIGHBOR",
    "ROUND", "RSQRT", "SELECT_V2", "SHAPE", "SIN", "SLICE", "SOFTMAX", "SPACE_TO_BATCH_ND", "SPACE_TO_DEPTH",
    "SPLIT", "SPLIT_V", "SQRT", "SQUARE", "SQUARED_DIFFERENCE", "SQUEEZE", "STRIDED_SLICE", "SUB", "SUM",
    "SVDF", "TANH", "TRANSPOSE", "TRANSPOSE_CONV", "UNIDIRECTIONAL_SEQUENCE_LSTM", "UNPACK", "VAR_HANDLE",
    "WHILE", "ZEROS_LIKE",
}


def load_schema():
    try:
        from tensorflow.lite.python import schema_py_generated as fb
        return fb.Model.GetRootAsModel, fb.BuiltinOperator
    except ImportError:
        from ethosu.vela.tflite.Model import Model
        from ethosu.vela.tflite.BuiltinOperator import BuiltinOperator
        return Model.GetRootAsModel, BuiltinOperator


def tensor_info(t):
    shape = [int(x) for x in t.ShapeAsNumpy()] if t.ShapeLength() else []
    q = t.Quantization()
    scale, zp = 0.0, 0
    if q is not None and q.ScaleLength():
        scale, zp = float(q.ScaleAsNumpy()[0]), int(q.ZeroPointAsNumpy()[0])
    return shape, t.Type(), scale, zp


def main(path):
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"Arquivo não existe: {p}\nArquivos .tflite encontrados aqui:\n  " +
                         "\n  ".join(str(f) for f in sorted(Path('.').rglob('*.tflite'))) )
    if p.stat().st_size == 0:
        raise SystemExit(f"Arquivo vazio (0 bytes): {p}. A exportação ou o Vela falhou; rode de novo.")

    get_root, builtin_ops = load_schema()
    names = {v: k for k, v in vars(builtin_ops).items() if not k.startswith("_")}
    model = get_root(p.read_bytes(), 0)
    sg = model.Subgraphs(0)

    # operadores: "ethos-u" roda na NPU, o resto roda na CPU (mais lento)
    ops = Counter()
    for k in range(sg.OperatorsLength()):
        oc = model.OperatorCodes(sg.Operators(k).OpcodeIndex())
        code = max(getattr(oc, "BuiltinCode", lambda: 0)(), getattr(oc, "DeprecatedBuiltinCode", lambda: 0)())
        if oc.CustomCode():
            ops[oc.CustomCode().decode()] += 1
        else:
            ops[names.get(code, str(code))] += 1

    ok = True

    def check(cond, msg):
        nonlocal ok
        print(("  OK    " if cond else "  ERRO  ") + msg)
        ok &= bool(cond)

    print(f"Modelo: {p} ({p.stat().st_size / 1024:.0f} KB)")
    print(f"  operadores: {dict(ops)}")
    if "ethos-u" in ops:
        cpu = {k: v for k, v in ops.items() if k != "ethos-u"}
        print(f"  Vela aplicado. Operadores fora da NPU (rodam na CPU): {cpu or 'nenhum'}")
    else:
        print("  AVISO: sem operador ethos-u, este arquivo NÃO passou pelo Vela (vai rodar só na CPU).")

    missing = sorted(k for k in ops if k not in FIRMWARE_OPS)
    check(not missing, f"todos os operadores existem no firmware" +
          (f" (FALTAM: {missing})" if missing else ""))
    floats = sorted(k for k in ("DEQUANTIZE", "QUANTIZE") if k in ops)
    if floats:
        print(f"  AVISO: {floats} no meio do grafo = trechos em ponto flutuante (lentos na CPU da placa).")

    check(sg.InputsLength() == 1 and sg.OutputsLength() == 1,
          f"1 entrada e 1 saída (tem {sg.InputsLength()} e {sg.OutputsLength()})")
    ishape, itype, s_in, z_in = tensor_info(sg.Tensors(sg.Inputs(0)))
    oshape, otype, s_out, z_out = tensor_info(sg.Tensors(sg.Outputs(0)))
    print(f"  entrada {ishape} tipo={itype} escala={s_in:.6f} zp={z_in}")
    print(f"  saída   {oshape} tipo={otype} escala={s_out:.6f} zp={z_out}")

    check(len(ishape) == 4 and ishape[0] == 1 and ishape[3] in (1, 3), "entrada no formato [1, H, W, 3]")
    h, w = (ishape[1], ishape[2]) if len(ishape) == 4 else (0, 0)
    check(h == w and h >= 32 and h % 32 == 0, f"entrada quadrada e múltipla de 32 ({h}x{w})")
    check(itype == INT8, "entrada int8")
    check(abs(s_in - 1 / 255) < 1e-4 and z_in == -128,
          "quantização da entrada = pixel/255 com zero point -128 (o firmware faz pixel-128)")

    n_expected = ((w >> 5) ** 2 + (w >> 4) ** 2 + (w >> 3) ** 2) * 3
    check(len(oshape) == 3 and oshape[0] == 1, "saída no formato [1, N, 5+classes]")
    if len(oshape) == 3:
        check(oshape[1] == n_expected,
              f"N = {oshape[1]} (o firmware exige {n_expected} para {w}x{w}: escalas 8/16/32 x 3 âncoras)")
        check(6 <= oshape[2] <= 85, f"5 + classes entre 6 e 85 (tem {oshape[2]})")
    check(otype == INT8, "saída int8")
    check(0 < s_out < 0.1, "escala da saída < 0.1 (o firmware entende coordenadas normalizadas 0..1)")

    print("\nResultado:", "compatível com o firmware" if ok else "INCOMPATÍVEL, veja os itens com ERRO")
    return 0 if ok else 1


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    sys.exit(main(sys.argv[1]))