#!/usr/bin/env python3
"""
Controla o Grove Vision AI V2 pela USB: faz UMA inferência a cada intervalo
(padrão 1 s), acende o LED do kit quando vê uma placa e salva a foto quando
a placa é confirmada.

O kit fica parado entre os pedidos; quem dita o ritmo é este script, mandando
comandos AT pela porta serial (o mesmo protocolo que o SenseCraft e a
biblioteca Arduino usam).

Regra para salvar: placa com confiança >= --limiar em --quadros inferências
seguidas, e pelo menos --espera segundos desde a última foto.

Onde salvar (--salvar):
  pc  (padrão) pede a imagem ao kit e grava o JPEG numa pasta do computador
  sd  manda o kit gravar no próprio cartão SD (pasta "Grove Vision AI (V2) Export")
  nao só acende o LED / registra no CSV, sem foto

LED (--led): acende enquanto houver placa acima do limiar e fica aceso mais
--led-segundos depois da última detecção.

Se o kit não responder, a inferência conta como "nenhuma placa" e o programa
segue. Se o cabo USB sair ou o kit reiniciar, ele espera e reconecta sozinho.

Instalação:  pip install pyserial
Uso:
  python grove_intervalo.py                         # acha a porta sozinho
  python grove_intervalo.py --led --salvar sd
  python grove_intervalo.py --porta /dev/ttyACM0 --intervalo 1 --limiar 65
Antes: feche o SenseCraft e qualquer outro programa usando a porta serial.
"""
import argparse
import base64
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    import serial
    from serial.tools import list_ports
except ImportError:
    raise SystemExit("Instale o pyserial:  pip install pyserial")

VID_PID = (0x1A86, 0x55D3)       # conversor USB-serial do Grove Vision AI V2
PREFIX = b"\r{"                  # cada mensagem do kit é um JSON que começa com \r{
CONNECTION_ERRORS = (serial.SerialException, OSError)


class GroveVisionAI:
    def __init__(self, port, baud=921600):
        self.ser = serial.Serial(port, baud, timeout=0.05)
        self.buf = b""
        time.sleep(0.2)
        self.ser.reset_input_buffer()

    def close(self):
        try:
            self.ser.close()
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    def _messages(self):
        """Lê o que chegou na serial e devolve as mensagens JSON completas
        (compactas ou formatadas em várias linhas)."""
        self.buf += self.ser.read(self.ser.in_waiting or 1)
        out = []
        dec = json.JSONDecoder()
        while True:
            start = self.buf.find(PREFIX)
            if start < 0:
                self.buf = self.buf[-1:]          # guarda um possível "\r" do próximo prefixo
                break
            text = self.buf[start + 1:].decode("latin-1")   # 1 byte = 1 caractere
            try:
                obj, end = dec.raw_decode(text)
            except json.JSONDecodeError:
                nxt = self.buf.find(PREFIX, start + 1)
                if nxt < 0:
                    self.buf = self.buf[start:]   # mensagem ainda incompleta: espera o resto
                    break
                self.buf = self.buf[nxt:]         # lixo/mensagem corrompida: pula para a próxima
                continue
            self.buf = self.buf[start + 1 + end:]
            if isinstance(obj, dict):
                out.append(obj)
        return out

    def _wait(self, name, msg_type, timeout):
        limit = time.monotonic() + timeout
        while time.monotonic() < limit:
            for m in self._messages():
                if m.get("type") == msg_type and name in str(m.get("name", "")):
                    return m
        raise TimeoutError(f"sem resposta {name} (type {msg_type}) em {timeout:.0f}s")

    def _discard_old(self):
        """Descarta respostas atrasadas de comandos anteriores."""
        self.ser.reset_input_buffer()
        self.buf = b""

    def command(self, body, name, timeout=3.0):
        """Envia AT+<body> e espera a resposta de operação (type 0)."""
        self._discard_old()
        self.ser.write(f"AT+{body}\r\n".encode())
        r = self._wait(name, 0, timeout)
        if r.get("code", 0) != 0:
            raise RuntimeError(f"AT+{body} falhou: código {r.get('code')} {r.get('data')}")
        return r

    def stop(self):
        """Interrompe uma inferência contínua que o SenseCraft tenha deixado rodando."""
        self.ser.write(b"AT+BREAK\r\n")
        time.sleep(0.3)
        self._discard_old()

    def led(self, on):
        """Acende/apaga o LED do kit (AT+LED não tem resposta)."""
        self.ser.write(f"AT+LED={1 if on else 0}\r\n".encode())

    def invoke(self, with_image=False, timeout=5.0):
        """Uma única inferência. Retorna o campo 'data' do evento (boxes, perf, image...)."""
        self.command(f"INVOKE=1,0,{0 if with_image else 1}", "INVOKE", timeout)
        ev = self._wait("INVOKE", 1, timeout)
        if ev.get("code", 0) != 0:
            raise RuntimeError(f"inferência falhou: código {ev.get('code')} {ev.get('data')}")
        return ev.get("data", {})


def find_port():
    for p in list_ports.comports():
        if (p.vid, p.pid) == VID_PID:
            return p.device
    return None


def connect(port_arg):
    """Abre o kit; se não estiver conectado, tenta de novo a cada 2 s."""
    aviso = False
    while True:
        port = port_arg or find_port()
        if port:
            try:
                kit = GroveVisionAI(port)
                kit.stop()
                print(f"[kit] conectado em {port}")
                return kit
            except CONNECTION_ERRORS as e:
                err = e
        else:
            err = "kit não encontrado nas portas USB"
        if not aviso:
            print(f"[kit] aguardando o kit ({err})... Ctrl+C para sair")
            aviso = True
        time.sleep(2)


def diagnostico(kit, tscore=None):
    """Mostra as respostas cruas do kit e salva o que a câmera está vendo."""
    def show(m):
        m = json.loads(json.dumps(m))
        img = m.get("data", {}).get("image") if isinstance(m.get("data"), dict) else None
        if img:
            m["data"]["image"] = f"<JPEG base64, {len(img)} caracteres>"
        print("   ", json.dumps(m, ensure_ascii=False)[:600])

    def raw(cmd, secs=1.5):
        print(f"\n>>> AT+{cmd}")
        kit._discard_old()
        kit.ser.write(f"AT+{cmd}\r\n".encode())
        got, limit = [], time.monotonic() + secs
        while time.monotonic() < limit:
            got += kit._messages()
        for m in got:
            show(m)
        if not got:
            print("    (nenhuma resposta)")
        return got

    for c in ("ID?", "VER?", "MODEL?", "ALGO?", "SENSOR?", "TSCORE?", "TIOU?"):
        raw(c)
    if tscore is not None:
        raw(f"TSCORE={tscore}")
    print("\n=== inferência com imagem ===")
    got = raw("INVOKE=1,0,0", secs=6)
    for m in got:
        d = m.get("data") if isinstance(m.get("data"), dict) else {}
        if d.get("image"):
            Path("diagnostico.jpg").write_bytes(base64.b64decode(d["image"]))
            print("\n    imagem da câmera salva em diagnostico.jpg")
        if m.get("type") == 1:
            print(f"    caixas recebidas: {d.get('boxes', 'campo boxes ausente')}")
    print("\nComo ler: sem nenhuma resposta = problema de comunicação; 'boxes': [] = o kit"
          " respondeu mas o modelo não detectou nada nessa imagem.")


def best_score(data, cls):
    """Maior confiança (0-100) entre as caixas da classe. Caixa = [x, y, w, h, score, classe]."""
    return max((b[4] for b in data.get("boxes", []) if len(b) >= 6 and b[5] == cls), default=0)


def save_photo(kit, args, score, pasta):
    if args.salvar == "pc":
        img = kit.invoke(with_image=True).get("image")
        if not img:
            raise RuntimeError("o kit não mandou a imagem")
        nome = pasta / (datetime.now().strftime("placa_%Y%m%d_%H%M%S_%f")[:-3] + f"_{score}.jpg")
        nome.write_bytes(base64.b64decode(img))
        return str(nome)
    kit.command('ACTION="save_jpeg()"', "ACTION")  # grava o quadro da próxima inferência
    try:
        kit.invoke(with_image=False)
    finally:
        kit.command('ACTION=""', "ACTION")          # remove a ação
    return "cartão SD"


def main():
    ap = argparse.ArgumentParser(description="Inferência a cada N segundos no Grove Vision AI V2 via USB")
    ap.add_argument("--porta", help="porta serial (padrão: detecta pelo VID/PID do kit)")
    ap.add_argument("--intervalo", type=float, default=1.0, help="segundos entre inferências")
    ap.add_argument("--limiar", type=int, default=30, help="confiança mínima 0-100 para contar como placa")
    ap.add_argument("--quadros", type=int, default=2, help="inferências seguidas acima do limiar para salvar")
    ap.add_argument("--espera", type=float, default=5.0, help="segundos mínimos entre fotos salvas")
    ap.add_argument("--classe", type=int, default=0, help="id da classe 'placa' no modelo")
    ap.add_argument("--salvar", choices=("pc", "sd", "nao"), default="sd", help="onde gravar as fotos")
    ap.add_argument("--pasta", default="fotos_placas", help="pasta das fotos no computador (--salvar pc)")
    ap.add_argument("--led", action="store_true", help="acender o LED do kit quando houver placa")
    ap.add_argument("--led-segundos", type=float, default=2.0, help="tempo que o LED fica aceso após a detecção")
    ap.add_argument("--log", default="deteccoes.csv", help="CSV com todas as inferências ('' desliga)")
    ap.add_argument("--tscore", type=int, default=None,
                    help="limiar de confiança do PRÓPRIO kit (0-100); caixas abaixo dele nem são enviadas")
    ap.add_argument("--debug", action="store_true", help="mostrar as caixas cruas de cada inferência")
    ap.add_argument("--diagnostico", action="store_true",
                    help="só testa a comunicação, mostra as respostas do kit e salva diagnostico.jpg")
    args = ap.parse_args()

    if args.diagnostico:
        kit = connect(args.porta)
        try:
            diagnostico(kit, args.tscore)
        finally:
            kit.close()
        return

    pasta = Path(args.pasta)
    if args.salvar == "pc":
        pasta.mkdir(parents=True, exist_ok=True)
    log_fh = open(args.log, "a", newline="") if args.log else None
    log = csv.writer(log_fh) if log_fh else None
    if log_fh and log_fh.tell() == 0:
        log.writerow(["hora", "score", "inferencia_ms", "salvou"])

    destino = {"pc": str(pasta), "sd": "cartão SD do kit", "nao": "não salva"}[args.salvar]
    print(f"[kit] 1 inferência a cada {args.intervalo}s | limiar {args.limiar} | {args.quadros} seguidas | "
          f"fotos: {destino} | LED: {'sim' if args.led else 'não'}")

    kit = connect(args.porta)
    if args.tscore is not None:
        try:
            kit.command(f"TSCORE={args.tscore}", "TSCORE")
            print(f"[kit] limiar interno do kit ajustado para {args.tscore}")
        except (TimeoutError, RuntimeError) as e:
            print(f"[kit] não consegui ajustar o TSCORE: {e}")
    if args.led:
        kit.led(False)
    seq, last_photo, last_seen, led_on = 0, -1e9, -1e9, False
    next_t = time.monotonic()
    try:
        while True:
            next_t += args.intervalo
            try:
                try:
                    data = kit.invoke(with_image=False)
                except (TimeoutError, RuntimeError) as e:
                    print(f"[erro] {e}  (contando como nenhuma placa)")
                    data = {}

                if args.debug:
                    print(f"    [debug] boxes={data.get('boxes', 'ausente')} chaves={sorted(data.keys())}")
                score = best_score(data, args.classe)
                perf = data.get("perf", [0, 0, 0])
                agora = time.monotonic()
                if score >= args.limiar:
                    seq, last_seen = seq + 1, agora
                else:
                    seq = 0

                if args.led:
                    want = agora - last_seen <= args.led_segundos
                    if want != led_on:
                        kit.led(want)
                        led_on = want

                salvou = ""
                if args.salvar != "nao" and seq >= args.quadros and agora - last_photo >= args.espera:
                    try:
                        salvou = save_photo(kit, args, score, pasta)
                        last_photo = agora
                        print(f">> foto salva ({salvou})")
                    except (TimeoutError, RuntimeError) as e:
                        print(f">> não salvou: {e}")
                    seq = 0

            except CONNECTION_ERRORS as e:
                print(f"[kit] conexão perdida ({e}); reconectando...")
                kit.close()
                kit = connect(args.porta)
                seq, led_on = 0, False
                if args.led:
                    kit.led(False)
                next_t = time.monotonic()
                continue

            hora = datetime.now().strftime("%H:%M:%S")
            inf_ms = perf[1] if len(perf) > 1 else "?"
            print(f"{hora} placa {score:3d} | inferência {inf_ms} ms | seguidas {seq}"
                  + (" | LED aceso" if led_on else ""))
            if log:
                log.writerow([datetime.now().isoformat(timespec="seconds"), score,
                              perf[1] if len(perf) > 1 else "", salvou])
                log_fh.flush()

            time.sleep(max(0.0, next_t - time.monotonic()))
            if time.monotonic() - next_t > args.intervalo:   # atrasou muito: não tenta "compensar"
                next_t = time.monotonic()
    except KeyboardInterrupt:
        print("\nencerrando")
    finally:
        try:
            if args.led:
                kit.led(False)
            if args.salvar == "sd":
                kit.command('ACTION=""', "ACTION")
        except Exception:
            pass
        kit.close()
        if log_fh:
            log_fh.close()


if __name__ == "__main__":
    sys.exit(main())