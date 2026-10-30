/*
 * grove_at.c - protocolo AT do Grove Vision AI V2 (SSCMA), sem dependência de hardware.
 *
 * O kit responde cada comando com mensagens JSON no formato  \r{ ... }\n
 *   type 0 = resposta ao comando, type 1 = evento (ex.: resultado da inferência)
 * Resultado da inferência:  "boxes": [[x, y, w, h, score, classe], ...]  (score 0-100)
 */
#include "grove_at.h"

#include <stdlib.h>
#include <string.h>

/* ------------------------- fila de recepção (ISR -> main) ------------------------- */
static volatile uint8_t  rx_buf[GROVE_RX_SIZE];
static volatile uint16_t rx_head, rx_tail;

void grove_rx_push(uint8_t b)          /* chamada pela interrupção da UART */
{
    uint16_t next = (uint16_t)((rx_head + 1u) % GROVE_RX_SIZE);
    if (next != rx_tail) {             /* fila cheia: descarta o byte */
        rx_buf[rx_head] = b;
        rx_head = next;
    }
}

static int rx_pop(void)
{
    if (rx_head == rx_tail) return -1;
    uint8_t b = rx_buf[rx_tail];
    rx_tail = (uint16_t)((rx_tail + 1u) % GROVE_RX_SIZE);
    return b;
}

/* ------------------------- montagem das mensagens ------------------------- */
static char     msg[GROVE_MSG_MAX];
static uint16_t msg_len;
static bool     in_msg;
static char     prev;

void grove_rx_flush(void)
{
    rx_tail = rx_head;                 /* descarta o que chegou antes do próximo comando */
    in_msg  = false;
    prev    = 0;
}

const char *grove_last_message(void) { return msg; }

bool grove_poll_message(void)
{
    int c;
    while ((c = rx_pop()) >= 0) {
        if (!in_msg) {
            if (prev == '\r' && c == '{') {        /* início: \r{ */
                in_msg  = true;
                msg_len = 0;
                msg[msg_len++] = '{';
            }
        } else {
            /* mensagens maiores que o buffer (ex.: com imagem) são cortadas;
               as caixas vêm antes da imagem, então continuam legíveis */
            if (msg_len < GROVE_MSG_MAX - 1) msg[msg_len++] = (char)c;
            if (c == '\n' && prev == '}') {        /* fim: }\n */
                msg[msg_len] = '\0';
                in_msg = false;
                prev   = (char)c;
                return true;
            }
        }
        prev = (char)c;
    }
    return false;
}

/* ------------------------- leitura de JSON (mínima) ------------------------- */
static const char *skip_ws(const char *p)
{
    while (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n') p++;
    return p;
}

/* devolve o ponteiro para o valor da chave "key" (já depois dos dois-pontos) */
static const char *json_value(const char *js, const char *key)
{
    char pattern[24];
    size_t n = strlen(key);
    if (n + 3 > sizeof(pattern)) return NULL;
    pattern[0] = '"';
    memcpy(pattern + 1, key, n);
    pattern[n + 1] = '"';
    pattern[n + 2] = '\0';

    const char *p = strstr(js, pattern);
    if (!p) return NULL;
    p = skip_ws(p + n + 2);
    if (*p != ':') return NULL;
    return skip_ws(p + 1);
}

bool grove_json_int(const char *js, const char *key, long *out)
{
    const char *p = json_value(js, key);
    if (!p) return false;
    char *end;
    long v = strtol(p, &end, 10);
    if (end == p) return false;
    *out = v;
    return true;
}

bool grove_json_str_contains(const char *js, const char *key, const char *text)
{
    const char *p = json_value(js, key);
    if (!p || *p != '"') return false;
    p++;
    const char *end = strchr(p, '"');
    if (!end) return false;
    size_t n = (size_t)(end - p), t = strlen(text);
    for (size_t i = 0; i + t <= n; i++)       /* procura "text" dentro do valor (ex.: "tag@INVOKE") */
        if (memcmp(p + i, text, t) == 0) return true;
    return false;
}

/* lê um número inteiro ou com casas decimais, sem usar ponto flutuante */
static const char *read_num(const char *p, long *out, bool *ok)
{
    char *end;
    long v = strtol(p, &end, 10);
    *ok = (end != p);
    if (*ok && *end == '.') {                 /* ignora a parte decimal */
        end++;
        while (*end >= '0' && *end <= '9') end++;
    }
    *out = v;
    return end;
}

int grove_best_score(const char *js, int classe)
{
    const char *p = json_value(js, "boxes");
    if (!p || *p != '[') return 0;
    p++;
    int best = 0;
    for (;;) {
        p = skip_ws(p);
        if (*p == ',') { p++; continue; }
        if (*p != '[') break;                 /* fim da lista (ou mensagem cortada) */
        p++;
        long v[6];
        int  n = 0;
        while (n < 6) {
            bool ok;
            p = read_num(skip_ws(p), &v[n], &ok);
            if (!ok) break;
            n++;
            p = skip_ws(p);
            if (*p == ',') p++; else break;
        }
        while (*p && *p != ']') p++;          /* pula até o fim desta caixa */
        if (*p != ']') break;
        p++;
        if (n >= 6 && v[5] == classe && v[4] > best) best = (int)v[4];
    }
    return best;
}
