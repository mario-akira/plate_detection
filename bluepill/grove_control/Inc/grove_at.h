#ifndef GROVE_AT_H
#define GROVE_AT_H

#include <stdbool.h>
#include <stdint.h>

#define GROVE_RX_SIZE 1024   /* fila de bytes recebidos (interrupção) */
#define GROVE_MSG_MAX 768    /* maior mensagem JSON guardada */

void        grove_rx_push(uint8_t b);   /* chamar na interrupção de recepção da UART */
void        grove_rx_flush(void);       /* descarta mensagens antigas antes de um comando */
bool        grove_poll_message(void);   /* true quando uma mensagem completa chegou */
const char *grove_last_message(void);   /* a última mensagem completa (texto JSON) */

bool grove_json_int(const char *js, const char *key, long *out);
bool grove_json_str_contains(const char *js, const char *key, const char *text);
int  grove_best_score(const char *js, int classe);   /* maior score (0-100) da classe, 0 se nada */

#endif
