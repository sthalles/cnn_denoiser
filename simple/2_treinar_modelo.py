# -*- coding: utf-8 -*-
#
# Etapa 2: definir e treinar a rede R-CED (Park & Lee, 2017).
#
# Versao simplificada e autocontida do pipeline principal
# (scripts/models/park_rced.py + scripts/training/train.py).
# A arquitetura, a funcao de custo e os hiperparametros sao os mesmos
# do treino que gerou models/runs/park_rced_park_data/best.keras.
#
# A rede trabalha no dominio da magnitude do espectrograma:
#   entrada: log1p da magnitude da STFT do audio ruidoso (normalizada)
#   saida:   log1p da magnitude da STFT do audio limpo (normalizada)
# A fase do sinal ruidoso e reaproveitada na reconstrucao (etapa 3).
#
# Rodar dentro do container tensorflow-2026:
#   python simple/2_treinar_modelo.py

import json
import os
import sys
import time

import numpy as np
import tensorflow as tf


# ---------------------------------------------------------------------------
# Parametros (os mesmos do treino original)
# ---------------------------------------------------------------------------

PASTA_TFRECORDS = "tfrecords_simple"
PASTA_SAIDA = "models/runs/park_simples"

EPOCAS = 60
TAMANHO_LOTE = 16
TAXA_APRENDIZADO = 0.001
DECAIMENTO_PESOS = 1e-05
SEMENTE = 42

NUM_FREQS = 257     # bins de frequencia da STFT (512/2 + 1)
NUM_QUADROS = 251   # quadros de tempo em um segmento de 2 s

# Piramide de filtros da R-CED: comeca estreita, alarga no meio e volta
# a estreitar (16 camadas convolucionais no total)
FILTROS = [12, 16, 20, 24, 32, 32, 32, 32,
           24, 24, 20, 20, 16, 16, 12, 12]

# Tamanho do nucleo de cada camada (frequencia x tempo), um pouco mais
# largo no tempo do que na frequencia
NUCLEOS = [(9, 13), (9, 11), (9, 9), (7, 9),
           (7, 7), (5, 7), (5, 5), (5, 5),
           (5, 5), (5, 5), (5, 7), (7, 7),
           (7, 9), (9, 9), (9, 11), (9, 13)]


# ---------------------------------------------------------------------------
# Preparacao da GPU
# ---------------------------------------------------------------------------

def preparar_gpu():

    gpus = tf.config.experimental.list_physical_devices("GPU")

    if len(gpus) == 0:
        print("ERRO: nenhuma GPU encontrada. Abortando.")
        sys.exit(3)

    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    # precisao mista (FP16) para aproveitar os tensor cores
    tf.keras.mixed_precision.set_global_policy("mixed_float16")

    print("GPU configurada com precisao mista (FP16)")


# ---------------------------------------------------------------------------
# Leitura dos TFRecords
# ---------------------------------------------------------------------------

TAMANHO_MATRIZ = NUM_FREQS * NUM_QUADROS

DESCRICAO_EXEMPLO = {
    "noisy_real": tf.io.FixedLenFeature([TAMANHO_MATRIZ], tf.float32),
    "noisy_imag": tf.io.FixedLenFeature([TAMANHO_MATRIZ], tf.float32),
    "clean_real": tf.io.FixedLenFeature([TAMANHO_MATRIZ], tf.float32),
    "clean_imag": tf.io.FixedLenFeature([TAMANHO_MATRIZ], tf.float32),
    "n_freq": tf.io.FixedLenFeature([], tf.int64),
    "n_time": tf.io.FixedLenFeature([], tf.int64),
    "snr": tf.io.FixedLenFeature([1], tf.float32),
}


def interpretar_exemplo(exemplo_serializado):

    campos = tf.io.parse_single_example(exemplo_serializado, DESCRICAO_EXEMPLO)

    ruidoso_real = tf.reshape(campos["noisy_real"], [NUM_FREQS, NUM_QUADROS])
    ruidoso_imag = tf.reshape(campos["noisy_imag"], [NUM_FREQS, NUM_QUADROS])
    limpo_real = tf.reshape(campos["clean_real"], [NUM_FREQS, NUM_QUADROS])
    limpo_imag = tf.reshape(campos["clean_imag"], [NUM_FREQS, NUM_QUADROS])

    # magnitude = raiz(real^2 + imag^2); o epsilon evita raiz de zero
    epsilon = 1e-8
    magnitude_ruidoso = tf.sqrt(ruidoso_real * ruidoso_real + ruidoso_imag * ruidoso_imag + epsilon)
    magnitude_limpo = tf.sqrt(limpo_real * limpo_real + limpo_imag * limpo_imag + epsilon)

    # log1p comprime a faixa de valores, e cada exemplo ganha um canal
    entrada = tf.math.log1p(magnitude_ruidoso)[..., tf.newaxis]
    alvo = tf.math.log1p(magnitude_limpo)[..., tf.newaxis]

    return entrada, alvo


def montar_dataset(pasta_split, embaralhar):

    arquivos = []
    for nome in sorted(os.listdir(pasta_split)):
        if nome.endswith(".tfrecord"):
            arquivos.append(os.path.join(pasta_split, nome))

    dados = tf.data.TFRecordDataset(arquivos)
    dados = dados.map(interpretar_exemplo, num_parallel_calls=tf.data.AUTOTUNE)

    if embaralhar:
        dados = dados.shuffle(2048, reshuffle_each_iteration=True)

    dados = dados.batch(TAMANHO_LOTE, drop_remainder=True)
    dados = dados.prefetch(tf.data.AUTOTUNE)

    return dados


# ---------------------------------------------------------------------------
# Definicao da rede R-CED
# ---------------------------------------------------------------------------

def criar_modelo():

    entrada = tf.keras.layers.Input(shape=(NUM_FREQS, NUM_QUADROS, 1), name="noisy_logmag")

    numero_camadas = len(FILTROS)   # 16
    ativacoes = []                  # guarda a saida de cada camada

    x = entrada
    for i in range(numero_camadas):

        # Conexoes de atalho (skip connections) em espelho: a camada i da
        # segunda metade recebe tambem a saida da camada (15 - i) da
        # primeira metade
        if i > numero_camadas // 2:

            atalho = ativacoes[numero_camadas - 1 - i]

            # se o numero de canais nao bate, ajusta com convolucao 1x1
            if atalho.shape[-1] != x.shape[-1]:
                atalho = tf.keras.layers.Conv2D(x.shape[-1], 1, padding="same",
                                                use_bias=False)(atalho)

            x = tf.keras.layers.Add()([x, atalho])

        # bloco basico: convolucao + batch norm + ReLU
        x = tf.keras.layers.Conv2D(FILTROS[i], NUCLEOS[i], padding="same",
                                   use_bias=False, name=f"conv_{i:02d}")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)

        ativacoes.append(x)

    # projecao final para 1 canal: o log1p da magnitude limpa prevista
    # (saida em float32 por causa da precisao mista)
    saida = tf.keras.layers.Conv2D(1, 1, padding="same", activation="linear",
                                   dtype="float32", name="logmag_out")(x)

    return tf.keras.Model(entrada, saida, name="park_rced")


# funcao de custo: erro quadratico medio sobre o log da magnitude
def erro_quadratico_medio(y_verdadeiro, y_previsto):
    y_verdadeiro = tf.cast(y_verdadeiro, tf.float32)
    y_previsto = tf.cast(y_previsto, tf.float32)
    return tf.reduce_mean(tf.square(y_verdadeiro - y_previsto))


# metrica auxiliar: erro absoluto medio
def erro_absoluto_medio(y_verdadeiro, y_previsto):
    y_verdadeiro = tf.cast(y_verdadeiro, tf.float32)
    y_previsto = tf.cast(y_previsto, tf.float32)
    return tf.reduce_mean(tf.abs(y_verdadeiro - y_previsto))


# ---------------------------------------------------------------------------
# Programa principal
# ---------------------------------------------------------------------------

def main():

    preparar_gpu()

    tf.random.set_seed(SEMENTE)
    np.random.seed(SEMENTE)

    os.makedirs(PASTA_SAIDA, exist_ok=True)

    # informacoes gravadas pela etapa 1
    with open(os.path.join(PASTA_TFRECORDS, "dataset_info.json")) as f:
        informacoes = json.load(f)

    exemplos_treino = informacoes["train_examples"]
    exemplos_val = informacoes["val_examples"]

    passos_por_epoca = exemplos_treino // TAMANHO_LOTE
    passos_validacao = exemplos_val // TAMANHO_LOTE

    print(f"Exemplos de treino: {exemplos_treino} ({passos_por_epoca} passos por epoca)")
    print(f"Exemplos de validacao: {exemplos_val} ({passos_validacao} passos)")

    dados_treino = montar_dataset(os.path.join(PASTA_TFRECORDS, "train"), embaralhar=True)
    dados_treino = dados_treino.repeat()
    dados_val = montar_dataset(os.path.join(PASTA_TFRECORDS, "val"), embaralhar=False)

    modelo = criar_modelo()
    print(f"Parametros do modelo: {modelo.count_params()}")

    otimizador = tf.keras.optimizers.AdamW(
        learning_rate=TAXA_APRENDIZADO,
        weight_decay=DECAIMENTO_PESOS,
        clipnorm=1.0,
    )

    # jit_compile=False: o treino original do Park nao usou compilacao XLA
    modelo.compile(
        optimizer=otimizador,
        loss=erro_quadratico_medio,
        metrics=[erro_absoluto_medio],
        jit_compile=False,
    )

    # callbacks:
    #  - guarda o melhor modelo segundo a perda de validacao
    #  - reduz a taxa de aprendizado quando a validacao estaciona
    #  - para o treino se nao houver melhora por 10 epocas
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(PASTA_SAIDA, "best.keras"),
            save_best_only=True, monitor="val_loss", mode="min", verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", patience=4, factor=0.5, min_lr=1e-06, verbose=1),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.CSVLogger(
            filename=os.path.join(PASTA_SAIDA, "training.csv")),
        tf.keras.callbacks.TerminateOnNaN(),
    ]

    inicio = time.time()

    historico = modelo.fit(
        dados_treino,
        epochs=EPOCAS,
        steps_per_epoch=passos_por_epoca,
        validation_data=dados_val,
        validation_steps=passos_validacao,
        callbacks=callbacks,
        verbose=2,
    )

    duracao = time.time() - inicio

    modelo.save(os.path.join(PASTA_SAIDA, "last.keras"))

    # guarda o historico de treino em json
    historico_json = {}
    for chave in historico.history:
        valores = []
        for v in historico.history[chave]:
            valores.append(float(v))
        historico_json[chave] = valores
    historico_json["duracao_segundos"] = duracao

    with open(os.path.join(PASTA_SAIDA, "history.json"), "w") as f:
        json.dump(historico_json, f, indent=2)

    print(f"Treino concluido em {duracao / 60:.1f} minutos")
    print(f"Melhor modelo salvo em {PASTA_SAIDA}/best.keras")


if __name__ == "__main__":
    main()
