# -*- coding: utf-8 -*-
#
# Etapa 1: gerar os TFRecords a partir dos pares de audio limpo/ruidoso.
#
# Versao simplificada e autocontida do pipeline principal
# (scripts/data/build_tfrecords.py). Faz exatamente as mesmas contas,
# mas de forma sequencial (um arquivo por vez), sem paralelismo.
#
# O que o script faz:
#   1. Le o manifesto do dataset para saber quais arquivos sao de treino
#      e quais sao de validacao (a separacao e por locutor).
#   2. Calcula a constante global de normalizacao (max_mag) sobre uma
#      amostra de 300 arquivos de treino.
#   3. Para cada par (limpo, ruidoso): corta o audio em segmentos de
#      2 segundos, calcula a STFT de cada segmento, normaliza e grava
#      tudo em arquivos TFRecord.
#
# Rodar dentro do container tensorflow-2026:
#   python simple/1_gerar_tfrecords.py

import json
import os
import random
import re

import numpy as np
import librosa
import soundfile as sf
import tensorflow as tf


# ---------------------------------------------------------------------------
# Parametros (os mesmos do pipeline principal)
# ---------------------------------------------------------------------------

TAXA_AMOSTRAGEM = 16000        # 16 kHz

TAMANHO_FFT = 512              # janela de 32 ms
PASSO_STFT = 128               # passo de 8 ms (75% de sobreposicao)

DURACAO_SEGMENTO = 2.0         # segundos
AMOSTRAS_SEGMENTO = int(DURACAO_SEGMENTO * TAXA_AMOSTRAGEM)   # 32000

EXEMPLOS_POR_SHARD = 1024

PASTA_LIMPO = "dataset/CleanSpeech_training"
PASTA_RUIDOSO = "dataset/NoisySpeech_training"
ARQUIVO_MANIFESTO = "dataset/dataset_manifest.json"
PASTA_SAIDA = "tfrecords_simple"

# Nome dos arquivos ruidosos: noisy123_SNRdb_10.0_clnsp123.wav
PADRAO_RUIDOSO = re.compile(r"noisy(\d+)_SNRdb_([\d.]+)_(clnsp\d+)\.wav$")


# ---------------------------------------------------------------------------
# Funcoes auxiliares
# ---------------------------------------------------------------------------

def carregar_audio(caminho):

    audio, sr = sf.read(caminho, dtype="float32")

    # se for estereo, tira a media dos canais
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    if sr != TAXA_AMOSTRAGEM:
        raise RuntimeError(f"{caminho}: taxa {sr}, esperado {TAXA_AMOSTRAGEM}")

    return audio


def calcular_stft(audio):

    # STFT com janela de Hann, igual em todo o projeto
    return librosa.stft(
        audio,
        n_fft=TAMANHO_FFT,
        hop_length=PASSO_STFT,
        win_length=TAMANHO_FFT,
        window="hann",
        center=True,
    )


def coletar_pares(split, manifesto):

    # ids que pertencem a este split (treino ou validacao)
    ids_do_split = set()
    for entrada in manifesto["splits"][split]:
        ids_do_split.add(entrada["id"])

    pares = []
    for nome in sorted(os.listdir(PASTA_RUIDOSO)):

        m = PADRAO_RUIDOSO.match(nome)
        if m is None:
            continue

        indice = int(m.group(1))
        if indice not in ids_do_split:
            continue

        snr = float(m.group(2))
        nome_limpo = m.group(3) + ".wav"

        caminho_limpo = os.path.join(PASTA_LIMPO, nome_limpo)
        caminho_ruidoso = os.path.join(PASTA_RUIDOSO, nome)

        if not os.path.exists(caminho_limpo):
            continue

        pares.append((caminho_limpo, caminho_ruidoso, snr))

    return pares


def calcular_max_mag(pares_treino):

    # Estatistica global de normalizacao: percentil 99,9 da magnitude da
    # STFT em 300 arquivos ruidosos de treino sorteados; fica o maior valor.
    # (mesma semente e mesma conta do pipeline principal, entao o resultado
    # e identico: 17.226323333741856)

    gerador = np.random.default_rng(0)
    indices = gerador.choice(len(pares_treino), size=min(300, len(pares_treino)), replace=False)

    percentis = []
    contador = 0
    for i in indices:

        par = pares_treino[int(i)]
        audio = carregar_audio(par[1])          # arquivo ruidoso
        magnitude = np.abs(calcular_stft(audio))
        percentis.append(float(np.percentile(magnitude, 99.9)))

        contador = contador + 1
        if contador % 50 == 0:
            print(f"  max_mag: {contador}/{len(indices)} arquivos")

    return float(np.max(percentis))


def montar_exemplo(ruidoso_real, ruidoso_imag, limpo_real, limpo_imag, snr):

    # Um exemplo do TFRecord guarda as STFTs (parte real e imaginaria)
    # do segmento ruidoso e do segmento limpo, ja normalizadas.

    n_freq = ruidoso_real.shape[0]
    n_tempo = ruidoso_real.shape[1]

    def lista_float(matriz):
        return tf.train.Feature(
            float_list=tf.train.FloatList(value=matriz.astype(np.float32).flatten()))

    def inteiro(valor):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[int(valor)]))

    campos = {
        "noisy_real": lista_float(ruidoso_real),
        "noisy_imag": lista_float(ruidoso_imag),
        "clean_real": lista_float(limpo_real),
        "clean_imag": lista_float(limpo_imag),
        "n_freq": inteiro(n_freq),
        "n_time": inteiro(n_tempo),
        "snr": lista_float(np.array([snr], dtype=np.float32)),
    }

    exemplo = tf.train.Example(features=tf.train.Features(feature=campos))
    return exemplo.SerializeToString()


def gravar_split(pares, nome_split, max_mag):

    pasta = os.path.join(PASTA_SAIDA, nome_split)
    os.makedirs(pasta, exist_ok=True)

    inverso = 1.0 / max(max_mag, 1e-8)

    numero_shard = 0
    exemplos_no_shard = 0
    total_exemplos = 0

    caminho_shard = os.path.join(pasta, f"{nome_split}-{numero_shard:05d}.tfrecord")
    escritor = tf.io.TFRecordWriter(caminho_shard)

    contador_pares = 0
    for caminho_limpo, caminho_ruidoso, snr in pares:

        limpo = carregar_audio(caminho_limpo)
        ruidoso = carregar_audio(caminho_ruidoso)

        # garante o mesmo tamanho
        menor = min(len(limpo), len(ruidoso))
        limpo = limpo[:menor]
        ruidoso = ruidoso[:menor]

        # segmentos de 2 s, sem sobreposicao
        inicio = 0
        while inicio + AMOSTRAS_SEGMENTO <= menor:

            pedaco_limpo = limpo[inicio:inicio + AMOSTRAS_SEGMENTO]
            pedaco_ruidoso = ruidoso[inicio:inicio + AMOSTRAS_SEGMENTO]

            stft_limpo = calcular_stft(pedaco_limpo)
            stft_ruidoso = calcular_stft(pedaco_ruidoso)

            registro = montar_exemplo(
                np.real(stft_ruidoso) * inverso,
                np.imag(stft_ruidoso) * inverso,
                np.real(stft_limpo) * inverso,
                np.imag(stft_limpo) * inverso,
                snr,
            )

            # fecha o shard atual quando enche e abre o proximo
            if exemplos_no_shard >= EXEMPLOS_POR_SHARD:
                escritor.close()
                numero_shard = numero_shard + 1
                exemplos_no_shard = 0
                caminho_shard = os.path.join(pasta, f"{nome_split}-{numero_shard:05d}.tfrecord")
                escritor = tf.io.TFRecordWriter(caminho_shard)

            escritor.write(registro)
            exemplos_no_shard = exemplos_no_shard + 1
            total_exemplos = total_exemplos + 1

            inicio = inicio + AMOSTRAS_SEGMENTO

        contador_pares = contador_pares + 1
        if contador_pares % 500 == 0:
            print(f"  [{nome_split}] {contador_pares}/{len(pares)} pares processados")

    escritor.close()

    return total_exemplos, numero_shard + 1


# ---------------------------------------------------------------------------
# Programa principal
# ---------------------------------------------------------------------------

def main():

    with open(ARQUIVO_MANIFESTO) as f:
        manifesto = json.load(f)

    print("Coletando pares de arquivos...")
    pares_treino = coletar_pares("train", manifesto)
    pares_val = coletar_pares("val", manifesto)
    print(f"  treino: {len(pares_treino)} pares, validacao: {len(pares_val)} pares")

    print("Calculando constante de normalizacao (max_mag)...")
    max_mag = calcular_max_mag(pares_treino)
    print(f"  max_mag = {max_mag}")

    # Embaralha a ordem dos pares de treino antes de gravar, para os
    # exemplos ja sairem misturados nos shards (o leitor do treino usa um
    # buffer de embaralhamento pequeno em relacao ao dataset inteiro).
    random.seed(42)
    random.shuffle(pares_treino)

    print("Gravando TFRecords de treino...")
    exemplos_treino, shards_treino = gravar_split(pares_treino, "train", max_mag)

    print("Gravando TFRecords de validacao...")
    exemplos_val, shards_val = gravar_split(pares_val, "val", max_mag)

    # guarda as informacoes que o treino e a avaliacao precisam
    numero_quadros = AMOSTRAS_SEGMENTO // PASSO_STFT + 1     # 251
    numero_freqs = TAMANHO_FFT // 2 + 1                      # 257

    informacoes = {
        "n_freq": numero_freqs,
        "n_time": numero_quadros,
        "sample_rate": TAXA_AMOSTRAGEM,
        "n_fft": TAMANHO_FFT,
        "hop_length": PASSO_STFT,
        "segment_duration": DURACAO_SEGMENTO,
        "segment_overlap": 0.0,
        "max_mag": max_mag,
        "train_examples": exemplos_treino,
        "train_shards": shards_treino,
        "val_examples": exemplos_val,
        "val_shards": shards_val,
        "manifest_path": ARQUIVO_MANIFESTO,
    }

    os.makedirs(PASTA_SAIDA, exist_ok=True)
    with open(os.path.join(PASTA_SAIDA, "dataset_info.json"), "w") as f:
        json.dump(informacoes, f, indent=2)

    np.savez(os.path.join(PASTA_SAIDA, "norm_stats.npz"), max_mag=max_mag)

    print("Pronto.")
    print(f"  treino: {exemplos_treino} exemplos em {shards_treino} shards")
    print(f"  validacao: {exemplos_val} exemplos em {shards_val} shards")


if __name__ == "__main__":
    main()
