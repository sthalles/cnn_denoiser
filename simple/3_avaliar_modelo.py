# -*- coding: utf-8 -*-
#
# Etapa 3: usar o modelo treinado para remover ruido e medir a qualidade.
#
# Versao simplificada e autocontida do pipeline principal
# (scripts/evaluation/evaluate.py). As contas sao as mesmas, entao os
# numeros reproduzem a avaliacao original (mesmos 200 arquivos de
# validacao, sorteados com a mesma semente).
#
# Para cada arquivo ruidoso da validacao:
#   1. corta o audio em segmentos de 2 s com 50% de sobreposicao
#   2. calcula a STFT de cada segmento e passa a magnitude pela rede
#   3. reconstroi o audio com a magnitude prevista e a fase do ruidoso
#   4. junta os segmentos (overlap-add) e mede o SDR antes e depois
#
# Rodar dentro do container tensorflow-2026 (sem o Keras legado):
#   env -u TF_USE_LEGACY_KERAS python simple/3_avaliar_modelo.py

import json
import os
import re

import numpy as np
import librosa
import soundfile as sf
import tensorflow as tf


# ---------------------------------------------------------------------------
# Parametros (os mesmos do pipeline principal)
# ---------------------------------------------------------------------------

TAXA_AMOSTRAGEM = 16000

TAMANHO_FFT = 512
PASSO_STFT = 128

AMOSTRAS_SEGMENTO = 32000      # 2 segundos
PASSO_SEGMENTO = 16000         # 50% de sobreposicao entre segmentos

# Constante global de normalizacao, calculada pela etapa 1 sobre o
# conjunto de treino (percentil 99,9 da magnitude da STFT em 300
# arquivos sorteados). Precisa ser a MESMA usada no treino.
MAX_MAG = 17.226323333741856

CAMINHO_MODELO = "models/runs/park_rced_park_data/best.keras"

PASTA_LIMPO = "dataset/CleanSpeech_training"
PASTA_RUIDOSO = "dataset/NoisySpeech_training"
ARQUIVO_MANIFESTO = "dataset/dataset_manifest.json"

PASTA_RESULTADOS = "simple/resultados_avaliacao"

NUM_ARQUIVOS = 200             # quantos arquivos de validacao avaliar
NUM_AUDIOS_SALVOS = 4          # quantos exemplos de audio gravar em disco

PADRAO_RUIDOSO = re.compile(r"noisy(\d+)_SNRdb_([\d.]+)_(clnsp\d+)\.wav$")


# ---------------------------------------------------------------------------
# Funcoes auxiliares
# ---------------------------------------------------------------------------

def carregar_audio(caminho):

    audio, sr = sf.read(caminho, dtype="float32")

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    if sr != TAXA_AMOSTRAGEM:
        raise RuntimeError(f"{caminho}: taxa {sr}, esperado {TAXA_AMOSTRAGEM}")

    return audio


def calcular_stft(audio):

    return librosa.stft(
        audio,
        n_fft=TAMANHO_FFT,
        hop_length=PASSO_STFT,
        win_length=TAMANHO_FFT,
        window="hann",
        center=True,
    )


def calcular_istft(stft):

    return librosa.istft(
        stft,
        hop_length=PASSO_STFT,
        win_length=TAMANHO_FFT,
        window="hann",
        center=True,
    )


def cortar_em_segmentos(audio):

    # segmentos de 2 s andando de 1 s em 1 s (50% de sobreposicao)
    segmentos = []
    inicio = 0
    while inicio + AMOSTRAS_SEGMENTO <= len(audio):
        pedaco = audio[inicio:inicio + AMOSTRAS_SEGMENTO]
        segmentos.append((inicio, pedaco))
        inicio = inicio + PASSO_SEGMENTO

    # se sobrar um trecho no final sem cobertura completa, acrescenta um
    # ultimo segmento alinhado com o fim do audio
    if len(audio) > AMOSTRAS_SEGMENTO:
        resto = len(audio) - AMOSTRAS_SEGMENTO
        if resto > PASSO_SEGMENTO and resto < AMOSTRAS_SEGMENTO:
            inicio_final = len(audio) - AMOSTRAS_SEGMENTO
            segmentos.append((inicio_final, audio[inicio_final:]))

    return segmentos


def juntar_segmentos(segmentos, tamanho_total):

    # overlap-add: soma os segmentos ponderados por uma janela de Hann e
    # divide pela soma das janelas
    saida = np.zeros(tamanho_total)
    pesos = np.zeros(tamanho_total)
    janela = np.hanning(AMOSTRAS_SEGMENTO)

    for inicio, pedaco in segmentos:
        fim = min(inicio + AMOSTRAS_SEGMENTO, tamanho_total)
        n = fim - inicio
        saida[inicio:fim] = saida[inicio:fim] + pedaco[:n] * janela[:n]
        pesos[inicio:fim] = pesos[inicio:fim] + janela[:n]

    pesos = np.maximum(pesos, 1e-8)
    return saida / pesos


def remover_ruido(modelo, ruidoso):

    segmentos = cortar_em_segmentos(ruidoso)

    # prepara a entrada da rede para cada segmento
    stfts = []
    entradas = []
    for inicio, pedaco in segmentos:
        stft = calcular_stft(pedaco)
        magnitude = np.abs(stft)
        entrada = np.log1p(magnitude / MAX_MAG).astype(np.float32)
        stfts.append(stft)
        entradas.append(entrada[..., np.newaxis])

    entradas = np.array(entradas)

    # passa todos os segmentos pela rede de uma vez
    previsoes = modelo.predict(entradas, verbose=0)

    # reconstroi cada segmento com a magnitude prevista e a fase ruidosa
    segmentos_limpos = []
    for i in range(len(segmentos)):

        log_magnitude = previsoes[i][..., 0].astype(np.float64)
        magnitude = np.expm1(log_magnitude)
        magnitude = np.maximum(magnitude, 0.0)
        magnitude = magnitude * MAX_MAG

        fase = np.angle(stfts[i])
        stft_estimada = magnitude * np.exp(1j * fase)

        audio_estimado = calcular_istft(stft_estimada)

        # garante exatamente 32000 amostras
        if len(audio_estimado) >= AMOSTRAS_SEGMENTO:
            audio_estimado = audio_estimado[:AMOSTRAS_SEGMENTO]
        else:
            faltam = AMOSTRAS_SEGMENTO - len(audio_estimado)
            audio_estimado = np.pad(audio_estimado, (0, faltam))

        segmentos_limpos.append((segmentos[i][0], audio_estimado))

    return juntar_segmentos(segmentos_limpos, len(ruidoso))


def calcular_sdr(referencia, estimado):

    # SDR = 10 * log10(potencia do sinal / potencia do erro)
    referencia = referencia.flatten()
    estimado = estimado.flatten()

    menor = min(len(referencia), len(estimado))
    referencia = referencia[:menor]
    estimado = estimado[:menor]

    erro = referencia - estimado
    potencia_sinal = np.sum(referencia ** 2) + 1e-8
    potencia_erro = np.sum(erro ** 2) + 1e-8

    return 10 * np.log10(potencia_sinal / potencia_erro)


# ---------------------------------------------------------------------------
# Programa principal
# ---------------------------------------------------------------------------

def main():

    # GPU com precisao mista, como no treino
    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if len(gpus) > 0:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")

    modelo = tf.keras.models.load_model(CAMINHO_MODELO, compile=False)
    print(f"Modelo carregado de {CAMINHO_MODELO}")

    with open(ARQUIVO_MANIFESTO) as f:
        manifesto = json.load(f)

    ids_validacao = set()
    for entrada in manifesto["splits"]["val"]:
        ids_validacao.add(entrada["id"])

    # lista todos os arquivos ruidosos da validacao
    arquivos_val = []
    for nome in sorted(os.listdir(PASTA_RUIDOSO)):

        m = PADRAO_RUIDOSO.match(nome)
        if m is None:
            continue

        indice = int(m.group(1))
        if indice not in ids_validacao:
            continue

        snr = float(m.group(2))
        nome_limpo = m.group(3)
        arquivos_val.append((indice, nome, snr, nome_limpo))

    print(f"Arquivos de validacao encontrados: {len(arquivos_val)}")

    # sorteia 200 arquivos (mesma semente da avaliacao original, entao
    # os arquivos escolhidos sao os mesmos)
    gerador = np.random.default_rng(0)
    if len(arquivos_val) > NUM_ARQUIVOS:
        indices = gerador.choice(len(arquivos_val), size=NUM_ARQUIVOS, replace=False)
        escolhidos = []
        for i in indices:
            escolhidos.append(arquivos_val[int(i)])
        arquivos_val = escolhidos

    pasta_audio = os.path.join(PASTA_RESULTADOS, "audio")
    os.makedirs(pasta_audio, exist_ok=True)

    resultados_por_snr = {}
    linhas_por_arquivo = []
    audios_gravados = 0
    contador = 0

    for indice, nome_ruidoso, snr, nome_limpo in arquivos_val:

        limpo = carregar_audio(os.path.join(PASTA_LIMPO, nome_limpo + ".wav"))
        ruidoso = carregar_audio(os.path.join(PASTA_RUIDOSO, nome_ruidoso))

        menor = min(len(limpo), len(ruidoso))
        limpo = limpo[:menor]
        ruidoso = ruidoso[:menor]

        estimado = remover_ruido(modelo, ruidoso)
        estimado = estimado[:menor]

        sdr_antes = calcular_sdr(limpo, ruidoso)
        sdr_depois = calcular_sdr(limpo, estimado)
        melhora = sdr_depois - sdr_antes

        if snr not in resultados_por_snr:
            resultados_por_snr[snr] = []
        resultados_por_snr[snr].append(melhora)

        linhas_por_arquivo.append({
            "id": indice,
            "arquivo": nome_ruidoso,
            "snr": snr,
            "sdr_antes": float(sdr_antes),
            "sdr_depois": float(sdr_depois),
            "melhora": float(melhora),
        })

        # grava alguns exemplos para ouvir depois
        if audios_gravados < NUM_AUDIOS_SALVOS:
            base = f"exemplo_id{indice}_snr{snr:.0f}"
            sf.write(os.path.join(pasta_audio, base + "_limpo.wav"),
                     limpo.astype(np.float32), TAXA_AMOSTRAGEM)
            sf.write(os.path.join(pasta_audio, base + "_ruidoso.wav"),
                     ruidoso.astype(np.float32), TAXA_AMOSTRAGEM)
            sf.write(os.path.join(pasta_audio, base + "_limpado.wav"),
                     estimado.astype(np.float32), TAXA_AMOSTRAGEM)
            audios_gravados = audios_gravados + 1

        contador = contador + 1
        if contador % 25 == 0:
            print(f"{contador}/{len(arquivos_val)} arquivos avaliados")

    # tabela final por SNR de entrada
    print()
    print("SNR entrada | arquivos | melhora media no SDR")
    print("------------|----------|---------------------")

    todas_melhoras = []
    for snr in sorted(resultados_por_snr.keys()):
        valores = resultados_por_snr[snr]
        media = float(np.mean(valores))
        print(f"  {snr:5.0f} dB  |   {len(valores):3d}    |    {media:+.2f} dB")
        for v in valores:
            todas_melhoras.append(v)

    media_geral = float(np.mean(todas_melhoras))
    print(f"\nMelhora media geral no SDR: {media_geral:+.2f} dB")
    print(f"({len(linhas_por_arquivo)} arquivos avaliados)")

    # guarda os numeros em json
    resumo = {
        "modelo": CAMINHO_MODELO,
        "n_arquivos": len(linhas_por_arquivo),
        "melhora_media_geral_dB": media_geral,
        "por_snr": {},
    }
    for snr in sorted(resultados_por_snr.keys()):
        valores = resultados_por_snr[snr]
        resumo["por_snr"][f"{snr:g}"] = {
            "n": len(valores),
            "melhora_media_dB": float(np.mean(valores)),
            "melhora_mediana_dB": float(np.median(valores)),
        }

    with open(os.path.join(PASTA_RESULTADOS, "resumo.json"), "w") as f:
        json.dump(resumo, f, indent=2)

    with open(os.path.join(PASTA_RESULTADOS, "por_arquivo.json"), "w") as f:
        json.dump(linhas_por_arquivo, f, indent=2)

    print(f"\nResultados salvos em {PASTA_RESULTADOS}/")


if __name__ == "__main__":
    main()
