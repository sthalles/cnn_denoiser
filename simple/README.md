# Versão simples do pipeline (R-CED)

Esta pasta é uma **destilação simplificada e autocontida** do pipeline
principal do repositório (`scripts/`), reescrita para leitura linear — por
exemplo, para servir de apêndice ou material de estudo do TCC. Os scripts
não importam nada do resto do repositório: cada etapa é um arquivo único,
com laços explícitos, sem paralelismo e com comentários em português.

As **contas são exatamente as mesmas** do pipeline principal, então os
resultados também são os mesmos (verificação abaixo).

## Arquivos

| Arquivo | O que faz |
|---|---|
| `1_gerar_tfrecords.py` | Corta os pares limpo/ruidoso em segmentos de 2 s, calcula as STFTs, normaliza e grava TFRecords (equivale a `scripts/data/build_tfrecords.py`) |
| `2_treinar_modelo.py` | Define a R-CED (Park & Lee, 2017) e treina com o mesmo protocolo do modelo publicado (equivale a `scripts/models/park_rced.py` + `scripts/training/train.py`) |
| `3_avaliar_modelo.py` | Remove o ruído dos arquivos de validação e mede a melhora de SDR por SNR de entrada (equivale a `scripts/evaluation/evaluate.py`) |
| `demonstracao.ipynb` | Demonstração em um arquivo só: formas de onda, espectrogramas e áudio antes/depois |

## Como rodar

Tudo dentro do container `tensorflow-2026` (ver `CLAUDE.md` na raiz):

```bash
docker exec tensorflow-2026 bash -c "cd /workspace/cnn_denoiser && \
  python simple/1_gerar_tfrecords.py"          # gera tfrecords_simple/ (~70 GB)

docker exec tensorflow-2026 bash -c "cd /workspace/cnn_denoiser && \
  python simple/2_treinar_modelo.py"           # treina (~14 h na RTX 4070)

docker exec tensorflow-2026 bash -c "cd /workspace/cnn_denoiser && \
  env -u TF_USE_LEGACY_KERAS python simple/3_avaliar_modelo.py"
```

A etapa 3 funciona sem rodar as etapas 1 e 2: por padrão ela avalia o modelo
já treinado (`models/runs/park_rced_park_data/best.keras`). Para avaliar um
modelo retreinado pela etapa 2, troque a constante `CAMINHO_MODELO`.

## Verificação de equivalência

Conferido em 2026-08-09, no container, contra o pipeline principal:

- **Avaliação** — `3_avaliar_modelo.py` sorteia os mesmos 200 arquivos de
  validação (mesma semente) e reproduz a avaliação publicada; diferença
  máxima por arquivo de 0,002 dB (ruído numérico de GPU):

  | SNR entrada | Publicado | Reproduzido |
  |---|---|---|
  | 0 dB | +11,70 dB | +11,70 dB |
  | 5 dB | +9,91 dB | +9,91 dB |
  | 10 dB | +7,84 dB | +7,84 dB |
  | 15 dB | +5,03 dB | +5,03 dB |
  | **média geral** | **+8,55 dB** | **+8,55 dB** |

- **Arquitetura** — o modelo de `2_treinar_modelo.py` tem os mesmos 352.361
  parâmetros; os pesos de `best.keras` carregam direto nele
  (`load_weights`) e as predições saem **bit a bit idênticas** às do modelo
  original.

- **Dados** — os exemplos serializados por `1_gerar_tfrecords.py` são
  **byte a byte idênticos** aos de `scripts/data/build_tfrecords.py` para os
  mesmos pares, e a lista de pares coletada é idêntica (12.960 de treino).

## Constante de normalização

`MAX_MAG = 17.226323333741856` é a estatística global calculada pela etapa 1
sobre o conjunto de treino (percentil 99,9 da magnitude da STFT em 300
arquivos sorteados com semente fixa). O valor precisa ser o mesmo no treino
e na inferência.

## Diferenças intencionais (não mudam os resultados)

- Processamento sequencial (o pipeline principal usa multiprocessing e
  `tf.data.interleave`); a etapa 1 embaralha a ordem dos pares na gravação
  para compensar a leitura sequencial no treino.
- Nomes dos shards e mensagens de progresso diferentes.
- A etapa 3 grava 4 exemplos de áudio (o original grava 8).
