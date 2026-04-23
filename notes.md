# Notas de estudo

## 1. Sobre a escolha dos modelos

### Imagem

`runwayml/stable-diffusion-v1-5` foi escolhido por ser um modelo conhecido e recorrente em tutoriais.

### Vídeo

`cerspense/zeroscope_v2_576w` foi escolhido como ponto de partida realista para estudo, sabendo que vídeo é pesado.

## 2. Sobre CPU

Rodar em CPU é útil para aprender a estrutura do projeto, mas não é a melhor experiência para inferência pesada.

Para estudo, isso continua valendo porque permite:

- validar dependências
- entender a arquitetura
- executar fluxos menores

## 3. Sobre a GTX 1080

A GTX 1080 ainda pode ser útil para aprendizado e POCs, mas pede ajustes conservadores.

Por isso o perfil `gtx1080` reduz:

- resolução de vídeo
- quantidade de frames
- quantidade de steps

## 4. Sobre dtypes

Nesta POC:

- `float32` é o padrão mais seguro para CPU
- `float16` é preferido em CUDA para aliviar memória quando o cenário permitir

Em alguns ambientes específicos, pode ser necessário experimentar manualmente.

## 5. Sobre o foco do projeto

O objetivo aqui não é atingir o melhor resultado visual possível.

O objetivo é aprender:

- como o projeto se organiza
- como o hardware influencia
- como mudar modelos e parâmetros
- como separar responsabilidades de forma limpa
