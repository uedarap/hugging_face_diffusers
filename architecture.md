# Arquitetura da POC

## Objetivo

A arquitetura foi desenhada para manter uma única base de projeto para múltiplos hardwares, com separação clara entre:

- configuração
- perfis
- pipelines
- exportação
- validação de ambiente

## Camadas

### 1. Entrada do usuário

Os scripts `generate_image.py`, `generate_video.py` e `check_environment.py` recebem argumentos via `argparse`.

Esses scripts são responsáveis por:

- ler prompt e overrides
- chamar a resolução de configuração
- acionar a lógica principal

### 2. Configuração

`src/config.py` é a camada central de configuração.

Ela combina:

- valores padrão do `config.yaml`
- variáveis de ambiente
- ajustes do perfil escolhido
- argumentos informados na CLI

Essa ordem foi escolhida para deixar o projeto previsível e fácil de estudar.

### 3. Perfis

`src/profiles.py` define perfis intencionais.

Os perfis não substituem toda a configuração.  
Eles aplicam apenas overrides relevantes para certos cenários.

Isso deixa claro:

- o que muda por hardware
- por que muda
- onde mudar depois

### 4. Device e dtype

`src/utils/device.py` concentra:

- detecção de CUDA
- resolução final de device
- escolha sugerida de dtype
- resumo do hardware

Separar essa lógica evita espalhar decisões de hardware por todo o projeto.

### 5. Pipelines

`src/pipelines/image_generation.py` e `src/pipelines/video_generation.py` encapsulam a parte específica do Diffusers.

Esses módulos:

- carregam o modelo
- configuram o pipeline
- executam a inferência
- retornam o resultado pronto para exportação

### 6. Exportação

`src/utils/image_export.py` e `src/utils/video_export.py` cuidam do salvamento.

Essa separação ajuda porque exportação não é a mesma coisa que inferência.

### 7. Observabilidade

`src/utils/logging_utils.py` organiza o logger do projeto para manter a saída do terminal legível.

## Fluxo resumido

1. o usuário roda um script CLI
2. a configuração é carregada e validada
3. o perfil aplica overrides coerentes
4. o device final é resolvido
5. o pipeline adequado é carregado
6. a geração acontece
7. o arquivo é salvo em disco

## Escolhas didáticas

Algumas escolhas foram feitas para estudo:

- uso de `dataclasses` para clareza
- comentários explicando trade-offs
- divisão de módulos por responsabilidade
- defaults conservadores para vídeo
- tratamento explícito de erros

## O que trocar primeiro no futuro

As extensões mais naturais são:

- adicionar novos perfis
- trocar modelos no YAML
- adicionar parâmetros extras no CLI
- expor via API web
- salvar histórico em banco ou JSON
