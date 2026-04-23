# POC local Ãºnica de geraÃ§Ã£o de imagem e vÃ­deo com Python + Hugging Face Diffusers

## 1. VisÃ£o geral

Esta POC foi criada para **estudo e aprendizado** sobre geraÃ§Ã£o de **imagem por prompt** e **vÃ­deo por prompt** usando **Python + Hugging Face Diffusers**.

O foco aqui nÃ£o Ã© produÃ§Ã£o. O foco Ã©:

- entender os conceitos
- ter uma base simples e organizada
- conseguir rodar em hardwares diferentes
- aprender a separar configuraÃ§Ã£o, perfis, pipelines e utilitÃ¡rios
- deixar o projeto fÃ¡cil de adaptar depois

Esta soluÃ§Ã£o foi pensada como **uma Ãºnica base de projeto**, e nÃ£o como duas POCs separadas.

Ela suporta:

- seleÃ§Ã£o de device: `auto`, `cpu`, `cuda`
- perfis de execuÃ§Ã£o: `auto`, `cpu_safe`, `gtx1080`, `high_quality`
- configuraÃ§Ã£o centralizada com `config.yaml`
- variÃ¡veis de ambiente com `.env.example`
- scripts CLI com `argparse`
- validaÃ§Ã£o de ambiente com `check_environment.py`

---

## 2. Objetivo do projeto

Com esta POC vocÃª consegue aprender, na prÃ¡tica:

1. o que Ã© a biblioteca **Diffusers**
2. como gerar **imagem por prompt**
3. como gerar **vÃ­deo por prompt**
4. como adaptar a execuÃ§Ã£o conforme o hardware disponÃ­vel
5. como estruturar um projeto reutilizÃ¡vel com **configuraÃ§Ã£o + perfis**
6. como separar bem cÃ³digo, configuraÃ§Ã£o e utilitÃ¡rios

---

## 3. O que Ã© Diffusers

`diffusers` Ã© uma biblioteca Python da Hugging Face voltada para **modelos de difusÃ£o**.

Ela oferece:

- pipelines prontos para inferÃªncia
- carregamento de modelos hospedados no Hugging Face Hub
- componentes reutilizÃ¡veis para imagem, vÃ­deo, Ã¡udio e outras tarefas generativas
- integraÃ§Ã£o com `torch` / PyTorch

Em termos prÃ¡ticos, o Diffusers ajuda vocÃª a:

- escolher um modelo
- carregar esse modelo
- configurar parÃ¢metros de geraÃ§Ã£o
- executar a inferÃªncia
- receber o resultado final como imagem, lista de frames ou outros artefatos

---

## 4. Conceitos principais

### 4.1 O que Ã© um pipeline

Um **pipeline** Ã© uma camada de alto nÃ­vel que organiza vÃ¡rias etapas internas de inferÃªncia.

Por exemplo, em geraÃ§Ã£o de imagem por texto, o pipeline cuida de:

- interpretar o prompt
- preparar embeddings de texto
- executar o processo de difusÃ£o
- decodificar o resultado para uma imagem final

Em vez de vocÃª montar cada peÃ§a manualmente, o pipeline jÃ¡ encapsula esse fluxo.

### 4.2 O que Ã© um modelo de difusÃ£o

Um **modelo de difusÃ£o** Ã© um modelo generativo que aprende a reconstruir dados a partir de ruÃ­do.

De forma simplificada:

1. o processo comeÃ§a de algo parecido com ruÃ­do aleatÃ³rio
2. o modelo vai removendo ruÃ­do em vÃ¡rias etapas
3. a cada etapa, ele Ã© guiado pelo prompt
4. no fim, surge a imagem ou sequÃªncia de frames coerente com o texto

### 4.3 Como funciona geraÃ§Ã£o de imagem em alto nÃ­vel

Em alto nÃ­vel:

1. vocÃª passa um prompt
2. o pipeline converte esse texto em representaÃ§Ãµes internas
3. o processo iterativo de difusÃ£o roda por vÃ¡rios `steps`
4. o resultado Ã© decodificado para uma imagem
5. a imagem Ã© salva em disco

### 4.4 Como funciona geraÃ§Ã£o de vÃ­deo em alto nÃ­vel

A lÃ³gica Ã© parecida, mas vÃ­deo adiciona uma complexidade importante:

- em vez de uma Ãºnica imagem, vocÃª precisa gerar **vÃ¡rios frames**
- esses frames precisam ter alguma consistÃªncia temporal
- isso aumenta muito o custo computacional e o consumo de memÃ³ria

EntÃ£o, na prÃ¡tica:

1. vocÃª envia um prompt
2. o pipeline de vÃ­deo gera uma sequÃªncia de frames
3. esses frames sÃ£o exportados como um arquivo de vÃ­deo, por exemplo `.mp4`

### 4.5 Por que vÃ­deo Ã© mais pesado que imagem

VÃ­deo Ã© mais pesado porque:

- gera mÃºltiplos frames em vez de um Ãºnico quadro
- exige mais memÃ³ria para tensores intermediÃ¡rios
- costuma usar modelos maiores
- precisa manter coerÃªncia entre quadros

Por isso, perfis para vÃ­deo normalmente reduzem:

- resoluÃ§Ã£o
- nÃºmero de frames
- nÃºmero de steps

### 4.6 CPU e GPU impactam como?

Em IA generativa, a GPU geralmente acelera muito a inferÃªncia.

- **CPU**: mais lenta, mas Ãºtil para estudo e compatibilidade
- **GPU NVIDIA com CUDA**: acelera o processamento e permite resoluÃ§Ãµes/steps maiores
- **GPU melhor no futuro**: permite perfis mais agressivos, mais qualidade e vÃ­deos menos limitados

### 4.7 Por que perfis ajudam

Perfis ajudam porque vocÃª nÃ£o quer alterar dezenas de parÃ¢metros manualmente a cada mÃ¡quina.

Em vez disso, vocÃª escolhe um perfil com intenÃ§Ã£o clara, como:

- `cpu_safe`: prioriza compatibilidade
- `gtx1080`: tenta equilibrar VRAM limitada e uso realista
- `high_quality`: prioriza qualidade para hardwares mais fortes

### 4.8 Biblioteca vs runtime/app

`diffusers` Ã© uma **biblioteca**.  
Ela nÃ£o Ã© um aplicativo final pronto para uso do mesmo jeito que algumas interfaces grÃ¡ficas sÃ£o.

Isso significa que:

- vocÃª escreve scripts Python
- escolhe modelos
- monta a configuraÃ§Ã£o
- controla execuÃ§Ã£o e exportaÃ§Ã£o

Essa POC existe justamente para mostrar como transformar a biblioteca em uma aplicaÃ§Ã£o local simples e reutilizÃ¡vel.

---

## 5. DiferenÃ§a entre Diffusers, Transformers, Ollama e LM Studio

### Diffusers

- biblioteca Python da Hugging Face
- especializada em modelos generativos baseados em difusÃ£o
- muito usada para imagem e vÃ­deo

### Transformers

- biblioteca Python da Hugging Face
- focada em arquiteturas Transformer
- muito usada para LLMs, classificaÃ§Ã£o, embeddings, traduÃ§Ã£o, etc.

### Ollama

- runtime/aplicativo para rodar modelos localmente
- muito associado a LLMs
- foca em experiÃªncia de uso e distribuiÃ§Ã£o local

### LM Studio

- aplicativo desktop para uso local de modelos
- muito associado a LLMs e inferÃªncia local
- oferece UI, gerenciamento e execuÃ§Ã£o simplificada

Resumo:

- **Diffusers** e **Transformers** sÃ£o bibliotecas para desenvolver
- **Ollama** e **LM Studio** sÃ£o ferramentas/apps para executar modelos com foco em praticidade

---

## 6. EstratÃ©gia multiambiente desta POC

Esta POC usa trÃªs camadas para adaptaÃ§Ã£o de hardware:

1. **detecÃ§Ã£o automÃ¡tica**
   - verifica se CUDA estÃ¡ disponÃ­vel
   - escolhe `cpu` ou `cuda` no modo `auto`

2. **configuraÃ§Ã£o centralizada**
   - `config.yaml` define defaults do projeto
   - `.env` pode sobrescrever alguns valores

3. **perfis**
   - perfis aplicam conjuntos coerentes de parÃ¢metros
   - permitem trocar o comportamento sem editar o cÃ³digo

Com isso, a mesma base pode ser usada em:

- mÃ¡quina somente CPU
- mÃ¡quina com GTX 1080
- mÃ¡quina futura com GPU mais forte

---

## 7. Modelos escolhidos

### 7.1 Modelo de imagem

Modelo padrÃ£o:

- `runwayml/stable-diffusion-v1-5`

Por que foi escolhido:

- Ã© muito conhecido no ecossistema
- Ã© comum em exemplos e tutoriais
- funciona bem como base de estudo
- Ã© relativamente fÃ¡cil trocar depois
- tem suporte maduro no Diffusers

### 7.2 Modelo de vÃ­deo

Modelo padrÃ£o:

- `cerspense/zeroscope_v2_576w`

Por que foi escolhido:

- Ã© um modelo conhecido para text-to-video no ecossistema Diffusers
- Ã© adequado como prova de conceito
- permite mostrar bem que vÃ­deo Ã© mais pesado que imagem
- Ã© fÃ¡cil de trocar via configuraÃ§Ã£o

### 7.3 ObservaÃ§Ã£o importante sobre vÃ­deo

Mesmo com esse modelo, geraÃ§Ã£o de vÃ­deo continua sendo pesada.

Por isso esta POC:

- usa configuraÃ§Ãµes conservadoras por padrÃ£o
- reduz resoluÃ§Ã£o/frames/steps em perfis mais modestos
- trata vÃ­deo como experimento de estudo

### 7.4 Como trocar os modelos depois

VocÃª pode trocar:

- no `config.yaml`
- em um perfil em `src/profiles.py`
- por argumento CLI, se quiser evoluir o projeto mais tarde

Normalmente basta alterar o identificador do repositÃ³rio do modelo:

```yaml
models:
  image_model: "runwayml/stable-diffusion-v1-5"
  video_model: "cerspense/zeroscope_v2_576w"
```

---

## 8. Arquitetura da soluÃ§Ã£o

O projeto foi dividido em responsabilidades separadas:

- `config.yaml`
  - defaults do projeto

- `.env.example`
  - exemplos de variÃ¡veis de ambiente

- `generate_image.py`
  - CLI principal para gerar imagem

- `generate_video.py`
  - CLI principal para gerar vÃ­deo

- `check_environment.py`
  - valida o ambiente local antes de gerar

- `src/config.py`
  - carrega e mescla configuraÃ§Ã£o de arquivo, ambiente, perfil e CLI

- `src/profiles.py`
  - define perfis prontos e explica o objetivo de cada um

- `src/pipelines/`
  - encapsula a lÃ³gica de carregamento e execuÃ§Ã£o dos pipelines Diffusers

- `src/utils/`
  - logging, detecÃ§Ã£o de device, exportaÃ§Ã£o de imagem e vÃ­deo

---

## 9. Estrutura de pastas

```text
hugging_face_diffusers/
â”œâ”€â”€ .env.example
â”œâ”€â”€ README.md
â”œâ”€â”€ architecture.md
â”œâ”€â”€ check_environment.py
â”œâ”€â”€ config.yaml
â”œâ”€â”€ generate_image.py
â”œâ”€â”€ generate_video.py
â”œâ”€â”€ notes.md
â”œâ”€â”€ requirements.txt
â”œâ”€â”€ run_examples.py
â”œâ”€â”€ outputs/
â”‚   â”œâ”€â”€ images/
â”‚   â””â”€â”€ videos/
â””â”€â”€ src/
    â”œâ”€â”€ __init__.py
    â”œâ”€â”€ config.py
    â”œâ”€â”€ profiles.py
    â”œâ”€â”€ pipelines/
    â”‚   â”œâ”€â”€ __init__.py
    â”‚   â”œâ”€â”€ image_generation.py
    â”‚   â””â”€â”€ video_generation.py
    â””â”€â”€ utils/
        â”œâ”€â”€ __init__.py
        â”œâ”€â”€ device.py
        â”œâ”€â”€ image_export.py
        â”œâ”€â”€ logging_utils.py
        â””â”€â”€ video_export.py
```

---

## 10. ExplicaÃ§Ã£o dos arquivos principais

### `config.yaml`

Arquivo principal de configuraÃ§Ã£o.  
Guarda defaults como:

- modo de device
- perfil
- modelos
- resoluÃ§Ã£o
- steps
- frames
- seed
- saÃ­das

### `.env.example`

Mostra como usar variÃ¡veis de ambiente para sobrescrever valores sem editar o YAML.

### `src/profiles.py`

Centraliza perfis reutilizÃ¡veis com objetivo claro.

### `src/config.py`

Une:

- config do YAML
- variÃ¡veis de ambiente
- perfil escolhido
- overrides do CLI

TambÃ©m valida valores e resolve `device` e `dtype`.

### `check_environment.py`

Script didÃ¡tico para estudar:

- Python instalado
- presenÃ§a de dependÃªncias
- versÃ£o do PyTorch
- CUDA
- device selecionado
- diretÃ³rios
- avisos por hardware

### `generate_image.py`

Executa geraÃ§Ã£o de imagem via prompt.

### `generate_video.py`

Executa geraÃ§Ã£o de vÃ­deo via prompt.

---

## 11. Como criar o ambiente virtual

### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### Linux/macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

## 12. Como instalar dependÃªncias

Instale primeiro os pacotes do projeto:

```bash
pip install -r requirements.txt
```

---

## 13. Como instalar PyTorch adequadamente

O PyTorch depende muito do seu ambiente.

### CPU

Exemplo:

```bash
pip install torch torchvision torchaudio
```

### CUDA

O ideal Ã© seguir a pÃ¡gina oficial do PyTorch e instalar a versÃ£o compatÃ­vel com sua GPU, driver e CUDA Runtime.

Exemplo genÃ©rico:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

ObservaÃ§Ã£o importante:

- a string `cu121` Ã© sÃ³ um exemplo
- confirme no site do PyTorch a combinaÃ§Ã£o correta para sua mÃ¡quina

---

## 14. Como validar o ambiente

Antes de gerar qualquer coisa:

```bash
python check_environment.py
```

VocÃª tambÃ©m pode informar config e perfil:

```bash
python check_environment.py --config config.yaml --profile gtx1080 --device auto
```

Esse script ajuda a entender:

- se o ambiente estÃ¡ pronto
- qual device serÃ¡ usado
- se o perfil Ã© compatÃ­vel
- quais riscos existem para CPU ou GPU

---

## 15. Como executar geraÃ§Ã£o de imagem

Exemplo simples:

```bash
python generate_image.py --prompt "a futuristic city at sunrise" --profile cpu_safe
```

ForÃ§ando CUDA:

```bash
python generate_image.py --prompt "a cinematic robot in the rain" --device cuda --profile gtx1080
```

Alterando steps e seed:

```bash
python generate_image.py --prompt "a watercolor castle on a hill" --steps 20 --seed 123
```

Definindo saÃ­da manual:

```bash
python generate_image.py --prompt "an astronaut reading in a library" --output outputs/images/custom_image.png
```

---

## 16. Como executar geraÃ§Ã£o de vÃ­deo

Exemplo simples:

```bash
python generate_video.py --prompt "a small boat crossing a misty lake" --profile cpu_safe
```

Perfil para GTX 1080:

```bash
python generate_video.py --prompt "a cinematic drone fly-through in a neon city" --profile gtx1080 --device cuda
```

Mudando frames e steps:

```bash
python generate_video.py --prompt "clouds moving over mountains at sunset" --frames 12 --steps 15
```

Definindo saÃ­da manual:

```bash
python generate_video.py --prompt "a futuristic corridor with moving lights" --output outputs/videos/custom_video.mp4
```

---

## 17. Como escolher perfil

Perfis disponÃ­veis:

- `auto`
- `cpu_safe`
- `gtx1080`
- `high_quality`

### `cpu_safe`

Use quando:

- a mÃ¡quina nÃ£o tem GPU
- vocÃª quer priorizar compatibilidade
- aceita execuÃ§Ã£o lenta

### `gtx1080`

Use quando:

- hÃ¡ CUDA disponÃ­vel
- a VRAM Ã© limitada
- vocÃª quer um equilÃ­brio mais realista para GPU intermediÃ¡ria/antiga

### `high_quality`

Use quando:

- vocÃª tem GPU mais forte
- quer mais qualidade
- aceita custo computacional maior

### `auto`

Use quando:

- vocÃª quer deixar o sistema adaptar defaults de forma mais neutra

---

## 18. Como forÃ§ar CPU ou GPU

### Via configuraÃ§Ã£o

No `config.yaml`:

```yaml
runtime:
  device_mode: "cpu"
```

Ou:

```yaml
runtime:
  device_mode: "cuda"
```

### Via CLI

```bash
python generate_image.py --prompt "test prompt" --device cpu
python generate_image.py --prompt "test prompt" --device cuda
python generate_image.py --prompt "test prompt" --device auto
```

A prioridade de sobrescrita nesta POC Ã©:

1. argumentos CLI
2. variÃ¡veis de ambiente
3. perfil
4. `config.yaml`

---

## 19. Como alterar resoluÃ§Ã£o, frames e steps

### No `config.yaml`

```yaml
image:
  width: 512
  height: 512
  steps: 25

video:
  width: 320
  height: 192
  num_frames: 12
  steps: 15
```

### Via CLI

Imagem:

```bash
python generate_image.py --prompt "test" --steps 30
```

VÃ­deo:

```bash
python generate_video.py --prompt "test" --frames 16 --steps 20
```

---

## 20. ExplicaÃ§Ã£o do `config.yaml`

O `config.yaml` concentra valores padrÃ£o do projeto.

Ele foi separado em blocos:

- `runtime`
- `models`
- `image`
- `video`
- `outputs`
- `logging`

Isso torna o arquivo mais legÃ­vel e fÃ¡cil de evoluir.

---

## 21. ExplicaÃ§Ã£o do `.env.example`

O `.env.example` mostra como usar variÃ¡veis de ambiente para personalizar o projeto.

Exemplos:

- `HF_HOME`
- `HF_TOKEN`
- `DEVICE_MODE`
- `PERFORMANCE_PROFILE`
- `LOG_LEVEL`

Isso Ã© Ãºtil quando vocÃª:

- quer trocar comportamento sem editar YAML
- quer guardar token fora do cÃ³digo
- quer usar caminhos diferentes por mÃ¡quina

---

## 22. ExplicaÃ§Ã£o do `profiles.py`

`src/profiles.py` existe para deixar explÃ­cito:

- quais perfis existem
- qual o objetivo de cada um
- quais parÃ¢metros cada perfil altera
- por que cada perfil faz sentido

Assim, o projeto fica didÃ¡tico e fÃ¡cil de manter.

---

## 23. ExplicaÃ§Ã£o do `check_environment.py`

Esse script valida o ambiente antes da inferÃªncia.

Ele verifica:

- versÃ£o do Python
- dependÃªncias importantes
- instalaÃ§Ã£o do PyTorch
- disponibilidade de CUDA
- nome da GPU, quando disponÃ­vel
- resoluÃ§Ã£o do device final
- diretÃ³rios esperados
- warnings relevantes

Ele foi pensado para aprendizado, entÃ£o o cÃ³digo estÃ¡ comentado.

---

## 24. LimitaÃ§Ãµes de hardware

### CPU

- pode funcionar para estudo
- serÃ¡ lenta para imagem
- pode ser muito lenta ou inviÃ¡vel para vÃ­deo

### GTX 1080

- consegue atender melhor imagem
- vÃ­deo ainda exige bastante cuidado
- resoluÃ§Ãµes, frames e steps devem ser moderados

### GPU mais forte

- permite perfis mais agressivos
- melhora tempo de geraÃ§Ã£o
- reduz algumas limitaÃ§Ãµes prÃ¡ticas de vÃ­deo

---

## 25. Tratamento de erros

A POC tenta tratar erros comuns com mensagens claras:

- dependÃªncias ausentes
- falta de CUDA quando o modo exige GPU
- configuraÃ§Ã£o invÃ¡lida
- valores inconsistentes
- erro de memÃ³ria
- falha ao salvar imagem
- falha ao exportar vÃ­deo
- modelo incompatÃ­vel

Mesmo assim, em IA generativa hÃ¡ muitas variaÃ§Ãµes de ambiente. EntÃ£o, em alguns cenÃ¡rios, ainda serÃ¡ necessÃ¡rio ajustar manualmente:

- modelo
- resoluÃ§Ã£o
- dtype
- steps
- frames

---

## 26. Troubleshooting

### Erro: `No module named ...`

Instale as dependÃªncias com:

```bash
pip install -r requirements.txt
```

### Erro relacionado a `torch` ou CUDA

Provavelmente a instalaÃ§Ã£o do PyTorch nÃ£o corresponde ao seu ambiente.  
Reinstale com a combinaÃ§Ã£o correta conforme a documentaÃ§Ã£o oficial do PyTorch.

### Erro de memÃ³ria na GPU

Tente:

- reduzir resoluÃ§Ã£o
- reduzir `steps`
- reduzir `num_frames`
- usar perfil `gtx1080` ou `cpu_safe`
- trocar `dtype` para `float16` em CUDA, quando suportado

### VÃ­deo muito lento

Isso Ã© esperado em muitos ambientes.  
Tente:

- usar resoluÃ§Ã£o menor
- menos frames
- menos steps
- comeÃ§ar validando imagem antes de vÃ­deo

### `device=cuda` mas CUDA nÃ£o foi detectado

Verifique:

- se a GPU NVIDIA estÃ¡ visÃ­vel no sistema
- se os drivers estÃ£o corretos
- se o PyTorch instalado Ã© a build com CUDA

### O modelo nÃ£o baixa

Verifique:

- conexÃ£o com internet
- necessidade de autenticaÃ§Ã£o no Hugging Face
- se o modelo escolhido existe e estÃ¡ acessÃ­vel

---

## 27. Fluxo ponta a ponta

O fluxo desta POC Ã©:

1. vocÃª escolhe um script (`generate_image.py` ou `generate_video.py`)
2. passa um prompt e, opcionalmente, `profile`, `device`, `seed`, etc.
3. o script carrega `config.yaml`
4. aplica sobrescritas do `.env`
5. aplica o perfil selecionado
6. aplica sobrescritas do CLI
7. resolve o `device` final (`cpu` ou `cuda`)
8. resolve `dtype` adequado ao contexto
9. carrega o pipeline Diffusers correspondente
10. executa a geraÃ§Ã£o
11. salva o resultado em `outputs/images` ou `outputs/videos`

---

## 28. O que eu estou aprendendo com essa POC

Esta POC ensina objetivamente:

- o papel da biblioteca Diffusers
- como carregar pipelines prontos
- como configurar modelos por arquivo
- como separar cÃ³digo e configuraÃ§Ã£o
- como usar perfis para mÃºltiplos hardwares
- como detectar CPU vs CUDA
- como gerar imagem por prompt
- como gerar vÃ­deo por prompt
- como tratar erros comuns de inferÃªncia local
- como organizar uma base Python didÃ¡tica para evoluÃ§Ãµes futuras

---

## 29. Como evoluir essa POC depois

EvoluÃ§Ãµes sugeridas:

- integrar com FastAPI
- criar interface web com Gradio ou Streamlit
- integrar com AnythingLLM como serviÃ§o auxiliar
- adicionar `image-to-image`
- adicionar `image-to-video`
- adicionar histÃ³rico das geraÃ§Ãµes em JSON
- adicionar cache local de execuÃ§Ãµes
- criar presets extras por hardware
- suportar mÃºltiplos backends de exportaÃ§Ã£o
- adicionar fila de jobs
- adicionar scheduler configurÃ¡vel
- adicionar negative prompts por CLI
- adicionar suporte futuro a GPUs mais fortes com perfis dedicados

---

## 30. PrÃ³ximos passos recomendados

Se vocÃª estiver estudando, a sequÃªncia sugerida Ã©:

1. rodar `check_environment.py`
2. gerar uma imagem simples
3. entender o merge de config, env, perfil e CLI
4. trocar o perfil
5. trocar o modelo de imagem
6. testar vÃ­deo com poucos frames
7. ajustar parÃ¢metros conforme o hardware real

---

## 31. Arquivos do projeto

Os conteÃºdos completos dos arquivos estÃ£o no prÃ³prio repositÃ³rio criado por esta POC:

- [README.md](/c:/Users/rueda/Desktop/iPort/hugging_face_diffusers/README.md)
- [architecture.md](/c:/Users/rueda/Desktop/iPort/hugging_face_diffusers/architecture.md)
- [notes.md](/c:/Users/rueda/Desktop/iPort/hugging_face_diffusers/notes.md)
- [config.yaml](/c:/Users/rueda/Desktop/iPort/hugging_face_diffusers/config.yaml)
- [.env.example](/c:/Users/rueda/Desktop/iPort/hugging_face_diffusers/.env.example)
- [generate_image.py](/c:/Users/rueda/Desktop/iPort/hugging_face_diffusers/generate_image.py)
- [generate_video.py](/c:/Users/rueda/Desktop/iPort/hugging_face_diffusers/generate_video.py)
- [check_environment.py](/c:/Users/rueda/Desktop/iPort/hugging_face_diffusers/check_environment.py)
- [run_examples.py](/c:/Users/rueda/Desktop/iPort/hugging_face_diffusers/run_examples.py)

---

## 32. ObservaÃ§Ã£o final

Esta POC foi desenhada para ser:

- simples
- funcional
- didÃ¡tica
- organizada
- fÃ¡cil de estudar
- fÃ¡cil de adaptar

Ela nÃ£o tenta esconder as limitaÃ§Ãµes reais de hardware. Pelo contrÃ¡rio: usa essas limitaÃ§Ãµes como parte do aprendizado.


---


# Resumo de execução

Para executar o projeto no Windows PowerShell, o fluxo básico é:

1. criar o ambiente virtual

```powershell
python -m venv .venv
```

2. ativar o ambiente virtual

```powershell
.\.venv\Scripts\Activate.ps1
```

Se o PowerShell bloquear a ativação de scripts, rode antes:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

3. instalar as dependências do projeto

```powershell
pip install -r requirements.txt
```

4. instalar o PyTorch adequado para sua máquina

Exemplo para CPU:

```powershell
pip install torch torchvision torchaudio
```

Exemplo para Windows com GPU NVIDIA e CUDA:

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

Observação importante:

- a combinação de `CUDA`, `Python` e `PyTorch` pode mudar ao longo do tempo
- se esse comando falhar, consulte o instalador oficial do PyTorch e escolha a combinação atual para o seu ambiente
- em Windows, confirme também se a GPU aparece no `nvidia-smi`

5. validar o ambiente

```powershell
python check_environment.py
```

Exemplo validando CUDA:

```powershell
python check_environment.py --profile gtx1080 --device cuda
```

6. gerar uma imagem

Exemplo conservador:

```powershell
python generate_image.py --prompt "a futuristic city at sunrise" --profile cpu_safe
```

Exemplo com GPU:

```powershell
python generate_image.py --prompt "a cinematic robot in the rain" --device cuda --profile gtx1080
```

Outros exemplos úteis:

```powershell
python generate_image.py --prompt "an astronaut reading in a library" --steps 20 --seed 123
python generate_image.py --prompt "a watercolor castle on a hill" --output outputs/images/teste.png
```

A imagem gerada vai para `outputs/images/`, a menos que você passe `--output`.

Sobre usar ambiente virtual no Windows: ele serve para isolar as dependências do projeto. Isso evita misturar versões de `torch`, `diffusers`, `transformers` e outros pacotes com projetos diferentes ou com o Python global da máquina.

Na prática, o `venv` ajuda porque:

- evita conflito entre projetos
- deixa a instalação mais previsível
- facilita testar versões diferentes de bibliotecas
- permite apagar e recriar o ambiente se algo quebrar
- evita "sujar" o Python global do Windows

Sem ambiente virtual, é comum acontecer:

- um projeto instalar uma versão de `torch`
- outro projeto precisar de outra versão
- tudo começar a conflitar no mesmo Python do sistema

No seu caso, como IA local costuma depender bastante de versão de `torch`, CUDA e bibliotecas associadas, usar `venv` é especialmente recomendado.
