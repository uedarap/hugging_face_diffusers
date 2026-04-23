# POC local única de geração de imagem e vídeo com Python + Hugging Face Diffusers

## 1. Visão geral

Esta POC foi criada para **estudo e aprendizado** sobre geração de **imagem por prompt** e **vídeo por prompt** usando **Python + Hugging Face Diffusers**.

O foco aqui não é produção. O foco é:

- entender os conceitos
- ter uma base simples e organizada
- conseguir rodar em hardwares diferentes
- aprender a separar configuração, perfis, pipelines e utilitários
- deixar o projeto fácil de adaptar depois

Esta solução foi pensada como **uma única base de projeto**, e não como duas POCs separadas.

Ela suporta:

- seleção de device: `auto`, `cpu`, `cuda`
- perfis de execução: `auto`, `cpu_safe`, `gtx1080`, `high_quality`
- configuração centralizada com `config.yaml`
- variáveis de ambiente com `.env.example`
- scripts CLI com `argparse`
- validação de ambiente com `check_environment.py`

---

## 2. Objetivo do projeto

Com esta POC você consegue aprender, na prática:

1. o que é a biblioteca **Diffusers**
2. como gerar **imagem por prompt**
3. como gerar **vídeo por prompt**
4. como adaptar a execução conforme o hardware disponível
5. como estruturar um projeto reutilizável com **configuração + perfis**
6. como separar bem código, configuração e utilitários

---

## 3. O que é Diffusers

`diffusers` é uma biblioteca Python da Hugging Face voltada para **modelos de difusão**.

Ela oferece:

- pipelines prontos para inferência
- carregamento de modelos hospedados no Hugging Face Hub
- componentes reutilizáveis para imagem, vídeo, áudio e outras tarefas generativas
- integração com `torch` / PyTorch

Em termos práticos, o Diffusers ajuda você a:

- escolher um modelo
- carregar esse modelo
- configurar parâmetros de geração
- executar a inferência
- receber o resultado final como imagem, lista de frames ou outros artefatos

---

## 4. Conceitos principais

### 4.1 O que é um pipeline

Um **pipeline** é uma camada de alto nível que organiza várias etapas internas de inferência.

Por exemplo, em geração de imagem por texto, o pipeline cuida de:

- interpretar o prompt
- preparar embeddings de texto
- executar o processo de difusão
- decodificar o resultado para uma imagem final

Em vez de você montar cada peça manualmente, o pipeline já encapsula esse fluxo.

### 4.2 O que é um modelo de difusão

Um **modelo de difusão** é um modelo generativo que aprende a reconstruir dados a partir de ruído.

De forma simplificada:

1. o processo começa de algo parecido com ruído aleatório
2. o modelo vai removendo ruído em várias etapas
3. a cada etapa, ele é guiado pelo prompt
4. no fim, surge a imagem ou sequência de frames coerente com o texto

### 4.3 Como funciona geração de imagem em alto nível

Em alto nível:

1. você passa um prompt
2. o pipeline converte esse texto em representações internas
3. o processo iterativo de difusão roda por vários `steps`
4. o resultado é decodificado para uma imagem
5. a imagem é salva em disco

### 4.4 Como funciona geração de vídeo em alto nível

A lógica é parecida, mas vídeo adiciona uma complexidade importante:

- em vez de uma única imagem, você precisa gerar **vários frames**
- esses frames precisam ter alguma consistência temporal
- isso aumenta muito o custo computacional e o consumo de memória

Então, na prática:

1. você envia um prompt
2. o pipeline de vídeo gera uma sequência de frames
3. esses frames são exportados como um arquivo de vídeo, por exemplo `.mp4`

### 4.5 Por que vídeo é mais pesado que imagem

Vídeo é mais pesado porque:

- gera múltiplos frames em vez de um único quadro
- exige mais memória para tensores intermediários
- costuma usar modelos maiores
- precisa manter coerência entre quadros

Por isso, perfis para vídeo normalmente reduzem:

- resolução
- número de frames
- número de steps

### 4.6 CPU e GPU impactam como?

Em IA generativa, a GPU geralmente acelera muito a inferência.

- **CPU**: mais lenta, mas útil para estudo e compatibilidade
- **GPU NVIDIA com CUDA**: acelera o processamento e permite resoluções/steps maiores
- **GPU melhor no futuro**: permite perfis mais agressivos, mais qualidade e vídeos menos limitados

### 4.7 Por que perfis ajudam

Perfis ajudam porque você não quer alterar dezenas de parâmetros manualmente a cada máquina.

Em vez disso, você escolhe um perfil com intenção clara, como:

- `cpu_safe`: prioriza compatibilidade
- `gtx1080`: tenta equilibrar VRAM limitada e uso realista
- `high_quality`: prioriza qualidade para hardwares mais fortes

### 4.8 Biblioteca vs runtime/app

`diffusers` é uma **biblioteca**.  
Ela não é um aplicativo final pronto para uso do mesmo jeito que algumas interfaces gráficas são.

Isso significa que:

- você escreve scripts Python
- escolhe modelos
- monta a configuração
- controla execução e exportação

Essa POC existe justamente para mostrar como transformar a biblioteca em uma aplicação local simples e reutilizável.

---

## 5. Diferença entre Diffusers, Transformers, Ollama e LM Studio

### Diffusers

- biblioteca Python da Hugging Face
- especializada em modelos generativos baseados em difusão
- muito usada para imagem e vídeo

### Transformers

- biblioteca Python da Hugging Face
- focada em arquiteturas Transformer
- muito usada para LLMs, classificação, embeddings, tradução, etc.

### Ollama

- runtime/aplicativo para rodar modelos localmente
- muito associado a LLMs
- foca em experiência de uso e distribuição local

### LM Studio

- aplicativo desktop para uso local de modelos
- muito associado a LLMs e inferência local
- oferece UI, gerenciamento e execução simplificada

Resumo:

- **Diffusers** e **Transformers** são bibliotecas para desenvolver
- **Ollama** e **LM Studio** são ferramentas/apps para executar modelos com foco em praticidade

---

## 6. Estratégia multiambiente desta POC

Esta POC usa três camadas para adaptação de hardware:

1. **detecção automática**
   - verifica se CUDA está disponível
   - escolhe `cpu` ou `cuda` no modo `auto`

2. **configuração centralizada**
   - `config.yaml` define defaults do projeto
   - `.env` pode sobrescrever alguns valores

3. **perfis**
   - perfis aplicam conjuntos coerentes de parâmetros
   - permitem trocar o comportamento sem editar o código

Com isso, a mesma base pode ser usada em:

- máquina somente CPU
- máquina com GTX 1080
- máquina futura com GPU mais forte

---

## 7. Modelos escolhidos

### 7.1 Modelo de imagem

Modelo padrão:

- `runwayml/stable-diffusion-v1-5`

Por que foi escolhido:

- é muito conhecido no ecossistema
- é comum em exemplos e tutoriais
- funciona bem como base de estudo
- é relativamente fácil trocar depois
- tem suporte maduro no Diffusers

### 7.2 Modelo de vídeo

Modelo padrão:

- `cerspense/zeroscope_v2_576w`

Por que foi escolhido:

- é um modelo conhecido para text-to-video no ecossistema Diffusers
- é adequado como prova de conceito
- permite mostrar bem que vídeo é mais pesado que imagem
- é fácil de trocar via configuração

### 7.3 Observação importante sobre vídeo

Mesmo com esse modelo, geração de vídeo continua sendo pesada.

Por isso esta POC:

- usa configurações conservadoras por padrão
- reduz resolução/frames/steps em perfis mais modestos
- trata vídeo como experimento de estudo

### 7.4 Como trocar os modelos depois

Você pode trocar:

- no `config.yaml`
- em um perfil em `src/profiles.py`
- por argumento CLI, se quiser evoluir o projeto mais tarde

Normalmente basta alterar o identificador do repositório do modelo:

```yaml
models:
  image_model: "runwayml/stable-diffusion-v1-5"
  video_model: "cerspense/zeroscope_v2_576w"
```

---

## 8. Arquitetura da solução

O projeto foi dividido em responsabilidades separadas:

- `config.yaml`
  - defaults do projeto

- `.env.example`
  - exemplos de variáveis de ambiente

- `generate_image.py`
  - CLI principal para gerar imagem

- `generate_video.py`
  - CLI principal para gerar vídeo

- `check_environment.py`
  - valida o ambiente local antes de gerar

- `src/config.py`
  - carrega e mescla configuração de arquivo, ambiente, perfil e CLI

- `src/profiles.py`
  - define perfis prontos e explica o objetivo de cada um

- `src/pipelines/`
  - encapsula a lógica de carregamento e execução dos pipelines Diffusers

- `src/utils/`
  - logging, detecção de device, exportação de imagem e vídeo

---

## 9. Estrutura de pastas

```text
hugging_face_diffusers/
├── .env.example
├── README.md
├── architecture.md
├── check_environment.py
├── config.yaml
├── generate_image.py
├── generate_video.py
├── notes.md
├── requirements.txt
├── run_examples.py
├── outputs/
│   ├── images/
│   └── videos/
└── src/
    ├── __init__.py
    ├── config.py
    ├── profiles.py
    ├── pipelines/
    │   ├── __init__.py
    │   ├── image_generation.py
    │   └── video_generation.py
    └── utils/
        ├── __init__.py
        ├── device.py
        ├── image_export.py
        ├── logging_utils.py
        └── video_export.py
```

---

## 10. Explicação dos arquivos principais

### `config.yaml`

Arquivo principal de configuração.  
Guarda defaults como:

- modo de device
- perfil
- modelos
- resolução
- steps
- frames
- seed
- saídas

### `.env.example`

Mostra como usar variáveis de ambiente para sobrescrever valores sem editar o YAML.

### `src/profiles.py`

Centraliza perfis reutilizáveis com objetivo claro.

### `src/config.py`

Une:

- config do YAML
- variáveis de ambiente
- perfil escolhido
- overrides do CLI

Também valida valores e resolve `device` e `dtype`.

### `check_environment.py`

Script didático para estudar:

- Python instalado
- presença de dependências
- versão do PyTorch
- CUDA
- device selecionado
- diretórios
- avisos por hardware

### `generate_image.py`

Executa geração de imagem via prompt.

### `generate_video.py`

Executa geração de vídeo via prompt.

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

## 12. Como instalar dependências

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

O ideal é seguir a página oficial do PyTorch e instalar a versão compatível com sua GPU, driver e CUDA Runtime.

Exemplo genérico:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Observação importante:

- a string `cu121` é só um exemplo
- confirme no site do PyTorch a combinação correta para sua máquina

---

## 14. Como validar o ambiente

Antes de gerar qualquer coisa:

```bash
python check_environment.py
```

Você também pode informar config e perfil:

```bash
python check_environment.py --config config.yaml --profile gtx1080 --device auto
```

Esse script ajuda a entender:

- se o ambiente está pronto
- qual device será usado
- se o perfil é compatível
- quais riscos existem para CPU ou GPU

---

## 15. Como executar geração de imagem

Exemplo simples:

```bash
python generate_image.py --prompt "a futuristic city at sunrise" --profile cpu_safe
```

Forçando CUDA:

```bash
python generate_image.py --prompt "a cinematic robot in the rain" --device cuda --profile gtx1080
```

Alterando steps e seed:

```bash
python generate_image.py --prompt "a watercolor castle on a hill" --steps 20 --seed 123
```

Definindo saída manual:

```bash
python generate_image.py --prompt "an astronaut reading in a library" --output outputs/images/custom_image.png
```

---

## 16. Como executar geração de vídeo

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

Definindo saída manual:

```bash
python generate_video.py --prompt "a futuristic corridor with moving lights" --output outputs/videos/custom_video.mp4
```

---

## 17. Como escolher perfil

Perfis disponíveis:

- `auto`
- `cpu_safe`
- `gtx1080`
- `high_quality`

### `cpu_safe`

Use quando:

- a máquina não tem GPU
- você quer priorizar compatibilidade
- aceita execução lenta

### `gtx1080`

Use quando:

- há CUDA disponível
- a VRAM é limitada
- você quer um equilíbrio mais realista para GPU intermediária/antiga

### `high_quality`

Use quando:

- você tem GPU mais forte
- quer mais qualidade
- aceita custo computacional maior

### `auto`

Use quando:

- você quer deixar o sistema adaptar defaults de forma mais neutra

---

## 18. Como forçar CPU ou GPU

### Via configuração

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

A prioridade de sobrescrita nesta POC é:

1. argumentos CLI
2. variáveis de ambiente
3. perfil
4. `config.yaml`

---

## 19. Como alterar resolução, frames e steps

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

Vídeo:

```bash
python generate_video.py --prompt "test" --frames 16 --steps 20
```

---

## 20. Explicação do `config.yaml`

O `config.yaml` concentra valores padrão do projeto.

Ele foi separado em blocos:

- `runtime`
- `models`
- `image`
- `video`
- `outputs`
- `logging`

Isso torna o arquivo mais legível e fácil de evoluir.

---

## 21. Explicação do `.env.example`

O `.env.example` mostra como usar variáveis de ambiente para personalizar o projeto.

Exemplos:

- `HF_HOME`
- `HF_TOKEN`
- `DEVICE_MODE`
- `PERFORMANCE_PROFILE`
- `LOG_LEVEL`

Isso é útil quando você:

- quer trocar comportamento sem editar YAML
- quer guardar token fora do código
- quer usar caminhos diferentes por máquina

---

## 22. Explicação do `profiles.py`

`src/profiles.py` existe para deixar explícito:

- quais perfis existem
- qual o objetivo de cada um
- quais parâmetros cada perfil altera
- por que cada perfil faz sentido

Assim, o projeto fica didático e fácil de manter.

---

## 23. Explicação do `check_environment.py`

Esse script valida o ambiente antes da inferência.

Ele verifica:

- versão do Python
- dependências importantes
- instalação do PyTorch
- disponibilidade de CUDA
- nome da GPU, quando disponível
- resolução do device final
- diretórios esperados
- warnings relevantes

Ele foi pensado para aprendizado, então o código está comentado.

---

## 24. Limitações de hardware

### CPU

- pode funcionar para estudo
- será lenta para imagem
- pode ser muito lenta ou inviável para vídeo

### GTX 1080

- consegue atender melhor imagem
- vídeo ainda exige bastante cuidado
- resoluções, frames e steps devem ser moderados

### GPU mais forte

- permite perfis mais agressivos
- melhora tempo de geração
- reduz algumas limitações práticas de vídeo

---

## 25. Tratamento de erros

A POC tenta tratar erros comuns com mensagens claras:

- dependências ausentes
- falta de CUDA quando o modo exige GPU
- configuração inválida
- valores inconsistentes
- erro de memória
- falha ao salvar imagem
- falha ao exportar vídeo
- modelo incompatível

Mesmo assim, em IA generativa há muitas variações de ambiente. Então, em alguns cenários, ainda será necessário ajustar manualmente:

- modelo
- resolução
- dtype
- steps
- frames

---

## 26. Troubleshooting

### Erro: `No module named ...`

Instale as dependências com:

```bash
pip install -r requirements.txt
```

### Erro relacionado a `torch` ou CUDA

Provavelmente a instalação do PyTorch não corresponde ao seu ambiente.  
Reinstale com a combinação correta conforme a documentação oficial do PyTorch.

### Erro de memória na GPU

Tente:

- reduzir resolução
- reduzir `steps`
- reduzir `num_frames`
- usar perfil `gtx1080` ou `cpu_safe`
- trocar `dtype` para `float16` em CUDA, quando suportado

### Vídeo muito lento

Isso é esperado em muitos ambientes.  
Tente:

- usar resolução menor
- menos frames
- menos steps
- começar validando imagem antes de vídeo

### `device=cuda` mas CUDA não foi detectado

Verifique:

- se a GPU NVIDIA está visível no sistema
- se os drivers estão corretos
- se o PyTorch instalado é a build com CUDA

### O modelo não baixa

Verifique:

- conexão com internet
- necessidade de autenticação no Hugging Face
- se o modelo escolhido existe e está acessível

---

## 27. Fluxo ponta a ponta

O fluxo desta POC é:

1. você escolhe um script (`generate_image.py` ou `generate_video.py`)
2. passa um prompt e, opcionalmente, `profile`, `device`, `seed`, etc.
3. o script carrega `config.yaml`
4. aplica sobrescritas do `.env`
5. aplica o perfil selecionado
6. aplica sobrescritas do CLI
7. resolve o `device` final (`cpu` ou `cuda`)
8. resolve `dtype` adequado ao contexto
9. carrega o pipeline Diffusers correspondente
10. executa a geração
11. salva o resultado em `outputs/images` ou `outputs/videos`

---

## 28. O que eu estou aprendendo com essa POC

Esta POC ensina objetivamente:

- o papel da biblioteca Diffusers
- como carregar pipelines prontos
- como configurar modelos por arquivo
- como separar código e configuração
- como usar perfis para múltiplos hardwares
- como detectar CPU vs CUDA
- como gerar imagem por prompt
- como gerar vídeo por prompt
- como tratar erros comuns de inferência local
- como organizar uma base Python didática para evoluções futuras

---

## 29. Como evoluir essa POC depois

Evoluções sugeridas:

- integrar com FastAPI
- criar interface web com Gradio ou Streamlit
- integrar com AnythingLLM como serviço auxiliar
- adicionar `image-to-image`
- adicionar `image-to-video`
- adicionar histórico das gerações em JSON
- adicionar cache local de execuções
- criar presets extras por hardware
- suportar múltiplos backends de exportação
- adicionar fila de jobs
- adicionar scheduler configurável
- adicionar negative prompts por CLI
- adicionar suporte futuro a GPUs mais fortes com perfis dedicados

---

## 30. Próximos passos recomendados

Se você estiver estudando, a sequência sugerida é:

1. rodar `check_environment.py`
2. gerar uma imagem simples
3. entender o merge de config, env, perfil e CLI
4. trocar o perfil
5. trocar o modelo de imagem
6. testar vídeo com poucos frames
7. ajustar parâmetros conforme o hardware real

---

## 31. Arquivos do projeto

Os conteúdos completos dos arquivos estão no próprio repositório criado por esta POC:

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

## 32. Observação final

Esta POC foi desenhada para ser:

- simples
- funcional
- didática
- organizada
- fácil de estudar
- fácil de adaptar

Ela não tenta esconder as limitações reais de hardware. Pelo contrário: usa essas limitações como parte do aprendizado.


---


# Resumo de execução

Para executar o gerador de imagens no Windows, o fluxo básico é:

ativar o ambiente virtual
.venv\Scripts\Activate.ps1

instalar dependências
pip install -r requirements.txt

instalar o PyTorch adequado para sua máquina, Exemplo CPU:
pip install torch torchvision torchaudio

validar o ambiente
python check_environment.py

gerar uma imagem
python generate_image.py --prompt "a futuristic city at sunrise" --profile cpu_safe

Exemplos úteis:
python generate_image.py --prompt "a cinematic robot in the rain" --device cuda --profile gtx1080
python generate_image.py --prompt "an astronaut reading in a library" --steps 20 --seed 123
python generate_image.py --prompt "a watercolor castle on a hill" --output outputs/images/teste.png

A imagem gerada vai para outputs/images/, a menos que você passe --output.

Sobre usar ambiente virtual no Windows: ele serve para isolar as dependências do projeto. Isso evita misturar versões de torch, diffusers, transformers e outros pacotes com projetos diferentes ou com o Python global da máquina.

Na prática, o venv ajuda porque:

evita conflito entre projetos
deixa a instalação mais previsível
facilita testar versões diferentes de bibliotecas
permite apagar e recriar o ambiente se algo quebrar
evita “sujar” o Python global do Windows
Sem ambiente virtual, é comum acontecer:

um projeto instalar uma versão de torch
outro projeto precisar de outra versão
tudo começar a conflitar no mesmo Python do sistema
No seu caso, como IA local costuma depender bastante de versão de torch, CUDA e libs associadas, usar venv é especialmente recomendado.

Se quiser, eu posso te passar agora um passo a passo exato 
CPU only
GTX 1080 com CUDA
configurar o .env para baixar os modelos dentro da pasta do projeto