# Matrioska - Sistema de Orquestração de LLM com Estado Compartilhado

![Matrioska](https://live.staticflickr.com/8646/16075618524_6f3b5b199e_b.jpg)

## 📋 Visão Geral

**Matrioska** é um sistema avançado de orquestração para modelos de linguagem grande (LLMs) que implementa uma arquitetura modular com estado compartilhado. Inspirado no conceito das bonecas russas, o sistema decompõe tarefas complexas em módulos especializados que se comunicam através de um quadro branco compartilhado.

## 🎯 Funcionalidades Principais

- **🧩 Decomposição Arquitetônica**: Divide automaticamente tarefas complexas em módulos especializados
- **🧠 Estado Compartilhado**: Sistema de comunicação entre módulos via `shared_state`
- **💾 Persistência de Contexto**: Salva e restaura o progresso entre execuções
- **⚡ Execução em Hiperfoco**: Cada módulo executa com foco específico
- **🔗 Integração Inteligente**: Combina artefatos mantendo consistência
- **⏳ Simplicidade e Reaproveitamento**: Busca gerar códigos simplórios e usar CDNs/Bibliotecas

## 🏗️ Arquitetura

### Componentes Principais

1. **`LocalLLM`** - Wrapper para modelos Mistral com quantização 4-bit
2. **`MatrioskaOrchestrator`** - Orquestrador principal do pipeline
3. **`ContextManager`** - Gerenciador de estado e persistência
4. **`Architecture`** - Estrutura de dados para planejamento modular

### Fluxo de Execução

```
FASE 1: ARQUITETURA → FASE 2: EXECUÇÃO → FASE 3: MONTAGEM
    ↓                      ↓                    ↓
 Decomposição        Execução Modular    Integração Final
```

## 🚀 Como Usar

### Instalação

```bash
pip install transformers accelerate bitsandbytes torch sentencepiece protobuf
```

### Execução Básica

```python
from matrioska import LocalLLM, MatrioskaOrchestrator

# Inicializar modelo
llm = LocalLLM("mistralai/Mistral-7B-Instruct-v0.3")
orchestrator = MatrioskaOrchestrator(llm, base_path="/content")

# Executar tarefa
result = orchestrator.run("Criar sistema de gerenciamento de biblioteca com dashboard")
```

### Estrutura de Diretórios

```
/content/
├── matrioska_artifacts/     # Artefatos gerados por módulo
├── matrioska_checkpoints/   # Estado compartilhado e arquitetura
│   ├── shared_state.json    # Quadro branco compartilhado
│   └── architecture.json    # Plano arquitetural
└── matrioska_results.zip    # Download de resultados
```

## 📖 Sistema de Módulos

### Especificação de Módulo

```python
@dataclass
class ModuleSpec:
    id: str                    # Identificador único
    name: str                  # Nome descritivo
    description: str           # Descrição da funcionalidade
    inputs: str               # Dependências de entrada
    outputs: str              # Saídas esperadas
    dependencies: List[str]   # Módulos predecessores
    rules: str                # Regras específicas
    shared_state_reads: List[str]  # Chaves de leitura
    shared_state_writes: List[str] # Chaves de escrita
```

### Exemplo de Comunicação

```python
# Módulo A gera IDs
shared_state_updates = {
    "element_ids": ["#loginForm", "#bookList", "#dashboardStats"],
    "page_structure": {"login": "...", "catalog": "..."}
}

# Módulo B consome IDs
context = context_manager.get_shared_context(["element_ids"])
```

## 🔧 Configuração do Modelo

### Quantização 4-bit

```python
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)
```

### Parâmetros de Geração

- **max_tokens**: 4000
- **temperature**: 0.3
- **top_p**: 0.85
- **do_sample**: True

## 📊 Prompt de Arquitetura

O sistema usa um prompt especializado (`ARCHITECT_SYSTEM_PROMPT`) para decompor tarefas, definindo:

- **Objetivo geral** do projeto
- **Módulos especializados** com dependências
- **Contratos de comunicação** via shared_state
- **Manuais específicos** para cada módulo

## 💡 Casos de Uso

### Desenvolvimento Web
```python
result = orchestrator.run('''
Criar aplicação React com:
- Autenticação JWT
- CRUD de produtos
- Dashboard administrativo
- Design responsivo
''')
```

### Processamento de Dados
```python
result = orchestrator.run('''
Sistema de análise de dados com:
- Extração de APIs REST
- Limpeza e transformação
- Visualizações interativas
- Relatórios automáticos
''')
```

## 🎨 Exemplo de Saída

```
🪆 MATRIOSKA ORCHESTRATOR - Hiperfoco + SharedState
================================================================================

🏗️  FASE 1: ARQUITETURA
--------------------------------------------------------------------------------
📋 Tarefa: 'Library management system with dashboard'

✓ Projeto: Library Management System
✓ Objetivo: Create a complete library management system with authentication, CRUD, and dashboard
✓ Módulos: 3
   1. HTML Structure 📖[] ✍️['element_ids', 'page_structure']
   2. CSS Styling 📖['element_ids', 'page_structure'] ✍️['css_classes', 'color_scheme']
   3. Authentication Logic 📖['element_ids'] ✍️['auth_api', 'storage_keys']

⚡ FASE 2: EXECUÇÃO (Hiperfoco + Comunicação)
--------------------------------------------------------------------------------
🎯 HTML Structure
   ✓ Gerado (1542 chars)

🎯 CSS Styling
   📖 Lendo contexto: ['element_ids', 'page_structure']
   ✍️ Escreveu: ['css_classes', 'color_scheme']
   ✓ Gerado (2387 chars)

🔧 FASE 3: MONTAGEM
--------------------------------------------------------------------------------
🔗 Integrando artefatos...

✅ RESULTADO FINAL
================================================================================
📦 Library Management System
🎯 Create a complete library management system with authentication, CRUD, and dashboard

📂 Artefatos: 3
🧠 SharedState Keys: ['element_ids', 'page_structure', 'css_classes', 'color_scheme', 'auth_api']

🔗 Resultado Integrado:
--------------------------------------------------------------------------------
[Sistema completo integrado...]
```

## 🔄 Gestão de Estado

### Shared State
- **Persistente**: Sobrevive entre reinicializações
- **Estruturado**: Dicionário JSON serializável
- **Seletivo**: Módulos acessam apenas chaves relevantes

### Checkpoints
- Arquitetura salva em `architecture.json`
- Estado compartilhado em `shared_state.json`
- Artefatos individuais em arquivos texto

## 📦 Exportação de Resultados

```python
# Download completo dos resultados
from google.colab import files
!zip -r matrioska_results.zip /content/matrioska_artifacts /content/matrioska_checkpoints
files.download('matrioska_results.zip')
```

## 🛠️ Requisitos Técnicos

- **GPU**: NVIDIA T4 (16GB VRAM) ou superior
- **RAM**: 16GB+
- **Python**: 3.8+
- **Bibliotecas**: transformers, torch, bitsandbytes, sentencepiece

## 🔮 Roadmap

- [ ] Suporte a múltiplos modelos LLM
- [ ] Interface web para monitoramento
- [ ] Sistema de plugins para módulos customizados
- [ ] Otimização de memória para projetos grandes
- [ ] Integração com controle de versão

## 📄 Licença

Este projeto é destinado para fins de pesquisa e desenvolvimento.

---

**Matrioska**: Transformando complexidade em modularidade inteligente 🪆