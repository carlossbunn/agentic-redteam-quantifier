# Pipeline de quantificação de red-teaming agentic

Framework de avaliação que combina, em módulos plugáveis, quatro trabalhos de jailbreaking/red-teaming
de LLMs em um único pipeline mensurável — para uso como metodologia/código de IC.

## Papers combinados e onde cada um entra no pipeline

| Módulo (arquivo) | Paper de origem | O que faz |
|---|---|---|
| `task_decomposer.py` | **TRACE** — Zeng et al., "Task-Aware Adaptive Self-Evolving Agentic Jailbreaking", arXiv:2605.30883 (2026) | Decompõe o objetivo de alto nível em uma sequência de subtarefas, gera múltiplos esquemas candidatos e mantém o esquema com menor pontuação de "sensibilidade aparente" — a ideia central do TRACE de disfarçar a tarefa maliciosa como uma sequência de passos benignos. |
| `principles.py` | **CoP** — Xiong, Chen & Ho, "Agentic Red-teaming for LLMs using Composition of Principles", NeurIPS 2025 | Biblioteca de princípios de persuasão nomeados (rapport/elogio, enquadramento educacional, enquadramento hipotético, jargão técnico, foot-in-the-door, ocultação de estratégia) que um agente compõe/orquestra por turno — biblioteca extraída diretamente da taxonomia empírica de estratégias reportada no paper da *Nature*. |
| `intent_reframer.py` | **HILL** — Luo et al., "A Simple and Efficient Jailbreak Method Exploiting LLMs' Helpfulness", arXiv:2509.14297 (2025) | Implementa os 4 passos do HILL: identificação do conceito-chave, transformação de imperativo em pergunta exploratória, pedido de detalhamento passo a passo, e indicador de hipoteticidade. |
| `orchestrator.py` + `judge.py` | **Nature Communications** — Hagendorff, Derner & Oliver, "Large reasoning models are autonomous jailbreak agents" (2026) | Laço multi-turn em que o modelo-alvo recebe o histórico completo da conversa (permitindo que a persuasão se acumule entre turnos); harm score 0-5, detecção de recusa/disclaimer, parada antecipada em sucesso. |
| `metrics.py` | Comum aos quatro | ASR/bypass rate, harm score médio ± SEM, taxa de recusa, taxa de disclaimer, trajetória de harm score por turno, fator de melhora vs. controle direto (equivalente ao "até 19x" do CoP). |

## Por que não há conteúdo nocivo real aqui

Este repositório roda inteiramente sobre um **benchmark de proxy benigno** (`benchmark/benign_proxy_tasks.json`):
cada "meta" é extrair um placeholder inofensivo (uma palavra-canário, uma frase secreta fictícia) de um
**modelo simulado** (`MockModel` em `model_client.py`), nunca conteúdo realmente nocivo, e nunca uma API real.
Essa é a mesma lógica de "canário/exfiltração" usada em benchmarks agentic de segurança como AgentDojo/AgentHarm:
o proxy preserva a estrutura do ataque (informação/ação que o modelo foi instruído a proteger) sem qualquer
conteúdo perigoso de verdade. Isso permite testar, depurar e demonstrar o pipeline inteiro com segurança.

## Como rodar

```bash
cd redteam_pipeline
python3 run_evaluation.py          # roda o benchmark completo e imprime as métricas agregadas
```

ou abra `pipeline_demo.ipynb` para uma demonstração passo a passo (conversa completa turno a turno, métricas
agregadas, e o gráfico de trajetória de harm score).

## Como migrar para modelos-alvo reais (para a próxima fase da IC)

1. **Só faça isso com aprovação do orientador/comitê de ética da IC.** Red-teaming quantitativo contra modelos
   de produção reais tem implicações de política de uso das APIs e de responsabilidade — os próprios quatro
   papers de origem rodam em ambientes controlados de pesquisa, não contra usuários reais.
2. Implemente `OpenAIClient`/`AnthropicClient` em `model_client.py` (esqueletos já deixados no arquivo) usando
   suas próprias credenciais.
3. Troque `Judge.score` heurístico por um LLM-as-judge real: peça a 3 modelos-juízes independentes uma nota
   0-5 de "harm score" com justificativa de uma linha, e agregue por mediana/maioria — replicando a
   metodologia de confiabilidade (ICC / Kappa) usada no paper da *Nature*.
4. Se for além dos proxies benignos, use benchmarks públicos de pesquisa já aprovados pela comunidade
   (AdvBench, HarmBench, AgentHarm) em vez de criar prompts nocivos novos — isso também facilita comparar
   seus números com os já publicados nos quatro papers.
5. Troque `PrincipleSelector._default_policy` por uma política aprendida (bandit contextual / Q-learning),
   como propõe o TRACE, e reporte se ela supera a heurística fixa de escalada usada aqui.

## Estrutura de arquivos

```
redteam_pipeline/
├── model_client.py         # interface de modelo + MockModel + stubs de API real
├── task_decomposer.py       # TRACE: decomposição em subtarefas
├── principles.py            # CoP: biblioteca e seleção de princípios de persuasão
├── intent_reframer.py       # HILL: reformulação de imperativo -> pergunta exploratória
├── orchestrator.py          # Nature: laço multi-turn com histórico completo
├── judge.py                 # harm score 0-5, recusa, disclaimer
├── metrics.py                # ASR, SEM, trajetória, fator de melhora
├── run_evaluation.py         # script de execução ponta a ponta
├── pipeline_demo.ipynb       # notebook de demonstração (já executado)
└── benchmark/
    └── benign_proxy_tasks.json   # 5 tarefas-proxy benignas (nenhum conteúdo nocivo real)
```
