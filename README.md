# 🤖 Agente EDA Avançado

## Análise Exploratória de Dados Inteligente com IA

Sistema completo de análise exploratória de dados utilizando **LangChain**, **Google Gemini** e **Streamlit** para realizar análises automatizadas e inteligentes de datasets CSV.

---

## 📋 Índice

- [Características](#-características)
- [Requisitos](#-requisitos)
- [Instalação](#-instalação)
- [Configuração](#-configuração)
- [Como Usar](#-como-usar)
- [Arquitetura](#-arquitetura)
- [Ferramentas Disponíveis](#-ferramentas-disponíveis)
- [Exemplos de Uso](#-exemplos-de-uso)
- [Estrutura do Projeto](#-estrutura-do-projeto)
- [Solução de Problemas](#-solução-de-problemas)

---

## ✨ Características

### Análises Disponíveis

- **Análise Descritiva**: Resumo completo dos dados (tipos, estatísticas, qualidade)
- **Análise de Distribuições**: Histogramas interativos com estatísticas de assimetria
- **Análise de Correlações**: Matriz de correlação com heatmap interativo
- **Detecção de Outliers**: Métodos IQR e Z-score com boxplots
- **Análise de Clustering**: K-means para identificação de padrões
- **Análise Temporal**: Identificação de tendências e padrões temporais
- **Conclusões Consolidadas**: Síntese automática de todas as descobertas

### Recursos Inteligentes

- **Memória Contextual**: Sistema de memória que conecta análises anteriores
- **Visualizações Interativas**: Gráficos Plotly totalmente interativos
- **Recomendações Automáticas**: IA sugere próximos passos e ações
- **Insights Acionáveis**: Conclusões práticas baseadas em evidências
- **Interface Responsiva**: Layout adaptativo para diferentes resoluções

---

## 🔧 Requisitos

### Python Version
- Python 3.9 ou superior

### Dependências Principais

```
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
plotly>=5.17.0
langchain>=0.1.0
langchain-google-genai>=0.0.6
scikit-learn>=1.3.0
scipy>=1.11.0
pydantic>=2.0.0
```

---

## 📦 Instalação

### 1. Clone o repositório

```bash
git clone <seu-repositorio>
cd eda_agent
```

### 2. Crie um ambiente virtual

```bash
python -m venv venv

# Linux/Mac
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Instale as dependências

```bash
pip install -r requirements.txt
```

### 4. Crie o arquivo requirements.txt

```txt
streamlit==1.28.0
pandas==2.0.3
numpy==1.24.3
plotly==5.17.0
langchain==0.1.0
langchain-google-genai==0.0.6
scikit-learn==1.3.0
scipy==1.11.0
pydantic==2.5.0
matplotlib==3.7.2
seaborn==0.12.2
```

---

## ⚙️ Configuração

### Obter API Key do Google Gemini

1. Acesse: https://aistudio.google.com/app/apikey
2. Faça login com sua conta Google
3. Crie uma nova API Key
4. Copie a chave gerada

### Configurar a Aplicação

Existem duas formas de configurar a API Key:

#### Opção 1: Via Interface (Recomendado)
- Insira a API Key diretamente na interface do Streamlit
- Clique em "Conectar API"

#### Opção 2: Via Variável de Ambiente
```bash
# Linux/Mac
export GOOGLE_API_KEY="sua-api-key-aqui"

# Windows
set GOOGLE_API_KEY=sua-api-key-aqui
```

---

## 🚀 Como Usar

### Iniciar a Aplicação

```bash
streamlit run app.py
```

A aplicação abrirá automaticamente no navegador em `http://localhost:8501`

### Fluxo de Uso

1. **Carregar Dados**
   - Clique em "Browse files" na sidebar
   - Selecione um arquivo CSV
   - Aguarde o carregamento

2. **Conectar API**
   - Insira sua API Key do Gemini
   - Clique em "Conectar API"
   - Aguarde confirmação de conexão

3. **Fazer Análises**
   - Use os botões de análise rápida, OU
   - Digite perguntas customizadas no campo de texto
   - Clique em "Analisar"

4. **Visualizar Resultados**
   - Leia a análise textual
   - Explore os gráficos interativos
   - Revise as recomendações

5. **Gerar Conclusões**
   - Após várias análises, clique em "Conclusões Gerais"
   - Receba um relatório consolidado completo

---

## 🏗️ Arquitetura

### Componentes Principais

```
┌─────────────────────────────────────────┐
│         Interface Streamlit             │
│         (app.py)                        │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│         EDA Agent                       │
│         (agents/eda_agent.py)           │
│  ┌──────────────────────────────────┐   │
│  │  LangChain Agent                 │   │
│  │  + Google Gemini LLM             │   │
│  └──────────────────────────────────┘   │
└──────┬────────────────┬─────────────┬───┘
       │                │             │
       ▼                ▼             ▼
┌─────────────┐  ┌────────────┐  ┌──────────────┐
│  Analyzer   │  │   Memory   │  │   Executor   │
│  (utils/)   │  │ (memory/)  │  │   (utils/)   │
└─────────────┘  └────────────┘  └──────────────┘
```

### Fluxo de Dados

1. **Usuário** → Pergunta no Streamlit
2. **Streamlit** → Envia para EDA Agent
3. **Agent** → Processa com LangChain + Gemini
4. **Agent** → Seleciona e executa ferramentas apropriadas
5. **Ferramentas** → Geram análises e gráficos
6. **Memory** → Armazena insights para contexto futuro
7. **Agent** → Retorna resultado completo
8. **Streamlit** → Exibe análise e visualizações

---

## 🛠️ Ferramentas Disponíveis

### 1. data_summary_tool
**Função**: Resumo completo do dataset  
**Retorna**: 
- Dimensões do dataset
- Tipos de variáveis
- Qualidade dos dados (nulos, duplicatas)
- Estatísticas descritivas

**Quando usar**: Primeira análise de qualquer dataset

---

### 2. distribution_analysis_tool
**Função**: Análise de distribuições com histogramas  
**Retorna**:
- Histogramas interativos
- Média, mediana, desvio padrão
- Assimetria (skewness) e interpretação

**Quando usar**: Para entender padrões de distribuição das variáveis

---

### 3. correlation_analysis_tool
**Função**: Análise de correlações entre variáveis  
**Parâmetros**:
- `method`: "pearson", "spearman", "kendall"
- `threshold`: Limite para correlações significativas (padrão: 0.5)

**Retorna**:
- Matriz de correlação interativa
- Lista de correlações significativas
- Recomendações sobre multicolinearidade

**Quando usar**: Para identificar relacionamentos entre variáveis

---

### 4. outlier_detection_tool
**Função**: Detecção de outliers  
**Parâmetros**:
- `method`: "iqr" ou "zscore"

**Retorna**:
- Boxplots interativos
- Quantidade e porcentagem de outliers
- Valores extremos identificados

**Quando usar**: Para identificar valores atípicos nos dados

---

### 5. clustering_analysis_tool
**Função**: Análise de agrupamentos  
**Parâmetros**:
- `n_clusters`: Número de clusters (padrão: 3)
- `method`: "kmeans"

**Retorna**:
- Scatter plot dos clusters
- Estatísticas de cada cluster
- Tamanho dos grupos

**Quando usar**: Para identificar segmentos naturais nos dados

---

### 6. temporal_analysis_tool
**Função**: Análise de padrões temporais  
**Parâmetros**:
- `time_column`: Nome da coluna temporal (auto-detecta se None)

**Retorna**:
- Gráficos de séries temporais
- Período analisado
- Recomendações sobre sazonalidade

**Quando usar**: Quando há coluna de data/tempo no dataset

---

### 7. python_analysis_tool
**Função**: Execução de código Python customizado  
**Parâmetros**:
- `code`: String com código Python

**Quando usar**: Para análises específicas não cobertas por outras ferramentas

**Exemplo**:
```python
# Disponível no contexto: df, pd, np, plt, px, go
code = """
print(df['Amount'].describe())
fig = px.histogram(df, x='Amount')
fig.show()
"""
```

---

### 8. generate_conclusions_tool
**Função**: Gera conclusões consolidadas de todas as análises  
**Retorna**:
- Resumo de análises realizadas
- Padrões identificados
- Descobertas principais
- Insights sobre qualidade dos dados
- Recomendações consolidadas
- Próximos passos sugeridos

**Quando usar**: Após realizar várias análises para obter síntese completa

---

### 9. memory_summary_tool
**Função**: Acessa histórico de análises  
**Retorna**: Resumo do que já foi analisado

**Quando usar**: Para verificar o contexto das análises anteriores

---

## 💡 Exemplos de Uso

### Exemplo 1: Análise Completa do Kaggle Credit Card Fraud

```python
# Sequência recomendada de perguntas:

1. "Faça um resumo completo dos dados incluindo tipos, estatísticas e qualidade"

2. "Analise a distribuição das variáveis numéricas V1 a V5 com histogramas"

3. "Existe correlação entre as variáveis? Mostre a matriz de correlação"

4. "Detecte e analise outliers em todas as variáveis numéricas usando método IQR"

5. "Identifique padrões e agrupe os dados em 3 clusters"

6. "Analise padrões temporais na coluna Time"

7. "Quais são suas conclusões consolidadas sobre os dados analisados?"
```

### Exemplo 2: Perguntas Específicas

```python
# Análise focada em uma variável
"Analise a distribuição da variável Amount"

# Correlação específica
"Qual a correlação entre V1 e V2?"

# Outliers em coluna específica
"Existem outliers na coluna Amount?"

# Clustering customizado
"Agrupe os dados em 5 clusters usando kmeans"

# Código Python customizado
"Execute o seguinte código para calcular a média por classe: 
df.groupby('Class')['Amount'].mean()"
```

### Exemplo 3: Análise Iterativa

```python
1. "Faça um resumo dos dados"
   ↓
2. [Vê que há coluna Class] "Qual a distribuição da variável Class?"
   ↓
3. [Vê desbalanceamento] "Quantos casos de fraude vs normais existem?"
   ↓
4. [Entende o problema] "Há correlação entre as variáveis V* e Class?"
   ↓
5. "Quais suas conclusões sobre padrões de fraude?"
```

---

## 📁 Estrutura do Projeto

```
eda_agent/
│
├── app.py                          # Interface Streamlit principal
│
├── agents/
│   └── eda_agent.py               # Agente principal com LangChain
│
├── utils/
│   ├── advanced_analyzer.py       # Ferramentas de análise
│   └── export_utils.py            # Utilitários de exportação
│
├── memory/
│   └── enhanced_memory.py         # Sistema de memória
│
├── config.py                       # Configurações da aplicação
│
├── requirements.txt                # Dependências Python
│
└── README.md                       # Este arquivo
```

---


## 🔍 Solução de Problemas

### Erro: "Gemini Não Conectado"

**Causa**: API Key não configurada ou inválida  
**Solução**:
1. Verifique se a API Key está correta
2. Teste a chave em: https://aistudio.google.com
3. Certifique-se de ter carregado um CSV antes de conectar

---

### Análise muito lenta

**Possíveis causas**:
- Dataset muito grande
- Muitas colunas sendo analisadas
- Limite de taxa da API

**Soluções**:
- Reduza o número de colunas analisadas
- Use amostragem para datasets grandes
- Aguarde alguns segundos entre análises
- Considere usar `max_iterations` menor no AgentExecutor

---

### Erro de memória

**Causa**: Dataset muito grande  
**Solução**:
```python
# Carregar apenas parte dos dados
df = pd.read_csv('arquivo.csv', nrows=10000)

# Ou amostrar
df = df.sample(n=10000, random_state=42)
```

---

## 📊 Métricas de Performance

### Tempos Médios (dataset 100k linhas, 30 colunas)

- Carregamento de dados: 1-3 segundos
- Análise de resumo: 2-5 segundos
- Análise de correlação: 3-7 segundos
- Detecção de outliers: 2-6 segundos
- Análise de clustering: 5-10 segundos
- Geração de conclusões: 3-8 segundos

### Limites Recomendados

- Linhas: até 500k (acima disso, use amostragem)
- Colunas: até 100 (acima disso, selecione colunas relevantes)
- Análises por sessão: ilimitado (memória usa deque com max_interactions=100)

---

## 🔐 Segurança

### Boas Práticas

1. **Nunca commite API Keys** no código
2. Use variáveis de ambiente ou entrada manual
3. Adicione `.env` ao `.gitignore`
4. Rotacione API Keys periodicamente
5. Monitore uso da API no Google Cloud Console

### Arquivo .gitignore recomendado

```
# API Keys
.env
config_local.py

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/

# Streamlit
.streamlit/secrets.toml

# Data
*.csv
*.xlsx
data/
```

---

## 🤝 Contribuindo

Melhorias bem-vindas:

1. Fork o projeto
2. Crie uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

---

## 📝 Licença

Este projeto é desenvolvido para fins educacionais.

---

## 📧 Contato

Para dúvidas ou sugestões, entre em contato através do repositório.

---

## 🙏 Agradecimentos

- LangChain pela framework de agentes
- Google pelo Gemini AI
- Streamlit pela interface
- Plotly pelas visualizações
- Kaggle pelos datasets de exemplo

---

## 📚 Referências

- [LangChain Documentation](https://python.langchain.com/)
- [Google Gemini API](https://ai.google.dev/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Plotly Python](https://plotly.com/python/)
- [Scikit-learn](https://scikit-learn.org/)

---

**Versão**: 1.0.0  
**Última Atualização**: Setembro 2025  
**Status**: Produção