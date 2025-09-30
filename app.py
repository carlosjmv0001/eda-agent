import streamlit as st  
import pandas as pd  
import numpy as np  
from datetime import datetime  
from typing import Dict, List, Any, Optional  
import plotly.graph_objects as go
  
from agents.eda_agent import EDAAgent  
  
st.set_page_config(  
    page_title="Agente EDA Avançado - Análise Exploratória de Dados",  
    page_icon="🤖",  
    layout="wide",  
    initial_sidebar_state="expanded"  
)  
  
st.markdown("""  
<style>  
    .main-header {  
        font-size: 2.5rem;  
        font-weight: bold;  
        color: #1f77b4;  
        text-align: center;  
        margin-bottom: 2rem;  
    }  
    .analysis-result {  
        background-color: #ffffff;  
        padding: 1.5rem;  
        border-radius: 0.5rem;  
        border: 1px solid #e0e0e0;  
        margin: 1rem 0;  
    }
    .plot-container {
        margin: 1.5rem 0;
        padding: 1rem;
        border: 1px solid #e0e0e0;
        border-radius: 0.5rem;
    }
</style>  
""", unsafe_allow_html=True)  
  
def initialize_session():  
    """Initialize session state variables"""  
    if 'eda_agent' not in st.session_state:  
        st.session_state.eda_agent = None  
    if 'df' not in st.session_state:  
        st.session_state.df = None  
    if 'analysis_history' not in st.session_state:  
        st.session_state.analysis_history = []  
    if 'gemini_connected' not in st.session_state:  
        st.session_state.gemini_connected = False
    if 'current_api_key' not in st.session_state:
        st.session_state.current_api_key = None
  
def create_eda_agent(df: pd.DataFrame, api_key: str) -> EDAAgent:  
    """Create and initialize the EDA agent"""  
    try:  
        agent = EDAAgent(df, api_key)  
        st.session_state.gemini_connected = True
        st.session_state.current_api_key = api_key
        return agent  
    except Exception as e:  
        st.error(f"Erro ao criar agente: {e}")
        st.session_state.gemini_connected = False
        return None  
  
def display_data_overview(df: pd.DataFrame):  
    """Display comprehensive data overview"""  
    st.subheader("📊 Visão Geral dos Dados")  
      
    col1, col2, col3, col4 = st.columns(4)  
      
    with col1:  
        st.metric("Linhas", f"{df.shape[0]:,}")  
    with col2:  
        st.metric("Colunas", df.shape[1])  
    with col3:  
        st.metric("Valores Nulos", f"{df.isnull().sum().sum():,}")  
    with col4:  
        st.metric("Duplicatas", f"{df.duplicated().sum():,}")  
      
    numeric_cols = df.select_dtypes(include=[np.number]).columns  
    categorical_cols = df.select_dtypes(include=['object']).columns  
    datetime_cols = df.select_dtypes(include=['datetime64']).columns  
      
    col1, col2, col3 = st.columns(3)  
    with col1:  
        st.metric("Colunas Numéricas", len(numeric_cols))  
    with col2:  
        st.metric("Colunas Categóricas", len(categorical_cols))  
    with col3:  
        st.metric("Colunas Temporais", len(datetime_cols))  
  
def display_analysis_result(result: Dict[str, Any]):  
    """Display analysis results with plots"""  
    if result['success']:  
        st.markdown('<div class="analysis-result">', unsafe_allow_html=True)  
        st.markdown("### 🔍 Resultado da Análise")  
        
        # Display text analysis
        st.markdown(result['analysis'])  
          
        # Display metadata
        col1, col2 = st.columns(2)
        with col1:
            if result.get('analysis_type'):  
                st.info(f"📌 Tipo de análise: **{result['analysis_type']}**")  
        with col2:
            if result.get('memory_context', 0) > 0:  
                st.success(f"🧠 Contexto: {result['memory_context']} análises anteriores")  
        
        # Display plots
        if result.get('plots'):
            st.markdown("---")
            st.markdown("### 📊 Visualizações Geradas")
            
            plots = result['plots']
            num_plots = len(plots)
            
            if num_plots == 1:
                # Single plot - full width
                for plot_name, fig in plots.items():
                    st.markdown(f"#### {plot_name}")
                    st.plotly_chart(fig, use_container_width=True, key=f"plot_{plot_name}_{datetime.now().timestamp()}")
            
            elif num_plots == 2:
                # Two plots side by side
                col1, col2 = st.columns(2)
                for idx, (plot_name, fig) in enumerate(plots.items()):
                    with col1 if idx == 0 else col2:
                        st.markdown(f"#### {plot_name}")
                        st.plotly_chart(fig, use_container_width=True, key=f"plot_{plot_name}_{datetime.now().timestamp()}")
            
            else:
                # Multiple plots - two per row
                plot_items = list(plots.items())
                for i in range(0, num_plots, 2):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if i < num_plots:
                            plot_name, fig = plot_items[i]
                            st.markdown(f"#### {plot_name}")
                            st.plotly_chart(fig, use_container_width=True, key=f"plot_{plot_name}_{datetime.now().timestamp()}")
                    
                    with col2:
                        if i + 1 < num_plots:
                            plot_name, fig = plot_items[i + 1]
                            st.markdown(f"#### {plot_name}")
                            st.plotly_chart(fig, use_container_width=True, key=f"plot_{plot_name}_{datetime.now().timestamp()}")
            
            st.success(f"✅ {num_plots} gráfico(s) gerado(s) com sucesso!")
          
        st.markdown('</div>', unsafe_allow_html=True)  
    else:  
        st.error(f"❌ Erro na análise: {result.get('error', 'Erro desconhecido')}")
        if result.get('error_details'):
            with st.expander("Ver detalhes do erro"):
                st.code(result['error_details'])
  
def main():  
    st.markdown('<h1 class="main-header">🤖 Agente EDA Avançado</h1>', unsafe_allow_html=True)  
    st.markdown("*Análise Exploratória de Dados Inteligente com IA*")
    st.markdown("---")  
      
    initialize_session()  
      
    with st.sidebar:  
        st.header("⚙️ Configurações")  
          
        st.subheader("🔑 Google Gemini API")  
        api_key = st.text_input(  
            "API Key:",  
            type="password",  
            placeholder="Digite sua chave API...",  
            help="Obtenha em: https://aistudio.google.com/app/apikey"  
        )  
        
        if st.button("🔌 Conectar API", type="primary", use_container_width=True):
            if not api_key:
                st.error("❌ Por favor, insira uma API Key válida")
            elif st.session_state.df is None:
                st.warning("⚠️ Carregue um arquivo CSV primeiro")
            else:
                with st.spinner("Conectando ao Gemini..."):
                    st.session_state.eda_agent = create_eda_agent(st.session_state.df, api_key)
                    if st.session_state.eda_agent:
                        st.success("✅ Conectado com sucesso!")
                        st.rerun()
          
        if st.session_state.gemini_connected:  
            st.success("✅ Gemini Conectado")  
        else:  
            st.warning("⚠️ Gemini Não Conectado")  
          
        st.markdown("---")  
          
        st.subheader("📁 Upload de Dados")  
        uploaded_file = st.file_uploader(  
            "Arquivo CSV:",  
            type=['csv'],  
            help="Selecione um arquivo CSV para análise"  
        )  
          
        if uploaded_file is not None:  
            try:  
                df = pd.read_csv(uploaded_file)  
                st.session_state.df = df  
                st.success("✅ Dados carregados!")  
                st.info(f"Dimensões: {df.shape[0]} × {df.shape[1]}")
                
                if api_key and not st.session_state.eda_agent:
                    with st.spinner("Inicializando agente..."):
                        st.session_state.eda_agent = create_eda_agent(df, api_key)
                        if st.session_state.eda_agent:
                            st.rerun()
                  
            except Exception as e:  
                st.error(f"Erro ao carregar arquivo: {e}")  
          
        st.markdown("---")  
          
        if st.session_state.df is not None and st.session_state.eda_agent:  
            st.subheader("🚀 Análises Rápidas")  
              
            quick_analyses = [  
                ("📈 Resumo dos Dados", "Faça um resumo completo dos dados incluindo tipos, estatísticas e qualidade"),  
                ("📊 Análise de Distribuições", "Analise a distribuição de todas as variáveis numéricas com histogramas"),
                ("🔗 Análise de Correlações", "Analise as correlações entre todas as variáveis numéricas"),  
                ("⚠️ Detecção de Outliers", "Detecte e analise outliers em todas as variáveis numéricas"),  
                ("🎯 Análise de Clusters", "Identifique padrões e agrupamentos nos dados"),  
                ("⏰ Análise Temporal", "Analise padrões e tendências temporais"),  
                ("💡 Conclusões Gerais", "Quais são suas conclusões consolidadas sobre todos os dados analisados?")  
            ]  
              
            for label, question in quick_analyses:  
                if st.button(label, key=f"quick_{label}", use_container_width=True):  
                    st.session_state.current_question = question  
                    st.rerun()
            
            st.markdown("---")
            
            if st.button("🗑️ Limpar Memória", use_container_width=True, help="Limpa o histórico de análises"):
                if st.session_state.eda_agent:
                    st.session_state.eda_agent.clear_memory()
                    st.session_state.analysis_history = []
                    st.success("Memória limpa!")
                    st.rerun()
      
    if st.session_state.df is not None:  
        display_data_overview(st.session_state.df)  
          
        tab1, tab2, tab3 = st.tabs(["🔍 Amostra dos Dados", "📊 Estatísticas", "🔧 Informações"])  
          
        with tab1:  
            st.dataframe(st.session_state.df.head(20), use_container_width=True)  
          
        with tab2:  
            st.dataframe(st.session_state.df.describe(), use_container_width=True)  
          
        with tab3:  
            info_df = pd.DataFrame({  
                'Coluna': st.session_state.df.columns,  
                'Tipo': st.session_state.df.dtypes,  
                'Nulos': st.session_state.df.isnull().sum(),  
                '% Nulos': (st.session_state.df.isnull().sum() / len(st.session_state.df) * 100).round(2),
                'Únicos': st.session_state.df.nunique()  
            })  
            st.dataframe(info_df, use_container_width=True)  
          
        st.markdown("---")  
          
        st.subheader("💬 Interface de Análise")  
          
        question = st.text_area(  
            "Faça sua pergunta sobre os dados:",  
            value=st.session_state.get('current_question', ''),  
            placeholder="Ex: Existe correlação entre as variáveis? Há outliers nos dados? Quais padrões você identifica? Quais suas conclusões?",  
            height=100,  
            key="question_input"  
        )  
          
        col1, col2, col3 = st.columns([2, 1, 1])  
          
        with col1:  
            analyze_btn = st.button("🔍 Analisar", type="primary", use_container_width=True)  
          
        with col2:  
            if st.button("🧹 Limpar Campo", use_container_width=True):  
                st.session_state.current_question = ""  
                st.rerun()  
          
        with col3:  
            if st.button("📋 Ver Histórico", use_container_width=True):  
                if st.session_state.eda_agent:  
                    with st.expander("📚 Histórico de Análises", expanded=True):
                        summary = st.session_state.eda_agent.get_analysis_summary()  
                        st.markdown(summary)
        
        st.markdown("---")
          
        if analyze_btn and question:  
            if not st.session_state.eda_agent:  
                st.error("❌ Configure a API key do Gemini e conecte primeiro!")  
            else:  
                with st.spinner("🤖 Analisando dados... Isso pode levar alguns segundos."):  
                    try:
                        result = st.session_state.eda_agent.analyze(question)  
                          
                        display_analysis_result(result)  
                          
                        st.session_state.analysis_history.append({  
                            'timestamp': datetime.now(),  
                            'question': question,  
                            'result': result  
                        })
                        
                        # Clear the question field after successful analysis
                        st.session_state.current_question = ""
                        
                    except Exception as e:
                        st.error(f"❌ Erro durante análise: {str(e)}")
                        st.info("💡 Dica: Verifique se a API Key está correta e se há conexão com a internet")

        if st.session_state.analysis_history:  
            st.markdown("---")  
            st.subheader("📚 Últimas Análises Realizadas")  
              
            with st.expander("Ver últimas 5 análises", expanded=False):  
                for i, entry in enumerate(reversed(st.session_state.analysis_history[-5:])):  
                    st.markdown(f"**{i+1}. {entry['question'][:80]}{'...' if len(entry['question']) > 80 else ''}**")  
                    st.caption(f"⏰ {entry['timestamp'].strftime('%d/%m/%Y %H:%M:%S')}")  
                    if entry['result']['success']:  
                        st.success(f"✅ Análise concluída - Tipo: {entry['result'].get('analysis_type', 'N/A')}")
                        if entry['result'].get('plots'):
                            st.info(f"📊 {len(entry['result']['plots'])} gráfico(s) gerado(s)")
                    else:  
                        st.error("❌ Erro na análise")  
                    st.markdown("---")  
      
    else:  
        st.markdown("## 🎯 Bem-vindo ao Agente EDA Avançado")  
          
        col1, col2 = st.columns([2, 1])  
          
        with col1:  
            st.markdown("""  
            ### 🚀 Capacidades do Agente  
              
            Este agente utiliza **LangChain** e **Google Gemini** para fornecer análises exploratórias avançadas:  
              
            #### 📊 Análises Disponíveis:  
            - **Análise Descritiva**: Tipos de dados, estatísticas, distribuições  
            - **Detecção de Outliers**: Métodos IQR, Z-score com visualizações
            - **Análise de Correlações**: Matrizes de correlação interativas
            - **Análise de Distribuições**: Histogramas e estatísticas detalhadas
            - **Clustering**: K-means, análise de agrupamentos  
            - **Análise Temporal**: Tendências, sazonalidade, padrões  
            - **Código Python Dinâmico**: Execução de análises customizadas  
            - **Conclusões Consolidadas**: Síntese de todos os insights descobertos
              
            #### 🧠 Recursos Inteligentes:  
            - **Memória Contextual**: Lembra e conecta análises anteriores  
            - **Recomendações Automáticas**: Sugere próximos passos  
            - **Visualizações Interativas**: Gráficos Plotly dinâmicos  
            - **Insights Acionáveis**: Conclusões práticas baseadas em evidências
            - **Geração de Conclusões**: Síntese automática de descobertas
            """)  
          
        with col2:  
            st.markdown("### 🔧 Como Usar")  
            st.info("""  
            1. **Carregue** um arquivo CSV  
            2. **Configure** sua API key do Gemini
            3. **Clique** em "Conectar API"
            4. **Faça perguntas** sobre seus dados  
            5. **Explore** as análises sugeridas  
            6. **Peça conclusões** consolidadas
            """)  
              
            st.markdown("### 📋 Exemplos de Perguntas")  
            example_questions = [  
                "Quais são os tipos de dados?",  
                "Analise a distribuição das variáveis",
                "Existe correlação entre as variáveis?",  
                "Há outliers nos dados?",  
                "Identifique padrões e clusters",  
                "Analise tendências temporais",  
                "Quais suas conclusões sobre os dados?"  
            ]  
              
            for q in example_questions:  
                st.code(q, language=None)
        
        st.markdown("---")
        st.info("💡 **Dica**: Para o desafio do Kaggle Credit Card Fraud, comece perguntando sobre o resumo dos dados e depois explore correlações e outliers!")
  
if __name__ == "__main__":  
    main()