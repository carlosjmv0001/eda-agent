from typing import Dict, List, Any, Optional, Tuple
from langchain_core.tools import tool  
from langchain.agents import create_tool_calling_agent, AgentExecutor  
from langchain_core.prompts import ChatPromptTemplate  
from langchain_google_genai import ChatGoogleGenerativeAI  
from utils.advanced_analyzer import AdvancedDataAnalyzer, PythonCodeExecutor  
from memory.enhanced_memory import EnhancedSessionMemory  
import pandas as pd  
import plotly.graph_objects as go
import json

class EDAAgent:  
    """Enhanced EDA Agent with LangChain integration and visualization support"""  
      
    def __init__(self, df: pd.DataFrame, api_key: str):  
        self.df = df  
        self.analyzer = AdvancedDataAnalyzer(df)  
        self.memory = EnhancedSessionMemory()  
        self.code_executor = PythonCodeExecutor()
        self.generated_plots = {}  # Store plots for display
          
        # Initialize LLM
        try:
            self.llm = ChatGoogleGenerativeAI(  
                model="gemini-2.5-pro",
                google_api_key=api_key,  
                temperature=0.1  
            )
        except Exception as e:
            raise Exception(f"Erro ao inicializar Gemini: {str(e)}")
          
        # Create tools  
        self.tools = self._create_tools()  
          
        # Create agent  
        self.agent = self._create_agent()  
      
    def _create_tools(self) -> List:  
        """Create analysis tools for the agent"""  
          
        @tool  
        def python_analysis_tool(code: str) -> str:  
            """Execute Python code for data analysis. Use for custom analysis and visualizations.
            The dataframe is available as 'df'. Use plotly for visualizations."""  
            result = self.code_executor.execute(code, {'df': self.df})  
              
            if result['success']:  
                # Check if any plots were created
                output_text = result['output']
                if not output_text:
                    output_text = "Code executed successfully (no output)"
                return f"Code executed successfully:\n{output_text}"  
            else:  
                return f"Error executing code: {result['error']}"  
          
        @tool  
        def correlation_analysis_tool(method: str = "pearson", threshold: float = 0.5) -> str:  
            """Analyze correlations between numeric variables. Returns analysis text and generates correlation heatmap."""  
            result = self.analyzer.correlation_analysis_tool(method, threshold)  
            # Store plots for later display
            if result.get('plots'):
                self.generated_plots.update(result['plots'])
            return result['analysis']  
          
        @tool  
        def outlier_detection_tool(method: str = "iqr") -> str:  
            """Detect outliers in numeric variables using IQR or zscore methods. Generates boxplots."""  
            result = self.analyzer.outlier_detection_tool(method)
            if result.get('plots'):
                self.generated_plots.update(result['plots'])
            return result['analysis']  
          
        @tool  
        def clustering_analysis_tool(n_clusters: int = 3, method: str = "kmeans") -> str:  
            """Perform clustering analysis. Generates scatter plot of clusters."""  
            result = self.analyzer.clustering_analysis_tool(n_clusters, method)
            if result.get('plots'):
                self.generated_plots.update(result['plots'])
            return result['analysis']  
          
        @tool  
        def temporal_analysis_tool(time_column: str = None) -> str:  
            """Analyze temporal patterns. Generates time series plots."""  
            result = self.analyzer.temporal_analysis_tool(time_column)
            if result.get('plots'):
                self.generated_plots.update(result['plots'])
            return result['analysis']
        
        @tool
        def distribution_analysis_tool(column: str = None) -> str:
            """Analyze distribution of variables. Generates histograms and distribution plots."""
            result = self.analyzer.distribution_analysis_tool(column)
            if result.get('plots'):
                self.generated_plots.update(result['plots'])
            return result['analysis']
          
        @tool  
        def data_summary_tool() -> str:  
            """Get a comprehensive summary of the dataset including types, statistics, and data quality."""  
            summary = f"""  
## Resumo do Dataset  
  
**Dimensões:** {self.df.shape[0]} linhas × {self.df.shape[1]} colunas  

**Colunas do Dataset:**
{', '.join(self.df.columns.tolist())}

**Tipos de Variáveis:**  
- Numéricas: {len(self.analyzer.numeric_columns)} colunas
  Colunas: {', '.join(self.analyzer.numeric_columns[:10])}
- Categóricas: {len(self.analyzer.categorical_columns)} colunas
  Colunas: {', '.join(self.analyzer.categorical_columns[:10])}
  
**Qualidade dos Dados:**  
- Valores nulos: {self.df.isnull().sum().sum()} ({(self.df.isnull().sum().sum() / (self.df.shape[0] * self.df.shape[1]) * 100):.2f}%)
- Duplicatas: {self.df.duplicated().sum()}  
  
**Estatísticas Básicas das Variáveis Numéricas:**  
{self.df.describe().to_string()}  
            """  
            return summary  
          
        @tool  
        def memory_summary_tool() -> str:  
            """Get a summary of all previous analyses and generate consolidated conclusions."""  
            return self.memory.generate_summary()
        
        @tool
        def generate_conclusions_tool() -> str:
            """Generate comprehensive conclusions based on all analyses performed in this session."""
            return self.memory.generate_comprehensive_conclusions()
          
        return [  
            python_analysis_tool,
            correlation_analysis_tool,  
            outlier_detection_tool,  
            clustering_analysis_tool,  
            temporal_analysis_tool,
            distribution_analysis_tool,
            data_summary_tool,  
            memory_summary_tool,
            generate_conclusions_tool
        ]  
      
    def _create_agent(self) -> AgentExecutor:  
        """Create the LangChain agent with tools"""  
        system_prompt = """  
Você é um agente especialista em Análise Exploratória de Dados (EDA) com capacidades avançadas.  
  
Suas principais habilidades incluem:  
- Análise estatística descritiva e inferencial  
- Detecção de outliers e anomalias  
- Análise de correlações e relacionamentos entre variáveis
- Clustering e segmentação de dados  
- Análise temporal e de tendências  
- Análise de distribuições
- Geração de visualizações interativas  
- Execução de código Python personalizado  
  
Você tem acesso às seguintes ferramentas:  
- data_summary_tool: Para resumo completo dos dados (USE PRIMEIRO para entender o dataset)
- correlation_analysis_tool: Para análise de correlações (GERA GRÁFICOS)
- outlier_detection_tool: Para detecção de outliers (GERA GRÁFICOS)
- clustering_analysis_tool: Para análise de agrupamentos (GERA GRÁFICOS)
- temporal_analysis_tool: Para análise temporal (GERA GRÁFICOS)
- distribution_analysis_tool: Para análise de distribuições (GERA GRÁFICOS)
- python_analysis_tool: Para código Python customizado e visualizações especiais
- memory_summary_tool: Para acessar análises anteriores  
- generate_conclusions_tool: Para gerar conclusões consolidadas de todas as análises

IMPORTANTE:
1. Para perguntas sobre conclusões, SEMPRE use generate_conclusions_tool
2. Sempre que uma ferramenta gerar gráficos, mencione "Gráfico gerado" na resposta
3. Use data_summary_tool para entender o dataset antes de análises complexas
4. Para questões sobre distribuição, histogramas, use distribution_analysis_tool
5. Forneça insights acionáveis e recomendações baseadas em evidências
6. Mantenha contexto das análises anteriores usando memory_summary_tool
7. Seja específico e detalhado nas respostas

Quando perguntarem sobre conclusões, tendências gerais ou "o que você aprendeu":
- Use generate_conclusions_tool para consolidar todos os insights
- Apresente padrões identificados
- Destaque descobertas importantes
- Forneça recomendações práticas
        """  
          
        prompt = ChatPromptTemplate.from_messages([  
            ("system", system_prompt),  
            ("human", "{input}"),  
            ("placeholder", "{agent_scratchpad}")  
        ])  
          
        agent = create_tool_calling_agent(self.llm, self.tools, prompt)  
        return AgentExecutor(
            agent=agent, 
            tools=self.tools, 
            verbose=True, 
            max_iterations=10,  # Increased for complex analyses
            handle_parsing_errors=True
        )  
      
    def analyze(self, question: str) -> Dict[str, Any]:  
        """Main analysis method that uses the agent"""  
        try:
            # Clear previous plots
            self.generated_plots = {}
            
            # Get contextual memory  
            relevant_memory = self.memory.get_contextual_memory(question)  
              
            # Add memory context to question if relevant  
            context = ""  
            if relevant_memory:  
                context = "\n\nContexto de análises anteriores:\n"  
                for entry in relevant_memory:  
                    context += f"- {entry.question}: {', '.join(entry.key_findings[:2])}\n"  
              
            enhanced_question = question + context  
              
            # Execute agent  
            result = self.agent.invoke({"input": enhanced_question})  
              
            # Determine analysis type  
            analysis_type = self._determine_analysis_type(question)  
              
            # Store in memory  
            analysis_result = {  
                'analysis': result['output'],  
                'plots': self.generated_plots.copy(),
                'insights': self._extract_insights(result['output']),  
                'recommendations': self._extract_recommendations(result['output'])
            }  
              
            self.memory.add_analysis(question, analysis_result, analysis_type)  
              
            return {  
                'success': True,  
                'analysis': result['output'],  
                'plots': self.generated_plots.copy(),
                'analysis_type': analysis_type,  
                'memory_context': len(relevant_memory)  
            }  
              
        except Exception as e:  
            import traceback
            error_details = traceback.format_exc()
            return {  
                'success': False,  
                'error': str(e),  
                'error_details': error_details,
                'analysis': f"Erro durante a análise: {e}",
                'plots': {}
            }
    
    def _extract_insights(self, text: str) -> List[str]:
        """Extract insights from analysis text"""
        insights = []
        lines = text.split('\n')
        for line in lines:
            if any(keyword in line.lower() for keyword in ['insight:', 'descoberta:', 'padrão:', 'encontrado:', 'identificado:']):
                insights.append(line.strip())
        return insights[:5]
    
    def _extract_recommendations(self, text: str) -> List[str]:
        """Extract recommendations from analysis text"""
        recommendations = []
        lines = text.split('\n')
        in_rec_section = False
        for line in lines:
            if 'recomenda' in line.lower():
                in_rec_section = True
            if in_rec_section and line.strip().startswith('-'):
                recommendations.append(line.strip())
        return recommendations[:5]
      
    def _determine_analysis_type(self, question: str) -> str:  
        """Determine the type of analysis based on question"""  
        question_lower = question.lower()  
          
        if any(word in question_lower for word in ['conclus', 'aprend', 'resumo geral', 'insights gerais']):
            return 'conclusions'
        elif any(word in question_lower for word in ['correlação', 'relação', 'correlation', 'relaciona']):  
            return 'correlation'  
        elif any(word in question_lower for word in ['outlier', 'anomalia', 'atípico', 'anomal']):  
            return 'outlier'  
        elif any(word in question_lower for word in ['cluster', 'agrupamento', 'grupo', 'segmenta']):  
            return 'clustering'  
        elif any(word in question_lower for word in ['tempo', 'temporal', 'tendência', 'time', 'trend']):  
            return 'temporal'  
        elif any(word in question_lower for word in ['distribuição', 'histograma', 'distribution', 'histogram']):  
            return 'distribution'
        elif any(word in question_lower for word in ['tipo', 'types', 'describe', 'resumo', 'summary']):
            return 'summary'
        else:  
            return 'general'  
      
    def get_analysis_summary(self) -> str:  
        """Get a summary of all analyses performed"""  
        return self.memory.generate_summary()
    
    def get_comprehensive_conclusions(self) -> str:
        """Get comprehensive conclusions from all analyses"""
        return self.memory.generate_comprehensive_conclusions()
      
    def clear_memory(self) -> None:  
        """Clear analysis memory"""  
        self.memory = EnhancedSessionMemory()