import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as sns  
import plotly.express as px  
import plotly.graph_objects as go  
from plotly.subplots import make_subplots  
from scipy import stats  
from sklearn.cluster import KMeans  
from sklearn.preprocessing import StandardScaler  
import warnings  
from typing import Dict, List, Any, Optional  
import io  
import sys  
  
warnings.filterwarnings('ignore')  

class PythonCodeExecutor:  
    """Execute Python code for data analysis"""
    
    def __init__(self):
        pass
      
    def execute(self, code: str, globals_dict: Dict[str, Any] = None) -> Dict[str, Any]:  
        """Execute Python code with DataFrame context"""  
        if globals_dict is None:  
            globals_dict = {}  
          
        globals_dict.update({  
            'pd': pd,  
            'np': np,  
            'plt': plt,  
            'sns': sns,  
            'px': px,  
            'go': go,  
            'stats': stats
        })  
          
        old_stdout = sys.stdout  
        sys.stdout = captured_output = io.StringIO()  
          
        try:  
            exec(code, globals_dict)  
            output = captured_output.getvalue()  
            return {  
                'success': True,  
                'output': output,  
                'globals': globals_dict  
            }  
        except Exception as e:  
            return {  
                'success': False,  
                'error': str(e),  
                'output': captured_output.getvalue()  
            }  
        finally:  
            sys.stdout = old_stdout  
  
class AdvancedDataAnalyzer:  
    """Enhanced data analyzer with dynamic code generation and advanced analytics"""  
      
    def __init__(self, df: pd.DataFrame):  
        self.df = df  
        self.numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()  
        self.categorical_columns = df.select_dtypes(include=['object']).columns.tolist()  
        self.datetime_columns = df.select_dtypes(include=['datetime64']).columns.tolist()  
        self.analysis_cache = {}
    
    def distribution_analysis_tool(self, column: str = None) -> Dict[str, Any]:
        """Analyze distribution of variables with histograms and statistics"""
        plots = {}
        
        if column and column in self.df.columns:
            # Analyze specific column
            columns_to_analyze = [column]
        else:
            # Analyze all numeric columns (limit to 6 for performance)
            columns_to_analyze = self.numeric_columns[:6]
        
        if not columns_to_analyze:
            return {
                'analysis': "⚠️ No numeric columns available for distribution analysis.",
                'plots': {},
                'insights': [],
                'recommendations': []
            }
        
        analysis_text = "## Análise de Distribuições\n\n"
        insights = []
        
        for col in columns_to_analyze:
            # Create histogram with distribution curve
            fig = go.Figure()
            
            # Histogram
            fig.add_trace(go.Histogram(
                x=self.df[col],
                name='Distribuição',
                nbinsx=50,
                opacity=0.7
            ))
            
            fig.update_layout(
                title=f'Distribuição: {col}',
                xaxis_title=col,
                yaxis_title='Frequência',
                showlegend=True,
                height=400
            )
            
            plots[f'Distribution_{col}'] = fig
            
            # Calculate statistics
            mean_val = self.df[col].mean()
            median_val = self.df[col].median()
            std_val = self.df[col].std()
            skew_val = self.df[col].skew()
            
            analysis_text += f"### {col}:\n"
            analysis_text += f"- Média: {mean_val:.2f}\n"
            analysis_text += f"- Mediana: {median_val:.2f}\n"
            analysis_text += f"- Desvio Padrão: {std_val:.2f}\n"
            analysis_text += f"- Assimetria (Skewness): {skew_val:.2f}\n"
            
            # Interpret skewness
            if abs(skew_val) < 0.5:
                analysis_text += "  - Distribuição aproximadamente simétrica\n"
            elif skew_val > 0:
                analysis_text += "  - Distribuição assimétrica à direita (cauda longa à direita)\n"
                insights.append(f"{col} tem distribuição assimétrica à direita")
            else:
                analysis_text += "  - Distribuição assimétrica à esquerda (cauda longa à esquerda)\n"
                insights.append(f"{col} tem distribuição assimétrica à esquerda")
            
            analysis_text += "\n"
        
        recommendations = []
        if any(abs(self.df[col].skew()) > 1 for col in columns_to_analyze):
            recommendations.append("Considere transformações (log, sqrt) para variáveis muito assimétricas")
        
        return {
            'analysis': analysis_text,
            'plots': plots,
            'insights': insights,
            'recommendations': recommendations
        }
          
    def correlation_analysis_tool(self, method: str = "pearson", threshold: float = 0.5) -> Dict[str, Any]:  
        """Analyze correlations between numeric variables"""  
        if len(self.numeric_columns) < 2:  
            return {  
                'analysis': "⚠️ Insuficientes variáveis numéricas para análise de correlação.",  
                'plots': {},  
                'insights': [],  
                'recommendations': []  
            }  
          
        corr_matrix = self.df[self.numeric_columns].corr(method=method)  
          
        fig = px.imshow(  
            corr_matrix,  
            title=f'Matriz de Correlação ({method.title()})',  
            color_continuous_scale='RdBu_r',  
            aspect='auto',
            labels=dict(color="Correlação")
        )  
        fig.update_layout(width=900, height=700)  
          
        significant_corrs = []  
        for i in range(len(corr_matrix.columns)):  
            for j in range(i+1, len(corr_matrix.columns)):  
                corr_value = corr_matrix.iloc[i, j]  
                if abs(corr_value) >= threshold:  
                    significant_corrs.append({  
                        'var1': corr_matrix.columns[i],  
                        'var2': corr_matrix.columns[j],  
                        'correlation': corr_value,  
                        'strength': self._correlation_strength(abs(corr_value))  
                    })  
          
        insights = []  
        if significant_corrs:  
            strongest = max(significant_corrs, key=lambda x: abs(x['correlation']))  
            insights.append(f"Correlação mais forte: {strongest['var1']} ↔ {strongest['var2']} ({strongest['correlation']:.3f})")  
              
            positive_corrs = [c for c in significant_corrs if c['correlation'] > 0]  
            negative_corrs = [c for c in significant_corrs if c['correlation'] < 0]  
              
            insights.append(f"Encontradas {len(positive_corrs)} correlações positivas e {len(negative_corrs)} negativas significativas")  
        else:  
            insights.append("Nenhuma correlação significativa encontrada acima do threshold")  
          
        recommendations = []  
        if len(significant_corrs) > 5:  
            recommendations.append("Considere seleção de features devido à alta multicolinearidade")  
        if any(abs(c['correlation']) > 0.9 for c in significant_corrs):  
            recommendations.append("Algumas variáveis são altamente correlacionadas - considere remover features redundantes")  
          
        analysis_text = self._format_correlation_analysis(significant_corrs, method)  
          
        return {  
            'analysis': analysis_text,  
            'plots': {'Correlation_Matrix': fig},  
            'insights': insights,  
            'recommendations': recommendations  
        }  
      
    def outlier_detection_tool(self, method: str = "iqr") -> Dict[str, Any]:  
        """Advanced outlier detection"""  
        if not self.numeric_columns:  
            return {  
                'analysis': "⚠️ Sem variáveis numéricas para análise de outliers.",  
                'plots': {},  
                'insights': [],  
                'recommendations': []  
            }  
          
        outlier_results = {}  
        plots = {}  
          
        for col in self.numeric_columns[:6]:  
            if method == "iqr":  
                outliers = self._detect_outliers_iqr(col)  
            elif method == "zscore":  
                outliers = self._detect_outliers_zscore(col)  
            else:  
                outliers = self._detect_outliers_iqr(col)  
              
            outlier_results[col] = outliers  
              
            fig = px.box(  
                self.df,   
                y=col,   
                title=f'Detecção de Outliers: {col}',  
                points="outliers"  
            )
            fig.update_layout(height=400)
            plots[f'Outliers_{col}'] = fig  
          
        analysis_text = self._format_outlier_analysis(outlier_results, method)  
          
        insights = []  
        total_outliers = sum(len(outliers) for outliers in outlier_results.values())  
        insights.append(f"Total de outliers detectados: {total_outliers}")  
          
        if outlier_results:
            most_outliers_col = max(outlier_results.keys(), key=lambda k: len(outlier_results[k]))  
            insights.append(f"Coluna com mais outliers: {most_outliers_col} ({len(outlier_results[most_outliers_col])} outliers)")  
          
        recommendations = []  
        outlier_percentage = (total_outliers / len(self.df)) * 100 if len(self.df) > 0 else 0
        if outlier_percentage > 5:  
            recommendations.append("Alta porcentagem de outliers detectada - investigue a qualidade dos dados")  
        if outlier_percentage > 10:  
            recommendations.append("Considere métodos de tratamento de outliers (remoção, transformação ou capping)")  

        return {  
            'analysis': analysis_text,  
            'plots': plots,  
            'insights': insights,  
            'recommendations': recommendations  
        }  
      
    def clustering_analysis_tool(self, n_clusters: int = 3, method: str = "kmeans") -> Dict[str, Any]:
        """Perform clustering analysis"""
        if len(self.numeric_columns) < 2:
            return {
                'analysis': "⚠️ Insuficientes variáveis numéricas para clustering.",
                'plots': {},
                'insights': [],
                'recommendations': []
            }
        
        X = self.df[self.numeric_columns].fillna(self.df[self.numeric_columns].mean())
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        if method == "kmeans":
            clusterer = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = clusterer.fit_predict(X_scaled)
        else:
            return {
                'analysis': f"⚠️ Método {method} não implementado ainda.",
                'plots': {},
                'insights': [],
                'recommendations': []
            }
        
        cluster_df = self.df.copy()
        cluster_df['Cluster'] = clusters
        cluster_stats = cluster_df.groupby('Cluster')[self.numeric_columns].mean()
        
        plots = {}
        if len(self.numeric_columns) >= 2:
            fig = px.scatter(
                cluster_df,
                x=self.numeric_columns[0],
                y=self.numeric_columns[1],
                color='Cluster',
                title=f'Análise de Clustering ({method}, k={n_clusters})',
                labels={'color': 'Cluster'}
            )
            fig.update_layout(height=500)
            plots['Clustering_Scatter'] = fig
        
        analysis_text = self._format_clustering_analysis(cluster_stats, method, n_clusters)
        
        insights = [
            f"Identificados {n_clusters} clusters distintos",
            f"Maior cluster: {pd.Series(clusters).value_counts().idxmax()} com {pd.Series(clusters).value_counts().max()} amostras"
        ]
        
        recommendations = [
            "Analise as características dos clusters para identificar padrões",
            "Considere usar informações de cluster para segmentação"
        ]
        
        return {
            'analysis': analysis_text,
            'plots': plots,
            'insights': insights,
            'recommendations': recommendations
        }
      
    def temporal_analysis_tool(self, time_column: str = None) -> Dict[str, Any]:  
        """Analyze temporal patterns"""  
        if time_column is None:  
            time_candidates = [col for col in self.df.columns if 'time' in col.lower() or 'date' in col.lower() or 'tempo' in col.lower()]
            if not time_candidates:  
                return {  
                    'analysis': "⚠️ Nenhuma coluna temporal detectada no dataset.",  
                    'plots': {},  
                    'insights': [],  
                    'recommendations': []  
                }  
            time_column = time_candidates[0]  
          
        if time_column not in self.df.columns:  
            return {  
                'analysis': f"⚠️ Coluna '{time_column}' não encontrada no dataset.",  
                'plots': {},  
                'insights': [],  
                'recommendations': []  
            }  
          
        time_data = pd.to_datetime(self.df[time_column], errors='coerce')  
        if time_data.isna().all():  
            time_data = pd.to_datetime(self.df[time_column], unit='s', errors='coerce')  
          
        plots = {}  
          
        for col in self.numeric_columns[:3]:  
            fig = px.line(  
                x=time_data,  
                y=self.df[col],  
                title=f'Padrão Temporal: {col}',  
                labels={'x': 'Tempo', 'y': col}  
            )
            fig.update_layout(height=400)
            plots[f'Temporal_{col}'] = fig  
          
        analysis_text = self._format_temporal_analysis(time_column, time_data)  
          
        insights = [  
            f"Análise temporal baseada na coluna: {time_column}",  
            f"Período: {time_data.min()} a {time_data.max()}"  
        ]  
          
        recommendations = [  
            "Procure por padrões sazonais ou tendências",  
            "Considere engenharia de features baseada em tempo"  
        ]  
          
        return {  
            'analysis': analysis_text,  
            'plots': plots,  
            'insights': insights,  
            'recommendations': recommendations  
        }  
      
    def _correlation_strength(self, corr_value: float) -> str:  
        if corr_value >= 0.9:  
            return "muito forte"  
        elif corr_value >= 0.7:  
            return "forte"  
        elif corr_value >= 0.5:  
            return "moderada"  
        elif corr_value >= 0.3:  
            return "fraca"  
        else:  
            return "muito fraca"  
      
    def _detect_outliers_iqr(self, column: str) -> List[int]:  
        Q1 = self.df[column].quantile(0.25)  
        Q3 = self.df[column].quantile(0.75)  
        IQR = Q3 - Q1  
        lower_bound = Q1 - 1.5 * IQR  
        upper_bound = Q3 + 1.5 * IQR  
        outlier_mask = (self.df[column] < lower_bound) | (self.df[column] > upper_bound)  
        return self.df[outlier_mask].index.tolist()  
      
    def _detect_outliers_zscore(self, column: str, threshold: float = 3) -> List[int]:  
        z_scores = np.abs(stats.zscore(self.df[column].fillna(self.df[column].mean())))  
        outlier_mask = z_scores > threshold  
        return self.df[outlier_mask].index.tolist()  
      
    def _format_correlation_analysis(self, correlations: List[Dict], method: str) -> str:  
        analysis = f"## Análise de Correlações ({method.title()})\n\n"  
        if correlations:  
            analysis += "### Correlações Significativas Encontradas:\n"  
            for corr in correlations[:10]:  
                direction = "positiva" if corr['correlation'] > 0 else "negativa"  
                analysis += f"- **{corr['var1']}** ↔ **{corr['var2']}**: "  
                analysis += f"Correlação {direction} {corr['strength']} ({corr['correlation']:.3f})\n"  
        else:  
            analysis += "Nenhuma correlação significativa encontrada acima do threshold.\n"  
        return analysis  
      
    def _format_outlier_analysis(self, outlier_results: Dict, method: str) -> str:  
        analysis = f"## Análise de Outliers (Método: {method.upper()})\n\n"  
        for col, outliers in outlier_results.items():  
            outlier_count = len(outliers)  
            outlier_pct = (outlier_count / len(self.df)) * 100 if len(self.df) > 0 else 0
            analysis += f"### {col}:\n"  
            analysis += f"- Outliers detectados: {outlier_count} ({outlier_pct:.2f}%)\n"  
            if outlier_count > 0:  
                outlier_values = self.df.loc[outliers, col]  
                analysis += f"- Valores extremos: {outlier_values.min():.2f} a {outlier_values.max():.2f}\n"
            analysis += "\n"
        return analysis  
      
    def _format_clustering_analysis(self, cluster_stats: pd.DataFrame, method: str, n_clusters: int) -> str:  
        analysis = f"## Análise de Agrupamentos ({method.upper()}, {n_clusters} clusters)\n\n"  
        for cluster_id in cluster_stats.index:  
            analysis += f"### Cluster {cluster_id}:\n"
            analysis += f"Tamanho: {len(cluster_stats)} elementos\n"
            for col in cluster_stats.columns[:5]:  # Limit columns
                analysis += f"- {col} (média): {cluster_stats.loc[cluster_id, col]:.2f}\n"  
            analysis += "\n"  
        return analysis  
      
    def _format_temporal_analysis(self, time_column: str, time_data: pd.Series) -> str:  
        analysis = f"## Análise Temporal\n\n"  
        analysis += f"### Coluna temporal: {time_column}\n"  
        analysis += f"- Período: {time_data.min()} até {time_data.max()}\n"  
        duration = (time_data.max() - time_data.min()).days if pd.notna(time_data.max()) and pd.notna(time_data.min()) else 0
        analysis += f"- Duração total: {duration} dias\n"  
        return analysis