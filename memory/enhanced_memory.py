from typing import List, Dict, Any, Optional  
from datetime import datetime  
from collections import deque  
from pydantic import BaseModel, Field  
  
class AnalysisMemory(BaseModel):  
    """Enhanced memory for storing analysis insights and patterns"""  
    timestamp: str  
    question: str  
    analysis_type: str  
    key_findings: List[str]  
    visualizations: List[str]  
    data_patterns: Dict[str, Any]  
    recommendations: List[str]  
  
class EnhancedSessionMemory:  
    """Enhanced memory system for tracking analysis history and generating conclusions"""  
      
    def __init__(self, max_interactions: int = 100):  
        self.max_interactions = max_interactions  
        self.analysis_history: deque = deque(maxlen=max_interactions)  
        self.data_insights: Dict[str, Any] = {}  
        self.conversation_summary: str = ""
        self.global_patterns: Dict[str, Any] = {}
          
    def add_analysis(self, question: str, analysis_result: Dict[str, Any],   
                    analysis_type: str = "general") -> None:  
        """Add comprehensive analysis to memory"""  
        memory_entry = AnalysisMemory(  
            timestamp=datetime.now().isoformat(),  
            question=question,  
            analysis_type=analysis_type,  
            key_findings=self._extract_key_findings(analysis_result),  
            visualizations=list(analysis_result.get('plots', {}).keys()),  
            data_patterns=self._extract_patterns(analysis_result),  
            recommendations=self._extract_recommendations(analysis_result)  
        )  
          
        self.analysis_history.append(memory_entry)  
        self._update_insights(memory_entry)
        self._update_global_patterns(memory_entry)
      
    def _extract_key_findings(self, analysis_result: Dict[str, Any]) -> List[str]:  
        """Extract key findings from analysis"""  
        findings = []  
        analysis_text = analysis_result.get('analysis', '')  
          
        lines = analysis_text.split('\n')  
        for line in lines:  
            stripped = line.strip()
            if stripped.startswith('-') or stripped.startswith('•'):  
                findings.append(stripped)
            elif any(keyword in line.lower() for keyword in ['encontrado', 'identificado', 'detectado', 'observado']):
                findings.append(stripped)
          
        return findings[:10]
      
    def _extract_patterns(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:  
        """Extract data patterns from analysis"""  
        patterns = {}  
        analysis_text = analysis_result.get('analysis', '').lower()
          
        # Correlation patterns
        if 'correlação' in analysis_text or 'correlation' in analysis_text:  
            patterns['has_correlations'] = True
            if 'forte' in analysis_text or 'strong' in analysis_text:
                patterns['strong_correlations'] = True
          
        # Outlier patterns  
        if 'outlier' in analysis_text or 'atípico' in analysis_text:  
            patterns['has_outliers'] = True
            if any(word in analysis_text for word in ['alta porcentagem', 'high percentage', 'muitos outliers']):
                patterns['many_outliers'] = True
              
        # Distribution patterns  
        if 'assimétrica' in analysis_text or 'asymmetric' in analysis_text or 'skew' in analysis_text:  
            patterns['asymmetric_distributions'] = True
        
        # Clustering patterns
        if 'cluster' in analysis_text or 'agrupamento' in analysis_text:
            patterns['has_clusters'] = True
        
        # Temporal patterns
        if 'tendência' in analysis_text or 'trend' in analysis_text or 'padrão temporal' in analysis_text:
            patterns['has_temporal_patterns'] = True
              
        return patterns  
      
    def _extract_recommendations(self, analysis_result: Dict[str, Any]) -> List[str]:  
        """Extract recommendations from analysis"""  
        recommendations = []  
        analysis_text = analysis_result.get('analysis', '')  
          
        if 'recomenda' in analysis_text.lower() or 'consider' in analysis_text.lower():  
            rec_section = analysis_text.split('Recomendações' if 'Recomendações' in analysis_text else 'ecomenda')[1] if 'ecomenda' in analysis_text else analysis_text
            lines = rec_section.split('\n')  
            for line in lines:  
                if line.strip().startswith('-'):  
                    recommendations.append(line.strip())
          
        return recommendations[:5]
      
    def _update_insights(self, memory_entry: AnalysisMemory) -> None:  
        """Update global insights based on new analysis"""  
        analysis_type = memory_entry.analysis_type  
          
        if analysis_type not in self.data_insights:  
            self.data_insights[analysis_type] = {  
                'count': 0,  
                'patterns': {},  
                'last_analysis': None,
                'key_findings': []
            }  
          
        self.data_insights[analysis_type]['count'] += 1  
        self.data_insights[analysis_type]['last_analysis'] = memory_entry.timestamp
        self.data_insights[analysis_type]['key_findings'].extend(memory_entry.key_findings[:3])
          
        for pattern, value in memory_entry.data_patterns.items():  
            self.data_insights[analysis_type]['patterns'][pattern] = value
    
    def _update_global_patterns(self, memory_entry: AnalysisMemory) -> None:
        """Update global patterns across all analyses"""
        for pattern, value in memory_entry.data_patterns.items():
            if pattern not in self.global_patterns:
                self.global_patterns[pattern] = {'count': 0, 'analyses': []}
            self.global_patterns[pattern]['count'] += 1
            self.global_patterns[pattern]['analyses'].append(memory_entry.analysis_type)
      
    def get_contextual_memory(self, question: str, n: int = 3) -> List[AnalysisMemory]:  
        """Get relevant memory entries based on question context"""  
        question_lower = question.lower()  
        relevant_entries = []  
          
        for entry in reversed(self.analysis_history):  
            relevance_score = 0  
              
            # Question similarity  
            if any(word in entry.question.lower() for word in question_lower.split() if len(word) > 3):  
                relevance_score += 2  
              
            # Analysis type relevance  
            type_keywords = {
                'correlation': ['correlação', 'relação', 'correlation', 'relaciona'],
                'outlier': ['outlier', 'anomalia', 'atípico'],
                'distribution': ['distribuição', 'histograma', 'distribution'],
                'clustering': ['cluster', 'agrupamento', 'grupo'],
                'temporal': ['tempo', 'temporal', 'time', 'trend'],
                'conclusions': ['conclus', 'aprend', 'insights']
            }
            
            for atype, keywords in type_keywords.items():
                if any(word in question_lower for word in keywords) and entry.analysis_type == atype:
                    relevance_score += 3
                    break
              
            if relevance_score > 0:  
                relevant_entries.append((entry, relevance_score))  
          
        relevant_entries.sort(key=lambda x: x[1], reverse=True)  
        return [entry for entry, _ in relevant_entries[:n]]  
      
    def generate_summary(self) -> str:  
        """Generate a summary of all analyses performed"""  
        if not self.analysis_history:  
            return "Nenhuma análise realizada ainda."  
          
        summary_parts = []  
        summary_parts.append(f"## Resumo das Análises Realizadas\n")  
        summary_parts.append(f"Total de análises: {len(self.analysis_history)}\n")  
          
        # Analysis types summary  
        type_counts = {}  
        for entry in self.analysis_history:  
            type_counts[entry.analysis_type] = type_counts.get(entry.analysis_type, 0) + 1  
          
        summary_parts.append("\n### Tipos de Análises:")  
        for analysis_type, count in type_counts.items():  
            summary_parts.append(f"- {analysis_type}: {count} análises")  
          
        # Key patterns found  
        if self.global_patterns:  
            summary_parts.append("\n### Padrões Identificados:")  
            for pattern, data in self.global_patterns.items():  
                summary_parts.append(f"- {pattern}: detectado em {data['count']} análises")  
          
        return "\n".join(summary_parts)
    
    def generate_comprehensive_conclusions(self) -> str:
        """Generate comprehensive conclusions from all analyses performed"""
        if not self.analysis_history:
            return "Nenhuma análise foi realizada ainda. Por favor, faça algumas análises primeiro."
        
        conclusions = []
        conclusions.append("# Conclusões Consolidadas da Análise Exploratória de Dados\n")
        conclusions.append(f"*Baseado em {len(self.analysis_history)} análises realizadas*\n")
        
        # 1. Overview of analyses
        conclusions.append("## 1. Resumo das Análises Realizadas")
        type_counts = {}
        for entry in self.analysis_history:
            type_counts[entry.analysis_type] = type_counts.get(entry.analysis_type, 0) + 1
        
        for atype, count in type_counts.items():
            conclusions.append(f"- **{atype}**: {count} análise(s)")
        conclusions.append("")
        
        # 2. Key Patterns Discovered
        conclusions.append("## 2. Principais Padrões Identificados")
        if self.global_patterns:
            for pattern, data in self.global_patterns.items():
                pattern_name = pattern.replace('_', ' ').title()
                conclusions.append(f"- **{pattern_name}**: Identificado em {data['count']} análise(s)")
                conclusions.append(f"  - Detectado em: {', '.join(set(data['analyses']))}")
        else:
            conclusions.append("- Nenhum padrão global identificado ainda")
        conclusions.append("")
        
        # 3. Key Findings by Analysis Type
        conclusions.append("## 3. Principais Descobertas por Tipo de Análise")
        for atype, insights in self.data_insights.items():
            if insights['key_findings']:
                conclusions.append(f"\n### {atype.title()}:")
                unique_findings = list(set(insights['key_findings']))[:5]
                for finding in unique_findings:
                    conclusions.append(f"  {finding}")
        conclusions.append("")
        
        # 4. Data Quality Insights
        conclusions.append("## 4. Insights sobre Qualidade dos Dados")
        quality_issues = []
        if self.global_patterns.get('has_outliers'):
            quality_issues.append("- Outliers detectados: Podem indicar problemas de qualidade ou valores genuinamente extremos")
        if self.global_patterns.get('many_outliers'):
            quality_issues.append("- Alta concentração de outliers: Requer investigação aprofundada")
        if self.global_patterns.get('asymmetric_distributions'):
            quality_issues.append("- Distribuições assimétricas: Considere transformações para normalização")
        
        if quality_issues:
            conclusions.extend(quality_issues)
        else:
            conclusions.append("- Dados aparentam boa qualidade geral")
        conclusions.append("")
        
        # 5. Relationships and Correlations
        conclusions.append("## 5. Relacionamentos entre Variáveis")
        if self.global_patterns.get('has_correlations'):
            conclusions.append("- Correlações significativas foram identificadas entre variáveis")
            if self.global_patterns.get('strong_correlations'):
                conclusions.append("- Algumas correlações são particularmente fortes")
                conclusions.append("- **Recomendação**: Considere análise de multicolinearidade para modelagem")
        else:
            conclusions.append("- Variáveis aparentam ser relativamente independentes")
        conclusions.append("")
        
        # 6. Clustering and Segmentation
        if self.global_patterns.get('has_clusters'):
            conclusions.append("## 6. Segmentação e Agrupamentos")
            conclusions.append("- Clusters distintos foram identificados nos dados")
            conclusions.append("- **Oportunidade**: Use segmentação para análises direcionadas")
            conclusions.append("")
        
        # 7. Temporal Patterns
        if self.global_patterns.get('has_temporal_patterns'):
            conclusions.append("## 7. Padrões Temporais")
            conclusions.append("- Tendências temporais foram identificadas")
            conclusions.append("- **Recomendação**: Considere análise de séries temporais")
            conclusions.append("")
        
        # 8. Consolidated Recommendations
        conclusions.append("## 8. Recomendações Consolidadas")
        all_recommendations = []
        for entry in self.analysis_history:
            all_recommendations.extend(entry.recommendations)
        
        unique_recommendations = list(set(all_recommendations))[:8]
        if unique_recommendations:
            for rec in unique_recommendations:
                conclusions.append(f"{rec}")
        else:
            conclusions.append("- Continue explorando os dados com análises mais específicas")
        conclusions.append("")
        
        # 9. Next Steps
        conclusions.append("## 9. Próximos Passos Sugeridos")
        
        # Suggest missing analyses
        performed_types = set(type_counts.keys())
        all_types = {'correlation', 'outlier', 'distribution', 'clustering', 'temporal', 'summary'}
        missing_types = all_types - performed_types
        
        if missing_types:
            conclusions.append("### Análises Recomendadas:")
            type_descriptions = {
                'correlation': 'Análise de correlações para entender relacionamentos',
                'outlier': 'Detecção de outliers para identificar anomalias',
                'distribution': 'Análise de distribuições para entender padrões',
                'clustering': 'Clustering para identificar segmentos',
                'temporal': 'Análise temporal para identificar tendências',
                'summary': 'Resumo geral dos dados'
            }
            for mtype in missing_types:
                if mtype in type_descriptions:
                    conclusions.append(f"- {type_descriptions[mtype]}")
        
        conclusions.append("\n### Ações Gerais:")
        conclusions.append("- Validar descobertas com stakeholders")
        conclusions.append("- Documentar insights para referência futura")
        conclusions.append("- Considerar análises preditivas se apropriado")
        
        return "\n".join(conclusions)
    
    def clear(self) -> None:
        """Clear all memory"""
        self.analysis_history.clear()
        self.data_insights.clear()
        self.conversation_summary = ""
        self.global_patterns.clear()