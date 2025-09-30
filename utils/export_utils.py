import pandas as pd  
import matplotlib.pyplot as plt  
import plotly.graph_objects as go  
from reportlab.lib.pagesizes import letter, A4  
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image  
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle  
from reportlab.lib.units import inch  
from datetime import datetime  
import io  
import base64  
from typing import Dict, List, Any  
  
class ReportExporter:  
    """Export analysis results to various formats"""  
      
    def __init__(self):  
        self.styles = getSampleStyleSheet()  
        self.custom_styles = self._create_custom_styles()  
      
    def _create_custom_styles(self) -> Dict[str, ParagraphStyle]:  
        """Create custom paragraph styles"""  
        return {  
            'CustomTitle': ParagraphStyle(  
                'CustomTitle',  
                parent=self.styles['Heading1'],  
                fontSize=18,  
                spaceAfter=30,  
                textColor='#1f77b4'  
            ),  
            'CustomHeading': ParagraphStyle(  
                'CustomHeading',  
                parent=self.styles['Heading2'],  
                fontSize=14,  
                spaceAfter=12,  
                textColor='#333333'  
            )  
        }  
      
    def export_to_pdf(self, analysis_results: List[Dict[str, Any]],   
                     filename: str = "eda_report.pdf") -> str:  
        """Export analysis results to PDF report"""  
          
        doc = SimpleDocTemplate(filename, pagesize=A4)  
        story = []  
          
        # Title  
        title = Paragraph("Relatório de Análise Exploratória de Dados",   
                         self.custom_styles['CustomTitle'])  
        story.append(title)  
        story.append(Spacer(1, 12))  
          
        # Timestamp  
        timestamp = Paragraph(f"Gerado em: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}",   
                             self.styles['Normal'])  
        story.append(timestamp)  
        story.append(Spacer(1, 20))  
          
        # Analysis results  
        for i, result in enumerate(analysis_results, 1):  
            # Question  
            question_title = Paragraph(f"{i}. {result['question']}",   
                                     self.custom_styles['CustomHeading'])  
            story.append(question_title)  
            story.append(Spacer(1, 6))  
              
            # Analysis text  
            analysis_text = result['analysis'].replace('\n', '<br/>')  
            analysis_para = Paragraph(analysis_text, self.styles['Normal'])  
            story.append(analysis_para)  
            story.append(Spacer(1, 12))  
              
            # Add plots if available  
            if result.get('plots'):  
                for plot_name, fig in result['plots'].items():  
                    # Convert plot to image  
                    img_buffer = io.BytesIO()  
                    fig.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')  
                    img_buffer.seek(0)  
                      
                    # Add to PDF  
                    img = Image(img_buffer, width=6*inch, height=4*inch)  
                    story.append(img)  
                    story.append(Spacer(1, 12))  
          
        doc.build(story)  
        return filename  
      
    def export_data_summary(self, df: pd.DataFrame) -> Dict[str, Any]:  
        """Export comprehensive data summary"""  
        summary = {  
            'basic_info': {  
                'shape': df.shape,  
                'columns': df.columns.tolist(),  
                'dtypes': df.dtypes.to_dict(),  
                'memory_usage': df.memory_usage(deep=True).sum(),  
                'missing_values': df.isnull().sum().to_dict(),  
                'duplicates': df.duplicated().sum()  
            },  
            'numeric_summary': {},  
            'categorical_summary': {}  
        }  
          
        # Numeric columns summary  
        numeric_cols = df.select_dtypes(include=['number']).columns  
        if len(numeric_cols) > 0:  
            summary['numeric_summary'] = df[numeric_cols].describe().to_dict()  
          
        # Categorical columns summary  
        categorical_cols = df.select_dtypes(include=['object']).columns  
        for col in categorical_cols:  
            summary['categorical_summary'][col] = {  
                'unique_count': df[col].nunique(),  
                'top_values': df[col].value_counts().head(5).to_dict(),  
                'missing_count': df[col].isnull().sum()  
            }  
          
        return summary