"""
ReportingAgent - Tạo báo cáo Markdown từ findings và LLM summaries
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import os
import sys

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


class ReportingAgent:
    """
    Agent để tạo báo cáo Markdown từ findings và LLM summaries.
    Nhận kết quả từ StaticAnalysisAgent và LLMOrchestratorAgent để tạo báo cáo tổng hợp.
    """
    
    def __init__(self, output_dir: str = "reports"):
        """
        Initialize ReportingAgent
        
        Args:
            output_dir: Thư mục để lưu báo cáo
        """
        self.logger = logging.getLogger(__name__)
        self.output_dir = output_dir
        
        # Tạo thư mục output nếu chưa tồn tại
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.logger.info(f"ReportingAgent initialized with output directory: {self.output_dir}")
    
    def generate_report(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        LangGraph node function để tạo báo cáo từ findings và LLM analysis
        
        Args:
            state: LangGraph state chứa static_analysis_results và llm_analysis
            
        Returns:
            Updated state với report information
        """
        self.logger.info("ReportingAgent generating report...")
        
        try:
            # Extract data từ state
            static_analysis_results = state.get('static_analysis_results', {})
            llm_analysis = state.get('llm_analysis', {})
            filename = state.get('filename', 'unknown_file')
            
            if not static_analysis_results and not llm_analysis:
                self.logger.warning("No analysis results found in state")
                return self._update_state_with_error(state, "No analysis results available for reporting")
            
            # Generate report
            report_content = self._create_markdown_report(
                static_analysis_results, 
                llm_analysis, 
                filename
            )
            
            # Save report to file
            report_filename = self._save_report(report_content, filename)
            
            # Update state với report information
            updated_state = state.copy()
            updated_state['report'] = {
                'filename': report_filename,
                'content': report_content,
                'generated_at': datetime.now().isoformat(),
                'output_path': os.path.join(self.output_dir, report_filename)
            }
            updated_state['current_agent'] = 'reporter'
            updated_state['processing_status'] = 'report_generated'
            
            self.logger.info(f"Report generated successfully: {report_filename}")
            return updated_state
            
        except Exception as e:
            self.logger.error(f"Error in ReportingAgent: {e}")
            return self._update_state_with_error(state, str(e))
    
    def _create_markdown_report(self, 
                               static_results: Dict[str, Any], 
                               llm_analysis: Dict[str, Any], 
                               filename: str) -> str:
        """
        Tạo nội dung báo cáo Markdown
        
        Args:
            static_results: Kết quả từ StaticAnalysisAgent
            llm_analysis: Kết quả từ LLMOrchestratorAgent
            filename: Tên file được phân tích
            
        Returns:
            Nội dung báo cáo Markdown
        """
        report_lines = []
        
        # Header
        report_lines.extend([
            f"# 📊 Code Analysis Report",
            f"",
            f"**File:** `{filename}`  ",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
            f"**Analysis Tool:** DeepCode-Insight  ",
            f"",
            "---",
            ""
        ])
        
        # Executive Summary từ LLM
        if llm_analysis and llm_analysis.get('summary'):
            report_lines.extend([
                "## 🎯 Executive Summary",
                "",
                llm_analysis['summary'],
                "",
                "---",
                ""
            ])
        
        # Static Analysis Results
        if static_results:
            report_lines.extend(self._format_static_analysis_section(static_results))
        
        # LLM Analysis Results
        if llm_analysis:
            report_lines.extend(self._format_llm_analysis_section(llm_analysis))
        
        # Recommendations và Action Items
        if llm_analysis and llm_analysis.get('recommendations'):
            report_lines.extend(self._format_recommendations_section(llm_analysis['recommendations']))
        
        # Footer
        report_lines.extend([
            "---",
            "",
            "## 📝 Report Information",
            "",
            f"- **Generated by:** DeepCode-Insight ReportingAgent",
            f"- **Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"- **File Analyzed:** `{filename}`"
        ])
        
        if llm_analysis and llm_analysis.get('llm_metadata'):
            metadata = llm_analysis['llm_metadata']
            report_lines.extend([
                f"- **LLM Model:** {metadata.get('model_used', 'Unknown')}",
                f"- **Analysis Type:** {metadata.get('analysis_type', 'Standard')}"
            ])
        
        report_lines.extend([
            "",
            "*This report was automatically generated. Please review findings and recommendations carefully.*"
        ])
        
        return "\n".join(report_lines)
    
    def _format_static_analysis_section(self, static_results: Dict[str, Any]) -> List[str]:
        """Format static analysis results section"""
        lines = [
            "## 🔍 Static Analysis Results",
            ""
        ]
        
        # Metrics
        if 'metrics' in static_results:
            metrics = static_results['metrics']
            lines.extend([
                "### 📈 Code Metrics",
                "",
                "| Metric | Value |",
                "|--------|-------|"
            ])
            
            for metric_name, metric_value in metrics.items():
                formatted_name = metric_name.replace('_', ' ').title()
                if isinstance(metric_value, float):
                    formatted_value = f"{metric_value:.2f}"
                else:
                    formatted_value = str(metric_value)
                lines.append(f"| {formatted_name} | {formatted_value} |")
            
            lines.extend(["", ""])
        
        # Issues
        if 'static_issues' in static_results:
            issues = static_results['static_issues']
            if issues:
                lines.extend([
                    "### ⚠️ Issues Found",
                    ""
                ])
                
                for issue_type, issue_list in issues.items():
                    if issue_list:
                        formatted_type = issue_type.replace('_', ' ').title()
                        lines.extend([
                            f"#### {formatted_type}",
                            ""
                        ])
                        
                        for issue in issue_list:
                            if isinstance(issue, dict):
                                issue_desc = self._format_issue_description(issue)
                                lines.append(f"- {issue_desc}")
                            else:
                                lines.append(f"- {str(issue)}")
                        
                        lines.append("")
            else:
                lines.extend([
                    "### ✅ No Issues Found",
                    "",
                    "Static analysis did not identify any issues in this file.",
                    ""
                ])
        
        # Suggestions
        if 'suggestions' in static_results and static_results['suggestions']:
            lines.extend([
                "### 💡 Suggestions",
                ""
            ])
            
            for suggestion in static_results['suggestions']:
                lines.append(f"- {suggestion}")
            
            lines.append("")
        
        lines.extend(["---", ""])
        return lines
    
    def _format_llm_analysis_section(self, llm_analysis: Dict[str, Any]) -> List[str]:
        """Format LLM analysis results section"""
        lines = [
            "## 🤖 AI-Powered Analysis",
            ""
        ]
        
        # Detailed Analysis
        if llm_analysis.get('detailed_analysis'):
            lines.extend([
                "### 📋 Detailed Analysis",
                "",
                llm_analysis['detailed_analysis'],
                "",
                ""
            ])
        
        # Priority Issues
        if llm_analysis.get('priority_issues'):
            lines.extend([
                "### 🚨 Priority Issues",
                ""
            ])
            
            for issue in llm_analysis['priority_issues']:
                if isinstance(issue, dict):
                    priority = issue.get('priority', 'Medium')
                    description = issue.get('description', 'No description')
                    action = issue.get('action', 'Review required')
                    
                    # Emoji based on priority
                    priority_emoji = {
                        'High': '🔴',
                        'Medium': '🟡', 
                        'Low': '🟢'
                    }.get(priority, '⚪')
                    
                    lines.extend([
                        f"#### {priority_emoji} {priority} Priority",
                        f"**Issue:** {description}",
                        f"**Action:** {action}",
                        ""
                    ])
                else:
                    lines.append(f"- {str(issue)}")
            
            lines.append("")
        
        # Code Quality Assessment
        if llm_analysis.get('code_quality_assessment'):
            lines.extend([
                "### 🎯 Code Quality Assessment",
                "",
                llm_analysis['code_quality_assessment'],
                "",
                ""
            ])
        
        # Improvement Suggestions
        if llm_analysis.get('improvement_suggestions'):
            lines.extend([
                "### 🚀 Improvement Suggestions",
                ""
            ])
            
            for suggestion in llm_analysis['improvement_suggestions']:
                if isinstance(suggestion, dict):
                    title = suggestion.get('title', 'Improvement')
                    description = suggestion.get('description', 'No description')
                    effort = suggestion.get('effort', 'Medium')
                    
                    # Emoji based on effort
                    effort_emoji = {
                        'Low': '🟢',
                        'Medium': '🟡',
                        'High': '🔴'
                    }.get(effort, '⚪')
                    
                    lines.extend([
                        f"- **{title}** {effort_emoji} _{effort} effort_",
                        f"  {description}",
                        ""
                    ])
                else:
                    lines.append(f"- {str(suggestion)}")
            
            lines.append("")
        
        lines.extend(["---", ""])
        return lines
    
    def _format_recommendations_section(self, recommendations: List[Dict[str, Any]]) -> List[str]:
        """Format recommendations section"""
        lines = [
            "## 📋 Action Items & Recommendations",
            ""
        ]
        
        if not recommendations:
            lines.extend([
                "No specific recommendations available.",
                ""
            ])
            return lines
        
        # Group by priority/effort
        high_priority = []
        medium_priority = []
        low_priority = []
        
        for rec in recommendations:
            if isinstance(rec, dict):
                effort = rec.get('effort', 'Medium').lower()
                if effort in ['high', 'critical']:
                    high_priority.append(rec)
                elif effort in ['low', 'easy']:
                    low_priority.append(rec)
                else:
                    medium_priority.append(rec)
            else:
                medium_priority.append({'description': str(rec), 'effort': 'Medium'})
        
        # High Priority
        if high_priority:
            lines.extend([
                "### 🔴 High Priority Actions",
                ""
            ])
            for rec in high_priority:
                desc = rec.get('description', str(rec))
                lines.append(f"- [ ] {desc}")
            lines.append("")
        
        # Medium Priority
        if medium_priority:
            lines.extend([
                "### 🟡 Medium Priority Actions",
                ""
            ])
            for rec in medium_priority:
                desc = rec.get('description', str(rec))
                lines.append(f"- [ ] {desc}")
            lines.append("")
        
        # Low Priority
        if low_priority:
            lines.extend([
                "### 🟢 Low Priority Actions",
                ""
            ])
            for rec in low_priority:
                desc = rec.get('description', str(rec))
                lines.append(f"- [ ] {desc}")
            lines.append("")
        
        return lines
    
    def _format_issue_description(self, issue: Dict[str, Any]) -> str:
        """Format individual issue description"""
        parts = []
        
        if 'type' in issue:
            parts.append(f"**{issue['type']}**")
        
        if 'name' in issue:
            parts.append(f"in `{issue['name']}`")
        
        if 'line' in issue:
            parts.append(f"(line {issue['line']})")
        
        if 'message' in issue:
            parts.append(f"- {issue['message']}")
        
        return " ".join(parts) if parts else str(issue)
    
    def _save_report(self, content: str, filename: str) -> str:
        """
        Lưu báo cáo vào file
        
        Args:
            content: Nội dung báo cáo
            filename: Tên file gốc
            
        Returns:
            Tên file báo cáo đã lưu
        """
        # Tạo tên file báo cáo
        base_name = os.path.splitext(os.path.basename(filename))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"report_{base_name}_{timestamp}.md"
        
        # Đường dẫn đầy đủ
        report_path = os.path.join(self.output_dir, report_filename)
        
        # Lưu file
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        self.logger.info(f"Report saved to: {report_path}")
        return report_filename
    
    def _update_state_with_error(self, state: Dict[str, Any], error_message: str) -> Dict[str, Any]:
        """Update state với error information"""
        updated_state = state.copy()
        updated_state['error'] = error_message
        updated_state['current_agent'] = 'reporter'
        updated_state['processing_status'] = 'report_error'
        return updated_state


def create_reporting_agent(output_dir: str = "reports") -> ReportingAgent:
    """
    Convenience function để tạo ReportingAgent instance
    
    Args:
        output_dir: Thư mục output cho báo cáo
        
    Returns:
        ReportingAgent instance
    """
    return ReportingAgent(output_dir=output_dir)


def reporting_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node function wrapper cho ReportingAgent
    
    Args:
        state: LangGraph state
        
    Returns:
        Updated state với report information
    """
    # Create agent instance (in production, này nên được cached)
    agent = create_reporting_agent()
    return agent.generate_report(state)


def demo_reporting_agent():
    """Demo function để test ReportingAgent"""
    print("🚀 Demo ReportingAgent")
    print("=" * 50)
    
    # Sample static analysis results
    sample_static_results = {
        'filename': 'demo.py',
        'static_issues': {
            'missing_docstrings': [
                {
                    'type': 'missing_function_docstring',
                    'name': 'calculate_sum',
                    'line': 5,
                    'message': 'Function lacks docstring'
                },
                {
                    'type': 'missing_class_docstring', 
                    'name': 'Calculator',
                    'line': 15,
                    'message': 'Class lacks docstring'
                }
            ],
            'code_style': [
                {
                    'type': 'line_too_long',
                    'line': 25,
                    'message': 'Line exceeds 80 characters'
                }
            ]
        },
        'metrics': {
            'code_quality_score': 75.5,
            'maintainability_index': 68.2,
            'complexity_score': 3.1
        },
        'suggestions': [
            'Add docstrings to functions and classes',
            'Break long lines for better readability',
            'Consider adding type hints'
        ]
    }
    
    # Sample LLM analysis results
    sample_llm_analysis = {
        'filename': 'demo.py',
        'summary': 'The code shows good structure but lacks documentation. Several style improvements needed.',
        'detailed_analysis': 'The code demonstrates solid programming practices with clear variable names and logical flow. However, documentation is insufficient and some style guidelines are not followed.',
        'priority_issues': [
            {
                'priority': 'High',
                'description': 'Missing docstrings reduce code maintainability',
                'action': 'Add comprehensive docstrings to all functions and classes'
            },
            {
                'priority': 'Medium', 
                'description': 'Line length violations affect readability',
                'action': 'Refactor long lines to improve code readability'
            }
        ],
        'recommendations': [
            {
                'description': 'Implement comprehensive documentation strategy',
                'effort': 'Medium'
            },
            {
                'description': 'Set up automated code formatting with black',
                'effort': 'Low'
            },
            {
                'description': 'Add type hints for better code clarity',
                'effort': 'High'
            }
        ],
        'code_quality_assessment': 'Code quality is above average but has room for improvement in documentation and style consistency.',
        'improvement_suggestions': [
            {
                'title': 'Documentation Enhancement',
                'description': 'Add docstrings following Google or NumPy style guide',
                'effort': 'Medium'
            },
            {
                'title': 'Code Formatting',
                'description': 'Use automated formatters like black and isort',
                'effort': 'Low'
            },
            {
                'title': 'Type Safety',
                'description': 'Implement type hints and use mypy for type checking',
                'effort': 'High'
            }
        ],
        'llm_metadata': {
            'model_used': 'codellama',
            'analysis_type': 'comprehensive_code_review'
        }
    }
    
    # Sample LangGraph state
    sample_state = {
        'static_analysis_results': sample_static_results,
        'llm_analysis': sample_llm_analysis,
        'filename': 'demo.py',
        'current_agent': 'llm_orchestrator',
        'processing_status': 'llm_analysis_completed'
    }
    
    try:
        # Test ReportingAgent
        agent = create_reporting_agent(output_dir="demo_reports")
        
        print(f"📁 Output Directory: {agent.output_dir}")
        
        # Generate report
        print("\n📝 Generating report...")
        result_state = agent.generate_report(sample_state)
        
        if result_state.get('processing_status') == 'report_generated':
            report_info = result_state['report']
            print(f"✅ Report generated successfully!")
            print(f"📄 Report file: {report_info['filename']}")
            print(f"📍 Full path: {report_info['output_path']}")
            print(f"⏰ Generated at: {report_info['generated_at']}")
            
            # Show preview of report content
            print(f"\n📋 Report Preview (first 500 chars):")
            print("-" * 50)
            preview = report_info['content'][:500]
            print(preview)
            if len(report_info['content']) > 500:
                print("...")
            print("-" * 50)
            
        else:
            print(f"❌ Report generation failed: {result_state.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    demo_reporting_agent() 