"""
LLMOrchestratorAgent - Orchestrate LLM calls để phân tích findings từ StaticAnalysisAgent
"""

import logging
from typing import Dict, List, Optional, Any, Union
import json
import sys
import os

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from ..utils.llm_caller import OllamaLLMCaller, OllamaModel, OllamaResponse, OllamaAPIError, create_llm_caller


class LLMOrchestratorAgent:
    """
    Agent để orchestrate LLM calls cho code analysis.
    Nhận findings từ StaticAnalysisAgent, format summary prompt và gọi llm_caller
    """
    
    def __init__(self, 
                 model: Union[str, OllamaModel] = OllamaModel.CODELLAMA,
                 base_url: str = "http://localhost:11434",
                 timeout: int = 120):
        """
        Initialize LLMOrchestratorAgent
        
        Args:
            model: LLM model để sử dụng
            base_url: Ollama server URL
            timeout: Request timeout
        """
        self.logger = logging.getLogger(__name__)
        
        try:
            self.llm_caller = OllamaLLMCaller(
                model=model,
                base_url=base_url,
                timeout=timeout
            )
            self.logger.info(f"LLMOrchestratorAgent initialized with model: {self.llm_caller.model}")
        except Exception as e:
            self.logger.error(f"Failed to initialize LLMOrchestratorAgent: {e}")
            raise
    
    def process_findings(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        LangGraph node function để process findings từ StaticAnalysisAgent
        
        Args:
            state: LangGraph state chứa findings từ previous agents
            
        Returns:
            Updated state với LLM analysis results
        """
        self.logger.info("LLMOrchestratorAgent processing findings...")
        
        try:
            # Extract findings từ state
            static_analysis_results = state.get('static_analysis_results', {})
            code_content = state.get('code_content', '')
            filename = state.get('filename', '<unknown>')
            
            if not static_analysis_results:
                self.logger.warning("No static analysis results found in state")
                return self._update_state_with_error(state, "No static analysis results available")
            
            # Generate LLM analysis
            llm_analysis = self.analyze_findings_with_llm(
                static_analysis_results, 
                code_content, 
                filename
            )
            
            # Update state với LLM results
            updated_state = state.copy()
            updated_state['llm_analysis'] = llm_analysis
            updated_state['current_agent'] = 'llm_orchestrator'
            updated_state['processing_status'] = 'llm_analysis_completed'
            
            self.logger.info("LLM analysis completed successfully")
            return updated_state
            
        except Exception as e:
            self.logger.error(f"Error in LLMOrchestratorAgent: {e}")
            return self._update_state_with_error(state, str(e))
    
    def analyze_findings_with_llm(self, 
                                  static_results: Dict[str, Any], 
                                  code_content: str = "",
                                  filename: str = "<unknown>") -> Dict[str, Any]:
        """
        Analyze static analysis findings sử dụng LLM
        
        Args:
            static_results: Results từ StaticAnalysisAgent
            code_content: Original code content (optional)
            filename: Filename for context
            
        Returns:
            Dict chứa LLM analysis results
        """
        analysis_result = {
            'filename': filename,
            'summary': '',
            'detailed_analysis': '',
            'priority_issues': [],
            'recommendations': [],
            'code_quality_assessment': '',
            'improvement_suggestions': [],
            'llm_metadata': {
                'model_used': self.llm_caller.model,
                'analysis_type': 'comprehensive_code_review'
            }
        }
        
        try:
            # Format summary prompt
            summary_prompt = self._format_summary_prompt(static_results, filename)
            
            # Get LLM summary
            self.logger.debug("Generating LLM summary...")
            summary_response = self.llm_caller.generate(
                summary_prompt,
                temperature=0.3,  # Lower temperature for consistent analysis
                max_tokens=500
            )
            analysis_result['summary'] = summary_response.response
            
            # Generate detailed analysis nếu có code content
            if code_content:
                detailed_prompt = self._format_detailed_analysis_prompt(
                    static_results, code_content, filename
                )
                
                self.logger.debug("Generating detailed LLM analysis...")
                detailed_response = self.llm_caller.generate(
                    detailed_prompt,
                    code_snippet=code_content,
                    temperature=0.3,
                    max_tokens=800
                )
                analysis_result['detailed_analysis'] = detailed_response.response
            
            # Generate priority issues
            priority_prompt = self._format_priority_issues_prompt(static_results)
            priority_response = self.llm_caller.generate(
                priority_prompt,
                temperature=0.2,
                max_tokens=400
            )
            analysis_result['priority_issues'] = self._parse_priority_issues(
                priority_response.response
            )
            
            # Generate recommendations
            recommendations_prompt = self._format_recommendations_prompt(static_results)
            recommendations_response = self.llm_caller.generate(
                recommendations_prompt,
                temperature=0.4,
                max_tokens=600
            )
            analysis_result['recommendations'] = self._parse_recommendations(
                recommendations_response.response
            )
            
            # Generate code quality assessment
            quality_prompt = self._format_quality_assessment_prompt(static_results)
            quality_response = self.llm_caller.generate(
                quality_prompt,
                temperature=0.3,
                max_tokens=300
            )
            analysis_result['code_quality_assessment'] = quality_response.response
            
            # Generate improvement suggestions
            improvement_prompt = self._format_improvement_suggestions_prompt(static_results)
            improvement_response = self.llm_caller.generate(
                improvement_prompt,
                temperature=0.5,
                max_tokens=500
            )
            analysis_result['improvement_suggestions'] = self._parse_improvement_suggestions(
                improvement_response.response
            )
            
            self.logger.info("LLM analysis completed successfully")
            
        except OllamaAPIError as e:
            self.logger.error(f"LLM API error: {e}")
            analysis_result['error'] = f"LLM API error: {e.message}"
        except Exception as e:
            self.logger.error(f"Error in LLM analysis: {e}")
            analysis_result['error'] = f"Analysis error: {str(e)}"
        
        return analysis_result
    
    def _format_summary_prompt(self, static_results: Dict[str, Any], filename: str) -> str:
        """Format prompt cho LLM summary"""
        issues = static_results.get('static_issues', {})
        metrics = static_results.get('metrics', {})
        suggestions = static_results.get('suggestions', [])
        
        # Count total issues
        total_issues = sum(len(issue_list) for issue_list in issues.values())
        
        # Format issues summary
        issues_summary = []
        for issue_type, issue_list in issues.items():
            if issue_list:
                issues_summary.append(f"- {issue_type.replace('_', ' ').title()}: {len(issue_list)} issues")
        
        prompt = f"""Bạn là một chuyên gia code review. Hãy phân tích kết quả static analysis sau và tạo summary ngắn gọn.

File: {filename}
Total Issues: {total_issues}

Issues Found:
{chr(10).join(issues_summary) if issues_summary else "- Không có issues được phát hiện"}

Code Quality Metrics:
- Quality Score: {metrics.get('code_quality_score', 'N/A')}/100
- Maintainability Index: {metrics.get('maintainability_index', 'N/A')}/100
- Cyclomatic Complexity: {metrics.get('cyclomatic_complexity', 'N/A')}
- Lines of Code: {metrics.get('lines_of_code', 'N/A')}

Existing Suggestions: {len(suggestions)} suggestions

Hãy tạo một summary ngắn gọn (2-3 câu) về tình trạng code quality và những điểm cần chú ý nhất."""
        
        return prompt
    
    def _format_detailed_analysis_prompt(self, static_results: Dict[str, Any], 
                                       code_content: str, filename: str) -> str:
        """Format prompt cho detailed analysis"""
        issues = static_results.get('static_issues', {})
        
        # Get most critical issues
        critical_issues = []
        for issue_type, issue_list in issues.items():
            for issue in issue_list[:3]:  # Top 3 issues per type
                critical_issues.append(f"- {issue_type}: {issue.get('message', 'Unknown issue')} (Line {issue.get('line', '?')})")
        
        prompt = f"""Bạn là một senior developer đang review code. Hãy phân tích chi tiết code sau dựa trên static analysis findings.

File: {filename}

Critical Issues Found:
{chr(10).join(critical_issues[:10]) if critical_issues else "- Không có critical issues"}

Hãy phân tích:
1. Code structure và organization
2. Potential bugs hoặc security issues
3. Performance implications
4. Maintainability concerns
5. Best practices compliance

Đưa ra phân tích chi tiết và constructive feedback."""
        
        return prompt
    
    def _format_priority_issues_prompt(self, static_results: Dict[str, Any]) -> str:
        """Format prompt để identify priority issues"""
        issues = static_results.get('static_issues', {})
        
        all_issues = []
        for issue_type, issue_list in issues.items():
            for issue in issue_list:
                all_issues.append({
                    'type': issue_type,
                    'message': issue.get('message', ''),
                    'line': issue.get('line', 0),
                    'severity': self._estimate_severity(issue_type, issue)
                })
        
        # Sort by estimated severity
        all_issues.sort(key=lambda x: x['severity'], reverse=True)
        
        issues_text = []
        for issue in all_issues[:10]:  # Top 10 issues
            issues_text.append(f"- {issue['type']}: {issue['message']} (Line {issue['line']})")
        
        prompt = f"""Bạn là một tech lead đang prioritize issues để fix. Dựa trên danh sách issues sau, hãy identify top 5 priority issues cần fix ngay.

Issues Found:
{chr(10).join(issues_text) if issues_text else "- Không có issues"}

Hãy list top 5 priority issues theo format:
1. [Issue Type] - [Brief Description] - [Why Priority]
2. [Issue Type] - [Brief Description] - [Why Priority]
...

Focus vào issues có impact cao nhất đến code quality, security, hoặc maintainability."""
        
        return prompt
    
    def _format_recommendations_prompt(self, static_results: Dict[str, Any]) -> str:
        """Format prompt cho recommendations"""
        metrics = static_results.get('metrics', {})
        suggestions = static_results.get('suggestions', [])
        
        prompt = f"""Bạn là một software architect đang đưa ra recommendations để improve codebase.

Current Metrics:
- Quality Score: {metrics.get('code_quality_score', 'N/A')}/100
- Maintainability Index: {metrics.get('maintainability_index', 'N/A')}/100
- Comment Ratio: {metrics.get('comment_ratio', 'N/A')}

Existing Suggestions:
{chr(10).join(f"- {s}" for s in suggestions[:5]) if suggestions else "- Không có suggestions"}

Hãy đưa ra 5-7 actionable recommendations để improve code quality, bao gồm:
1. Immediate actions (có thể làm ngay)
2. Short-term improvements (1-2 weeks)
3. Long-term architectural changes (nếu cần)

Format: 
- [Recommendation] - [Expected Impact] - [Effort Level: Low/Medium/High]"""
        
        return prompt
    
    def _format_quality_assessment_prompt(self, static_results: Dict[str, Any]) -> str:
        """Format prompt cho overall quality assessment"""
        metrics = static_results.get('metrics', {})
        issues = static_results.get('static_issues', {})
        
        total_issues = sum(len(issue_list) for issue_list in issues.values())
        
        prompt = f"""Bạn là một code quality expert. Hãy đánh giá overall quality của code dựa trên metrics sau:

Quality Metrics:
- Overall Score: {metrics.get('code_quality_score', 'N/A')}/100
- Maintainability: {metrics.get('maintainability_index', 'N/A')}/100
- Complexity: {metrics.get('cyclomatic_complexity', 'N/A')}
- Total Issues: {total_issues}
- Lines of Code: {metrics.get('lines_of_code', 'N/A')}

Hãy đưa ra assessment ngắn gọn (3-4 câu) về:
1. Overall code quality level (Excellent/Good/Fair/Poor)
2. Main strengths
3. Key areas for improvement
4. Readiness for production (nếu applicable)"""
        
        return prompt
    
    def _format_improvement_suggestions_prompt(self, static_results: Dict[str, Any]) -> str:
        """Format prompt cho improvement suggestions"""
        issues = static_results.get('static_issues', {})
        metrics = static_results.get('metrics', {})
        
        # Identify main problem areas
        problem_areas = []
        if len(issues.get('missing_docstrings', [])) > 0:
            problem_areas.append("Documentation")
        if len(issues.get('complex_functions', [])) > 0:
            problem_areas.append("Code Complexity")
        if len(issues.get('code_smells', [])) > 0:
            problem_areas.append("Code Smells")
        if metrics.get('maintainability_index', 100) < 60:
            problem_areas.append("Maintainability")
        
        prompt = f"""Bạn là một senior developer mentor. Dựa trên analysis results, hãy suggest concrete improvement steps.

Main Problem Areas: {', '.join(problem_areas) if problem_areas else 'None identified'}

Current Quality Score: {metrics.get('code_quality_score', 'N/A')}/100

Hãy suggest 5-6 specific improvement actions theo format:
1. [Action] - [How to implement] - [Expected benefit]
2. [Action] - [How to implement] - [Expected benefit]
...

Focus vào actionable steps mà developer có thể implement ngay để improve code quality."""
        
        return prompt
    
    def _estimate_severity(self, issue_type: str, issue: Dict[str, Any]) -> int:
        """Estimate severity score cho issue (higher = more severe)"""
        severity_map = {
            'syntax_error': 10,
            'security_issue': 9,
            'high_complexity': 8,
            'god_class': 7,
            'too_many_parameters': 6,
            'long_function': 5,
            'missing_docstring': 3,
            'unused_import': 2,
            'long_line': 1
        }
        
        base_severity = severity_map.get(issue_type, 4)
        
        # Adjust based on issue details
        if 'complexity' in issue and issue['complexity'] > 15:
            base_severity += 2
        if 'count' in issue and issue['count'] > 10:
            base_severity += 1
            
        return base_severity
    
    def _parse_priority_issues(self, llm_response: str) -> List[Dict[str, str]]:
        """Parse LLM response thành structured priority issues"""
        issues = []
        lines = llm_response.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-')):
                # Remove numbering
                clean_line = line.lstrip('0123456789.- ')
                if ' - ' in clean_line:
                    parts = clean_line.split(' - ', 2)
                    if len(parts) >= 2:
                        issues.append({
                            'type': parts[0].strip(),
                            'description': parts[1].strip(),
                            'reason': parts[2].strip() if len(parts) > 2 else ''
                        })
        
        return issues[:5]  # Top 5 only
    
    def _parse_recommendations(self, llm_response: str) -> List[Dict[str, str]]:
        """Parse LLM response thành structured recommendations"""
        recommendations = []
        lines = llm_response.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if line and (line.startswith('-') or line[0].isdigit()):
                # Remove bullet points/numbering
                clean_line = line.lstrip('0123456789.- ')
                if ' - ' in clean_line:
                    parts = clean_line.split(' - ')
                    if len(parts) >= 2:
                        recommendations.append({
                            'action': parts[0].strip(),
                            'impact': parts[1].strip() if len(parts) > 1 else '',
                            'effort': parts[2].strip() if len(parts) > 2 else 'Medium'
                        })
        
        return recommendations
    
    def _parse_improvement_suggestions(self, llm_response: str) -> List[Dict[str, str]]:
        """Parse LLM response thành structured improvement suggestions"""
        suggestions = []
        lines = llm_response.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if line and (line.startswith('-') or line[0].isdigit()):
                # Remove bullet points/numbering
                clean_line = line.lstrip('0123456789.- ')
                if ' - ' in clean_line:
                    parts = clean_line.split(' - ')
                    if len(parts) >= 2:
                        suggestions.append({
                            'action': parts[0].strip(),
                            'implementation': parts[1].strip() if len(parts) > 1 else '',
                            'benefit': parts[2].strip() if len(parts) > 2 else ''
                        })
        
        return suggestions
    
    def _update_state_with_error(self, state: Dict[str, Any], error_message: str) -> Dict[str, Any]:
        """Update state với error information"""
        updated_state = state.copy()
        updated_state['llm_analysis'] = {
            'error': error_message,
            'status': 'failed'
        }
        updated_state['current_agent'] = 'llm_orchestrator'
        updated_state['processing_status'] = 'llm_analysis_failed'
        
        return updated_state
    
    def check_llm_health(self) -> bool:
        """Check if LLM service is available"""
        try:
            return self.llm_caller.check_health()
        except Exception as e:
            self.logger.error(f"LLM health check failed: {e}")
            return False
    
    def get_available_models(self) -> List[str]:
        """Get list of available LLM models"""
        try:
            return self.llm_caller.list_models()
        except Exception as e:
            self.logger.error(f"Failed to get available models: {e}")
            return []


# Convenience functions for LangGraph integration
def create_llm_orchestrator_agent(model: Union[str, OllamaModel] = OllamaModel.CODELLAMA,
                                 base_url: str = "http://localhost:11434",
                                 timeout: int = 120) -> LLMOrchestratorAgent:
    """
    Factory function để tạo LLMOrchestratorAgent
    
    Args:
        model: LLM model để sử dụng
        base_url: Ollama server URL
        timeout: Request timeout
        
    Returns:
        LLMOrchestratorAgent instance
    """
    return LLMOrchestratorAgent(model=model, base_url=base_url, timeout=timeout)


def llm_orchestrator_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node function wrapper cho LLMOrchestratorAgent
    
    Args:
        state: LangGraph state
        
    Returns:
        Updated state với LLM analysis
    """
    # Create agent instance (in production, này nên được cached)
    agent = create_llm_orchestrator_agent()
    return agent.process_findings(state)


# Demo function
def demo_llm_orchestrator():
    """Demo function để test LLMOrchestratorAgent"""
    print("🤖 === LLMOrchestratorAgent Demo ===")
    
    # Sample static analysis results
    sample_static_results = {
        'filename': 'demo.py',
        'static_issues': {
            'missing_docstrings': [
                {'type': 'missing_function_docstring', 'name': 'calculate', 'line': 5, 'message': "Function 'calculate' thiếu docstring"},
                {'type': 'missing_class_docstring', 'name': 'Calculator', 'line': 10, 'message': "Class 'Calculator' thiếu docstring"}
            ],
            'complex_functions': [
                {'type': 'too_many_parameters', 'name': 'complex_func', 'line': 20, 'count': 8, 'message': "Function 'complex_func' có quá nhiều parameters (8)"}
            ],
            'code_smells': [
                {'type': 'long_line', 'line': 15, 'length': 150, 'message': "Line quá dài (150 characters)"}
            ],
            'unused_imports': []
        },
        'metrics': {
            'code_quality_score': 75.5,
            'maintainability_index': 68.2,
            'cyclomatic_complexity': 12,
            'lines_of_code': 85,
            'comment_ratio': 0.15
        },
        'suggestions': [
            'Thêm docstrings cho 2 functions/classes để cải thiện documentation',
            'Refactor 1 complex functions để cải thiện maintainability',
            'Chia 1 long lines để tuân thủ PEP 8'
        ]
    }
    
    sample_code = '''
def calculate(a, b, c, d, e, f, g, h):
    result = a + b + c + d + e + f + g + h
    if result > 100:
        return result * 2
    elif result > 50:
        return result * 1.5
    else:
        return result

class Calculator:
    def __init__(self):
        self.history = []
    
    def add(self, x, y):
        return x + y
'''
    
    # Sample LangGraph state
    sample_state = {
        'static_analysis_results': sample_static_results,
        'code_content': sample_code,
        'filename': 'demo.py',
        'current_agent': 'static_analyzer',
        'processing_status': 'static_analysis_completed'
    }
    
    try:
        # Test LLMOrchestratorAgent
        agent = create_llm_orchestrator_agent()
        
        print(f"🔗 LLM Model: {agent.llm_caller.model}")
        print(f"🌐 Base URL: {agent.llm_caller.base_url}")
        
        # Check health
        if agent.check_llm_health():
            print("✅ LLM service is available")
            
            # Process findings
            print("\n🔄 Processing findings with LLM...")
            result_state = agent.process_findings(sample_state)
            
            if 'llm_analysis' in result_state:
                analysis = result_state['llm_analysis']
                
                print(f"\n📋 LLM Analysis Results:")
                print(f"📁 File: {analysis.get('filename', 'N/A')}")
                
                if 'error' in analysis:
                    print(f"❌ Error: {analysis['error']}")
                else:
                    print(f"\n📝 Summary:")
                    print(f"  {analysis.get('summary', 'N/A')}")
                    
                    print(f"\n🎯 Priority Issues ({len(analysis.get('priority_issues', []))}):")
                    for i, issue in enumerate(analysis.get('priority_issues', [])[:3], 1):
                        print(f"  {i}. {issue.get('type', 'Unknown')}: {issue.get('description', 'N/A')}")
                    
                    print(f"\n💡 Recommendations ({len(analysis.get('recommendations', []))}):")
                    for i, rec in enumerate(analysis.get('recommendations', [])[:3], 1):
                        print(f"  {i}. {rec.get('action', 'N/A')} - {rec.get('effort', 'Medium')} effort")
                    
                    print(f"\n🏆 Quality Assessment:")
                    print(f"  {analysis.get('code_quality_assessment', 'N/A')}")
                    
                    print(f"\n🔧 Improvement Suggestions ({len(analysis.get('improvement_suggestions', []))}):")
                    for i, suggestion in enumerate(analysis.get('improvement_suggestions', [])[:3], 1):
                        print(f"  {i}. {suggestion.get('action', 'N/A')}")
            
            print("\n✅ Demo completed successfully!")
            
        else:
            print("❌ LLM service is not available. Please check Ollama server.")
            print("💡 Start Ollama server: ollama serve")
            print("💡 Pull CodeLlama model: ollama pull codellama")
            
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    demo_llm_orchestrator() 