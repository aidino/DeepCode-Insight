"""
LLMOrchestratorAgent - Orchestrate LLM calls với RAG context và Chain-of-Thought prompting
"""

import logging
from typing import Dict, List, Optional, Any, Union
import json
import sys
import os

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from ..utils.llm_interface import (
        BaseLLMProvider, LLMProviderFactory, LLMProvider, LLMResponse,
        create_llm_provider
    )
    from ..utils.llm_caller import OllamaModel
    from .rag_context import RAGContextAgent
except ImportError:
    from utils.llm_interface import (
        BaseLLMProvider, LLMProviderFactory, LLMProvider, LLMResponse,
        create_llm_provider
    )
    from utils.llm_caller import OllamaModel
    from rag_context import RAGContextAgent


class LLMOrchestratorAgent:
    """
    Enhanced Agent để orchestrate LLM calls với RAG context và Chain-of-Thought prompting.
    Hỗ trợ multiple LLM providers (Ollama, OpenAI, Gemini) và tích hợp với RAGContextAgent.
    """
    
    def __init__(self, 
                 provider: str = "ollama",
                 model: str = "codellama",
                 rag_context_agent: Optional[RAGContextAgent] = None,
                 enable_rag: bool = True,
                 enable_chain_of_thought: bool = True,
                 **provider_kwargs):
        """
        Initialize LLMOrchestratorAgent
        
        Args:
            provider: LLM provider (ollama, openai, gemini)
            model: Model name
            rag_context_agent: RAGContextAgent instance (optional)
            enable_rag: Enable RAG context retrieval
            enable_chain_of_thought: Enable Chain-of-Thought prompting
            **provider_kwargs: Additional provider-specific arguments
        """
        self.logger = logging.getLogger(__name__)
        self.enable_rag = enable_rag
        self.enable_chain_of_thought = enable_chain_of_thought
        
        try:
            # Initialize LLM provider
            self.llm_provider = create_llm_provider(
                provider=provider,
                model=model,
                **provider_kwargs
            )
            
            # Initialize RAG context agent if enabled
            self.rag_agent = rag_context_agent
            if self.enable_rag and not self.rag_agent:
                try:
                    self.rag_agent = RAGContextAgent()
                    self.logger.info("RAGContextAgent initialized successfully")
                except Exception as e:
                    self.logger.warning(f"Failed to initialize RAGContextAgent: {e}. RAG disabled.")
                    self.enable_rag = False
            
            self.logger.info(f"LLMOrchestratorAgent initialized with provider: {provider}, model: {model}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize LLMOrchestratorAgent: {e}")
            raise
    
    def process_findings(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhanced LangGraph node function với RAG context và Chain-of-Thought
        
        Args:
            state: LangGraph state chứa findings từ previous agents
            
        Returns:
            Updated state với enhanced LLM analysis results
        """
        self.logger.info("LLMOrchestratorAgent processing findings with enhanced capabilities...")
        
        try:
            # Extract findings từ state
            static_analysis_results = state.get('static_analysis_results', {})
            code_content = state.get('code_content', '')
            filename = state.get('filename', '<unknown>')
            
            if not static_analysis_results:
                self.logger.warning("No static analysis results found in state")
                return self._update_state_with_error(state, "No static analysis results available")
            
            # Get RAG context if enabled
            rag_context = None
            if self.enable_rag and self.rag_agent and code_content:
                rag_context = self._get_rag_context(code_content, filename, static_analysis_results)
            
            # Generate enhanced LLM analysis với RAG và Chain-of-Thought
            llm_analysis = self.analyze_findings_with_enhanced_llm(
                static_analysis_results, 
                code_content, 
                filename,
                rag_context
            )
            
            # Update state với enhanced LLM results
            updated_state = state.copy()
            updated_state['llm_analysis'] = llm_analysis
            updated_state['current_agent'] = 'llm_orchestrator'
            updated_state['processing_status'] = 'llm_analysis_completed'
            updated_state['rag_enabled'] = self.enable_rag
            updated_state['chain_of_thought_enabled'] = self.enable_chain_of_thought
            
            self.logger.info("Enhanced LLM analysis completed successfully")
            return updated_state
            
        except Exception as e:
            self.logger.error(f"Error in LLMOrchestratorAgent: {e}")
            return self._update_state_with_error(state, str(e))
    
    def _get_rag_context(self, 
                        code_content: str, 
                        filename: str, 
                        static_results: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Get relevant context từ RAG system
        
        Args:
            code_content: Source code
            filename: File name
            static_results: Static analysis results
            
        Returns:
            RAG context dictionary hoặc None
        """
        try:
            # Index current code if not already indexed
            self.rag_agent.index_code_file(code_content, filename)
            
            # Create query based on static analysis findings
            query_text = self._create_rag_query(static_results, filename)
            
            # Query RAG system
            rag_results = self.rag_agent.query_with_context(
                query_text=query_text,
                top_k=5,
                generate_response=False  # We'll generate our own response
            )
            
            return {
                'query': query_text,
                'relevant_chunks': rag_results.get('chunks', []),
                'context_summary': rag_results.get('summary', ''),
                'metadata': rag_results.get('metadata', {})
            }
            
        except Exception as e:
            self.logger.warning(f"Failed to get RAG context: {e}")
            return None
    
    def _create_rag_query(self, static_results: Dict[str, Any], filename: str) -> str:
        """Create query cho RAG system dựa trên static analysis findings"""
        issues = static_results.get('static_issues', {})
        
        # Extract key issues for query
        query_parts = [f"code analysis for {filename}"]
        
        for issue_type, issue_list in issues.items():
            if issue_list:
                query_parts.append(f"{issue_type.replace('_', ' ')}")
        
        return " ".join(query_parts)
    
    def analyze_findings_with_enhanced_llm(self, 
                                          static_results: Dict[str, Any], 
                                          code_content: str = "",
                                          filename: str = "<unknown>",
                                          rag_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Enhanced analysis với RAG context và Chain-of-Thought prompting
        
        Args:
            static_results: Results từ StaticAnalysisAgent
            code_content: Original code content
            filename: Filename for context
            rag_context: RAG context từ vector database
            
        Returns:
            Dict chứa enhanced LLM analysis results
        """
        analysis_result = {
            'filename': filename,
            'summary': '',
            'detailed_analysis': '',
            'priority_issues': [],
            'recommendations': [],
            'code_quality_assessment': '',
            'improvement_suggestions': [],
            'solution_suggestions': [],  # New: Chain-of-Thought solutions
            'rag_context_used': rag_context is not None,
            'llm_metadata': {
                'provider': self.llm_provider.__class__.__name__,
                'model_used': self.llm_provider.model,
                'analysis_type': 'enhanced_code_review_with_rag_and_cot',
                'rag_enabled': self.enable_rag,
                'chain_of_thought_enabled': self.enable_chain_of_thought
            }
        }
        
        try:
            # 1. Generate summary với RAG context
            summary_prompt = self._format_enhanced_summary_prompt(
                static_results, filename, rag_context
            )
            summary_response = self._generate_with_provider(
                summary_prompt,
                temperature=0.3,
                max_tokens=600
            )
            analysis_result['summary'] = summary_response.response
            
            # 2. Generate detailed analysis với Chain-of-Thought
            if code_content:
                detailed_prompt = self._format_chain_of_thought_analysis_prompt(
                    static_results, code_content, filename, rag_context
                )
                detailed_response = self._generate_with_provider(
                    detailed_prompt,
                    temperature=0.3,
                    max_tokens=1000
                )
                analysis_result['detailed_analysis'] = detailed_response.response
            
            # 3. Generate priority issues với enhanced context
            priority_prompt = self._format_enhanced_priority_issues_prompt(
                static_results, rag_context
            )
            priority_response = self._generate_with_provider(
                priority_prompt,
                temperature=0.2,
                max_tokens=500
            )
            analysis_result['priority_issues'] = self._parse_priority_issues(
                priority_response.response
            )
            
            # 4. Generate solution suggestions với Chain-of-Thought
            if self.enable_chain_of_thought:
                solution_prompt = self._format_chain_of_thought_solution_prompt(
                    static_results, code_content, rag_context
                )
                solution_response = self._generate_with_provider(
                    solution_prompt,
                    temperature=0.4,
                    max_tokens=800
                )
                analysis_result['solution_suggestions'] = self._parse_solution_suggestions(
                    solution_response.response
                )
            
            # 5. Generate recommendations với RAG context
            recommendations_prompt = self._format_enhanced_recommendations_prompt(
                static_results, rag_context
            )
            recommendations_response = self._generate_with_provider(
                recommendations_prompt,
                temperature=0.4,
                max_tokens=700
            )
            analysis_result['recommendations'] = self._parse_recommendations(
                recommendations_response.response
            )
            
            # 6. Generate quality assessment
            quality_prompt = self._format_enhanced_quality_assessment_prompt(
                static_results, rag_context
            )
            quality_response = self._generate_with_provider(
                quality_prompt,
                temperature=0.3,
                max_tokens=400
            )
            analysis_result['code_quality_assessment'] = quality_response.response
            
            # 7. Generate improvement suggestions
            improvement_prompt = self._format_enhanced_improvement_suggestions_prompt(
                static_results, rag_context
            )
            improvement_response = self._generate_with_provider(
                improvement_prompt,
                temperature=0.5,
                max_tokens=600
            )
            analysis_result['improvement_suggestions'] = self._parse_improvement_suggestions(
                improvement_response.response
            )
            
            self.logger.info("Enhanced LLM analysis completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error in enhanced LLM analysis: {e}")
            analysis_result['error'] = f"Enhanced analysis error: {str(e)}"
        
        return analysis_result
    
    def _generate_with_provider(self, 
                               prompt: str,
                               system_prompt: Optional[str] = None,
                               **kwargs) -> LLMResponse:
        """Generate response using configured LLM provider"""
        return self.llm_provider.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            **kwargs
        )
    
    def _format_enhanced_summary_prompt(self, 
                                       static_results: Dict[str, Any], 
                                       filename: str,
                                       rag_context: Optional[Dict[str, Any]]) -> str:
        """Format enhanced summary prompt với RAG context"""
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
        
        # Add RAG context if available
        rag_context_text = ""
        if rag_context and rag_context.get('relevant_chunks'):
            rag_context_text = f"""

Relevant Code Context từ codebase:
{chr(10).join([f"- {chunk.get('text', '')[:100]}..." for chunk in rag_context['relevant_chunks'][:3]])}"""
        
        prompt = f"""Bạn là một chuyên gia code review với kinh nghiệm sâu về software engineering best practices. 
Hãy phân tích kết quả static analysis sau và tạo summary ngắn gọn nhưng sâu sắc.

File: {filename}
Total Issues: {total_issues}

Issues Found:
{chr(10).join(issues_summary) if issues_summary else "- Không có issues được phát hiện"}

Code Quality Metrics:
- Quality Score: {metrics.get('code_quality_score', 'N/A')}/100
- Maintainability Index: {metrics.get('maintainability_index', 'N/A')}/100
- Cyclomatic Complexity: {metrics.get('cyclomatic_complexity', 'N/A')}
- Lines of Code: {metrics.get('lines_of_code', 'N/A')}

Existing Suggestions: {len(suggestions)} suggestions{rag_context_text}

Hãy tạo một summary ngắn gọn (3-4 câu) về:
1. Tình trạng tổng thể của code quality
2. Những điểm cần chú ý nhất
3. Mức độ ưu tiên của việc refactor
4. Khuyến nghị tổng quát"""
        
        return prompt
    
    def _format_chain_of_thought_analysis_prompt(self, 
                                                static_results: Dict[str, Any], 
                                                code_content: str, 
                                                filename: str,
                                                rag_context: Optional[Dict[str, Any]]) -> str:
        """Format Chain-of-Thought analysis prompt"""
        issues = static_results.get('static_issues', {})
        
        # Get most critical issues
        critical_issues = []
        for issue_type, issue_list in issues.items():
            for issue in issue_list[:3]:  # Top 3 issues per type
                critical_issues.append(f"- {issue_type}: {issue.get('message', 'Unknown issue')} (Line {issue.get('line', '?')})")
        
        # Add RAG context if available
        rag_context_text = ""
        if rag_context and rag_context.get('relevant_chunks'):
            rag_context_text = f"""

Related Code Patterns từ codebase:
{chr(10).join([f"- {chunk.get('text', '')[:150]}..." for chunk in rag_context['relevant_chunks'][:2]])}"""
        
        prompt = f"""Bạn là một senior software architect đang thực hiện deep code review. 
Sử dụng Chain-of-Thought reasoning để phân tích code một cách có hệ thống.

File: {filename}

Critical Issues Found:
{chr(10).join(critical_issues[:10]) if critical_issues else "- Không có critical issues"}

Code to Analyze:
```
{code_content[:2000]}{'...' if len(code_content) > 2000 else ''}
```{rag_context_text}

Hãy thực hiện Chain-of-Thought analysis theo các bước sau:

**Bước 1: Code Structure Analysis**
- Phân tích kiến trúc và organization của code
- Đánh giá design patterns được sử dụng
- Xác định coupling và cohesion

**Bước 2: Issue Impact Assessment**
- Phân tích từng issue về mức độ nghiêm trọng
- Đánh giá impact đến maintainability, performance, security
- Xác định root causes

**Bước 3: Risk Evaluation**
- Đánh giá rủi ro khi không fix các issues
- Xác định dependencies và side effects
- Ước tính effort để fix

**Bước 4: Solution Strategy**
- Đề xuất approach để giải quyết issues
- Ưu tiên thứ tự fix
- Đề xuất refactoring strategy

Hãy trình bày analysis theo format trên với reasoning rõ ràng cho mỗi bước."""
        
        return prompt
    
    def _format_enhanced_priority_issues_prompt(self, static_results: Dict[str, Any], 
                                               rag_context: Optional[Dict[str, Any]]) -> str:
        """Format enhanced priority issues prompt với RAG context"""
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
        
        # Add RAG context if available
        rag_context_text = ""
        if rag_context and rag_context.get('relevant_chunks'):
            rag_context_text = f"""

Similar Issues từ codebase:
{chr(10).join([f"- {chunk.get('text', '')[:120]}..." for chunk in rag_context['relevant_chunks'][:2]])}"""
        
        prompt = f"""Bạn là một tech lead với kinh nghiệm về software architecture đang prioritize issues để fix. 
Dựa trên danh sách issues sau và context từ codebase, hãy identify top 5 priority issues cần fix ngay.

Issues Found:
{chr(10).join(issues_text) if issues_text else "- Không có issues"}{rag_context_text}

Hãy list top 5 priority issues theo format:
1. [Issue Type] - [Brief Description] - [Why Priority] - [Impact Level]
2. [Issue Type] - [Brief Description] - [Why Priority] - [Impact Level]
...

Focus vào issues có impact cao nhất đến code quality, security, maintainability, và performance."""
        
        return prompt
    
    def _format_chain_of_thought_solution_prompt(self, static_results: Dict[str, Any], 
                                                code_content: str,
                                                rag_context: Optional[Dict[str, Any]]) -> str:
        """Format Chain-of-Thought solution prompt"""
        issues = static_results.get('static_issues', {})
        
        # Extract key issues for solution
        solution_parts = []
        for issue_type, issue_list in issues.items():
            for issue in issue_list[:3]:  # Top 3 issues per type
                solution_parts.append(f"- {issue_type}: {issue.get('message', 'Unknown issue')} (Line {issue.get('line', '?')})")
        
        prompt = f"""Bạn là một senior developer đang đề xuất solutions để fix issues được phát hiện.

Issues Found:
{chr(10).join(solution_parts) if solution_parts else "- Không có issues"}

Hãy đề xuất 5-7 specific solutions để fix issues được phát hiện.

Format: 
- [Solution] - [How to implement] - [Expected benefit]"""
        
        return prompt
    
    def _format_enhanced_recommendations_prompt(self, static_results: Dict[str, Any], 
                                               rag_context: Optional[Dict[str, Any]]) -> str:
        """Format enhanced recommendations prompt với RAG context"""
        metrics = static_results.get('metrics', {})
        suggestions = static_results.get('suggestions', [])
        
        # Add RAG context if available
        rag_context_text = ""
        if rag_context and rag_context.get('relevant_chunks'):
            rag_context_text = f"""

Best Practices từ codebase:
{chr(10).join([f"- {chunk.get('text', '')[:100]}..." for chunk in rag_context['relevant_chunks'][:3]])}"""
        
        prompt = f"""Bạn là một software architect với kinh nghiệm về enterprise software development đang đưa ra recommendations để improve codebase.

Current Metrics:
- Quality Score: {metrics.get('code_quality_score', 'N/A')}/100
- Maintainability Index: {metrics.get('maintainability_index', 'N/A')}/100
- Comment Ratio: {metrics.get('comment_ratio', 'N/A')}

Existing Suggestions:
{chr(10).join(f"- {s}" for s in suggestions[:5]) if suggestions else "- Không có suggestions"}{rag_context_text}

Hãy đưa ra 5-7 actionable recommendations để improve code quality, bao gồm:
1. Immediate actions (có thể làm ngay)
2. Short-term improvements (1-2 weeks)
3. Long-term architectural changes (nếu cần)

Format: 
- [Recommendation] - [Expected Impact] - [Effort Level: Low/Medium/High] - [Priority: High/Medium/Low]"""
        
        return prompt
    
    def _format_enhanced_quality_assessment_prompt(self, static_results: Dict[str, Any], 
                                                   rag_context: Optional[Dict[str, Any]]) -> str:
        """Format enhanced quality assessment prompt với RAG context"""
        metrics = static_results.get('metrics', {})
        issues = static_results.get('static_issues', {})
        
        total_issues = sum(len(issue_list) for issue_list in issues.values())
        
        # Add RAG context if available
        rag_context_text = ""
        if rag_context and rag_context.get('relevant_chunks'):
            rag_context_text = f"""

Quality Benchmarks từ codebase:
{chr(10).join([f"- {chunk.get('text', '')[:100]}..." for chunk in rag_context['relevant_chunks'][:2]])}"""
        
        prompt = f"""Bạn là một code quality expert với kinh nghiệm về software engineering standards. 
Hãy đánh giá overall quality của code dựa trên metrics sau và context từ codebase.

Quality Metrics:
- Overall Score: {metrics.get('code_quality_score', 'N/A')}/100
- Maintainability: {metrics.get('maintainability_index', 'N/A')}/100
- Complexity: {metrics.get('cyclomatic_complexity', 'N/A')}
- Total Issues: {total_issues}
- Lines of Code: {metrics.get('lines_of_code', 'N/A')}{rag_context_text}

Hãy đưa ra assessment ngắn gọn (4-5 câu) về:
1. Overall code quality level (Excellent/Good/Fair/Poor) với justification
2. Main strengths và competitive advantages
3. Key areas for improvement với specific recommendations
4. Readiness for production và risk assessment
5. Comparison với industry standards"""
        
        return prompt
    
    def _format_enhanced_improvement_suggestions_prompt(self, static_results: Dict[str, Any], 
                                                       rag_context: Optional[Dict[str, Any]]) -> str:
        """Format enhanced improvement suggestions prompt với RAG context"""
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
        
        # Add RAG context if available
        rag_context_text = ""
        if rag_context and rag_context.get('relevant_chunks'):
            rag_context_text = f"""

Improvement Patterns từ codebase:
{chr(10).join([f"- {chunk.get('text', '')[:120]}..." for chunk in rag_context['relevant_chunks'][:3]])}"""
        
        prompt = f"""Bạn là một senior developer mentor với kinh nghiệm về code refactoring và best practices. 
Dựa trên analysis results và patterns từ codebase, hãy suggest concrete improvement steps.

Main Problem Areas: {', '.join(problem_areas) if problem_areas else 'None identified'}

Current Quality Score: {metrics.get('code_quality_score', 'N/A')}/100{rag_context_text}

Hãy suggest 5-6 specific improvement actions theo format:
1. [Action] - [How to implement] - [Expected benefit] - [Timeline]
2. [Action] - [How to implement] - [Expected benefit] - [Timeline]
...

Focus vào actionable steps mà developer có thể implement ngay để improve code quality, 
với reference đến successful patterns từ codebase."""
        
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
    
    def _parse_solution_suggestions(self, llm_response: str) -> List[Dict[str, str]]:
        """Parse LLM response thành structured solution suggestions"""
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
                            'solution': parts[0].strip(),
                            'implementation': parts[1].strip() if len(parts) > 1 else '',
                            'benefit': parts[2].strip() if len(parts) > 2 else ''
                        })
        
        return suggestions
    
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
            return self.llm_provider.check_health()
        except Exception as e:
            self.logger.error(f"LLM health check failed: {e}")
            return False
    
    def get_available_models(self) -> List[str]:
        """Get list of available LLM models"""
        try:
            return self.llm_provider.list_models()
        except Exception as e:
            self.logger.error(f"Failed to get available models: {e}")
            return []


# Convenience functions for LangGraph integration
def create_llm_orchestrator_agent(provider: str = "ollama",
                                 model: str = "codellama",
                                 rag_context_agent: Optional[RAGContextAgent] = None,
                                 enable_rag: bool = True,
                                 enable_chain_of_thought: bool = True,
                                 **provider_kwargs) -> LLMOrchestratorAgent:
    """
    Factory function để tạo LLMOrchestratorAgent
    
    Args:
        provider: LLM provider (ollama, openai, gemini)
        model: Model name
        rag_context_agent: RAGContextAgent instance (optional)
        enable_rag: Enable RAG context retrieval
        enable_chain_of_thought: Enable Chain-of-Thought prompting
        **provider_kwargs: Additional provider-specific arguments
        
    Returns:
        LLMOrchestratorAgent instance
    """
    return LLMOrchestratorAgent(provider=provider, model=model, rag_context_agent=rag_context_agent, enable_rag=enable_rag, enable_chain_of_thought=enable_chain_of_thought, **provider_kwargs)


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
        
        print(f"🔗 LLM Provider: {agent.llm_provider.__class__.__name__}")
        print(f"🔗 LLM Model: {agent.llm_provider.model}")
        print(f"🤖 RAG Enabled: {agent.enable_rag}")
        print(f"🧠 Chain-of-Thought Enabled: {agent.enable_chain_of_thought}")
        
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
                    
                    if analysis.get('solution_suggestions'):
                        print(f"\n💡 Solution Suggestions ({len(analysis.get('solution_suggestions', []))}):")
                        for i, solution in enumerate(analysis.get('solution_suggestions', [])[:3], 1):
                            print(f"  {i}. {solution.get('solution', 'N/A')}")
                    
                    print(f"\n📊 Enhanced Features:")
                    print(f"  🤖 RAG Context Used: {analysis.get('rag_context_used', False)}")
                    print(f"  🧠 Chain-of-Thought: {analysis.get('llm_metadata', {}).get('chain_of_thought_enabled', False)}")
                    print(f"  🔗 Provider: {analysis.get('llm_metadata', {}).get('provider', 'Unknown')}")
            
            print("\n✅ Enhanced demo completed successfully!")
            
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