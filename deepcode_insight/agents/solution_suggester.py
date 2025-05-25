"""
Solution Suggestion Agent - Refines raw LLM solutions thành actionable suggestions
"""

import logging
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass

from ..utils.llm_interface import create_llm_provider, LLMResponse, LLMProvider

logger = logging.getLogger(__name__)


@dataclass
class RefinedSolution:
    """Structured representation của refined solution"""
    original_solution: str
    refined_title: str
    description: str
    implementation_steps: List[str]
    prerequisites: List[str]
    estimated_effort: str
    impact_level: str
    risk_assessment: str
    code_examples: List[str]
    related_patterns: List[str]
    success_metrics: List[str]
    confidence_score: float


class SolutionSuggestionAgent:
    """
    Agent để refine raw LLM solutions thành actionable, detailed suggestions
    """
    
    def __init__(self, 
                 provider: str = "ollama",
                 model: str = "codellama",
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None,
                 temperature: float = 0.3,
                 max_tokens: int = 2000,
                 **kwargs):
        """
        Initialize SolutionSuggestionAgent
        
        Args:
            provider: LLM provider (ollama, openai, gemini)
            model: Model name
            api_key: API key for external providers
            base_url: Base URL for local providers
            temperature: Generation temperature (lower = more focused)
            max_tokens: Maximum tokens per response
            **kwargs: Additional provider-specific parameters
        """
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        try:
            self.llm_provider = create_llm_provider(
                provider=provider,
                model=model,
                api_key=api_key,
                base_url=base_url,
                **kwargs
            )
            logger.info(f"SolutionSuggestionAgent initialized với {provider}:{model}")
        except Exception as e:
            logger.error(f"Failed to initialize LLM provider: {e}")
            raise
    
    def refine_solutions(self, 
                        raw_solutions: List[Dict[str, Any]], 
                        code_context: str,
                        filename: str,
                        static_analysis_context: Optional[Dict[str, Any]] = None) -> List[RefinedSolution]:
        """
        Refine raw solutions thành detailed, actionable suggestions
        
        Args:
            raw_solutions: List of raw solution dictionaries
            code_context: Original code content
            filename: File being analyzed
            static_analysis_context: Additional context from static analysis
            
        Returns:
            List of RefinedSolution objects
        """
        refined_solutions = []
        
        for i, raw_solution in enumerate(raw_solutions):
            try:
                logger.debug(f"Refining solution {i+1}/{len(raw_solutions)}")
                
                refined = self._refine_single_solution(
                    raw_solution, 
                    code_context, 
                    filename,
                    static_analysis_context
                )
                
                if refined:
                    refined_solutions.append(refined)
                    
            except Exception as e:
                logger.error(f"Error refining solution {i+1}: {e}")
                # Continue với solutions khác
                continue
        
        logger.info(f"Successfully refined {len(refined_solutions)}/{len(raw_solutions)} solutions")
        return refined_solutions
    
    def _refine_single_solution(self, 
                               raw_solution: Dict[str, Any],
                               code_context: str,
                               filename: str,
                               static_analysis_context: Optional[Dict[str, Any]] = None) -> Optional[RefinedSolution]:
        """
        Refine một single solution
        
        Args:
            raw_solution: Raw solution dictionary
            code_context: Code content
            filename: File name
            static_analysis_context: Static analysis context
            
        Returns:
            RefinedSolution object hoặc None nếu refinement fails
        """
        try:
            # Extract raw solution text
            solution_text = self._extract_solution_text(raw_solution)
            if not solution_text or solution_text.strip() == "":
                logger.warning("Empty solution text, skipping refinement")
                return None
            
            # Create refinement prompt
            prompt = self._create_refinement_prompt(
                solution_text, 
                code_context, 
                filename,
                static_analysis_context
            )
            
            # Generate refined solution
            response = self.llm_provider.generate(
                prompt=prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            # Parse response thành structured format
            refined = self._parse_refined_solution(response.response, solution_text)
            
            return refined
            
        except Exception as e:
            logger.error(f"Error in _refine_single_solution: {e}")
            return None
    
    def _extract_solution_text(self, raw_solution: Dict[str, Any]) -> str:
        """Extract solution text từ raw solution dictionary"""
        
        # Handle string input directly
        if isinstance(raw_solution, str):
            return raw_solution.strip()
        
        # Handle dict input
        if isinstance(raw_solution, dict):
            # Check if dict is empty
            if not raw_solution:
                return ""
            
            # Try different possible keys
            possible_keys = ['solution', 'suggestion', 'recommendation', 'text', 'description']
            
            for key in possible_keys:
                if key in raw_solution and raw_solution[key]:
                    return str(raw_solution[key]).strip()
            
            # If no standard key found, try to concatenate available info
            parts = []
            for key, value in raw_solution.items():
                if value and isinstance(value, str):
                    parts.append(f"{key}: {value}")
            
            if parts:
                return " | ".join(parts)
        
        # Last resort: convert to string
        return str(raw_solution).strip()
    
    def _create_refinement_prompt(self, 
                                 solution_text: str,
                                 code_context: str,
                                 filename: str,
                                 static_analysis_context: Optional[Dict[str, Any]] = None) -> str:
        """
        Create detailed refinement prompt
        """
        
        # Build context information
        context_info = f"File: {filename}\n"
        
        if static_analysis_context:
            if 'metrics' in static_analysis_context:
                metrics = static_analysis_context['metrics']
                context_info += f"Code Quality Score: {metrics.get('code_quality_score', 'N/A')}\n"
                context_info += f"Complexity: {metrics.get('cyclomatic_complexity', 'N/A')}\n"
            
            if 'static_issues' in static_analysis_context:
                issue_count = sum(len(issues) for issues in static_analysis_context['static_issues'].values())
                context_info += f"Total Issues Found: {issue_count}\n"
        else:
            context_info += f"Code Quality Score: N/A\n"
            context_info += f"Complexity: N/A\n"
        
        # Truncate code context if too long
        code_preview = code_context[:500] + "..." if len(code_context) > 500 else code_context
        
        prompt = f"""Bạn là một senior software engineer và technical lead với expertise trong code refactoring và best practices. 

Nhiệm vụ của bạn là refine một raw solution suggestion thành một detailed, actionable recommendation.

**Context Information:**
{context_info}

**Code Preview:**
```
{code_preview}
```

**Raw Solution to Refine:**
{solution_text}

**Instructions:**
Hãy refine solution này thành một comprehensive, actionable recommendation. Provide detailed analysis theo format sau:

**REFINED_TITLE:** [Clear, specific title cho solution]

**DESCRIPTION:** [Detailed explanation của problem và tại sao solution này effective]

**IMPLEMENTATION_STEPS:**
1. [Specific step 1 với technical details]
2. [Specific step 2 với technical details]
3. [Continue với all necessary steps]

**PREREQUISITES:**
- [Requirement 1: tools, knowledge, dependencies needed]
- [Requirement 2: ...]

**ESTIMATED_EFFORT:** [Time estimate: hours/days/weeks với justification]

**IMPACT_LEVEL:** [High/Medium/Low với explanation]

**RISK_ASSESSMENT:** [Potential risks và mitigation strategies]

**CODE_EXAMPLES:**
```python
# Example 1: Before (problematic code)
[Show current problematic pattern]

# Example 2: After (improved code)
[Show how code should look after implementing solution]
```

**RELATED_PATTERNS:** [Design patterns, principles, hoặc best practices liên quan]

**SUCCESS_METRICS:** [How to measure if solution was successful]

**CONFIDENCE_SCORE:** [0.0-1.0 score indicating confidence in this solution]

Hãy focus vào making the solution:
1. **Specific và actionable** - clear steps developer có thể follow
2. **Technically accurate** - correct implementation details
3. **Contextually relevant** - appropriate cho codebase và situation
4. **Risk-aware** - acknowledge potential issues và mitigation
5. **Measurable** - clear success criteria

Respond in Vietnamese với technical terms in English khi appropriate."""

        return prompt
    
    def _parse_refined_solution(self, response: str, original_solution: str) -> Optional[RefinedSolution]:
        """
        Parse LLM response thành RefinedSolution object
        """
        try:
            # Initialize default values
            refined_title = "Improved Solution"
            description = ""
            implementation_steps = []
            prerequisites = []
            estimated_effort = "Unknown"
            impact_level = "Medium"
            risk_assessment = ""
            code_examples = []
            related_patterns = []
            success_metrics = []
            confidence_score = 0.5
            
            # Parse sections từ response
            sections = self._split_response_into_sections(response)
            
            # Extract each section
            refined_title = sections.get('REFINED_TITLE', refined_title).strip()
            description = sections.get('DESCRIPTION', description).strip()
            
            # Parse implementation steps
            impl_text = sections.get('IMPLEMENTATION_STEPS', '')
            implementation_steps = self._parse_numbered_list(impl_text)
            
            # Parse prerequisites
            prereq_text = sections.get('PREREQUISITES', '')
            prerequisites = self._parse_bullet_list(prereq_text)
            
            # Extract simple fields
            estimated_effort = sections.get('ESTIMATED_EFFORT', estimated_effort).strip()
            impact_level = sections.get('IMPACT_LEVEL', impact_level).strip()
            risk_assessment = sections.get('RISK_ASSESSMENT', risk_assessment).strip()
            
            # Parse code examples
            code_examples_text = sections.get('CODE_EXAMPLES', '')
            code_examples = self._extract_code_blocks(code_examples_text)
            
            # Parse related patterns
            patterns_text = sections.get('RELATED_PATTERNS', '')
            related_patterns = self._parse_comma_separated(patterns_text)
            
            # Parse success metrics
            metrics_text = sections.get('SUCCESS_METRICS', '')
            success_metrics = self._parse_bullet_list(metrics_text)
            
            # Parse confidence score
            confidence_text = sections.get('CONFIDENCE_SCORE', '0.5')
            confidence_score = self._parse_confidence_score(confidence_text)
            
            return RefinedSolution(
                original_solution=original_solution,
                refined_title=refined_title,
                description=description,
                implementation_steps=implementation_steps,
                prerequisites=prerequisites,
                estimated_effort=estimated_effort,
                impact_level=impact_level,
                risk_assessment=risk_assessment,
                code_examples=code_examples,
                related_patterns=related_patterns,
                success_metrics=success_metrics,
                confidence_score=confidence_score
            )
            
        except Exception as e:
            logger.error(f"Error parsing refined solution: {e}")
            return None
    
    def _split_response_into_sections(self, response: str) -> Dict[str, str]:
        """Split response thành sections based on headers"""
        sections = {}
        current_section = None
        current_content = []
        
        lines = response.split('\n')
        
        for line in lines:
            # Check if line is a section header (e.g., **REFINED_TITLE:** or **DESCRIPTION:**)
            if line.strip().startswith('**') and ':**' in line.strip():
                # Save previous section
                if current_section:
                    sections[current_section] = '\n'.join(current_content).strip()
                
                # Start new section - extract section name between ** and :**
                section_line = line.strip()
                start_idx = section_line.find('**') + 2
                end_idx = section_line.find(':**')
                if start_idx < end_idx:
                    current_section = section_line[start_idx:end_idx].strip()
                    current_content = []
                    
                    # If there's content after the header on the same line, include it
                    remaining_content = section_line[end_idx + 3:].strip()
                    if remaining_content:
                        current_content.append(remaining_content)
            else:
                # Add to current section content
                if current_section:
                    current_content.append(line)
        
        # Save last section
        if current_section:
            sections[current_section] = '\n'.join(current_content).strip()
        
        return sections
    
    def _parse_numbered_list(self, text: str) -> List[str]:
        """Parse numbered list từ text"""
        items = []
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            # Match patterns like "1. ", "2. ", etc.
            if line and (line[0].isdigit() or line.startswith('-') or line.startswith('•')):
                # Remove numbering/bullets
                clean_line = line
                if line[0].isdigit():
                    clean_line = line.split('.', 1)[1].strip() if '.' in line else line
                elif line.startswith('-') or line.startswith('•'):
                    clean_line = line[1:].strip()
                
                if clean_line:
                    items.append(clean_line)
        
        return items
    
    def _parse_bullet_list(self, text: str) -> List[str]:
        """Parse bullet list từ text"""
        items = []
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if line and (line.startswith('-') or line.startswith('•') or line.startswith('*')):
                clean_line = line[1:].strip()
                if clean_line:
                    items.append(clean_line)
        
        return items
    
    def _parse_comma_separated(self, text: str) -> List[str]:
        """Parse comma-separated values"""
        if not text.strip():
            return []
        
        items = [item.strip() for item in text.split(',')]
        return [item for item in items if item]
    
    def _extract_code_blocks(self, text: str) -> List[str]:
        """Extract code blocks từ text"""
        code_blocks = []
        lines = text.split('\n')
        in_code_block = False
        current_block = []
        
        for line in lines:
            if line.strip().startswith('```'):
                if in_code_block:
                    # End of code block
                    if current_block:
                        code_blocks.append('\n'.join(current_block))
                    current_block = []
                    in_code_block = False
                else:
                    # Start of code block
                    in_code_block = True
            elif in_code_block:
                current_block.append(line)
        
        # Handle case where code block wasn't properly closed
        if current_block:
            code_blocks.append('\n'.join(current_block))
        
        return code_blocks
    
    def _parse_confidence_score(self, text: str) -> float:
        """Parse confidence score từ text"""
        try:
            # Extract number từ text
            import re
            numbers = re.findall(r'[0-9]*\.?[0-9]+', text)
            if numbers:
                score = float(numbers[0])
                # Ensure score is between 0 and 1
                if score > 1.0:
                    score = score / 100.0  # Convert percentage
                return max(0.0, min(1.0, score))
        except:
            pass
        
        return 0.5  # Default confidence
    
    def process_solutions(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        LangGraph node function để process solutions
        
        Args:
            state: LangGraph state containing solution data
            
        Returns:
            Updated state với refined solutions
        """
        try:
            logger.info("SolutionSuggestionAgent processing solutions...")
            
            # Extract required data từ state
            raw_solutions = state.get('llm_analysis', {}).get('solution_suggestions', [])
            code_content = state.get('code_content', '')
            filename = state.get('filename', 'unknown.py')
            static_analysis_results = state.get('static_analysis_results', {})
            
            if not raw_solutions:
                logger.warning("No raw solutions found in state")
                return {
                    **state,
                    'refined_solutions': [],
                    'current_agent': 'solution_suggester',
                    'processing_status': 'no_solutions_to_refine'
                }
            
            # Refine solutions
            refined_solutions = self.refine_solutions(
                raw_solutions=raw_solutions,
                code_context=code_content,
                filename=filename,
                static_analysis_context=static_analysis_results
            )
            
            # Convert to serializable format
            refined_solutions_data = [
                {
                    'original_solution': sol.original_solution,
                    'refined_title': sol.refined_title,
                    'description': sol.description,
                    'implementation_steps': sol.implementation_steps,
                    'prerequisites': sol.prerequisites,
                    'estimated_effort': sol.estimated_effort,
                    'impact_level': sol.impact_level,
                    'risk_assessment': sol.risk_assessment,
                    'code_examples': sol.code_examples,
                    'related_patterns': sol.related_patterns,
                    'success_metrics': sol.success_metrics,
                    'confidence_score': sol.confidence_score
                }
                for sol in refined_solutions
            ]
            
            logger.info(f"Successfully refined {len(refined_solutions)} solutions")
            
            return {
                **state,
                'refined_solutions': refined_solutions_data,
                'current_agent': 'solution_suggester',
                'processing_status': 'solution_refinement_completed',
                'refinement_metadata': {
                    'total_raw_solutions': len(raw_solutions),
                    'successfully_refined': len(refined_solutions),
                    'refinement_success_rate': len(refined_solutions) / len(raw_solutions) if raw_solutions else 0,
                    'provider': self.provider,
                    'model': self.model
                }
            }
            
        except Exception as e:
            logger.error(f"Error in SolutionSuggestionAgent.process_solutions: {e}")
            return {
                **state,
                'refined_solutions': [],
                'current_agent': 'solution_suggester',
                'processing_status': 'solution_refinement_failed',
                'error': str(e)
            }
    
    def check_health(self) -> bool:
        """Check if agent is healthy và ready to process"""
        try:
            return self.llm_provider.check_health()
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    def get_available_models(self) -> List[str]:
        """Get list of available models"""
        try:
            return self.llm_provider.list_models()
        except Exception as e:
            logger.error(f"Error getting available models: {e}")
            return []


def create_solution_suggester_agent(provider: str = "ollama",
                                   model: str = "codellama",
                                   **kwargs) -> SolutionSuggestionAgent:
    """
    Factory function để create SolutionSuggestionAgent
    
    Args:
        provider: LLM provider
        model: Model name
        **kwargs: Additional parameters
        
    Returns:
        SolutionSuggestionAgent instance
    """
    return SolutionSuggestionAgent(
        provider=provider,
        model=model,
        **kwargs
    )


def solution_suggester_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node function cho solution suggestion refinement
    
    Args:
        state: Current graph state
        
    Returns:
        Updated state với refined solutions
    """
    agent = create_solution_suggester_agent()
    return agent.process_solutions(state) 