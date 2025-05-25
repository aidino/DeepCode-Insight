"""
Common utilities for DeepCode-Insight

Các utility functions chung được sử dụng bởi nhiều components
"""

import os
import re
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
from .interfaces import AnalysisLanguage


def detect_language_from_filename(filename: str) -> AnalysisLanguage:
    """Detect programming language from filename extension"""
    ext = Path(filename).suffix.lower()
    
    language_map = {
        '.py': AnalysisLanguage.PYTHON,
        '.java': AnalysisLanguage.JAVA,
        '.js': AnalysisLanguage.JAVASCRIPT,
        '.jsx': AnalysisLanguage.JAVASCRIPT,
        '.ts': AnalysisLanguage.TYPESCRIPT,
        '.tsx': AnalysisLanguage.TYPESCRIPT,
    }
    
    return language_map.get(ext, AnalysisLanguage.UNKNOWN)


def is_valid_code_file(filename: str, max_size_mb: int = 10) -> bool:
    """Check if file is a valid code file for analysis"""
    if not os.path.exists(filename):
        return False
    
    # Check file size
    file_size_mb = os.path.getsize(filename) / (1024 * 1024)
    if file_size_mb > max_size_mb:
        return False
    
    # Check if it's a supported language
    language = detect_language_from_filename(filename)
    return language != AnalysisLanguage.UNKNOWN


def normalize_line_endings(code: str) -> str:
    """Normalize line endings to Unix style"""
    return code.replace('\r\n', '\n').replace('\r', '\n')


def extract_line_from_code(code: str, line_number: int) -> str:
    """Extract a specific line from code (1-indexed)"""
    lines = code.split('\n')
    if 1 <= line_number <= len(lines):
        return lines[line_number - 1]
    return ""


def get_code_context(code: str, line_number: int, context_lines: int = 3) -> Dict[str, Any]:
    """Get code context around a specific line"""
    lines = code.split('\n')
    total_lines = len(lines)
    
    start_line = max(1, line_number - context_lines)
    end_line = min(total_lines, line_number + context_lines)
    
    context = {
        'target_line': line_number,
        'start_line': start_line,
        'end_line': end_line,
        'lines': []
    }
    
    for i in range(start_line, end_line + 1):
        if i <= total_lines:
            context['lines'].append({
                'number': i,
                'content': lines[i - 1],
                'is_target': i == line_number
            })
    
    return context


def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe usage"""
    # Remove or replace invalid characters
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Remove leading/trailing whitespace and dots
    sanitized = sanitized.strip(' .')
    # Ensure it's not empty
    if not sanitized:
        sanitized = "unnamed_file"
    return sanitized


def calculate_complexity_score(metrics: Dict[str, Any]) -> float:
    """Calculate a normalized complexity score from various metrics"""
    score = 0.0
    
    # Cyclomatic complexity (weight: 0.4)
    if 'cyclomatic_complexity' in metrics:
        cc = metrics['cyclomatic_complexity']
        # Normalize: 1-10 = good, 11-20 = moderate, 21+ = high
        cc_score = min(cc / 20.0, 1.0)
        score += cc_score * 0.4
    
    # Lines of code (weight: 0.2)
    if 'lines_of_code' in metrics:
        loc = metrics['lines_of_code']
        # Normalize: 1-100 = good, 101-300 = moderate, 301+ = high
        loc_score = min(loc / 300.0, 1.0)
        score += loc_score * 0.2
    
    # Number of parameters (weight: 0.2)
    if 'parameter_count' in metrics:
        params = metrics['parameter_count']
        # Normalize: 1-5 = good, 6-10 = moderate, 11+ = high
        param_score = min(params / 10.0, 1.0)
        score += param_score * 0.2
    
    # Nesting depth (weight: 0.2)
    if 'nesting_depth' in metrics:
        depth = metrics['nesting_depth']
        # Normalize: 1-3 = good, 4-6 = moderate, 7+ = high
        depth_score = min(depth / 6.0, 1.0)
        score += depth_score * 0.2
    
    return min(score, 1.0)


def format_issue_message(issue_type: str, details: Dict[str, Any]) -> str:
    """Format a standardized issue message"""
    templates = {
        'missing_docstring': "Missing docstring for {element_type} '{name}'",
        'unused_import': "Unused import: {import_name}",
        'complex_function': "Function '{name}' has high complexity (CC: {complexity})",
        'naming_violation': "Naming violation: {element_type} '{name}' should follow {convention}",
        'code_smell': "{smell_type}: {description}",
        'security_issue': "Security issue: {description}",
        'performance_issue': "Performance issue: {description}",
    }
    
    template = templates.get(issue_type, "{description}")
    
    try:
        return template.format(**details)
    except KeyError as e:
        return f"{issue_type}: Missing template parameter {e}"


def merge_analysis_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge multiple analysis results into a single result"""
    if not results:
        return {}
    
    merged = {
        'files_analyzed': len(results),
        'total_issues': 0,
        'issues_by_type': {},
        'issues_by_severity': {},
        'overall_metrics': {},
        'all_issues': [],
        'all_suggestions': []
    }
    
    for result in results:
        if 'static_issues' in result:
            for issue_type, issues in result['static_issues'].items():
                if isinstance(issues, list):
                    merged['total_issues'] += len(issues)
                    merged['issues_by_type'][issue_type] = merged['issues_by_type'].get(issue_type, 0) + len(issues)
                    merged['all_issues'].extend(issues)
        
        if 'suggestions' in result:
            merged['all_suggestions'].extend(result['suggestions'])
    
    # Calculate severity distribution
    for issue in merged['all_issues']:
        severity = issue.get('severity', 'warning')
        merged['issues_by_severity'][severity] = merged['issues_by_severity'].get(severity, 0) + 1
    
    return merged


def setup_logging(level: str = "INFO", format_string: Optional[str] = None) -> logging.Logger:
    """Setup standardized logging configuration"""
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_string,
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    return logging.getLogger("deepcode_insight")


def validate_state_schema(state: Dict[str, Any], required_fields: List[str]) -> Tuple[bool, List[str]]:
    """Validate that state contains required fields"""
    missing_fields = []
    
    for field in required_fields:
        if field not in state:
            missing_fields.append(field)
    
    return len(missing_fields) == 0, missing_fields


def safe_get_nested(data: Dict[str, Any], path: str, default: Any = None) -> Any:
    """Safely get nested dictionary value using dot notation"""
    keys = path.split('.')
    current = data
    
    try:
        for key in keys:
            current = current[key]
        return current
    except (KeyError, TypeError):
        return default


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """Truncate text to specified length"""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def is_test_file(filename: str) -> bool:
    """Check if file is a test file"""
    filename_lower = filename.lower()
    test_patterns = [
        'test_',
        '_test.',
        'tests/',
        '/test/',
        'spec_',
        '_spec.',
        'specs/',
        '/spec/'
    ]
    
    return any(pattern in filename_lower for pattern in test_patterns)


def get_file_stats(code: str) -> Dict[str, int]:
    """Get basic file statistics"""
    lines = code.split('\n')
    
    return {
        'total_lines': len(lines),
        'non_empty_lines': len([line for line in lines if line.strip()]),
        'comment_lines': len([line for line in lines if line.strip().startswith('#')]),
        'blank_lines': len([line for line in lines if not line.strip()]),
        'characters': len(code),
        'words': len(code.split())
    } 