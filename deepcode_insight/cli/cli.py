#!/usr/bin/env python3
"""
CLI cho DeepCode-Insight Analysis Tool
Tích hợp với complete LangGraph workflow
"""

import click
import os
import sys
from typing import Optional

# Add project root to path
sys.path.append(os.path.dirname(__file__))

from ..core.graph import create_analysis_workflow
from ..core.state import AgentState, DEFAULT_AGENT_STATE


@click.group()
@click.version_option(version="1.0.0")
def cli():
    """
    🚀 DeepCode-Insight - AI-Powered Code Analysis Tool
    
    Phân tích code với static analysis và LLM insights để tạo báo cáo chi tiết.
    """
    pass


@cli.command()
@click.option('--repo-url', '-r', help='Repository URL để analyze')
@click.option('--pr-id', '-p', type=int, help='Pull Request ID để analyze')
@click.option('--target-file', '-f', help='Specific file path để analyze')
@click.option('--output-dir', '-o', default='analysis_reports', help='Output directory cho reports')
@click.option('--config', '-c', help='Configuration file path')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
def analyze(repo_url: Optional[str], 
           pr_id: Optional[int], 
           target_file: Optional[str],
           output_dir: str,
           config: Optional[str],
           verbose: bool):
    """
    Analyze repository hoặc specific file với complete workflow
    
    Examples:
    
    \b
    # Analyze repository
    deepcode-insight analyze --repo-url https://github.com/user/repo
    
    \b
    # Analyze specific file in repository
    deepcode-insight analyze --repo-url https://github.com/user/repo --target-file src/main.py
    
    \b
    # Analyze Pull Request
    deepcode-insight analyze --repo-url https://github.com/user/repo --pr-id 123
    """
    
    if verbose:
        click.echo("🔧 Verbose mode enabled")
    
    # Validate inputs
    if not repo_url:
        click.echo("❌ Error: Repository URL is required", err=True)
        click.echo("Use --repo-url to specify repository", err=True)
        sys.exit(1)
    
    click.echo(f"🚀 Starting DeepCode-Insight Analysis")
    click.echo(f"📂 Repository: {repo_url}")
    
    if pr_id:
        click.echo(f"🔀 Pull Request: #{pr_id}")
    
    if target_file:
        click.echo(f"📄 Target file: {target_file}")
    
    click.echo(f"📁 Output directory: {output_dir}")
    
    try:
        # Create workflow
        if verbose:
            click.echo("🔧 Creating analysis workflow...")
        
        graph = create_analysis_workflow()
        
        # Prepare initial state
        initial_state: AgentState = {
            **DEFAULT_AGENT_STATE,
            'repo_url': repo_url,
            'pr_id': str(pr_id) if pr_id else None,
            'target_file': target_file,
            'config': {
                'output_dir': output_dir,
                'verbose': verbose,
                'cli_mode': True
            }
        }
        
        if verbose:
            click.echo("🔄 Running analysis workflow...")
        
        # Run workflow
        with click.progressbar(length=5, label='Analyzing') as bar:
            # Simulate progress (in real implementation, you'd hook into workflow events)
            result = graph.invoke(initial_state)
            bar.update(5)
        
        # Process results
        if result.get('processing_status') == 'report_generated':
            click.echo("✅ Analysis completed successfully!")
            
            # Show summary
            if result.get('static_analysis_results'):
                static_results = result['static_analysis_results']
                issues_count = sum(len(issues) for issues in static_results.get('static_issues', {}).values())
                quality_score = static_results.get('metrics', {}).get('code_quality_score', 'N/A')
                
                click.echo(f"🔍 Static Analysis: {issues_count} issues found")
                click.echo(f"📊 Code Quality Score: {quality_score}")
            
            if result.get('llm_analysis'):
                click.echo("🤖 LLM Analysis: Completed")
                if verbose:
                    summary = result['llm_analysis'].get('summary', '')
                    if summary:
                        click.echo(f"📝 Summary: {summary[:100]}...")
            
            if result.get('report'):
                report_path = result['report']['output_path']
                click.echo(f"📄 Report generated: {report_path}")
                
                # Open report option
                if click.confirm("🔍 Would you like to view the report?"):
                    try:
                        if sys.platform == "darwin":  # macOS
                            os.system(f"open '{report_path}'")
                        elif sys.platform == "linux":
                            os.system(f"xdg-open '{report_path}'")
                        elif sys.platform == "win32":
                            os.system(f"start '{report_path}'")
                        else:
                            click.echo(f"📖 Please open: {report_path}")
                    except Exception as e:
                        click.echo(f"⚠️ Could not open report automatically: {e}")
                        click.echo(f"📖 Please open manually: {report_path}")
        
        elif result.get('processing_status') == 'error':
            click.echo(f"❌ Analysis failed: {result.get('error', 'Unknown error')}", err=True)
            sys.exit(1)
        
        else:
            click.echo(f"⚠️ Analysis completed with status: {result.get('processing_status', 'Unknown')}")
            if result.get('error'):
                click.echo(f"⚠️ Warning: {result['error']}")
    
    except Exception as e:
        click.echo(f"❌ Unexpected error: {e}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


@cli.command()
@click.argument('file_path', type=click.Path(exists=True))
@click.option('--output-dir', '-o', default='analysis_reports', help='Output directory cho reports')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
def analyze_file(file_path: str, output_dir: str, verbose: bool):
    """
    Analyze local file với complete workflow
    
    Examples:
    
    \b
    # Analyze local Python file
    deepcode-insight analyze-file src/main.py
    
    \b
    # Analyze với custom output directory
    deepcode-insight analyze-file src/main.py --output-dir my_reports
    """
    
    if verbose:
        click.echo("🔧 Verbose mode enabled")
    
    click.echo(f"🚀 Starting DeepCode-Insight File Analysis")
    click.echo(f"📄 File: {file_path}")
    click.echo(f"📁 Output directory: {output_dir}")
    
    try:
        # Read file content
        if verbose:
            click.echo("📖 Reading file content...")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            code_content = f.read()
        
        if not code_content.strip():
            click.echo("⚠️ Warning: File is empty", err=True)
        
        # Create workflow
        if verbose:
            click.echo("🔧 Creating analysis workflow...")
        
        graph = create_analysis_workflow()
        
        # Prepare initial state
        initial_state: AgentState = {
            **DEFAULT_AGENT_STATE,
            'code_content': code_content,
            'filename': os.path.basename(file_path),
            'config': {
                'output_dir': output_dir,
                'verbose': verbose,
                'cli_mode': True,
                'local_file': True
            }
        }
        
        if verbose:
            click.echo("🔄 Running analysis workflow...")
        
        # Run workflow
        with click.progressbar(length=4, label='Analyzing') as bar:
            result = graph.invoke(initial_state)
            bar.update(4)
        
        # Process results (same as analyze command)
        if result.get('processing_status') == 'report_generated':
            click.echo("✅ Analysis completed successfully!")
            
            # Show summary
            if result.get('static_analysis_results'):
                static_results = result['static_analysis_results']
                issues_count = sum(len(issues) for issues in static_results.get('static_issues', {}).values())
                quality_score = static_results.get('metrics', {}).get('code_quality_score', 'N/A')
                
                click.echo(f"🔍 Static Analysis: {issues_count} issues found")
                click.echo(f"📊 Code Quality Score: {quality_score}")
            
            if result.get('llm_analysis'):
                click.echo("🤖 LLM Analysis: Completed")
            
            if result.get('report'):
                report_path = result['report']['output_path']
                click.echo(f"📄 Report generated: {report_path}")
                
                # Open report option
                if click.confirm("🔍 Would you like to view the report?"):
                    try:
                        if sys.platform == "darwin":  # macOS
                            os.system(f"open '{report_path}'")
                        elif sys.platform == "linux":
                            os.system(f"xdg-open '{report_path}'")
                        elif sys.platform == "win32":
                            os.system(f"start '{report_path}'")
                        else:
                            click.echo(f"📖 Please open: {report_path}")
                    except Exception as e:
                        click.echo(f"⚠️ Could not open report automatically: {e}")
                        click.echo(f"📖 Please open manually: {report_path}")
        
        elif result.get('processing_status') == 'error':
            click.echo(f"❌ Analysis failed: {result.get('error', 'Unknown error')}", err=True)
            sys.exit(1)
        
        else:
            click.echo(f"⚠️ Analysis completed with status: {result.get('processing_status', 'Unknown')}")
    
    except FileNotFoundError:
        click.echo(f"❌ File not found: {file_path}", err=True)
        sys.exit(1)
    
    except Exception as e:
        click.echo(f"❌ Unexpected error: {e}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


@cli.command()
def demo():
    """
    Chạy demo workflow với sample code
    """
    click.echo("🚀 Running DeepCode-Insight Demo")
    click.echo("=" * 50)
    
    try:
        from ..core.graph import run_analysis_demo
        run_analysis_demo()
        
        click.echo("\n✅ Demo completed successfully!")
        click.echo("📁 Check 'analysis_reports' directory for generated report")
        
    except Exception as e:
        click.echo(f"❌ Demo failed: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--check-llm', is_flag=True, help='Check LLM service availability')
@click.option('--check-deps', is_flag=True, help='Check dependencies')
def health(check_llm, check_deps):
    """
    Kiểm tra health của system và dependencies
    """
    click.echo("🏥 DeepCode-Insight Health Check")
    click.echo("=" * 40)
    
    # Check Python version
    python_version = sys.version.split()[0]
    click.echo(f"🐍 Python version: {python_version}")
    
    # Check dependencies
    try:
        import langgraph
        click.echo("✅ LangGraph: Available")
    except ImportError:
        click.echo("❌ LangGraph: Not installed")
    
    try:
        import tree_sitter
        click.echo("✅ Tree-sitter: Available")
    except ImportError:
        click.echo("❌ Tree-sitter: Not installed")
    
    try:
        import requests
        click.echo("✅ Requests: Available")
    except ImportError:
        click.echo("❌ Requests: Not installed")
    
    # Check LLM service
    if True:  # Always check LLM
        try:
            from agents.llm_orchestrator import create_llm_orchestrator_agent
            orchestrator = create_llm_orchestrator_agent()
            
            if orchestrator.check_llm_health():
                click.echo("✅ LLM Service: Available")
            else:
                click.echo("⚠️ LLM Service: Not available (will skip LLM analysis)")
        except Exception as e:
            click.echo(f"❌ LLM Service: Error - {e}")
    
    # Check output directory
    output_dir = "analysis_reports"
    if os.path.exists(output_dir):
        click.echo(f"✅ Output directory: {output_dir}")
    else:
        click.echo(f"📁 Output directory will be created: {output_dir}")
    
    click.echo("\n🎯 System ready for analysis!")


if __name__ == '__main__':
    cli() 