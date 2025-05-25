#!/usr/bin/env python3
"""
CLI cho LangGraph Demo
Sử dụng Click để tạo command line interface với validation
"""

import click
import re
import sys
from urllib.parse import urlparse


def validate_repo_url(ctx, param, value):
    """
    Validate repository URL format
    Hỗ trợ GitHub, GitLab, Bitbucket URLs
    """
    if not value:
        raise click.BadParameter('Repository URL is required')
    
    # Regex patterns for common git hosting services
    github_pattern = r'^https://github\.com/[\w\-\.]+/[\w\-\.]+/?$'
    gitlab_pattern = r'^https://gitlab\.com/[\w\-\.]+/[\w\-\.]+/?$'
    bitbucket_pattern = r'^https://bitbucket\.org/[\w\-\.]+/[\w\-\.]+/?$'
    
    # Remove trailing .git if present
    clean_url = value.rstrip('/').replace('.git', '')
    
    if not (re.match(github_pattern, clean_url) or 
            re.match(gitlab_pattern, clean_url) or
            re.match(bitbucket_pattern, clean_url)):
        raise click.BadParameter(
            'Invalid repository URL. Must be a valid GitHub, GitLab, or Bitbucket URL.\n'
            'Examples:\n'
            '  - https://github.com/owner/repo\n'
            '  - https://gitlab.com/owner/repo\n'
            '  - https://bitbucket.org/owner/repo'
        )
    
    return clean_url


def validate_pr_id(ctx, param, value):
    """
    Validate Pull Request/Merge Request ID
    Must be a positive integer
    """
    if not value:
        raise click.BadParameter('PR/MR ID is required')
    
    try:
        pr_id = int(value)
        if pr_id <= 0:
            raise ValueError()
        return pr_id
    except ValueError:
        raise click.BadParameter('PR/MR ID must be a positive integer')


@click.group()
@click.version_option(version='1.0.0', prog_name='LangGraph Demo CLI')
def cli():
    """
    🚀 LangGraph Demo CLI
    
    Command line interface cho LangGraph project với repository analysis features.
    """
    pass


@cli.command()
@click.option(
    '--repo-url', 
    required=True,
    callback=validate_repo_url,
    help='Repository URL (GitHub, GitLab, or Bitbucket)',
    prompt='Repository URL'
)
@click.option(
    '--pr-id',
    required=True,
    callback=validate_pr_id,
    help='Pull Request/Merge Request ID (positive integer)',
    prompt='PR/MR ID'
)
@click.option(
    '--output-format',
    type=click.Choice(['json', 'text', 'markdown'], case_sensitive=False),
    default='text',
    help='Output format (default: text)'
)
@click.option(
    '--verbose', '-v',
    is_flag=True,
    help='Enable verbose output'
)
def analyze(repo_url, pr_id, output_format, verbose):
    """
    🔍 Analyze a Pull Request using LangGraph agents
    
    Phân tích Pull Request bằng cách sử dụng hai agents:
    - Agent 1: Code review và analysis
    - Agent 2: Feedback và suggestions
    """
    if verbose:
        click.echo(f"🔍 Starting analysis...")
        click.echo(f"📂 Repository: {repo_url}")
        click.echo(f"🔢 PR/MR ID: {pr_id}")
        click.echo(f"📄 Output format: {output_format}")
        click.echo("=" * 50)
    
    # Simulate analysis process
    try:
        with click.progressbar(range(100), label='Analyzing PR') as bar:
            for i in bar:
                # Simulate work
                import time
                time.sleep(0.01)
        
        # Mock results based on format
        if output_format.lower() == 'json':
            result = {
                "repository": repo_url,
                "pr_id": pr_id,
                "status": "completed",
                "analysis": {
                    "agent_1_review": "Code looks good, proper structure",
                    "agent_2_feedback": "Consider adding more tests",
                    "overall_score": 8.5
                }
            }
            click.echo(click.style("\n✅ Analysis completed!", fg='green'))
            click.echo(f"📊 Results (JSON):\n{result}")
            
        elif output_format.lower() == 'markdown':
            click.echo(click.style("\n✅ Analysis completed!", fg='green'))
            click.echo(f"""
# PR Analysis Report

**Repository:** {repo_url}  
**PR ID:** {pr_id}

## Agent 1 Review
- Code structure is well organized
- Follows coding standards
- No major issues found

## Agent 2 Feedback  
- Consider adding unit tests
- Documentation could be improved
- Overall good implementation

**Score:** 8.5/10
""")
        else:  # text format
            click.echo(click.style("\n✅ Analysis completed!", fg='green'))
            click.echo(f"""
📊 PR Analysis Results:
─────────────────────────
Repository: {repo_url}
PR ID: {pr_id}

🤖 Agent 1 Review:
   • Code structure is well organized
   • Follows coding standards  
   • No major issues found

🦾 Agent 2 Feedback:
   • Consider adding unit tests
   • Documentation could be improved
   • Overall good implementation

📈 Overall Score: 8.5/10
""")
            
    except KeyboardInterrupt:
        click.echo(click.style("\n❌ Analysis cancelled by user", fg='red'))
        sys.exit(1)
    except Exception as e:
        click.echo(click.style(f"\n❌ Error during analysis: {e}", fg='red'))
        sys.exit(1)


@cli.command()
@click.option(
    '--repo-url',
    required=True, 
    callback=validate_repo_url,
    help='Repository URL to run demo workflow'
)
@click.option(
    '--pr-id',
    required=True,
    callback=validate_pr_id, 
    help='PR ID for demo workflow'
)
def demo(repo_url, pr_id):
    """
    🎭 Run LangGraph demo workflow
    
    Chạy demo workflow với repository và PR ID được chỉ định.
    """
    click.echo(f"🚀 Running LangGraph demo...")
    click.echo(f"📂 Repository: {repo_url}")
    click.echo(f"🔢 PR ID: {pr_id}")
    click.echo("=" * 50)
    
    # Import và chạy demo workflow với CodeFetcherAgent
    try:
        import sys
        import os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
        
        from src.graph import run_demo
        from agents.code_fetcher import CodeFetcherAgent
        
        click.echo("🤖 Initializing agents...")
        click.echo(f"📂 Repository: {repo_url}")
        click.echo(f"🔢 PR ID: {pr_id}")
        
        # Initialize CodeFetcherAgent
        click.echo("🔄 Initializing CodeFetcherAgent...")
        code_fetcher = CodeFetcherAgent()
        
        try:
            # Get repository info
            click.echo("📊 Getting repository information...")
            repo_info = code_fetcher.get_repository_info(repo_url)
            
            if 'error' not in repo_info:
                click.echo(f"✅ Repository: {repo_info['full_name']}")
                click.echo(f"🌟 Platform: {repo_info['platform']}")
                click.echo(f"📝 Latest commit: {repo_info['latest_commit']['message'][:50]}...")
            
            # Try to get PR diff
            click.echo(f"🔍 Fetching PR {pr_id} diff...")
            pr_diff = code_fetcher.get_pr_diff(repo_url, pr_id)
            
            if pr_diff['error']:
                click.echo(click.style(f"⚠️  PR diff warning: {pr_diff['error']}", fg='yellow'))
            else:
                click.echo(f"📄 Files changed: {len(pr_diff['files_changed'])}")
                click.echo(f"➕ Additions: {pr_diff['stats']['additions']}")
                click.echo(f"➖ Deletions: {pr_diff['stats']['deletions']}")
            
        finally:
            # Cleanup
            code_fetcher.cleanup()
        
        # Run original demo
        click.echo("\n🎭 Running LangGraph demo...")
        run_demo()
        
        click.echo(click.style("\n✅ Demo completed successfully!", fg='green'))
        click.echo(f"📝 Used repository: {repo_url}")
        click.echo(f"📝 Used PR ID: {pr_id}")
        
    except ImportError as e:
        click.echo(click.style(f"❌ Failed to import demo modules: {e}", fg='red'))
        sys.exit(1)
    except Exception as e:
        click.echo(click.style(f"❌ Demo failed: {e}", fg='red'))
        sys.exit(1)


@cli.command()
def validate():
    """
    ✅ Test input validation functions
    
    Test các validation functions với different inputs.
    """
    click.echo("🧪 Testing validation functions...")
    
    # Test valid URLs
    valid_urls = [
        "https://github.com/user/repo",
        "https://gitlab.com/user/repo",
        "https://bitbucket.org/user/repo"
    ]
    
    click.echo("\n✅ Testing valid URLs:")
    for url in valid_urls:
        try:
            result = validate_repo_url(None, None, url)
            click.echo(f"  ✓ {url} → {result}")
        except click.BadParameter as e:
            click.echo(f"  ✗ {url} → {e}")
    
    # Test invalid URLs
    invalid_urls = [
        "not-a-url",
        "https://example.com/repo",
        "github.com/user/repo",
        ""
    ]
    
    click.echo("\n❌ Testing invalid URLs:")
    for url in invalid_urls:
        try:
            validate_repo_url(None, None, url)
            click.echo(f"  ✗ {url} → Should have failed!")
        except click.BadParameter as e:
            click.echo(f"  ✓ {url} → {e}")
    
    # Test PR IDs
    valid_ids = ["1", "123", "9999"]
    invalid_ids = ["0", "-1", "abc", ""]
    
    click.echo("\n✅ Testing valid PR IDs:")
    for pr_id in valid_ids:
        try:
            result = validate_pr_id(None, None, pr_id)
            click.echo(f"  ✓ {pr_id} → {result}")
        except click.BadParameter as e:
            click.echo(f"  ✗ {pr_id} → {e}")
    
    click.echo("\n❌ Testing invalid PR IDs:")
    for pr_id in invalid_ids:
        try:
            validate_pr_id(None, None, pr_id)
            click.echo(f"  ✗ {pr_id} → Should have failed!")
        except click.BadParameter as e:
            click.echo(f"  ✓ {pr_id} → {e}")


@cli.command()
@click.option(
    '--repo-url',
    required=True,
    callback=validate_repo_url,
    help='Repository URL để analyze'
)
@click.option(
    '--pr-id',
    type=int,
    help='PR ID để fetch diff (optional)'
)
@click.option(
    '--list-files',
    is_flag=True,
    help='List files trong repository'
)
@click.option(
    '--get-info',
    is_flag=True,
    help='Get repository information'
)
def fetch(repo_url, pr_id, list_files, get_info):
    """
    🔄 Fetch repository data using CodeFetcherAgent
    
    Test CodeFetcherAgent functionality để clone repository,
    get PR diffs, list files, và repository information.
    """
    click.echo("🔄 Starting CodeFetcherAgent operations...")
    click.echo(f"📂 Repository: {repo_url}")
    
    try:
        from agents.code_fetcher import CodeFetcherAgent
        
        # Initialize agent
        agent = CodeFetcherAgent()
        
        try:
            # Get repository info
            if get_info:
                click.echo("\n📊 Getting repository information...")
                with click.progressbar(range(100), label='Fetching repo info') as bar:
                    for i in bar:
                        import time
                        time.sleep(0.01)
                
                info = agent.get_repository_info(repo_url)
                
                if 'error' in info:
                    click.echo(click.style(f"❌ Error: {info['error']}", fg='red'))
                else:
                    click.echo(click.style("\n✅ Repository Information:", fg='green'))
                    click.echo(f"  🏷️  Full name: {info['full_name']}")
                    click.echo(f"  🌟 Platform: {info['platform']}")
                    click.echo(f"  👤 Owner: {info['owner']}")
                    click.echo(f"  📦 Repo name: {info['repo_name']}")
                    click.echo(f"  📝 Latest commit: {info['latest_commit']['message'][:60]}...")
                    click.echo(f"  👨‍💻 Author: {info['latest_commit']['author']}")
                    click.echo(f"  📅 Date: {info['latest_commit']['date']}")
                    click.echo(f"  🌿 Branches: {', '.join(info['branches'][:5])}")
                    if info['tags']:
                        click.echo(f"  🏷️  Recent tags: {', '.join(info['tags'][-3:])}")
            
            # List files
            if list_files:
                click.echo("\n📁 Listing repository files...")
                files = agent.list_repository_files(repo_url)
                
                if files:
                    click.echo(click.style(f"\n✅ Found {len(files)} files:", fg='green'))
                    for i, file in enumerate(files[:20]):  # Show first 20 files
                        click.echo(f"  📄 {file}")
                    
                    if len(files) > 20:
                        click.echo(f"  ... and {len(files) - 20} more files")
                else:
                    click.echo(click.style("⚠️  No files found", fg='yellow'))
            
            # Get PR diff
            if pr_id:
                click.echo(f"\n🔍 Fetching PR {pr_id} diff...")
                with click.progressbar(range(100), label='Fetching PR diff') as bar:
                    for i in bar:
                        import time
                        time.sleep(0.02)
                
                pr_diff = agent.get_pr_diff(repo_url, pr_id)
                
                if pr_diff['error']:
                    click.echo(click.style(f"❌ PR Error: {pr_diff['error']}", fg='red'))
                else:
                    click.echo(click.style(f"\n✅ PR {pr_id} Analysis:", fg='green'))
                    click.echo(f"  📄 Files changed: {len(pr_diff['files_changed'])}")
                    click.echo(f"  ➕ Additions: {pr_diff['stats']['additions']}")
                    click.echo(f"  ➖ Deletions: {pr_diff['stats']['deletions']}")
                    click.echo(f"  💾 Total files: {pr_diff['stats']['files']}")
                    
                    if pr_diff['files_changed']:
                        click.echo(f"  📝 Changed files:")
                        for file in pr_diff['files_changed'][:10]:  # Show first 10
                            click.echo(f"    • {file}")
                        if len(pr_diff['files_changed']) > 10:
                            click.echo(f"    ... and {len(pr_diff['files_changed']) - 10} more")
                    
                    if pr_diff['commits']:
                        click.echo(f"  📋 Commits ({len(pr_diff['commits'])}):")
                        for commit in pr_diff['commits'][:3]:  # Show first 3
                            click.echo(f"    • {commit['sha']}: {commit['message'][:50]}...")
            
            click.echo(click.style("\n🎉 CodeFetcherAgent operations completed!", fg='green'))
            
        finally:
            # Always cleanup
            agent.cleanup()
            click.echo("🧹 Cleaned up temporary files")
            
    except ImportError as e:
        click.echo(click.style(f"❌ Failed to import CodeFetcherAgent: {e}", fg='red'))
        sys.exit(1)
    except Exception as e:
        click.echo(click.style(f"❌ Operation failed: {e}", fg='red'))
        sys.exit(1)


if __name__ == '__main__':
    cli() 