"""
Complete LangGraph Workflow cho DeepCode-Insight
Kết nối tất cả agents: UserInteraction -> CodeFetcher -> StaticAnalysis -> LLMOrchestrator -> Reporting
"""

import sys
import os
from typing import Dict, Any, Literal

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from langgraph.graph import StateGraph, START, END
from .state import AgentState, SimpleAgentState, DEFAULT_AGENT_STATE

# Import all agents
from ..agents.code_fetcher import CodeFetcherAgent
from ..agents.static_analyzer import StaticAnalysisAgent
from ..agents.llm_orchestrator import LLMOrchestratorAgent, create_llm_orchestrator_agent
from ..agents.reporter import ReportingAgent, create_reporting_agent


# ===== Agent Node Functions =====

def user_interaction_node(state: AgentState) -> Dict[str, Any]:
    """
    UserInteractionAgent node - Xử lý input từ user (CLI/API)
    Trong implementation này, chúng ta assume input đã được validate từ CLI
    """
    print(f"🤖 UserInteractionAgent: Processing input...")
    
    # Validate required inputs
    if not state.get('repo_url') and not state.get('code_content'):
        return {
            **state,
            'current_agent': 'user_interaction',
            'processing_status': 'error',
            'error': 'Either repo_url or code_content must be provided',
            'finished': True
        }
    
    # Update state
    updated_state = state.copy()
    updated_state.update({
        'current_agent': 'user_interaction',
        'processing_status': 'input_validated',
        'error': None
    })
    
    print(f"✅ UserInteractionAgent: Input validated successfully")
    return updated_state


def code_fetcher_node(state: AgentState) -> Dict[str, Any]:
    """
    CodeFetcherAgent node - Fetch code từ repository hoặc sử dụng provided code
    """
    print(f"🔄 CodeFetcherAgent: Fetching code...")
    
    try:
        # Nếu đã có code_content, skip fetching
        if state.get('code_content'):
            print(f"📝 CodeFetcherAgent: Using provided code content")
            return {
                **state,
                'current_agent': 'code_fetcher',
                'processing_status': 'code_available',
                'filename': state.get('filename', 'provided_code.py')
            }
        
        # Fetch code từ repository
        if state.get('repo_url'):
            fetcher = CodeFetcherAgent()
            
            # Get repository info
            repo_info = fetcher.get_repository_info(state['repo_url'])
            
            # Get PR diff nếu có PR ID
            pr_diff = None
            if state.get('pr_id'):
                try:
                    pr_diff = fetcher.get_pr_diff(state['repo_url'], int(state['pr_id']))
                except Exception as e:
                    print(f"⚠️ Warning: Could not fetch PR diff: {e}")
            
            # List repository files để get sample code
            files = fetcher.list_repository_files(state['repo_url'])
            
            # Find a Python file để analyze (simple heuristic)
            python_files = [f for f in files if f.endswith('.py') and not f.startswith('.')]
            
            if not python_files:
                return {
                    **state,
                    'current_agent': 'code_fetcher',
                    'processing_status': 'error',
                    'error': 'No Python files found in repository',
                    'finished': True
                }
            
            # Get content của first Python file
            target_file = state.get('target_file', python_files[0])
            code_content = fetcher.get_file_content(state['repo_url'], target_file)
            
            # Cleanup
            fetcher.cleanup()
            
            print(f"✅ CodeFetcherAgent: Successfully fetched code from {target_file}")
            
            return {
                **state,
                'current_agent': 'code_fetcher',
                'processing_status': 'code_fetched',
                'code_content': code_content,
                'filename': target_file,
                'repository_info': repo_info,
                'pr_diff': pr_diff
            }
        
        # No valid input
        return {
            **state,
            'current_agent': 'code_fetcher',
            'processing_status': 'error',
            'error': 'No valid code source provided',
            'finished': True
        }
        
    except Exception as e:
        print(f"❌ CodeFetcherAgent error: {e}")
        return {
            **state,
            'current_agent': 'code_fetcher',
            'processing_status': 'error',
            'error': f"Code fetching failed: {str(e)}",
            'finished': True
        }


def static_analysis_node(state: AgentState) -> Dict[str, Any]:
    """
    StaticAnalysisAgent node - Thực hiện static analysis
    """
    print(f"🔍 StaticAnalysisAgent: Analyzing code...")
    
    try:
        if not state.get('code_content'):
            return {
                **state,
                'current_agent': 'static_analyzer',
                'processing_status': 'error',
                'error': 'No code content available for analysis',
                'finished': True
            }
        
        # Run static analysis
        analyzer = StaticAnalysisAgent()
        analysis_results = analyzer.analyze_code(
            state['code_content'], 
            state.get('filename', '<unknown>')
        )
        
        print(f"✅ StaticAnalysisAgent: Analysis completed")
        
        return {
            **state,
            'current_agent': 'static_analyzer',
            'processing_status': 'static_analysis_completed',
            'static_analysis_results': analysis_results
        }
        
    except Exception as e:
        print(f"❌ StaticAnalysisAgent error: {e}")
        return {
            **state,
            'current_agent': 'static_analyzer',
            'processing_status': 'error',
            'error': f"Static analysis failed: {str(e)}",
            'finished': True
        }


def llm_orchestrator_node(state: AgentState) -> Dict[str, Any]:
    """
    LLMOrchestratorAgent node - Thực hiện LLM analysis
    """
    print(f"🤖 LLMOrchestratorAgent: Processing with LLM...")
    
    try:
        if not state.get('static_analysis_results'):
            return {
                **state,
                'current_agent': 'llm_orchestrator',
                'processing_status': 'error',
                'error': 'No static analysis results available for LLM processing',
                'finished': True
            }
        
        # Create LLM orchestrator
        orchestrator = create_llm_orchestrator_agent()
        
        # Check LLM health
        if not orchestrator.check_llm_health():
            print(f"⚠️ Warning: LLM service not available, skipping LLM analysis")
            return {
                **state,
                'current_agent': 'llm_orchestrator',
                'processing_status': 'llm_skipped',
                'llm_analysis': None
            }
        
        # Process findings với LLM
        llm_result = orchestrator.process_findings(state)
        
        print(f"✅ LLMOrchestratorAgent: LLM analysis completed")
        
        return llm_result
        
    except Exception as e:
        print(f"❌ LLMOrchestratorAgent error: {e}")
        # Continue workflow without LLM analysis
        print(f"⚠️ Continuing workflow without LLM analysis")
        return {
            **state,
            'current_agent': 'llm_orchestrator',
            'processing_status': 'llm_error',
            'llm_analysis': None,
            'error': f"LLM analysis failed: {str(e)} (continuing without LLM)"
        }


def reporting_node(state: AgentState) -> Dict[str, Any]:
    """
    ReportingAgent node - Tạo final report
    """
    print(f"📊 ReportingAgent: Generating report...")
    
    try:
        # Create reporting agent
        reporter = create_reporting_agent(output_dir="analysis_reports")
        
        # Generate report
        final_result = reporter.generate_report(state)
        
        # Mark workflow as finished
        final_result['finished'] = True
        
        print(f"✅ ReportingAgent: Report generated successfully")
        if final_result.get('report'):
            print(f"📄 Report saved to: {final_result['report']['output_path']}")
        
        return final_result
        
    except Exception as e:
        print(f"❌ ReportingAgent error: {e}")
        return {
            **state,
            'current_agent': 'reporter',
            'processing_status': 'error',
            'error': f"Report generation failed: {str(e)}",
            'finished': True
        }


# ===== Conditional Logic =====

def should_continue(state: AgentState) -> Literal["continue", "end"]:
    """
    Quyết định có tiếp tục workflow hay không
    """
    # Stop nếu có error hoặc finished
    if state.get('finished') or state.get('processing_status') == 'error':
        return "end"
    
    return "continue"


def route_after_user_interaction(state: AgentState) -> Literal["code_fetcher", "end"]:
    """
    Route sau UserInteractionAgent
    """
    if state.get('processing_status') == 'error':
        return "end"
    return "code_fetcher"


def route_after_code_fetcher(state: AgentState) -> Literal["static_analyzer", "end"]:
    """
    Route sau CodeFetcherAgent
    """
    if state.get('processing_status') in ['error', 'code_fetched', 'code_available']:
        if state.get('code_content'):
            return "static_analyzer"
    return "end"


def route_after_static_analysis(state: AgentState) -> Literal["llm_orchestrator", "end"]:
    """
    Route sau StaticAnalysisAgent
    """
    if state.get('processing_status') == 'static_analysis_completed':
        return "llm_orchestrator"
    return "end"


def route_after_llm_orchestrator(state: AgentState) -> Literal["reporter", "end"]:
    """
    Route sau LLMOrchestratorAgent
    """
    if state.get('processing_status') in ['llm_analysis_completed', 'llm_skipped', 'llm_error']:
        return "reporter"
    return "end"


# ===== Graph Creation =====

def create_analysis_workflow() -> StateGraph:
    """
    Tạo complete analysis workflow graph
    """
    # Khởi tạo graph với AgentState
    workflow = StateGraph(AgentState)
    
    # Thêm các nodes (agents)
    workflow.add_node("user_interaction", user_interaction_node)
    workflow.add_node("code_fetcher", code_fetcher_node)
    workflow.add_node("static_analyzer", static_analysis_node)
    workflow.add_node("llm_orchestrator", llm_orchestrator_node)
    workflow.add_node("reporter", reporting_node)
    
    # Thiết lập entry point
    workflow.add_edge(START, "user_interaction")
    
    # Thêm conditional edges để điều khiển flow
    workflow.add_conditional_edges(
        "user_interaction",
        route_after_user_interaction,
        {
            "code_fetcher": "code_fetcher",
            "end": END
        }
    )
    
    workflow.add_conditional_edges(
        "code_fetcher",
        route_after_code_fetcher,
        {
            "static_analyzer": "static_analyzer",
            "end": END
        }
    )
    
    workflow.add_conditional_edges(
        "static_analyzer",
        route_after_static_analysis,
        {
            "llm_orchestrator": "llm_orchestrator",
            "end": END
        }
    )
    
    workflow.add_conditional_edges(
        "llm_orchestrator",
        route_after_llm_orchestrator,
        {
            "reporter": "reporter",
            "end": END
        }
    )
    
    # Reporter always ends the workflow
    workflow.add_edge("reporter", END)
    
    # Compile graph
    return workflow.compile()


# ===== Legacy Demo Functions (Backward Compatibility) =====

def agent_1(state: SimpleAgentState) -> Dict[str, Any]:
    """
    Agent 1: Khởi tạo cuộc trò chuyện và gửi message (Legacy demo)
    """
    messages = state.get("messages", [])
    message_count = state.get("message_count", 0)
    
    print(f"🤖 Agent 1 đang xử lý message số {message_count + 1}")
    
    # Thêm message mới từ Agent 1
    new_message = f"Xin chào từ Agent 1! (Lần thứ {message_count + 1})"
    messages.append(new_message)
    
    print(f"📝 Agent 1 nói: {new_message}")
    
    return {
        "messages": messages,
        "current_agent": "agent_2",
        "message_count": message_count + 1,
        "finished": message_count >= 4  # Dừng sau 5 lần trao đổi
    }


def agent_2(state: SimpleAgentState) -> Dict[str, Any]:
    """
    Agent 2: Nhận và phản hồi message từ Agent 1 (Legacy demo)
    """
    messages = state.get("messages", [])
    message_count = state.get("message_count", 0)
    
    print(f"🦾 Agent 2 đang xử lý message số {message_count}")
    
    # Thêm message phản hồi từ Agent 2
    new_message = f"Chào Agent 1! Tôi đã nhận được message của bạn. (Phản hồi lần {message_count})"
    messages.append(new_message)
    
    print(f"📝 Agent 2 nói: {new_message}")
    
    return {
        "messages": messages,
        "current_agent": "agent_1",
        "message_count": message_count + 1,
        "finished": message_count >= 4  # Dừng sau 5 lần trao đổi
    }


def should_continue_simple(state: SimpleAgentState) -> Literal["agent_1", "agent_2", "end"]:
    """
    Quyết định agent nào sẽ xử lý tiếp theo (Legacy demo)
    """
    if state.get("finished", False):
        return "end"
    
    current_agent = state.get("current_agent", "agent_1")
    if current_agent == "agent_1":
        return "agent_1"
    else:
        return "agent_2"


def create_simple_graph() -> StateGraph:
    """
    Tạo simple demo graph (Legacy - backward compatibility)
    """
    # Khởi tạo graph với SimpleAgentState
    workflow = StateGraph(SimpleAgentState)
    
    # Thêm các nodes (agents)
    workflow.add_node("agent_1", agent_1)
    workflow.add_node("agent_2", agent_2)
    
    # Thiết lập entry point
    workflow.add_edge(START, "agent_1")
    
    # Thêm conditional edge để quyết định luồng
    workflow.add_conditional_edges(
        "agent_1",
        should_continue_simple,
        {
            "agent_2": "agent_2",
            "end": END
        }
    )
    
    workflow.add_conditional_edges(
        "agent_2", 
        should_continue_simple,
        {
            "agent_1": "agent_1",
            "end": END
        }
    )
    
    # Compile graph
    return workflow.compile()


# ===== Demo Functions =====

def run_analysis_demo():
    """
    Chạy complete analysis workflow demo
    """
    print("🚀 Bắt đầu DeepCode-Insight Analysis Workflow")
    print("=" * 60)
    
    # Tạo graph
    graph = create_analysis_workflow()
    
    # Sample code để analyze
    sample_code = '''
def calculate_sum(a, b):
    return a + b

class Calculator:
    def multiply(self, x, y):
        return x * y
        
def very_long_function_name_that_exceeds_the_recommended_line_length_limit():
    pass
'''
    
    # Initial state
    initial_state: AgentState = {
        **DEFAULT_AGENT_STATE,
        'code_content': sample_code,
        'filename': 'demo_analysis.py'
    }
    
    try:
        # Chạy graph
        print(f"🔄 Starting analysis workflow...")
        result = graph.invoke(initial_state)
        
        print("\n" + "=" * 60)
        print("📋 Analysis Workflow Summary:")
        print(f"📁 File analyzed: {result.get('filename', 'Unknown')}")
        print(f"🎯 Final status: {result.get('processing_status', 'Unknown')}")
        
        if result.get('error'):
            print(f"❌ Error: {result['error']}")
        
        if result.get('static_analysis_results'):
            static_results = result['static_analysis_results']
            print(f"🔍 Static analysis: {len(static_results.get('static_issues', {}))} issue types found")
            print(f"📊 Code quality score: {static_results.get('metrics', {}).get('code_quality_score', 'N/A')}")
        
        if result.get('llm_analysis'):
            print(f"🤖 LLM analysis: Completed")
            print(f"📝 Summary: {result['llm_analysis'].get('summary', 'N/A')[:100]}...")
        
        if result.get('report'):
            report_info = result['report']
            print(f"📄 Report generated: {report_info.get('filename', 'N/A')}")
            print(f"📍 Report path: {report_info.get('output_path', 'N/A')}")
        
        print(f"✅ Workflow completed successfully!")
        
    except Exception as e:
        print(f"❌ Workflow failed: {e}")
        import traceback
        traceback.print_exc()


def run_simple_demo():
    """
    Chạy simple demo với initial state (Legacy)
    """
    print("🚀 Bắt đầu LangGraph Demo - Hai Agents Trò Chuyện")
    print("=" * 50)
    
    # Tạo graph
    graph = create_simple_graph()
    
    # Initial state
    initial_state: SimpleAgentState = {
        "messages": [],
        "current_agent": "agent_1",
        "message_count": 0,
        "finished": False
    }
    
    # Chạy graph
    result = graph.invoke(initial_state)
    
    print("\n" + "=" * 50)
    print("📋 Tóm tắt cuộc trò chuyện:")
    for i, message in enumerate(result["messages"], 1):
        print(f"{i}. {message}")
    
    print(f"\n📊 Tổng số message: {result['message_count']}")
    print("✅ Demo hoàn thành!")


# Aliases for backward compatibility
create_graph = create_simple_graph
run_demo = run_simple_demo 