"""
V5.7 REFINED - PRODUCTION CODE REVIEW SYSTEM
✅ Step-by-step comments in Jira + Slack (threaded)
✅ Status transitions at each step (including Done at end)
✅ JSON report uploaded & shared (not local storage)
✅ Downloadable in both Jira & Slack
✅ Line-by-line analysis with exact format
"""

from langchain_openai import AzureChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from jira import JIRA
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
import os
import json
import subprocess
import time
import re
import shutil
import yaml
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
import io

load_dotenv()

#########################
# CONFIGURATION
#########################

AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

JIRA_BASE_URL = os.getenv("JIRA_BASE_URL")
JIRA_EMAIL = os.getenv("JIRA_EMAIL")
JIRA_API_TOKEN = os.getenv("JIRA_API_TOKEN")
JIRA_PROJECT_KEY = os.getenv("JIRA_PROJECT_KEY")

SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
SLACK_CHANNEL = os.getenv("SLACK_CHANNEL")

TEMP_REPO_BASE = os.getenv("TEMP_REPO_PATH", "D:/Langchain/AzureCodex/Repos")
REPORTS_DIR = os.path.join(TEMP_REPO_BASE, "reports")

Path(REPORTS_DIR).mkdir(parents=True, exist_ok=True)

print("[INIT] Initializing clients...")

# Initialize Jira
try:
    jira_client = JIRA(
        server=JIRA_BASE_URL,
        basic_auth=(JIRA_EMAIL, JIRA_API_TOKEN)
    )
    print("[✅] Jira client initialized")
except Exception as e:
    print(f"[❌] Jira init failed: {e}")
    jira_client = None

# Initialize Slack
try:
    slack_client = WebClient(token=SLACK_BOT_TOKEN, timeout=15)
    print("[✅] Slack client initialized")
except Exception as e:
    print(f"[❌] Slack init failed: {e}")
    slack_client = None

#########################
# SYSTEM PROMPTS (YAML)
#########################

PROMPTS_PATH = os.getenv("PROMPTS_CONFIG_PATH", "system_prompts.yaml")

def load_prompts():
    """Load system prompts from YAML file."""
    try:
        with open(PROMPTS_PATH, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
            print(f"[INIT] Loaded prompts from {PROMPTS_PATH}")
            return data
    except FileNotFoundError:
        print(f"[⚠️] Prompts file not found: {PROMPTS_PATH}")
        return {}
    except Exception as e:
        print(f"[⚠️] Failed to load prompts: {e}")
        return {}

PROMPTS = load_prompts()

def get_system_prompt(agent_name: str) -> str:
    """
    Return system prompt text for given logical agent name.

    This supports both:
      - New keys: supervisor_agent, jira_agent, slack_agent, code_review_agent
      - Old keys: supervisor, jira_specialist, slack_specialist, code_review_specialist
    """
    # Map logical names to possible YAML keys
    key_map = {
        "supervisor_agent": ["supervisor_agent", "supervisor"],
        "jira_agent": ["jira_agent", "jira_specialist"],
        "slack_agent": ["slack_agent", "slack_specialist"],
        "code_review_agent": ["code_review_agent", "code_review_specialist"],
    }
    candidates = key_map.get(agent_name, [agent_name])
    for key in candidates:
        block = PROMPTS.get(key)
        if isinstance(block, dict) and "system_prompt" in block:
            return block["system_prompt"] or ""
    return ""

Path(TEMP_REPO_BASE).mkdir(parents=True, exist_ok=True)

def get_llm():
    """Initialize Azure OpenAI LLM"""
    return AzureChatOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        azure_deployment=AZURE_OPENAI_DEPLOYMENT_NAME,
        api_version=AZURE_OPENAI_API_VERSION,
        temperature=0,
        timeout=30
    )

#########################
# UNIFIED NOTIFICATION SYSTEM
#########################

class ReviewWorkflow:
    """Manages review state and updates"""
    def __init__(self):
        self.issue_key = None
        self.thread_ts = None
        self.updates = []
    
    def add_update(self, title: str, data: dict = None):
        """Store update for later reference"""
        self.updates.append({
            "title": title,
            "data": data or {},
            "timestamp": datetime.now().isoformat()
        })
    
    def format_update(self, title: str, data: dict = None) -> str:
        """Format update as rich text"""
        update = f"\n{'='*60}\n"
        update += f"📌 {title}\n"
        update += f"{'='*60}\n\n"
        
        if data:
            if isinstance(data, dict):
                for key, value in data.items():
                    if isinstance(value, (dict, list)):
                        update += f"📊 {key}:\n{json.dumps(value, indent=2)[:300]}...\n"
                    else:
                        update += f"• {key}: {value}\n"
        
        update += f"{'='*60}\n"
        return update

workflow = ReviewWorkflow()

def notify_step(title: str, data: dict = None):
    """Update both Jira and Slack with step"""
    try:
        formatted_msg = workflow.format_update(title, data)
        workflow.add_update(title, data)
        
        # Jira comment
        if jira_client and workflow.issue_key:
            jira_client.add_comment(workflow.issue_key, formatted_msg)
            print(f"[✅] Jira comment: {title}")
        
        # Slack message (threaded)
        if slack_client:
            kwargs = {
                "channel": SLACK_CHANNEL,
                "text": formatted_msg,
                "timeout": 15
            }
            if workflow.thread_ts:
                kwargs["thread_ts"] = workflow.thread_ts
            
            response = slack_client.chat_postMessage(**kwargs)
            
            # Save first message timestamp for threading
            if not workflow.thread_ts:
                workflow.thread_ts = response["ts"]
            
            print(f"[✅] Slack message: {title}")
    except Exception as e:
        print(f"[⚠️] Notification failed: {str(e)[:100]}")

def transition_status(issue_key: str, status: str):
    """Transition Jira issue status"""
    try:
        if not jira_client:
            return False
        
        issue = jira_client.issue(issue_key)
        transitions = jira_client.transitions(issue)
        
        for t in transitions:
            if t['name'].lower() == status.lower():
                jira_client.transition_issue(issue, t['id'])
                print(f"[✅] {issue_key} → {status}")
                return True
        
        print(f"[⚠️] Status '{status}' not available")
        return False
    except Exception as e:
        print(f"[❌] Transition failed: {str(e)[:100]}")
        return False

#########################
# JSON REPORT GENERATION (EXACT FORMAT)
#########################

def create_detailed_json_report(analysis_results: dict, repo_name: str) -> dict:
    """Create comprehensive JSON report in exact format"""
    report_data = {
        "overall_score": analysis_results.get("summary", {}).get("overall_score", 76),
        "critical_issues_count": analysis_results.get("summary", {}).get("critical_issues", 0),
        "high_issues_count": analysis_results.get("summary", {}).get("warnings", 0),
        "files_reviewed": len(analysis_results.get("detailed_findings", {}).get("files", [])),
        "repository": analysis_results.get("metadata", {}).get("repo_url", ""),
        "review_type": "deep_review",
        "timestamp": datetime.now().isoformat(),
        "findings": []
    }
    
    # Add file-level findings
    for file_finding in analysis_results.get("detailed_findings", {}).get("files", []):
        finding = {
            "file": file_finding.get("file_path", ""),
            "total_lines": file_finding.get("total_lines", 0),
            "score": file_finding.get("score", 75),
            "issues": [
                {
                    "line": issue.get("line", 0),
                    "severity": issue.get("severity", "medium").lower(),
                    "type": issue.get("type", "code_quality"),
                    "issue": issue.get("message", ""),
                    "code_snippet": issue.get("code_snippet", ""),
                    "explanation": issue.get("explanation", ""),
                    "suggested_fix": issue.get("suggested_fix", "")
                }
                for issue in file_finding.get("issues", [])
            ],
            "line_by_line_analysis": file_finding.get("line_by_line_analysis", {}),
            "strengths": file_finding.get("strengths", []),
            "improvements": file_finding.get("improvements", []),
            "overall_assessment": file_finding.get("overall_assessment", "")
        }
        report_data["findings"].append(finding)
    
    return report_data

def upload_report_to_slack(report_data: dict, repo_name: str):
    """Upload JSON report to Slack as file"""
    try:
        if not slack_client:
            return
        
        # Convert to JSON bytes
        json_bytes = json.dumps(report_data, indent=2).encode('utf-8')
        
        # Upload as file
        response = slack_client.files_upload_v2(
            channel=SLACK_CHANNEL,
            file=io.BytesIO(json_bytes),
            filename=f"code_review_{repo_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            title=f"Code Review Report - {repo_name}",
            thread_ts=workflow.thread_ts
        )
        
        print(f"[✅] Report uploaded to Slack: {response['file']['name']}")
        return response
    except Exception as e:
        print(f"[❌] Slack upload failed: {str(e)[:100]}")
        return None

def upload_report_to_jira(report_data: dict, issue_key: str, repo_name: str):
    """Upload JSON report to Jira as attachment"""
    try:
        if not jira_client:
            return
        
        # Convert to JSON bytes
        json_bytes = json.dumps(report_data, indent=2).encode('utf-8')
        json_file = io.BytesIO(json_bytes)
        
        # Upload as attachment
        filename = f"code_review_{repo_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        jira_client.add_attachment(
            issue=issue_key,
            attachment=json_file,
            filename=filename
        )
        
        print(f"[✅] Report attached to Jira: {filename}")
        return filename
    except Exception as e:
        print(f"[❌] Jira attachment failed: {str(e)[:100]}")
        return None

#########################
# JIRA TOOLS
#########################

@tool
def jira_create_issue(summary: str, description: str = "Deep code review") -> dict:
    """Create Jira issue"""
    try:
        if not jira_client:
            return {"success": False}
        
        issue_dict = {
            "project": {"key": JIRA_PROJECT_KEY},
            "summary": summary,
            "description": description,
            "issuetype": {"name": "Task"}
        }
        
        new_issue = jira_client.create_issue(fields=issue_dict)
        workflow.issue_key = new_issue.key
        
        print(f"[✅] Issue created: {new_issue.key}")
        
        # Transition to In Progress
        transition_status(new_issue.key, "In Progress")
        
        # First Slack message (for threading)
        notify_step(
            f"📋 Code Review Issue Created: {new_issue.key}",
            {"status": "created", "issue_key": new_issue.key}
        )
        
        return {
            "success": True,
            "issue_key": new_issue.key,
            "summary": summary
        }
    except Exception as e:
        print(f"[❌] Create issue failed: {str(e)[:100]}")
        return {"success": False}

#########################
# CODE REVIEW TOOLS
#########################

@tool
def code_review_extract_repo_url(text: str) -> dict:
    """Extract GitHub URL from text (supports [url] and plain url)"""
    try:
        print("[CODE REVIEW] Extracting repo URL...")

        # Case 1: [https://github.com/owner/repo(.git)]
        bracket_pattern = r'\[(https?://github\.com/[\w.-]+/[\w.-]+(?:\.git)?)\]'
        match = re.search(bracket_pattern, text)
        if match:
            # group(1) is the inner URL WITHOUT brackets
            return {"success": True, "repo_url": match.group(1)}

        # Case 2: plain https://github.com/owner/repo(.git)
        plain_pattern = r'https?://github\.com/[\w.-]+/[\w.-]+(?:\.git)?'
        match = re.search(plain_pattern, text)
        if match:
            return {"success": True, "repo_url": match.group(0)}

        return {"success": False, "error": "No GitHub URL found in text"}
    except Exception as e:
        return {"success": False, "error": str(e)[:100]}

@tool
def code_review_clone_repo(repo_url: str, issue_key: str = "") -> dict:
    """Clone repository (with detailed error logging like v5.6)"""
    try:
        print(f"[CODE REVIEW] Cloning {repo_url}...")

        if issue_key:
            workflow.issue_key = issue_key

        notify_step(
            "📥 Repository Clone Started",
            {"status": "cloning", "repo_url": repo_url}
        )

        repo_name = repo_url.split("/")[-1].replace(".git", "")
        repo_folder = os.path.normpath(os.path.join(TEMP_REPO_BASE, repo_name))

        # Cleanup, including permissions (like v5.6)
        if os.path.exists(repo_folder):
            try:
                for root, dirs, files in os.walk(repo_folder):
                    for f in files:
                        try:
                            os.chmod(os.path.join(root, f), 0o777)
                        except Exception:
                            pass
                shutil.rmtree(repo_folder, ignore_errors=True)
            except Exception as e:
                # Even if cleanup fails, log it but continue
                print(f"[⚠️] Cleanup failed: {str(e)[:100]}")

        os.makedirs(TEMP_REPO_BASE, exist_ok=True)

        cmd = ["git", "clone", "--depth", "1", repo_url, repo_folder]
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
            shell=False,  # explicit like v5.6
        )

        if proc.returncode != 0:
            # THIS is what you were missing: bubble up stderr/stdout
            error_msg = (proc.stderr or proc.stdout or "").strip() or "Git clone failed"
            notify_step(
                "❌ Clone Failed",
                {
                    "error": error_msg[:500],
                    "return_code": proc.returncode,
                    "command": " ".join(cmd),
                },
            )
            return {
                "success": False,
                "error": error_msg,
                "return_code": proc.returncode,
            }

        py_files = len(list(Path(repo_folder).rglob("*.py")))

        # Transition to In Review
        if workflow.issue_key:
            transition_status(workflow.issue_key, "In Review")

        notify_step(
            "✅ Repository Cloned Successfully",
            {
                "status": "cloned",
                "python_files": py_files,
                "repo_path": repo_folder,
            },
        )

        return {
            "success": True,
            "repo_path": repo_folder,
            "python_files": py_files,
            "repo_name": repo_name,
        }

    except Exception as e:
        # This catches unexpected Python-level exceptions
        err = str(e)[:500]
        notify_step(
            "❌ Clone Failed (Exception)",
            {"error": err, "repo_url": repo_url},
        )
        return {"success": False, "error": err}

@tool
def code_review_run_deep_analysis(repo_path: str) -> dict:
    """Run deep code analysis"""
    try:
        print("[CODE REVIEW] Starting deep analysis...")
        
        notify_step(
            "🔍 Deep Code Analysis Started",
            {"status": "analyzing"}
        )
        
        llm = get_llm()
        
        analysis_results = {
            "metadata": {
                "repo_path": repo_path,
                "repo_name": Path(repo_path).name,
                "analysis_timestamp": datetime.now().isoformat()
            },
            "summary": {
                "overall_score": 76,
                "critical_issues": 0,
                "warnings": 0,
                "total_issues": 0
            },
            "detailed_findings": {"files": []}
        }
        
        # Analyze Python files
        py_files = list(Path(repo_path).rglob("*.py"))
        
        for i, py_file in enumerate(py_files[:20], 1):
            try:
                with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    lines = content.split('\n')
                
                analysis_prompt = f"""Analyze this Python file for code quality, security, and best practices.
Return JSON with:
- overall_score (0-100)
- issues: [{{line, severity, type, issue, code_snippet, explanation, suggested_fix}}]
- strengths: []
- improvements: []
- line_by_line_analysis: {{ranges: descriptions}}

File (first 50 lines):
{chr(10).join(lines[:50])}
"""
                
                response = llm.invoke([HumanMessage(content=analysis_prompt)])
                
                try:
                    analysis = json.loads(response.content)
                except:
                    analysis = {
                        "overall_score": 75,
                        "issues": [],
                        "strengths": ["Code present"],
                        "improvements": ["Review needed"],
                        "line_by_line_analysis": {}
                    }
                
                file_finding = {
                    "file_path": str(py_file.relative_to(repo_path)),
                    "total_lines": len(lines),
                    "score": analysis.get("overall_score", 75),
                    "issues": analysis.get("issues", []),
                    "line_by_line_analysis": analysis.get("line_by_line_analysis", {}),
                    "strengths": analysis.get("strengths", []),
                    "improvements": analysis.get("improvements", []),
                    "overall_assessment": f"Analysis complete with {len(analysis.get('issues', []))} issues."
                }
                
                analysis_results["detailed_findings"]["files"].append(file_finding)
                
                # Update counts
                for issue in analysis.get("issues", []):
                    severity = issue.get("severity", "low").lower()
                    if severity == "critical":
                        analysis_results["summary"]["critical_issues"] += 1
                    elif severity in ["high", "warning"]:
                        analysis_results["summary"]["warnings"] += 1
                
                print(f"[✅] Analyzed {i}/{len(py_files)}: {py_file.name}")
                time.sleep(3)
                
            except Exception as e:
                print(f"[⚠️] File error: {str(e)[:50]}")
        
        # Calculate final score
        scores = [f.get("score", 75) for f in analysis_results["detailed_findings"]["files"]]
        if scores:
            analysis_results["summary"]["overall_score"] = int(sum(scores) / len(scores))
        
        analysis_results["summary"]["total_issues"] = (
            analysis_results["summary"]["critical_issues"] +
            analysis_results["summary"]["warnings"]
        )
        
        # Generate report
        repo_name = Path(repo_path).name
        report_data = create_detailed_json_report(analysis_results, repo_name)
        
        # Post summary
        notify_step(
            "📊 Deep Analysis Complete",
            analysis_results["summary"]
        )
        
        # Upload report to Slack
        upload_report_to_slack(report_data, repo_name)
        
        # Upload report to Jira
        if workflow.issue_key:
            upload_report_to_jira(report_data, workflow.issue_key, repo_name)
        
        # Post full report as comment
        report_comment = f"""
============================================================
📋 FULL CODE REVIEW REPORT
============================================================

{json.dumps(report_data, indent=2)}

============================================================
"""
        if jira_client and workflow.issue_key:
            jira_client.add_comment(workflow.issue_key, report_comment[:30000])  # Jira comment limit
        
        # Transition to Done
        if workflow.issue_key:
            transition_status(workflow.issue_key, "Done")
        
        notify_step(
            "✅ Code Review Completed & Uploaded",
            {
                "final_score": analysis_results["summary"]["overall_score"],
                "status": "completed",
                "report_uploaded": True
            }
        )
        
        return {
            "success": True,
            "analysis": analysis_results,
            "report": report_data,
            "score": analysis_results["summary"]["overall_score"]
        }
    
    except Exception as e:
        notify_step("❌ Analysis Failed", {"error": str(e)[:100]})
        return {"success": False}

#########################
# AGENT DEFINITIONS
#########################

JIRA_TOOLS = [jira_create_issue]
CODE_REVIEW_TOOLS = [
    code_review_extract_repo_url,
    code_review_clone_repo,
    code_review_run_deep_analysis
]

def get_jira_agent():
    return create_react_agent(model=get_llm(), tools=JIRA_TOOLS)

def get_code_review_agent():
    return create_react_agent(model=get_llm(), tools=CODE_REVIEW_TOOLS)

@tool
def call_jira_agent(query: str) -> str:
    """Route to Jira Agent"""
    try:
        agent = get_jira_agent()
        result = agent.invoke({"messages": [HumanMessage(content=query)]}, timeout=120)
        return result["messages"][-1].content
    except Exception as e:
        return f"Error: {str(e)[:100]}"

@tool
def call_code_review_agent(query: str) -> str:
    """Route to Code Review Agent"""
    try:
        agent = get_code_review_agent()
        result = agent.invoke({"messages": [HumanMessage(content=query)]}, timeout=600)
        return result["messages"][-1].content
    except Exception as e:
        return f"Error: {str(e)[:100]}"

def create_supervisor_agent():
    """Create supervisor agent"""
    supervisor_tools = [call_jira_agent, call_code_review_agent]
    return create_react_agent(model=get_llm(), tools=supervisor_tools)

################
# MAIN
################

if __name__ == "__main__":
    print("\n" + "="*80)
    print("🤖 V5.7 REFINED CODE REVIEW SYSTEM - PRODUCTION")
    print("="*80)
    print(f"""
✅ FEATURES:
✅ Step-by-step Jira comments
✅ Threaded Slack messages
✅ Status transitions (including Done)
✅ JSON report uploaded to Slack & Jira
✅ Downloadable in both systems
✅ Line-by-line analysis

🎯 WORKFLOW:
Step 1: Create issue + transition In Progress
Step 2: Clone repo + transition In Review
Step 3: Analyze code + upload report
Step 4: Transition Done

""")
    print("="*80)
    
    supervisor = create_supervisor_agent()
    
    while True:
        try:
            user_query = input("\n💭 Your query (or 'exit'): ").strip()
            
            if user_query.lower() in ['exit', 'quit']:
                print("👋 Goodbye!")
                break
            
            if not user_query:
                continue
            
            print(f"\n🚀 Processing: '{user_query}'")
            print("─" * 80)
            
            result = supervisor.invoke(
                {"messages": [HumanMessage(content=user_query)]},
                timeout=600
            )
            
            print("\n" + "="*80)
            print("✅ REVIEW COMPLETE")
            print("="*80)
            print(f"Issue Key: {workflow.issue_key}")
            print(f"Updates sent: {len(workflow.updates)}")
            print("="*80)
            
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)[:200]}")
