from typing import TypedDict, List, Optional, Annotated
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
import json
import asyncio
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET

SYSTEM_PROMPT = """You are an expert Medical Research Agent with deep knowledge of:
- Medical literature, clinical trials, and research
- Pharmacology, drug interactions, and pharmacokinetics
- Clinical guidelines (WHO, CDC, AHA, ADA, etc.)
- Pathophysiology and disease mechanisms
- Evidence-based medicine and diagnostic criteria

You have access to the following tools:
- search_pubmed: Search PubMed for medical literature by keyword
- search_web: Search the web for current medical news and guidelines
- lookup_drug_interactions: Look up drug-drug interactions
- get_drug_info: Get detailed pharmacology info about a drug

Guidelines:
1. Use tools when you need current data, specific drug info, or literature citations
2. Cite sources clearly (study name, year, journal when available)
3. Structure complex answers with ## headers and bullet points
4. Distinguish between established evidence (Level A) and emerging research (Level B/C)
5. For drug interactions, always state severity: MAJOR / MODERATE / MINOR

Always conclude with: "**Disclaimer:** This information is for educational and research purposes only. Always consult a qualified healthcare professional before making any medical decisions."
"""

@tool
def search_pubmed(query: str, max_results: int = 5) -> str:
    """Search PubMed for medical literature. Returns titles, authors, and abstracts."""
    try:
        base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        search_url = f"{base}esearch.fcgi?db=pubmed&term={urllib.parse.quote(query)}&retmax={max_results}&retmode=json"
        with urllib.request.urlopen(search_url, timeout=10) as resp:
            data = json.loads(resp.read())
        ids = data.get("esearchresult", {}).get("idlist", [])
        if not ids:
            return f"No PubMed results found for: {query}"
        fetch_url = f"{base}efetch.fcgi?db=pubmed&id={','.join(ids)}&rettype=abstract&retmode=xml"
        with urllib.request.urlopen(fetch_url, timeout=10) as resp:
            xml_data = resp.read()
        root = ET.fromstring(xml_data)
        results = []
        for article in root.findall(".//PubmedArticle")[:max_results]:
            title_el = article.find(".//ArticleTitle")
            title = title_el.text if title_el is not None else "No title"
            year_el = article.find(".//PubDate/Year")
            year = year_el.text if year_el is not None else "Unknown year"
            journal_el = article.find(".//Journal/Title")
            journal = journal_el.text if journal_el is not None else "Unknown journal"
            authors = []
            for author in article.findall(".//Author")[:3]:
                last = author.find("LastName")
                first = author.find("ForeName")
                if last is not None:
                    name = last.text
                    if first is not None:
                        name += f" {first.text[0]}"
                    authors.append(name)
            author_str = ", ".join(authors) + (" et al." if len(authors) >= 3 else "")
            abstract_el = article.find(".//AbstractText")
            abstract = abstract_el.text[:400] if abstract_el is not None and abstract_el.text else "No abstract available"
            pmid_el = article.find(".//PMID")
            pmid = pmid_el.text if pmid_el is not None else ""
            url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else ""
            results.append(
                f"**{title}**\nAuthors: {author_str} ({year}) — {journal}\n"
                f"Abstract: {abstract}...\nURL: {url}"
            )
        return f"PubMed results for '{query}':\n\n" + "\n\n---\n\n".join(results)
    except Exception as e:
        return f"PubMed search failed: {str(e)}"


@tool
def search_web(query: str) -> str:
    """Search the web for current medical guidelines, news, and information."""
    try:
        url = f"https://api.duckduckgo.com/?q={urllib.parse.quote(query + ' medical')}&format=json&no_html=1&skip_disambig=1"
        req = urllib.request.Request(url, headers={"User-Agent": "MedicalResearchAgent/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
        results = []
        if data.get("AbstractText"):
            results.append(f"**Summary:** {data['AbstractText']}\nSource: {data.get('AbstractURL', '')}")
        for topic in data.get("RelatedTopics", [])[:5]:
            if isinstance(topic, dict) and topic.get("Text"):
                results.append(f"• {topic['Text'][:300]}\n  {topic.get('FirstURL', '')}")
        if not results:
            return f"Web search for '{query}' returned limited results."
        return f"Web results for '{query}':\n\n" + "\n\n".join(results)
    except Exception as e:
        return f"Web search failed: {str(e)}"


@tool
def lookup_drug_interactions(drug1: str, drug2: str) -> str:
    """Look up interactions between two drugs."""
    try:
        def get_rxcui(drug_name: str) -> Optional[str]:
            search_url = f"https://rxnav.nlm.nih.gov/REST/rxcui.json?name={urllib.parse.quote(drug_name)}&search=1"
            req = urllib.request.Request(search_url, headers={"User-Agent": "MedicalResearchAgent/1.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
            return data.get("idGroup", {}).get("rxnormId", [None])[0]
        rxcui1 = get_rxcui(drug1)
        rxcui2 = get_rxcui(drug2)
        if not rxcui1 or not rxcui2:
            not_found = [d for d, r in [(drug1, rxcui1), (drug2, rxcui2)] if not r]
            return f"Could not find RxCUI for: {', '.join(not_found)}. Please check spelling."
        int_url = f"https://rxnav.nlm.nih.gov/REST/interaction/interaction.json?rxcui={rxcui1}&sources=ONCHigh"
        req = urllib.request.Request(int_url, headers={"User-Agent": "MedicalResearchAgent/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
        results = []
        for group in data.get("interactionTypeGroup", []):
            for itype in group.get("interactionType", []):
                for pair in itype.get("interactionPair", []):
                    concepts = pair.get("interactionConcept", [])
                    names = [c.get("minConceptItem", {}).get("name", "") for c in concepts]
                    if any(drug2.lower() in n.lower() for n in names):
                        severity = pair.get("severity", "Unknown").upper()
                        results.append(f"**Severity: {severity}**\n{pair.get('description', 'No description')}")
        if results:
            return f"Interactions between **{drug1}** and **{drug2}**:\n\n" + "\n\n".join(results)
        return f"No interaction data found between {drug1} and {drug2}. Always verify with a pharmacist."
    except Exception as e:
        return f"Drug interaction lookup failed: {str(e)}"


@tool
def get_drug_info(drug_name: str) -> str:
    """Get detailed drug information from FDA database."""
    try:
        def fetch_label(search_param: str):
            url = f"https://api.fda.gov/drug/label.json?search={search_param}&limit=1"
            req = urllib.request.Request(url, headers={"User-Agent": "MedicalResearchAgent/1.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                return json.loads(resp.read()).get("results", [])
        results_list = fetch_label(f"openfda.brand_name:{urllib.parse.quote(drug_name)}")
        if not results_list:
            results_list = fetch_label(f"openfda.generic_name:{urllib.parse.quote(drug_name)}")
        if not results_list:
            return f"No FDA drug information found for '{drug_name}'."
        label = results_list[0]
        openfda = label.get("openfda", {})
        def get_section(keys, max_len=500):
            for key in keys:
                val = label.get(key)
                if val:
                    text = val[0] if isinstance(val, list) else val
                    return text[:max_len] + ("..." if len(text) > max_len else "")
            return "Not available"
        return (
            f"## Drug Information: {', '.join(openfda.get('brand_name', [drug_name])[:3])}\n\n"
            f"**Generic Name:** {', '.join(openfda.get('generic_name', ['Unknown'])[:2])}\n"
            f"**Drug Class:** {', '.join(openfda.get('pharm_class_epc', ['Unknown'])[:3])}\n\n"
            f"**Mechanism of Action:**\n{get_section(['clinical_pharmacology', 'mechanism_of_action'])}\n\n"
            f"**Indications:**\n{get_section(['indications_and_usage'])}\n\n"
            f"**Warnings:**\n{get_section(['warnings', 'warnings_and_cautions'], 400)}\n\n"
            f"**Adverse Reactions:**\n{get_section(['adverse_reactions'], 400)}\n\n"
            f"*Source: FDA Drug Label Database*"
        )
    except Exception as e:
        return f"Drug info lookup failed: {str(e)}"


TOOLS = [search_pubmed, search_web, lookup_drug_interactions, get_drug_info]

# ── State uses add_messages reducer — this is the key fix ──────────────────────
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

class MedicalResearchAgent:
    def __init__(self, openai_api_key: str, model: str = "gpt-4o-mini"):
        self.llm = ChatOpenAI(
            api_key=openai_api_key,
            model=model,
            temperature=0.2,
            max_tokens=2048,
        ).bind_tools(TOOLS)
        self.tool_node = ToolNode(TOOLS)
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        graph = StateGraph(AgentState)
        graph.add_node("agent", self._agent_node)
        graph.add_node("tools", self.tool_node)
        graph.set_entry_point("agent")
        graph.add_conditional_edges("agent", self._should_continue, {
            "tools": "tools",
            "end": END
        })
        graph.add_edge("tools", "agent")
        return graph.compile()

    def _agent_node(self, state: AgentState) -> dict:
        messages = state["messages"]
        # Inject SystemMessage fresh at call time, never stored in state
        messages_to_send = [SystemMessage(content=SYSTEM_PROMPT)] + list(messages)
        response = self.llm.invoke(messages_to_send)
        # Return only the new message — add_messages reducer appends it correctly
        return {"messages": [response]}

    def _should_continue(self, state: AgentState) -> str:
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tools"
        return "end"

    async def run(self, messages: List[dict]) -> dict:
        lc_messages: List[BaseMessage] = []
        for m in messages:
            if m["role"] == "user":
                lc_messages.append(HumanMessage(content=m["content"]))
            elif m["role"] == "assistant" and m.get("content"):
                lc_messages.append(AIMessage(content=m["content"]))

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: self.graph.invoke({"messages": lc_messages})
        )

        tools_used = []
        for msg in result["messages"]:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    tools_used.append(tc["name"])

        final_content = ""
        for msg in reversed(result["messages"]):
            if isinstance(msg, AIMessage) and msg.content:
                final_content = msg.content
                break

        return {"response": final_content, "tools_used": tools_used}

    async def stream(self, messages: List[dict]):
        result = await self.run(messages)
        words = result["response"].split(" ")
        for i, word in enumerate(words):
            yield {"type": "token", "content": word + (" " if i < len(words) - 1 else "")}
            await asyncio.sleep(0.02)
        if result["tools_used"]:
            yield {"type": "tools_used", "tools": result["tools_used"]}