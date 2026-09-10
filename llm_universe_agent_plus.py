"""
LLM Universe Agent ++ (LangGraph + Tool-Calling, Screener + Fundamentals + News)
-------------------------------------------------------------------------------
- Input: list of symbols (US/KR mixed OK, e.g. ["AAPL","NVDA","005930.KS"]).
- Graph:
1) node_screener : compute momentum/trend/valuation metrics + score (weekly screener style)
2) node_funda_news : fetch key fundamentals (yfinance.info) + recent headlines (yfinance.news)
3) node_decide : LLM tool-calling agent that reads everything and outputs final picks JSON


- Stop/Take guidance: agent is instructed to use 5% stop-loss and 5% take-profit
- Tools: price_now, history_brief, fundamentals_tool, news_tool (for on-demand re-check)


Dependencies:
pip install -U "langchain>=0.2.17" "langchain-openai>=0.2.0" "langgraph>=0.2.39" \
"yfinance>=0.2.40" "pandas>=2.2" "numpy>=1.26,<2" "typing_extensions>=4.10"
# optional deepseek
# pip install -U langchain-deepseek


Env:
export OPENAI_API_KEY=... # or DEEPSEEK_API_KEY
# If routing ChatOpenAI to DeepSeek's API
# export OPENAI_BASE_URL=https://api.deepseek.com
"""

from __future__ import annotations
import os
import json
from typing import Dict, List, Optional, Annotated, Any
from typing_extensions import TypedDict


import numpy as np
import pandas as pd
import yfinance as yf

from datetime import datetime, timezone, timedelta


# LangChain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool, Tool

from langchain_openai import ChatOpenAI

# optional DeepSeek native
try:
    from langchain_deepseek import ChatDeepSeek # type: ignore
    _HAS_DEEPSEEK = True
except Exception:
    ChatDeepSeek = None # type: ignore
    _HAS_DEEPSEEK = False


# LangGraph
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# -----------------------------------------------------------------------------
# Helpers / indicators
# -----------------------------------------------------------------------------


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period, min_periods=period).mean()
    avg_loss = loss.rolling(period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def pct(a: float, b: float) -> Optional[float]:
    try:
        return float(a / b - 1.0)
    except Exception:
        return None
    
    
def inv_rank(series: pd.Series) -> pd.Series:
    """Rank higher = better (descending), map to [0,1]."""
    return series.rank(pct=True, ascending=False)


# -----------------------------------------------------------------------------
# Tools (agent can call to re-check)
# -----------------------------------------------------------------------------
@tool
def price_now(ticker: str) -> str:
    """Return latest close for ticker as JSON: {ticker, close, asof}."""
    t = ticker.strip().upper()
    tk = yf.Ticker(t)
    hist = tk.history(period="5d")
    if hist.empty:
        return json.dumps({"ticker": t, "error": "no price data"})
    last = float(hist["Close"].iloc[-1])
    return json.dumps({"ticker": t, "close": last, "asof": str(hist.index[-1])})

@tool
def history_brief(ticker: str, period: str = "6mo") -> str:
    """Return SMA20/50/200, RSI14, 1M%, 3M% as JSON for ticker."""
    t = ticker.strip().upper()
    df = yf.Ticker(t).history(period=period, interval="1d")
    if df.empty or "Close" not in df:
        return json.dumps({"ticker": t, "error": "no historical data"})
    close = df["Close"].dropna()
    out = {
        "ticker": t,
        "last_close": float(close.iloc[-1]),
        "sma20": float(close.rolling(20).mean().iloc[-1]) if len(close) >= 20 else None,
        "sma50": float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else None,
        "sma200": float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else None,
        "rsi14": float(rsi(close, 14).iloc[-1]) if len(close) >= 14 else None,
        "ret1m_pct": float(pct(close.iloc[-1], close.iloc[-21]) * 100) if len(close) > 21 else None,
        "ret3m_pct": float(pct(close.iloc[-1], close.iloc[-63]) * 100) if len(close) > 63 else None,
        }
    return json.dumps(out)

@tool
def fundamentals_tool(ticker: str) -> str:
    """Return key fundamentals as JSON using yfinance.info (best-effort)."""
    t = ticker.strip().upper()
    info = {}
    try:
        info = yf.Ticker(t).info or {}
    except Exception:
        pass
    keys = [
    "sector","industry","marketCap","beta","trailingPE","forwardPE","priceToBook",
    "profitMargins","operatingMargins","returnOnEquity","grossMargins","ebitdaMargins",
    "revenueGrowth","earningsGrowth","totalDebt","totalCash","currentRatio","dividendYield","payoutRatio"
    ]
    out = {k: info.get(k) for k in keys}
    return json.dumps({"ticker": t, "fundamentals": out})
    
@tool
def news_tool(
    ticker: str,
    limit: int = 8,
    since_days: int = 14,     # 최근 N일만
    include_links: bool = True
) -> str:
    """
    Return recent headlines for a ticker as JSON:
    {
    "ticker": "AAPL",
    "headlines": [
    {"title": "...", "source": "...", "time": "2025-10-29T12:34:56Z", "link": "..."},
    ...
    ]
    }
    """
    t = ticker.strip().upper()
    headlines = []
    try:
        items = getattr(yf.Ticker(t), "news", None) or []
        # 최신순 정렬(대개 이미 최신순이지만 안전하게)
        items = sorted(items, key=lambda n: n.get("providerPublishTime", 0), reverse=True)

        cutoff = None
        if since_days is not None and since_days > 0:
            cutoff = datetime.now(timezone.utc) - timedelta(days=since_days)

        for n in items:
            title = n.get("title") or n.get("content") or ""
            src   = n.get("publisher") or n.get("source") or ""
            ts    = n.get("providerPublishTime")  # epoch seconds
            link  = n.get("link") or n.get("url") or ""

            if not title:
                continue

            # 날짜 필터
            if cutoff and ts:
                dt = datetime.fromtimestamp(ts, tz=timezone.utc)
                if dt < cutoff:
                    continue
            else:
                dt = datetime.fromtimestamp(ts, tz=timezone.utc) if ts else None

            item = {
                "title": title,
                "source": src or None,
                "time": dt.isoformat().replace("+00:00", "Z") if dt else None,
            }
            if include_links:
                item["link"] = link or None

            headlines.append(item)
            if len(headlines) >= limit:
                break
    except Exception:
        pass
    
    return json.dumps({"ticker": t, "headlines": headlines}, ensure_ascii=False)


TOOLS: List[Tool] = [price_now, history_brief, fundamentals_tool, news_tool]

# -----------------------------------------------------------------------------
# LLM factory
# -----------------------------------------------------------------------------
def build_llm(model: Optional[str] = None, temperature: float = 0.2):
    if _HAS_DEEPSEEK and os.getenv("DEEPSEEK_API_KEY"):
        return ChatDeepSeek(model=model or os.getenv("DEEPSEEK_MODEL", "deepseek-chat"), temperature=temperature)
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("DEEPSEEK_BASE_URL") or None
    if base_url is None and os.getenv("DEEPSEEK_API_KEY"):
        base_url = "https://api.deepseek.com"
    return ChatOpenAI(api_key=api_key, base_url=base_url, model=model or os.getenv("OPENAI_MODEL", "gpt-4.1-mini"), temperature=temperature)

# -----------------------------------------------------------------------------
# State & Nodes
# -----------------------------------------------------------------------------
class State(TypedDict):
    messages: Annotated[list, add_messages]
    symbols: List[str]
    llm: object
    weights: Dict[str, float]
    top_k: int
    
    # Outputs
    screener: Dict[str, Any] # per-ticker metrics + scores
    fundamentals: Dict[str, Any] # per-ticker fundamentals
    headlines: Dict[str, Any] # per-ticker news
    decision: Dict[str, Any] # final JSON from agent
    

# Node 1: Screener metrics + score
def node_screener(state: State) -> State:
    syms = [s.strip().upper() for s in state["symbols"]]
    rows = []
    for t in syms:
        tk = yf.Ticker(t)
        try:
            df = tk.history(period="1y", interval="1d")
        except Exception:
            df = pd.DataFrame()
        if df.empty:
            rows.append({"ticker": t, "error": "no data"})
            continue

        close = df["Adj Close"].dropna() if "Adj Close" in df else df["Close"].dropna()
        vol = df.get("Volume", pd.Series(index=df.index, dtype=float)).fillna(0)
        dollar_vol20 = float((close * vol).rolling(20).mean().iloc[-1]) if len(close) >= 20 else None
        r1m = float(pct(close.iloc[-1], close.iloc[-21]) * 100) if len(close) > 21 else None
        r3m = float(pct(close.iloc[-1], close.iloc[-63]) * 100) if len(close) > 63 else None
        sma20_up = bool(close.iloc[-1] > close.rolling(20).mean().iloc[-1]) if len(close) >= 20 else None
        sma50_up = bool(close.iloc[-1] > close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else None
        sma200_up = bool(close.iloc[-1] > close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else None
        rsi14 = float(rsi(close, 14).iloc[-1]) if len(close) >= 14 else None
        
        # valuation proxy
        try:
            info = tk.info or {}
            pe = info.get("trailingPE") or info.get("forwardPE")
            pe = float(pe) if pe and pe > 0 else None
            
        except Exception:
            pe = None

        rows.append({
        "ticker": t,
        "last_close": float(close.iloc[-1]) if len(close) else None,
        "r1m_pct": r1m, "r3m_pct": r3m,
        "sma20_up": sma20_up, "sma50_up": sma50_up, "sma200_up": sma200_up,
        "rsi14": rsi14, "pe": pe, "dollar_vol20": dollar_vol20,
        })
        
        df = pd.DataFrame(rows).set_index("ticker") if rows else pd.DataFrame()
        if df.empty:
            state["screener"] = {"table": {}, "ranking": []}
            state["messages"].append(AIMessage(content="Screener: no data"))
            return state
        
        # scores
        mom_rank = 0.4 * inv_rank(df["r1m_pct"]) + 0.6 * inv_rank(df["r3m_pct"])
        trend_hits = df[["sma20_up","sma50_up","sma200_up"]].astype(float).sum(axis=1) # True=1.0
        trend_ratio = trend_hits / df[["sma20_up","sma50_up","sma200_up"]].notna().sum(axis=1).clip(lower=1)
        trend_rank = inv_rank(trend_ratio)
        inv_pe = 1 / df["pe"].replace({0: np.nan})
        val_rank = inv_rank(inv_pe).fillna(0.5)
        
        w = state["weights"]
        final = (
        (w.get("momentum",0.4) * mom_rank.fillna(0.5)) +
        (w.get("trend",0.4) * trend_rank.fillna(0.5)) +
        (w.get("valuation",0.2) * val_rank)
        )
        
        df["score_momentum"] = mom_rank
        df["score_trend"] = trend_rank
        df["score_valuation"] = val_rank
        df["score_final"] = final
        df = df.sort_values("score_final", ascending=False)
        
        # JSON-able
        table = df.round(4).replace({np.nan: None}).to_dict(orient="index")
        ranking = list(df.index)
        state["screener"] = {"table": table, "ranking": ranking}
        state["messages"].append(AIMessage(content=f"Screener computed for {len(df)} symbols."))
        return state
        
# Node 2: Fundamentals + News
def node_funda_news(state: State) -> State:
    fundamentals: Dict[str, Any] = {}
    headlines: Dict[str, Any] = {}
    for t in state["symbols"]:
        s = t.strip().upper()
        # fundamentals
        try:
            info = yf.Ticker(s).info or {}
        except Exception:
            info = {}
        keys = [
        "sector","industry","marketCap","beta","trailingPE","forwardPE","priceToBook",
        "profitMargins","operatingMargins","returnOnEquity","grossMargins","ebitdaMargins",
        "revenueGrowth","earningsGrowth","totalDebt","totalCash","currentRatio","dividendYield","payoutRatio"
        ]
        fundamentals[s] = {k: info.get(k) for k in keys}
        # news
        hs: List[str] = []
        try:
            news = getattr(yf.Ticker(s), "news", None)
            if news:
                for n in news[:8]:
                    title = n.get("title") or n.get("content") or ""
                    src = n.get("publisher") or n.get("source") or ""
                    if title:
                        hs.append(f"- {title} ({src})")
        except Exception:
            pass
        headlines[s] = hs
        
    state["fundamentals"] = fundamentals
    state["headlines"] = headlines
    state["messages"].append(AIMessage(content="Fundamentals & news collected."))
    return state

# Node 3: Agent decide using all data (with tool-calling)
def node_decide(state: State) -> State:
    from langgraph.prebuilt import create_react_agent
    
    llm = state["llm"]
    tools = TOOLS
    w = state["weights"]
    
    system_prompt = f"""
    You are a portfolio selection assistant.
    You get a screener table (momentum/trend/valuation scores), raw metrics, fundamentals, and headlines
    for multiple tickers. Rank them and recommend TOP_K to BUY now.
    Weights: momentum={w.get("momentum",0.4)}, trend={w.get("trend",0.4)}, valuation={w.get("valuation",0.2)}.

    Rules:
    - Use tools if you need to re-check specific numbers.
    - Prefer liquid names; if valuation missing, treat neutral.
    - Risk management: stop-loss -5%, take-profit +5% from entry.
    - Output STRICT JSON inside a code block with keys:
    {{
    "ranking": ["T1","T2",...],
    "top_k": ["T1",...],
    "reasons": {{"T1": "short evidence", ...}},
    "risk_notes": "short",
    "entry_exit": {{"T1": {{"stop":"-5%","take":"+5%"}}, ...}}
    }}
    Keep explanations brief and concrete, referencing both scores and fundamentals/news when relevant.
    """.strip()
    
    user_msg = f"""
    TOP_K = {state["top_k"]}
    SCREENER = {json.dumps(state["screener"], ensure_ascii=False)}
    FUNDAMENTALS = {json.dumps(state["fundamentals"], ensure_ascii=False)}
    HEADLINES = {json.dumps(state["headlines"], ensure_ascii=False)}
    """.strip()
    
    try:
        # Create ReAct agent with tools
        agent = create_react_agent(llm, tools, state_modifier=system_prompt)
        
        # Run agent
        result = agent.invoke({"messages": [HumanMessage(content=user_msg)]})
        
        # Extract final message
        final_messages = result.get("messages", [])
        text = ""
        if final_messages:
            last_msg = final_messages[-1]
            text = last_msg.content if hasattr(last_msg, 'content') else str(last_msg)
        
        # extract JSON from code block or raw
        import re
        m = re.search(r"```(?:json)?\s*\n(\{[\s\S]*?\})\s*\n```", text)
        raw = m.group(1) if m else text
        try:
            decision = json.loads(raw)
        except Exception:
            # Try to find JSON in the text
            m2 = re.search(r'\{[\s\S]*"ranking"[\s\S]*\}', text)
            if m2:
                try:
                    decision = json.loads(m2.group(0))
                except Exception:
                    decision = {"raw": text, "error": "Failed to parse JSON"}
            else:
                decision = {"raw": text, "error": "No JSON found"}
    except Exception as e:
        decision = {"error": str(e), "traceback": str(e.__class__.__name__)}
        
    state["decision"] = decision
    state["messages"].append(AIMessage(content=f"Decision prepared for top_k={state['top_k']}."))
    return state


# -----------------------------------------------------------------------------
# Build
# -----------------------------------------------------------------------------
def create_universe_graph():
    g = StateGraph(State)
    g.add_node("screener", node_screener)
    g.add_node("funda_news", node_funda_news)
    g.add_node("decide", node_decide)
    
    g.add_edge(START, "screener")
    g.add_edge("screener", "funda_news")
    g.add_edge("funda_news", "decide")
    g.add_edge("decide", END)
    return g.compile()


# -----------------------------------------------------------------------------
# Public API / CLI
# -----------------------------------------------------------------------------
def run_universe(symbols: List[str], top_k: int = 5, weights: Optional[Dict[str, float]] = None, model: Optional[str] = None, temperature: float = 0.2) -> Dict[str, Any]:
    weights = weights or {"momentum": 0.4, "trend": 0.4, "valuation": 0.2}
    app = create_universe_graph()
    llm = build_llm(model=model, temperature=temperature)
    init: State = {
        "messages": [HumanMessage(content=f"Analyze universe: {symbols}")],
        "symbols": symbols,
        "llm": llm,
        "weights": weights,
        "top_k": top_k,
        "screener": {},
        "fundamentals": {},
        "headlines": {},
        "decision": {},
    }
    
    state = app.invoke(init)
    return {"screener": state["screener"], "fundamentals": state["fundamentals"], "headlines": state["headlines"], "decision": state["decision"]}



def save_results_to_files(output: Dict[str, Any], symbols: List[str], output_dir: str = "output"):
    """Save analysis results to JSON, HTML, and Markdown files."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"analysis_{timestamp}"
    
    # 1) Save JSON
    json_path = os.path.join(output_dir, f"{base_name}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"✓ JSON saved: {json_path}")
    
    # 2) Save Markdown
    md_path = os.path.join(output_dir, f"{base_name}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# Stock Analysis Report\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Symbols:** {', '.join(symbols)}\n\n")
        
        # Decision summary
        decision = output.get("decision", {})
        f.write(f"## 📊 Investment Recommendation\n\n")
        if "top_k" in decision:
            f.write(f"### Top Picks\n")
            for ticker in decision.get("top_k", []):
                f.write(f"- **{ticker}**\n")
                reason = decision.get("reasons", {}).get(ticker, "N/A")
                f.write(f"  - Reason: {reason}\n")
                entry_exit = decision.get("entry_exit", {}).get(ticker, {})
                if entry_exit:
                    f.write(f"  - Stop Loss: {entry_exit.get('stop', 'N/A')}\n")
                    f.write(f"  - Take Profit: {entry_exit.get('take', 'N/A')}\n")
            f.write(f"\n")
        
        if "risk_notes" in decision:
            f.write(f"### Risk Notes\n{decision['risk_notes']}\n\n")
        
        # Screener table
        f.write(f"## 📈 Screener Results\n\n")
        screener = output.get("screener", {})
        ranking = screener.get("ranking", [])
        table = screener.get("table", {})
        if ranking and table:
            f.write(f"| Rank | Ticker | Score | R1M% | R3M% | RSI14 | P/E |\n")
            f.write(f"|------|--------|-------|------|------|-------|-----|\n")
            for i, ticker in enumerate(ranking, 1):
                data = table.get(ticker, {})
                score = data.get("score_final", "N/A")
                r1m = data.get("r1m_pct", "N/A")
                r3m = data.get("r3m_pct", "N/A")
                rsi = data.get("rsi14", "N/A")
                pe = data.get("pe", "N/A")
                
                # Format each value safely
                score_str = f"{score:.4f}" if isinstance(score, (int, float)) else str(score)
                r1m_str = f"{r1m:.2f}" if isinstance(r1m, (int, float)) else str(r1m)
                r3m_str = f"{r3m:.2f}" if isinstance(r3m, (int, float)) else str(r3m)
                rsi_str = f"{rsi:.2f}" if isinstance(rsi, (int, float)) else str(rsi)
                pe_str = f"{pe:.2f}" if isinstance(pe, (int, float)) else str(pe)
                
                f.write(f"| {i} | {ticker} | {score_str} | {r1m_str} | {r3m_str} | {rsi_str} | {pe_str} |\n")
        
        # Headlines
        f.write(f"\n## 📰 Recent News\n\n")
        headlines = output.get("headlines", {})
        for ticker in symbols:
            news_list = headlines.get(ticker.upper(), [])
            if news_list:
                f.write(f"### {ticker}\n")
                for headline in news_list[:5]:  # Top 5 news
                    f.write(f"{headline}\n")
                f.write(f"\n")
    
    print(f"✓ Markdown saved: {md_path}")
    
    # 3) Save HTML
    html_path = os.path.join(output_dir, f"{base_name}.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(f"""<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Stock Analysis Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 20px; border-radius: 8px; }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; border-left: 4px solid #3498db; padding-left: 10px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th {{ background-color: #3498db; color: white; padding: 12px; text-align: left; }}
        td {{ padding: 10px; border-bottom: 1px solid #ddd; }}
        tr:hover {{ background-color: #f5f5f5; }}
        .top-pick {{ background-color: #e8f5e9; padding: 15px; margin: 10px 0; border-radius: 5px; border-left: 4px solid #4caf50; }}
        .ticker {{ font-weight: bold; color: #2c3e50; font-size: 1.2em; }}
        .news-item {{ margin: 10px 0; padding: 10px; background-color: #f9f9f9; border-radius: 4px; }}
        .timestamp {{ color: #7f8c8d; font-size: 0.9em; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 Stock Analysis Report</h1>
        <p class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p><strong>Analyzed Symbols:</strong> {', '.join(symbols)}</p>
        
        <h2>💼 Investment Recommendation</h2>
""")
        
        decision = output.get("decision", {})
        for ticker in decision.get("top_k", []):
            reason = decision.get("reasons", {}).get(ticker, "N/A")
            entry_exit = decision.get("entry_exit", {}).get(ticker, {})
            f.write(f"""
        <div class="top-pick">
            <div class="ticker">{ticker}</div>
            <p><strong>Reason:</strong> {reason}</p>
            <p><strong>Stop Loss:</strong> {entry_exit.get('stop', 'N/A')} | <strong>Take Profit:</strong> {entry_exit.get('take', 'N/A')}</p>
        </div>
""")
        
        if "risk_notes" in decision:
            f.write(f"<p><strong>Risk Notes:</strong> {decision['risk_notes']}</p>\n")
        
        # Screener table
        f.write(f"\n<h2>📈 Screener Results</h2>\n<table>\n")
        f.write(f"<tr><th>Rank</th><th>Ticker</th><th>Score</th><th>R1M%</th><th>R3M%</th><th>RSI14</th><th>P/E</th></tr>\n")
        
        screener = output.get("screener", {})
        ranking = screener.get("ranking", [])
        table = screener.get("table", {})
        for i, ticker in enumerate(ranking, 1):
            data = table.get(ticker, {})
            score = data.get("score_final", "N/A")
            r1m = data.get("r1m_pct", "N/A")
            r3m = data.get("r3m_pct", "N/A")
            rsi = data.get("rsi14", "N/A")
            pe = data.get("pe", "N/A")
            
            # Format each value safely
            score_str = f"{score:.4f}" if isinstance(score, (int, float)) else str(score)
            r1m_str = f"{r1m:.2f}" if isinstance(r1m, (int, float)) else str(r1m)
            r3m_str = f"{r3m:.2f}" if isinstance(r3m, (int, float)) else str(r3m)
            rsi_str = f"{rsi:.2f}" if isinstance(rsi, (int, float)) else str(rsi)
            pe_str = f"{pe:.2f}" if isinstance(pe, (int, float)) else str(pe)
            
            f.write(f"<tr><td>{i}</td><td><strong>{ticker}</strong></td>"
                   f"<td>{score_str}</td>"
                   f"<td>{r1m_str}</td>"
                   f"<td>{r3m_str}</td>"
                   f"<td>{rsi_str}</td>"
                   f"<td>{pe_str}</td></tr>\n")
        
        f.write(f"</table>\n")
        
        # Headlines
        f.write(f"<h2>📰 Recent News</h2>\n")
        headlines = output.get("headlines", {})
        for ticker in symbols:
            news_list = headlines.get(ticker.upper(), [])
            if news_list:
                f.write(f"<h3>{ticker}</h3>\n")
                for headline in news_list[:5]:
                    f.write(f'<div class="news-item">{headline}</div>\n')
        
        f.write(f"""
    </div>
</body>
</html>
""")
    
    print(f"✓ HTML saved: {html_path}")
    return {"json": json_path, "markdown": md_path, "html": html_path}


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="LLM Universe Agent ++")
    p.add_argument("--symbols", type=str, required=True, help="Comma separated, e.g., AAPL,MSFT,005930.KS")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--w-momentum", type=float, default=0.4)
    p.add_argument("--w-trend", type=float, default=0.4)
    p.add_argument("--w-valuation", type=float, default=0.2)
    p.add_argument("--model", type=str, default=None)
    p.add_argument("--temp", type=float, default=0.2)
    p.add_argument("--output-dir", type=str, default="output", help="Output directory for results")
    p.add_argument("--save-files", action="store_true", help="Save results to files (JSON, HTML, MD)")
    args = p.parse_args()
    
    syms = [s.strip() for s in args.symbols.split(',') if s.strip()]
    weights = {"momentum": args.__dict__["w_momentum"], "trend": args.__dict__["w_trend"], "valuation": args.__dict__["w_valuation"]}
    
    out = run_universe(syms, top_k=args.top_k, weights=weights, model=args.model, temperature=args.temp)
    
    # Print to console
    print("\n== SCREENER ==\n", json.dumps(out["screener"], ensure_ascii=False, indent=2))
    print("\n== FUNDAMENTALS ==\n", json.dumps(out["fundamentals"], ensure_ascii=False, indent=2))
    print("\n== HEADLINES ==\n", json.dumps(out["headlines"], ensure_ascii=False, indent=2))
    print("\n== DECISION ==\n", json.dumps(out["decision"], ensure_ascii=False, indent=2))
    
    # Save to files if requested
    if args.save_files:
        print("\n" + "="*60)
        print("Saving results to files...")
        print("="*60)
        saved_files = save_results_to_files(out, syms, args.output_dir)
        print("\n✅ All files saved successfully!")
        print(f"📁 Output directory: {os.path.abspath(args.output_dir)}")