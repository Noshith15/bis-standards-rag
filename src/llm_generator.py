"""
LLM Generator using Anthropic Claude API
Generates expert rationale from retrieved BIS standards
This is what makes this a TRUE RAG system - not just retrieval
"""

import os, re


def generate_rationale(query: str, standards: list) -> str:
    """
    Call Claude to generate expert compliance rationale.
    Falls back to template if API unavailable.
    """
    try:
        import urllib.request, json
        
        context = "\n\n".join([
            f"Standard: {s['standard_id']}\nTitle: {s['title']}\nRelevance: {s['rationale']}"
            for s in standards[:3]
        ])
        
        prompt = f"""You are a BIS (Bureau of Indian Standards) compliance expert helping Indian MSEs.

A business has asked: "{query}"

The RAG system retrieved these standards:
{context}

In 2-3 sentences, explain WHY these standards are relevant to this business and what key compliance requirements they should be aware of. Be specific, practical, and helpful. Do not make up standards not listed above."""

        payload = json.dumps({
            "model": "claude-sonnet-4-6",
            "max_tokens": 200,
            "messages": [{"role": "user", "content": prompt}]
        }).encode()

        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read())
            return data['content'][0]['text']
    except Exception:
        # Fallback: template rationale
        if standards:
            top = standards[0]
            return f"The primary applicable standard is {top['standard_id']} — {top['title']}. {top['rationale']}. Review all recommended standards to ensure full compliance."
        return "Please consult the retrieved standards for compliance requirements."
