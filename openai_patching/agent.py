
from __future__ import annotations
import os
import json
from typing import Dict, Any, List
from dotenv import load_dotenv
from openai import OpenAI
from event import Event, EventState
from document_ranker import DocumentRanker

load_dotenv()


class Agent:
    def __init__(self, metadata_path: str = "labeled_files/metadata.json"):

        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata: Dict[str, Dict[str, Any]] = json.load(f)

        self.ranker = DocumentRanker(self.metadata)
        self.state = EventState()

        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = "gpt-5"

        self.system = (
            "You are a document routing assistant for P&ID drawings.\n"
            "\n"
            "You have access to a tool called rank_documents(query_items) that returns the most relevant page IDs.\n"
            "The dataset is METADATA (loaded from metadata.json), mapping page_id -> record.\n"
            "\n"
            "IMPORTANT: You do NOT see METADATA directly. You only see the transcript.\n"
            "\n"
            "Your job is to extract a SMALL list of meaningful query items from the user's request.\n"
            "Query items should be:\n"
            "- equipment tags (e.g., HX-101, P-201A)\n"
            "- line/service phrases (e.g., Cooling Water Supply, Cooling Water Return)\n"
            "- instrument tags if present (e.g., FIC-101)\n"
            "\n"
            "DO NOT include filler words like: show me, page, for, the, to, return me, etc.\n"
            "DO NOT include 'P&ID' as a query item.\n"
            "\n"
            "=== TRANSCRIPT FORMAT YOU WILL RECEIVE ===\n"
            "- User inputs:\n"
            "  [USER] <question>\n"
            "- Your reasoning:\n"
            "  [AGENT reason_step] <summary>\n"
            "- When you call the ranker:\n"
            "  [AGENT rank_documents REQUEST] {\"query_items\": [\"...\"]}\n"
            "  [TOOL rank_documents RESULT] {\"ok\": true, \"doc_ids\": [\"0007\", ...]}\n"
            "- When you finish:\n"
            "  [AGENT final_answer] <answer>\n"
            "\n"
            "=== TOOLS YOU CAN CALL ===\n"
            "1) reason_step(summary, next)\n"
            "   - next must be 'rank_documents' or 'final_answer'\n"
            "2) rank_documents(query_items)\n"
            "3) final_answer(answer)\n"
            "\n"
            "=== STRICT PROTOCOL (MUST FOLLOW) ===\n"
            "A) ALWAYS call reason_step first.\n"
            "B) ALWAYS set next='rank_documents' after reason_step.\n"
            "C) After you see [TOOL rank_documents RESULT], call reason_step again.\n"
            "D) Then call final_answer listing only the doc_ids (one per line).\n"
            "\n"
            "Keep the reasoning summary short and concrete."
        )

        self.tools = [
            {
                "type": "function",
                "name": "reason_step",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "summary": {"type": "string"},
                        "next": {"type": "string", "enum": ["rank_documents", "final_answer"]},
                    },
                    "required": ["summary", "next"],
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "rank_documents",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query_items": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Meaningful P&ID query items (equipment tags, service phrases, instrument tags)."
                        }
                    },
                    "required": ["query_items"],
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "final_answer",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "answer": {"type": "string"},
                    },
                    "required": ["answer"],
                    "additionalProperties": False,
                },
            },
        ]

    def rank_documents(self, query_items: List[str]) -> dict:
        try:
            doc_ids = self.ranker.rank(query_items)
            return {"ok": True, "doc_ids": doc_ids, "query_items": query_items}
        except Exception as e:
            return {"ok": False, "error": str(e), "query_items": query_items}

    def loop(self, user_input: str) -> str:
        def force(name: str):
            return {"type": "function", "name": name}

        self.state.update_state(Event("USER", "user_input", user_input, None))
        expected = "reason_step"

        while True:
            context = self.state.render()

            resp = self.client.responses.create(
                model=self.model,
                input=[
                    {"role": "system", "content": self.system},
                    {"role": "user", "content": context},
                ],
                tools=self.tools,
                tool_choice=force(expected),
                parallel_tool_calls=False,
            )

            tool_call = None
            for item in resp.output:
                if getattr(item, "type", None) == "function_call":
                    tool_call = item
                    break

            if tool_call is None:
                raise ValueError("No function_call returned by model.")

            name = tool_call.name
            if name != expected:
                raise ValueError(f"Incorrect tool call. Received {name}. Expected {expected}")

            if name == "reason_step":
                args = json.loads(tool_call.arguments)
                summary = args["summary"]
                next_tool = args["next"]

                print(f"[reason_step] REASONING:\n{summary}\n")
                self.state.update_state(Event("AGENT", "reason_step", summary, None))
                expected = next_tool
                continue

            if name == "rank_documents":
                args = json.loads(tool_call.arguments)
                query_items = args["query_items"]

                print("[rank_documents] REQUEST:\n" + json.dumps(args, ensure_ascii=False) + "\n")
                tool_payload = self.rank_documents(query_items)
                print(f"[TOOL rank_documents RESULT] => \n{tool_payload}\n")

                self.state.update_state(Event("AGENT", "rank_documents", args, tool_payload))
                expected = "reason_step"
                continue

            if name == "final_answer":
                args = json.loads(tool_call.arguments)
                answer = args["answer"]
                self.state.update_state(Event("AGENT", "final_answer", answer, None))
                return answer


if __name__ == "__main__":
    agent = Agent(metadata_path="labeled_files/metadata.json")

    print("\nAsk about the document set. Type 'exit' to quit.\n")
    while True:
        try:
            q = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not q:
            continue
        if q.lower() == "exit":
            print("Exiting.")
            break

        answer = agent.loop(q)
        print(f"\nAgent> {answer}\n")
