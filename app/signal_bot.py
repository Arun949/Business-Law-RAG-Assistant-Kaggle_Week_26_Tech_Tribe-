"""
signal_bot.py — Signal messenger bot for the Multi-Agent RAG pipeline.

Setup:
  1. Install Docker Desktop (https://www.docker.com/products/docker-desktop)
  2. Run signal-cli-rest-api container:
       docker run -d --name signal-api -p 8080:8080 ^
         -v signal-cli-config:/home/.local/share/signal-cli ^
         -e MODE=normal ^
         bbernhard/signal-cli-rest-api
  3. Register your Signal number via the REST API:
       curl -X POST "http://localhost:8080/v1/register/+33766977633"
       curl -X POST "http://localhost:8080/v1/register/+33766977633/verify/CODE"
     (replace CODE with the SMS verification code you receive)
  4. Set env vars in .env:
       SIGNAL_PHONE_NUMBER=+33766977633
       SIGNAL_SERVICE=http://localhost:8080
  5. pip install signalbot
  6. Run:  python app/signal_bot.py
"""

import os
import sys
import asyncio
import logging

sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
from signalbot import SignalBot, Command, Context
from signalbot.api import ConnectionMode

from agents import run_pipeline
from llm_judge import judge_faithfulness

load_dotenv()

SIGNAL_PHONE   = os.getenv("SIGNAL_PHONE_NUMBER", "").replace(" ", "")
SIGNAL_SERVICE = os.getenv("SIGNAL_SERVICE", "http://localhost:8080")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("signal_bot")


# ── RAG Command — answers any question ──────────────────────────────────────

class AskCommand(Command):
    """Handle every incoming message as a RAG query."""

    async def handle(self, ctx: Context):
        query = ctx.message.text.strip() if ctx.message.text else ""
        if not query:
            return

        log.info(f"Query received: {query}")

        # Handle special commands
        if query.lower() in ("/help", "/start"):
            await ctx.send(
                "Hi! I'm the Business Law RAG Assistant.\n\n"
                "Just send me any question about business law and I'll "
                "search the corpus + live web to answer.\n\n"
                "Commands:\n"
                "  /help  - show this message\n"
                "  /cost  - show session API cost"
            )
            return

        if query.lower() == "/cost":
            try:
                import json
                with open("data/cost_tracker.json") as f:
                    costs = json.load(f)
                await ctx.send(
                    f"Total API cost: ${costs.get('total_usd', 0):.4f}\n"
                    f"Total calls: {costs.get('calls', 0)}"
                )
            except Exception:
                await ctx.send("Cost tracker not available yet.")
            return

        # Run the 3-agent pipeline in a thread to avoid blocking
        try:
            await ctx.send("Processing your query...")

            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, run_pipeline, query)
            final_answer, corpus_chunks, web_results, query_cost, corpus_context = result

            # Format a clean response for Signal
            response_parts = [final_answer]

            # Only add sources if the answer is not a refusal / out-of-scope message
            refusal_phrases = (
                "i don't have information", "not in the available documents",
                "outside the scope", "i can only answer", "no information",
                "i can only answer questions related to",
            )
            is_refusal = any(p in final_answer.lower() for p in refusal_phrases)

            if not is_refusal:
                sources = []
                if corpus_chunks:
                    pages = sorted(set(str(c["page"]) for c in corpus_chunks[:3]))
                    sources.append(f"Corpus: p.{', '.join(pages)}")
                if web_results:
                    for r in web_results[:2]:
                        sources.append(f"Web: {r.get('title', '')[:40]}")
                if sources:
                    response_parts.append("\n---\nSources: " + " | ".join(sources))

            response_parts.append(f"\n[Cost: ${query_cost:.4f}]")

            await ctx.send("\n".join(response_parts))
            log.info(f"Answered query, cost=${query_cost:.4f}")

        except Exception as e:
            log.error(f"Pipeline error: {e}", exc_info=True)
            await ctx.send(f"Sorry, something went wrong: {str(e)[:200]}")


# ── Bot setup ────────────────────────────────────────────────────────────────

def main():
    if not SIGNAL_PHONE:
        print("ERROR: Set SIGNAL_PHONE_NUMBER in .env")
        print("  Example: SIGNAL_PHONE_NUMBER=+33766977633")
        sys.exit(1)

    config = {
        "signal_service": SIGNAL_SERVICE,
        "phone_number": SIGNAL_PHONE,
        "connection_mode": ConnectionMode.HTTP_ONLY,
    }

    bot = SignalBot(config)
    bot.register(AskCommand())

    print(f"Signal bot starting for {SIGNAL_PHONE}...")
    print(f"Connecting to signal-cli-rest-api at {SIGNAL_SERVICE}")
    print("Send any message to the bot's Signal number to query the RAG pipeline.")
    print("Press Ctrl+C to stop.\n")

    bot.start()


if __name__ == "__main__":
    main()
