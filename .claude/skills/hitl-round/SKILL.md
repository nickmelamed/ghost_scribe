---
name: hitl-round
description: Archive old review files, rebuild the review queue, launch the review app, then turn the reviews into labeled corrections. Only the owner starts this.
disable-model-invocation: true
---

Run one human review round. The owner does the reviewing. Nothing here
decides a case, and no verdict is final until a human confirms it.

1. Archive, never delete: move `hitl/reviews.jsonl` and
   `hitl/llm_reviews.jsonl`, if present, into
   `hitl/archive/<YYYY-MM-DD>/`. The archive stays gitignored. Show what
   was moved and ask before moving anything.
2. Rebuild the queue: `python -m stage2.build_review_queue`. Show the queue
   size and how many rows are AI-generated verdicts (always reviewed) versus
   gated human-written verdicts.
3. Tell the owner to run `streamlit run hitl/review_app.py` and review the
   queue. Do not review cases for them.
4. When the owner says the pass is done, check the count in
   `hitl/reviews.jsonl` against the queue size.
5. Run `python -m hitl.apply_corrections --source human`, and with
   `--source llm` only if an LLM pass was also run. Keep the two sources
   separate. Never report LLM reviews as human review.
6. Show how many reviewer labels differ from Stage 2, and how many carry the
   "possible false flag: non-native writing style" tag.
