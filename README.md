# AI-Generated Text Detection Flywheel

A production-style ML flywheel applied to academic-integrity triage: a
classical stylometric model as a fast first-pass filter, a fine-tuned small
LLM as an escalation layer for ambiguous essays, a teacher-facing
human-in-the-loop review queue, and drift monitoring as new AI-generation
sources appear. A dedicated fairness check tracks false-positive rates on
non-native English writing — a well-documented failure mode of AI-text
detectors — across every round.

**Status: work in progress.** This README will be filled in fully once all
phases are complete (see `CLAUDE.md` for the build spec and phase list).

## Framing

This is a decision-support triage tool for teachers, not an automated
accusation system. Every escalated/flagged case requires human confirmation
before any action is taken.

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Full methodology, metrics, fairness-gap trend, drift report, and
ethics/limitations section will be added in Phase 8.
