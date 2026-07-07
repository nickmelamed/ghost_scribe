# DAIGT Dataset — Preprocessing Summary

Source: [thedrcat/daigt-v2-train-dataset](https://www.kaggle.com/datasets/thedrcat/daigt-v2-train-dataset), the community-compiled DAIGT dataset (human essays from the Persuade corpus + AI essays from a range of LLMs), used because the official competition training set lacks multi-model generation-source labels.

## Schema (post-processing)

| column | type | meaning |
|---|---|---|
| text | str | essay text |
| label | int | 0 = human, 1 = AI-generated |
| generation_source | str | "human", or the generating model's name |
| prompt_name | str | essay prompt/topic |
| char_count | int | character length of text |
| word_count | int | whitespace-split word count |

Dropped 3 row(s) with `source=="train_essays"` and `label==1`: AI-generated but the specific generating model is not recorded in the source data, so no `generation_source` could be assigned.

## Held-out generation sources (unseen at initial training)

Standing in for "a new AI model appears later" (this dataset has no real timestamp axis — see CLAUDE.md). Reintroduced in Phase 6, used as the focal case in Phase 7 drift monitoring.

- `falcon_180b_v1`: 1055 rows
- `llama_70b_v1`: 1172 rows

## Generation-source counts, full dataset

| generation_source | label | count | held out |
|---|---|---|---|
| human | 0 | 27371 |  |
| chat_gpt_moth | 1 | 2421 |  |
| llama2_chat | 1 | 2421 |  |
| mistral7binstruct_v1 | 1 | 2421 |  |
| mistral7binstruct_v2 | 1 | 2421 |  |
| kingki19_palm | 1 | 1384 |  |
| llama_70b_v1 | 1 | 1172 | yes |
| falcon_180b_v1 | 1 | 1055 | yes |
| darragh_claude_v6 | 1 | 1000 |  |
| darragh_claude_v7 | 1 | 1000 |  |
| radek_500 | 1 | 500 |  |
| NousResearch/Llama-2-7b-chat-hf | 1 | 400 |  |
| mistralai/Mistral-7B-Instruct-v0.1 | 1 | 400 |  |
| cohere-command | 1 | 350 |  |
| palm-text-bison1 | 1 | 349 |  |
| radekgpt4 | 1 | 200 |  |

## Split sizes (seen pool only, held-out sources excluded)

| split | rows | human | AI |
|---|---|---|---|
| train | 29846 | 19159 | 10687 |
| val | 6396 | 4106 | 2290 |
| test | 6396 | 4106 | 2290 |
| held_out | 2227 | 0 | 2227 |

## Generation-source counts by split

| generation_source | train | val | test | held_out |
|---|---|---|---|---|
| NousResearch/Llama-2-7b-chat-hf | 280 | 60 | 60 | 0 |
| chat_gpt_moth | 1694 | 363 | 364 | 0 |
| cohere-command | 245 | 53 | 52 | 0 |
| darragh_claude_v6 | 700 | 150 | 150 | 0 |
| darragh_claude_v7 | 700 | 150 | 150 | 0 |
| falcon_180b_v1 | 0 | 0 | 0 | 1055 |
| human | 19159 | 4106 | 4106 | 0 |
| kingki19_palm | 969 | 207 | 208 | 0 |
| llama2_chat | 1695 | 363 | 363 | 0 |
| llama_70b_v1 | 0 | 0 | 0 | 1172 |
| mistral7binstruct_v1 | 1695 | 363 | 363 | 0 |
| mistral7binstruct_v2 | 1695 | 363 | 363 | 0 |
| mistralai/Mistral-7B-Instruct-v0.1 | 280 | 60 | 60 | 0 |
| palm-text-bison1 | 244 | 53 | 52 | 0 |
| radek_500 | 350 | 75 | 75 | 0 |
| radekgpt4 | 140 | 30 | 30 | 0 |
