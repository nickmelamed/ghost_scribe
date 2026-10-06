# Drift report across generation-source batches

Each generation source stands in for a step in time. This is a substitution, not real chronological drift. AI batches are compared with the AI essays in train and the human control with the human essays in train, using the round-1 models. PSI above 0.1 is moderate and above 0.2 is major. The noise column is the 95th percentile of the score PSI for a same-size bootstrap sample of the reference.

| Batch | Kind | Rows | Score PSI | Level | Noise p95 | Score KS | Features over 0.1 | Features over 0.2 | Score PSI vs own train rows | Auto-clear | Escalate | Auto-flag |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| human (control) | human control | 3809 | 0.006 | none | 0.005 | 0.029 | 0 | 0 | n/a | 78.5% | 21.1% | 0.4% |
| NousResearch/Llama-2-7b-chat-hf | seen AI source | 60 | 0.089 | none | 0.326 | 0.110 | 22 | 22 | 0.106 | 0.0% | 18.3% | 81.7% |
| chat_gpt_moth | seen AI source | 364 | 0.042 | none | 0.043 | 0.050 | 18 | 13 | 0.052 | 0.8% | 19.2% | 79.9% |
| cohere-command | seen AI source | 52 | 1.880 | major | 0.408 | 0.489 | 21 | 18 | 0.295 | 7.7% | 55.8% | 36.5% |
| darragh_claude_v6 | seen AI source | 150 | 0.150 | moderate | 0.128 | 0.173 | 23 | 20 | 0.158 | 0.0% | 17.3% | 82.7% |
| darragh_claude_v7 | seen AI source | 150 | 0.360 | major | 0.128 | 0.238 | 23 | 21 | 0.096 | 0.7% | 8.7% | 90.7% |
| kingki19_palm | seen AI source | 208 | 1.371 | major | 0.078 | 0.367 | 21 | 19 | 0.034 | 0.0% | 43.8% | 56.2% |
| llama2_chat | seen AI source | 363 | 0.052 | none | 0.044 | 0.056 | 23 | 21 | 0.025 | 0.0% | 19.0% | 81.0% |
| mistral7binstruct_v1 | seen AI source | 363 | 0.100 | moderate | 0.044 | 0.145 | 19 | 17 | 0.048 | 0.0% | 13.5% | 86.5% |
| mistral7binstruct_v2 | seen AI source | 363 | 0.076 | none | 0.044 | 0.116 | 19 | 13 | 0.041 | 0.8% | 28.9% | 70.2% |
| mistralai/Mistral-7B-Instruct-v0.1 | seen AI source | 60 | 0.197 | moderate | 0.326 | 0.158 | 23 | 23 | 0.109 | 0.0% | 20.0% | 80.0% |
| palm-text-bison1 | seen AI source | 52 | 3.236 | major | 0.408 | 0.582 | 22 | 22 | 0.238 | 5.8% | 71.2% | 23.1% |
| radek_500 | seen AI source | 75 | 0.412 | major | 0.257 | 0.209 | 23 | 23 | 0.116 | 0.0% | 8.0% | 92.0% |
| radekgpt4 | seen AI source | 30 | 0.947 | major | 1.565 | 0.193 | 23 | 23 | 0.215 | 0.0% | 10.0% | 90.0% |
| falcon_180b_v1 | unseen AI source | 1055 | 0.037 | none | 0.016 | 0.063 | 19 | 11 | n/a | 0.9% | 24.6% | 74.5% |
| llama_70b_v1 | unseen AI source | 1172 | 0.509 | major | 0.015 | 0.279 | 20 | 12 | n/a | 4.7% | 44.3% | 51.0% |

For an AI batch the auto-clear share is the share Stage 1 would clear with no review, so it is the miss rate. For the human control the auto-flag share is the false-flag rate.

## Top drifting features per batch

| Batch | Top three features by PSI |
|---|---|
| human (control) | word_count (0.01), char_count (0.01), noun_ratio (0.01) |
| NousResearch/Llama-2-7b-chat-hf | word_count (5.92), sentence_count (5.47), char_count (5.35) |
| chat_gpt_moth | sentence_count (4.19), char_count (3.12), word_count (2.98) |
| cohere-command | det_ratio (3.37), conj_ratio (1.43), adj_ratio (1.12) |
| darragh_claude_v6 | type_token_ratio (4.50), bigram_burstiness (3.15), avg_word_length (2.51) |
| darragh_claude_v7 | type_token_ratio (4.98), bigram_burstiness (3.57), avg_word_length (2.76) |
| kingki19_palm | flesch_reading_ease (4.03), avg_word_length (3.54), flesch_kincaid_grade (3.36) |
| llama2_chat | sentence_count (1.85), flesch_reading_ease (1.84), pronoun_ratio (1.83) |
| mistral7binstruct_v1 | type_token_ratio (2.43), bigram_burstiness (1.65), char_count (1.42) |
| mistral7binstruct_v2 | noun_ratio (0.82), type_token_ratio (0.65), sentence_count (0.60) |
| mistralai/Mistral-7B-Instruct-v0.1 | char_count (4.73), word_count (4.20), gunning_fog (3.39) |
| palm-text-bison1 | conj_ratio (2.94), bigram_burstiness (2.89), gunning_fog (2.60) |
| radek_500 | char_count (7.39), noun_ratio (6.23), word_count (5.93) |
| radekgpt4 | char_count (6.84), flesch_reading_ease (6.56), gunning_fog (6.40) |
| falcon_180b_v1 | sentence_count (0.83), avg_word_length (0.56), type_token_ratio (0.55) |
| llama_70b_v1 | type_token_ratio (1.24), bigram_burstiness (0.97), sentence_count (0.72) |

