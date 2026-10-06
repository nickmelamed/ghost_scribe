# Prompt-held-out evaluation

Stage 1 and Stage 2 were retrained without the held-out prompts, using the same hyperparameters as the main run. The escalation bands were recalibrated on val rows of the seen prompts. The test rows below come from the cleaned test split.

Held-out prompts (seed 42): Mandatory extracurricular activities, Car-free cities, "A Cowboy Who Rode the Waves".
Train rows kept 23408 of 29846. Stage 2 trained on 4387 escalated rows.

| Metric | Held-out prompts | Seen prompts | Main run, all prompts |
|---|---|---|---|
| Test rows | 1208 | 4891 |  |
| Escalation rate | 0.2930 | 0.2065 | 0.2172 |
| Stage-1-only accuracy | 0.9280 | 0.9636 | 0.9611 |
| Stage-1+2 accuracy | 0.9884 | 0.9947 | 0.9936 |
| Escalated rows | 354 | 1010 | 1325 |
| Escalated, Stage-1 accuracy | 0.7881 | 0.8426 | 0.8438 |
| Escalated, Stage-2 accuracy | 0.9944 | 0.9931 | 0.9932 |
| Escalated, Stage-2 95% interval | 0.9796 to 0.9984 | 0.9858 to 0.9966 |  |
| Human FPR, Stage-1 only | 0.0477 | 0.0340 |  |
| Human FPR, combined | 0.0087 | 0.0032 |  |
| Dangerous errors (said human, was AI) | 1 | 5 |  |
| Stage-2 fallbacks | 0 | 0 |  |

## Escalated rows by prompt

| Group | Prompt | Escalated rows | Stage-1 accuracy | Stage-2 accuracy |
|---|---|---|---|---|
| held out | "A Cowboy Who Rode the Waves" | 97 | 0.7938 | 0.9897 |
| held out | Car-free cities | 169 | 0.7811 | 1.0000 |
| held out | Mandatory extracurricular activities | 88 | 0.7955 | 0.9886 |
| seen | Cell phones at school | 44 | 0.9773 | 1.0000 |
| seen | Community service | 76 | 0.8289 | 1.0000 |
| seen | Distance learning | 97 | 0.8660 | 0.9897 |
| seen | Does the electoral college work? | 183 | 0.8251 | 1.0000 |
| seen | Driverless cars | 45 | 0.9111 | 1.0000 |
| seen | Exploring Venus | 90 | 0.8556 | 1.0000 |
| seen | Facial action coding system | 100 | 0.6900 | 0.9900 |
| seen | Grades for extracurricular activities | 53 | 0.9434 | 1.0000 |
| seen | Phones and driving | 44 | 0.8409 | 1.0000 |
| seen | Seeking multiple opinions | 145 | 0.8276 | 0.9862 |
| seen | Summer projects | 77 | 0.8442 | 0.9610 |
| seen | The Face on Mars | 56 | 0.9107 | 1.0000 |

## Reading this

This is one split of 3 of 15 prompts, so a different draw could give a different result. The intervals are Wilson 95% intervals on the escalated rows only. The main run column uses models trained on all prompts and so is not a like-for-like comparison. The seen-prompts column uses the same retrained models as the held-out column, so the difference between them reflects prompt shift.
