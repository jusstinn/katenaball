# Adoption notes for `david/adopt-docs`

**Status, Nov 13, 13:25 CET.** Of the tokens in part 0, the reproduction, hash and training-time ones are filled
in (four machines gave one LightGBM member and one dressed CSV for AW2; `train` took 26 minutes on the PC). Only
`PENDING_JUSTIN_NAME` is left, three places. Nothing here is adopted: the documents on this branch describe AW2
as the submitted model so that an adoption needs no new text, and they must not reach main before both names
are given.

Second version, 2026-11-13, after the merge of `david/a-adopt-prep`. Both technical documents now describe
**AW2** as the submitted model: AW (new SkillCorner labels, possession-length weights) with a TCN trained
9,600 steps instead of 4,800.

- Part 0: every PENDING token, with its line.
- Part 1: what I could not make true from the decision log.
- Part 2: every place that changes if **A2** is adopted instead, with the replacement text.
- Part 3: other statements to check.
- Part 4: what the documents need outside these two files.
- Part 5: what was cut from the short document.

Every number in the two documents is in `docs/decision-log.md` on this branch (checked by script: all 12
intervals of the short document and all 67 of the long one are in the log verbatim).

---

## Part 0. PENDING tokens

| Token | File and line | The sentence it marks |
|---|---|---|
| `PENDING_JUSTIN_NAME` | `docs/technical-document.md:79` | "Both of us put our names to it (PENDING_JUSTIN_NAME); one had by then seen a first score of the new labels." |
| `PENDING_JUSTIN_NAME` | `docs/technical-document-full.md:414` | "Both gave them (PENDING_JUSTIN_NAME)." (section 3.5, rule 2) |
| `PENDING_JUSTIN_NAME` | `docs/technical-document-full.md:583-584` | "Both team members put their names to rule 2 and to AW2 before the cut-off of Nov 14, 18:00 CET (PENDING_JUSTIN_NAME)." (section 3.5, "Adopted") |
| `PENDING_AW2_REPRO` | `docs/technical-document.md:218-219` | "AW2's final model did the same (PENDING_AW2_REPRO)." (section 7, "Model bytes") |
| `PENDING_AW2_REPRO` | `docs/technical-document-full.md:588` | "AW2's all-30 model was checked the same way (PENDING_AW2_REPRO)." (section 3.5, "Reproduction") |
| `PENDING_AW2_REPRO` | `docs/technical-document-full.md:847-848` | "Its all-30 model was checked on the same four kinds of machine, and they gave one LightGBM member and one dressed CSV (PENDING_AW2_REPRO)." (section 7) |
| `HASH_PENDING_AW2_GBDT` | `docs/technical-document-full.md:849` | Expected on the reference PC: LightGBM |
| `HASH_PENDING_AW2_TCN` | `docs/technical-document-full.md:850` | TCN |
| `HASH_PENDING_AW2_BUNDLE` | `docs/technical-document-full.md:850` | bundle |
| `HASH_PENDING_AW2_DRESSED_CSV` | `docs/technical-document-full.md:851` | the dressed match's CSV |
| `PENDING_AW2_TRAIN_TIME` | `docs/technical-document.md:244-245` | "about 30 minutes to train (PENDING_AW2_TRAIN_TIME)" (section 8, "Cost") |
| `PENDING_AW2_TRAIN_TIME` | `docs/technical-document-full.md:918-919` | "AW2's TCN trains twice as long: about 30 minutes (PENDING_AW2_TRAIN_TIME)." (section 8) |

The short document quotes no hash, as before, so it has no `HASH_PENDING_*`. Each token sits in brackets
right after the clause it marks; removing a token and its brackets leaves a correct sentence.
`scripts/build_document.py` does not stop on these tokens (it stops on `TODO(` only), so search for
`PENDING` before printing. The three `TODO(after the real run)` markers are unchanged (short document,
lines 88, 228, 246).

If AW2's cross-OS run shows something else than "one LightGBM member and one CSV on four machines", the
three `PENDING_AW2_REPRO` sentences have to be rewritten, not just unmarked. AW2's LightGBM member should
be AW's (`55de693c7ea2...`): the registration says the longer TCN changes nothing else.

---

## Part 1. What I could not make true from the log

1. **AW2 against F3 has no interval in the log.** The task gave `+0.0618 [+0.0462, +0.0772]` on the 8 folds
   and `+0.0563 [+0.0455, +0.0666]` on SkillCorner. The result entry has only "+0.0618 (AW2) on the 8
   folds, 8 of 8" and "+0.0563 ... (18 of 20)". Both documents quote the point values and the fold counts
   and no interval for these two. Every interval they do quote is one step: AW against F3 (4,800 steps) and
   AW2 against AW. To add the two intervals, put them in the log first, then in: short document line 111
   ("Against F3, AW2 is +0.0618 on the 8 folds (8 of 8)") and long document section 1 (third bullet) and
   section 3.5 ("Against F3, AW2 stands at").
2. **AW2 under rule 1 is not in the log**, and neither is "AW2 against AW on the standing truth: +0.0040".
   The documents say rule 1 was read for AW (-0.0224, 8 of 28, at 4,800 steps) and that the longer TCN's
   rule does not read the old label, which is what its registration says. They do not say that the longer
   TCN leaves the rule-1 verdict unchanged: nothing in the log shows it.
3. **AW2 against F3 at severe noise on the 8 folds** (rule 2's "not worse at severe") is not in the log.
   The documents give AW's (+0.0613, 6 of 8) and AW2 against AW on the 28 folds at severe (+0.0077
   [+0.0047, +0.0106], 24 of 28). The short document says "Neither step is worse at severe noise".
4. **The by-truth differences of the weights at 9,600 steps (+0.024, -0.011, -0.006)** are not printed in
   the log. I quote the table's values instead (IDSSE 0.6814 against 0.6571, Metrica 0.7530 against 0.7642,
   SkillCorner 0.6657 against 0.6712), and in the long document the two differences the log does print
   (+0.0155 [+0.0032, +0.0275] on the 8 folds, -0.0055 [-0.0131, +0.0024] on SkillCorner).
5. **The training time's parts (LightGBM 165 s, TCN 716 s)** are not in the log; it has "`train` 15 minutes"
   for AW on the PC. The long document quotes the 15 minutes; "about 30 minutes" for AW2 is the estimate
   the task gave, marked `PENDING_AW2_TRAIN_TIME` in both.
6. **AW2's predicted possessions per true one by dataset** are not in the log, only 0.98 (8 folds) and 1.23
   (SkillCorner). Both documents use those two, with F3's 0.73 and 0.89 beside them.
7. **A2 against F3 has no interval in the log either** (the task gave [+0.0355, +0.0575] and [+0.0528,
   +0.0706]). Part 2 uses the log's point values (+0.0464, 8 of 8; +0.0617, 20 of 20).
8. **The log's own list of what is still open** (result entry of A2 and AW2) includes "the equality of the
   pipeline's training with the gate's on a fold". Neither document claims it for AW2. Both say the
   path-equality check and the stress check were run with F and F3 only.

---

## Part 2. Where the text differs if A2 is adopted

A2 is A (the new labels, no weights) with the 9,600-step TCN. Its numbers, all from the log: 0.6838 on the
8 folds (+0.0464 against F3, 8/8); 0.6712 on the 20 SkillCorner folds against the action truth (+0.0617,
20/20); 0.6748 on all 28. A2 against A: +0.0067 [+0.0040, +0.0093], 23/28; at severe +0.0082 [+0.0050,
+0.0114], 24/28; on the 8 folds +0.0075 [+0.0034, +0.0120], 7/8; on SkillCorner +0.0064 [+0.0031, +0.0095],
16/20. A against F3: 0.6763, +0.0389 [+0.0276, +0.0498], 8/8; 0.6647, +0.0553 [+0.0461, +0.0641], 20/20;
rule 1 -0.0088 [-0.0150, -0.0025], 10/28 (score 0.6092); at severe on the 8 folds +0.0363 [+0.0254,
+0.0470], 8/8. Predicted possessions per true one for A2: 0.80 (8 folds) and 1.02 (SkillCorner).

**A switch to A2 leaves the registered procedure.** The registered test names AW over A (+0.0160 [+0.0041,
+0.0282] on the 8 folds, 5/8), and the same reading at 9,600 steps is +0.0155 [+0.0032, +0.0275], 6/8.
The texts S9b and F10c below say so. The reason in them is drafted from the log's "For the two who
decide" paragraph and has to be the deciders' own. What A2 gains in exchange: A's gate was repeated on
the PC, so A has a second machine where AW has none.

### 2a. Short document, `docs/technical-document.md`

**S1. Section 1, last bullet.**

> - The submitted model, A2, is F3 trained on other labels for the 20 SkillCorner matches, with a TCN
>   trained twice as long. The old rule scores 20 of its 28 folds against the labels A2 no longer learns, so
>   the new labels were judged under a second rule, declared before any score. On the 8 folds whose truth
>   they leave alone A2 scores 0.6838 against F3's 0.6374; on the old SkillCorner labels the relabelled
>   model loses. It is a bet on how the organisers describe the ground truth (sections 3 and 7).

**S2. Section 2, "LightGBM heads".** Delete "In the team head a training frame weighs 1 / L, where L is the
length in seconds of its possession, clipped to [1, 30]: spellF1 counts a possession once, however long it
lasts."

**S3. Section 2, "TCN".** "with the same two heads and the same weights in its team loss." -> "with the same
two heads."

**S4. Section 3, "Rule 2" and "The longer TCN".** "AW is F3 with the new SkillCorner labels (section 6) and
the weights." -> "A is F3 with the new SkillCorner labels (section 6)."; "the per-frame label AW stops
training on, so it cannot pass AW" -> "... A stops training on, so it cannot pass A"; "AW2 is AW with 9,600
TCN steps for 4,800, registered before its score against AW" -> "A2 is A with 9,600 TCN steps for 4,800,
registered before its score against A".

**S5. Section 3, last validation bullet.** "AW2 changes training only." -> "A2 changes training only."

**S6. Section 3, results table: the last six rows.**

> | A: F3 with the new SkillCorner labels | 8 IDSSE and Metrica (rule 2) | 0.6763 | +0.0389 [+0.0276, +0.0498] | 8/8 |
> | | 20 SkillCorner, new labels | 0.6647 | +0.0553 [+0.0461, +0.0641] | 20/20 |
> | | 28, old label (rule 1: fails) | 0.6092 | -0.0088 [-0.0150, -0.0025] | 10/28 |
> | **A2: A with a TCN trained 9,600 steps** | 28, new labels (its rule) | **0.6748** | **+0.0067 [+0.0040, +0.0093]** | **23/28** |
> | | 8 IDSSE and Metrica | 0.6838 | +0.0075 [+0.0034, +0.0120] | 7/8 |
> | | 20 SkillCorner, new labels | 0.6712 | +0.0064 [+0.0031, +0.0095] | 16/20 |

**S7. Section 3, the three bullets under the table.**

> - **Against F3, A2 is +0.0464 on the 8 folds (8 of 8)** and +0.0617 on the 20 SkillCorner folds on the new
>   labels (20 of 20), where F3 scores 0.6374 and 0.6094. Neither step is worse at severe noise. A2's rule
>   does not read the old label.
> - **How they were run.** A's gate was run on GitHub's Ubuntu runners and again on our PC: the same
>   verdicts, every mean within 0.0002, 27 of 28 folds identical to the last digit. The fold models of A2
>   were fitted on the runners only, and scored on our PC against F3's cached fold models.

Third bullet, heading only: "Evidence that does not rest on A's labels."

**S8. Section 4, second table.** The weights row's decision: "Failed rule 1 on folds won. Tried again with
the new labels (AW, below)". The sister row becomes:

> | AW, AW2: A and A2 with weights by possession length. AT, ATW: A and AW with F3's alive/dead training | 8 folds: AW 0.6923, +0.0548 [+0.0399, +0.0697], 8/8; AW2 0.6993; AT 0.6725; ATW 0.6964 | All met rule 2 and failed rule 1. The registered tests named AW: ahead of A by +0.0160 [+0.0041, +0.0282], 5/8. Not adopted (section 7) |

**S9a. Section 6.** "The SkillCorner labels, new in AW." -> "new in A."; "So AW builds the label" -> "So A
builds the label"; "The output": "AW2 predicts 0.98 possessions per true one on the 8 IDSSE and Metrica
folds and 1.23 on SkillCorner's new labels" -> "A2 predicts 0.80 possessions per true one on the 8 IDSSE and
Metrica folds and 1.02 on SkillCorner's new labels".

**S9b. Section 7.** First bullet: "under rule 1 AW, at 4,800 TCN steps, is -0.0224 against F3 on the 28
folds, with 8 folds won" -> "under rule 1 A, at 4,800 TCN steps, is -0.0088 against F3 on the 28 folds, with
10 folds won". The bullet "The weights help one kind of truth" is replaced by:

> - **We left the registered choice.** Our registrations said that AW, A with weights by possession length,
>   would replace A if it was ahead on the 8 folds with the interval above 0. It was: +0.0160 [+0.0041,
>   +0.0282], 5 of 8. We kept A's line all the same, after the scores were known. With the longer TCN the
>   weights gain on the six IDSSE folds (0.6814 against 0.6571) and lose on the two Metrica folds (0.7530
>   against 0.7642) and on SkillCorner's new labels (0.6657 against 0.6712). The provided match's ground
>   truth comes from Metrica's corrected tracking.

**S9c. Section 7, "Model bytes".** "The final models of A and of AW did not show that: each gave ..." stays.
"AW2's final model did the same (PENDING_AW2_REPRO)." -> "A2's final model did the same
(PENDING_A2_REPRO)."

**S10. Section 8.** Nothing but the token: A2's TCN is the same size, so "about 30 minutes" stands with its
token.

Unchanged for A2: section 5 (the same labels and row counts), the label rule, "The new SkillCorner labels
are our construction", "The development estimate is optimistic", section 9. The A2 text is shorter than
the AW2 text, so it stays within 5 pages; build it again all the same.

### 2b. Long document, `docs/technical-document-full.md`

**F1. Section 1.** The three bullets from "The submitted model, AW2" to "AW2: a longer-trained TCN" become:

> - The submitted model, A2, is F3 with two changes to its training and none to its inference path. They
>   came in two gated steps (section 3.5).
> - **A: new labels.** The 20 SkillCorner matches are relabelled from SkillCorner's own action events. The
>   standing rule scores 20 of its 28 folds against the label A stops training on, so A was judged under a
>   second rule, declared before any score. On the 8 IDSSE and Metrica folds, whose truth is untouched, A
>   scores 0.6763 against F3's 0.6374: +0.0389 [+0.0276, +0.0498], 8 of 8 folds won. On the 20 SkillCorner
>   folds scored against the new labels it scores 0.6647 against 0.6094. Under the standing rule, with
>   SkillCorner scored on the old label, it loses 0.0088 on the 28 folds and wins 10 of them.
> - **A2: a longer-trained TCN.** A2 is A with a TCN trained 9,600 steps instead of 4,800. It was
>   registered before its score under its own rule and beat A on the 28 folds with the new labels on
>   SkillCorner: +0.0067 [+0.0040, +0.0093], 23 of 28 folds won. A2 scores 0.6838 on the 8 IDSSE and Metrica
>   folds (F3: 0.6374; +0.0464, 8 of 8), 0.6712 on the 20 SkillCorner folds against the new labels (F3:
>   0.6094; +0.0617, 20 of 20), and 0.6748 against 0.6174 on all 28.

In the last bullet "AW2 is a bet" -> "A2 is a bet".

**F2. Section 2.3.** "the labels AW and AW2 train on" -> "the labels A and A2 train on"; TCN bullet "F3 and
AW trained 213 columns for 4,800" -> "F3 and A trained ..."; delete the "Possession-length weights" bullet
(its definition moves to section 3.5, F6b).

**F3. Section 2.4.** "F, F3, AW and AW2 all train" -> "F, F3, A and A2 all train"; "AW and AW2 keep F3's rows"
-> "A and A2 keep F3's rows".

**F4. Section 3.1.** Rule 2 bullet: "AW replaces the SkillCorner training labels" -> "A replaces ..."; "the
label AW stops learning" -> "A stops learning". "F3, AW and AW2 have no bench read" -> "F3, A and A2".
"It was not repeated with AW or AW2" (twice) -> "with A or A2".

**F5. Section 3.4, last bullet.** "until AW2 replaced it ... the model AW and AW2 are measured against" ->
"until A2 replaced it ... the model A and A2 are measured against".

**F6a. Section 3.5, heading and first paragraph.**

> ### 3.5 After F3: SkillCorner labels from its action events, and a longer TCN (A and A2, 2026-11-12 to 13)
>
> A is F3 with other training labels on the 20 SkillCorner matches. A2, the submitted model, is A with a TCN
> trained 9,600 steps instead of 4,800. Features, members, the LightGBM size, seeds, blend, decoder and
> priors are F3's, and no line of the inference path changes. This section takes A first, with the three
> candidates registered beside it, and the longer TCN at its end.

**F6b. Section 3.5, "The weights (candidate AW)".** First sentence:

> **The weights (candidate AW, not adopted).** AW is A plus possession-length weights in the LightGBM team
> head and in the TCN's team loss (`katena.models.lenweight`): a training frame weighs 1 / L, with L the
> length in seconds of the constant-label run it sits in, in the T = 2 labels it trains on, clipped to
> [1, 30].

**F6c. Section 3.5, tables.** Move the bold in the four-candidate table from the AW cell to the A cell of
"Rule 2: 8 IDSSE + Metrica folds"; in the table "The longer TCN against its own base" from the AW2 cell to
the A2 cell; in the spellF1 table from the AW2 column to the A2 column.

**F6d. Section 3.5, after the table of tests.** "So the registered procedure names AW." becomes:

> So the registered procedure names AW. We kept A's line all the same, after these scores were known, and
> that leaves the registration. The reason is in the next table: the weights' gain is all on the six IDSSE
> folds, and A is ahead of AW on Metrica's kind of truth and on SkillCorner against the action truth.

**F6e. Section 3.5, bullets under the longer TCN's tables.** "Against F3, AW2 stands at +0.0618 ... A2
stands at +0.0464 (8 of 8) and +0.0617 (20 of 20)." -> put A2 first. "Rule 1 was read for AW, not for AW2.
AW is -0.0224 ..." -> "Rule 1 was read for A, not for A2. A is -0.0088 [-0.0150, -0.0025] against F3 on the
28 folds with the frame truth on SkillCorner (10 of 28)."

**F6f. Section 3.5, "Adopted" and "Reproduction".**

> **Adopted.** A met rule 2. The registered tests named AW; both team members chose A's line, the simpler
> model, and A2 passed its own rule against A, so the longer TCN is part of it. Both put their names to
> rule 2 and to A2 before the cut-off of Nov 14, 18:00 CET (PENDING_JUSTIN_NAME). A2 is the submitted model.
>
> **Reproduction.** (first sentence unchanged) A2's all-30 model was checked the same way
> (PENDING_A2_REPRO). Section 7 has the hashes.

**F7. Section 4.** The row "A: SkillCorner training labels ..." is replaced by an AW row, and the A2 row by
an AW2 row:

> | AW: A with possession-length weights in both team heads | Rule 2: 0.6923 on the 8 IDSSE + Metrica folds, +0.0548 [+0.0399, +0.0697], 8/8; 0.6586 on SkillCorner against the action truth, +0.0491 [+0.0387, +0.0597], 19/20. Rule 1: -0.0224, 8/28 | Met rule 2, failed rule 1. The registered test named it over A (+0.0160 [+0.0041, +0.0282] on the 8 folds, 5/8). Not adopted (section 3.5) |
> | AW2: AW with a TCN trained 9,600 steps instead of 4,800 | Against AW, 28 folds with the action truth on SkillCorner: +0.0071 [+0.0046, +0.0096], 20/28. 0.6993 on the 8 IDSSE + Metrica folds, 0.6657 on SkillCorner against the action truth, 0.6753 on all 28 | Passed its rule, with exactly the 20 folds it asks for. Not used: it can only replace AW |

The S2 row's last sentence: "reopened once, in AW's registration, for the action labels" -> "reopened once,
as AW, for the action labels, and not adopted".

**F8. Section 5.** Nothing.

**F9. Section 6.** "(new in AW)" -> "(new in A)"; "AW derives the label" -> "A derives the label"; in "The
output": "AW2 predicts 0.98 ... and 1.23" -> "A2 predicts 0.80 possessions per true one on the 8 IDSSE and
Metrica folds and 1.02"; "By dataset the log has AW's figures, at 4,800 steps: 0.90 on IDSSE, 1.17 on
Metrica and 1.24 on SkillCorner" -> "By dataset the log has A's figures, at 4,800 steps: 0.73 on IDSSE, 0.97
on Metrica and 1.02 on SkillCorner".

**F10a. Section 7, first bullet.** "Under rule 1 AW, at 4,800 TCN steps, is -0.0224 against F3 on the 28
folds with 8 folds won" -> "Under rule 1 A, at 4,800 TCN steps, is -0.0088 [-0.0150, -0.0025] against F3 on
the 28 folds with 10 folds won"; "AW2 was not read under rule 1" -> "A2 was not read under rule 1".

**F10b. Section 7, "The action labels are our construction".** "In AW and AW2 these rows" -> "In A and A2".

**F10c. Section 7, "The weights help one kind of truth".** Replace the bullet with:

> - **We left the registered choice.** The registrations said AW would replace A if it was ahead on the 8
>   folds with the interval above 0, and it was (+0.0160 [+0.0041, +0.0282], 5 of 8). We kept A's line,
>   after the scores were known. All of AW's lead comes from the six IDSSE folds (0.6725 against 0.6453). On
>   the two Metrica folds AW is behind A (0.7517 against 0.7692), and on SkillCorner against the action
>   truth too (0.6586 against 0.6647; at severe noise -0.0106 [-0.0185, -0.0028]). With the longer TCN the
>   picture is the same: AW2 minus A2 is +0.0155 [+0.0032, +0.0275] on the 8 folds (6 of 8), 0.6814 against
>   0.6571 on IDSSE and 0.7530 against 0.7642 on Metrica, and -0.0055 [-0.0131, +0.0024] on SkillCorner
>   against the action truth. On all 28 folds AW2 scores 0.6753 and A2 0.6748. The provided match's ground
>   truth comes from Metrica's corrected tracking. Two Metrica folds are thin evidence, and the action
>   truth is our own construction: if the hidden truth is of IDSSE's kind, the choice costs score.

**F10d. Section 7, "Rule 2 is weaker evidence".** "AW was also chosen among four related candidates on those
8 folds; the two registered tests are there to keep that choice honest." -> "A's line was kept among four
related candidates after their scores on those 8 folds were known, against the registered test (above)."
"not AW itself" -> "not A itself". Delete "And AW2 passed its own rule with exactly the 20 of 28 folds it
asks for." (A2 won 23.)

**F10e. Section 7, "AW and AW2 were fitted on one kind of machine".** Replace the first three sentences:

> - **A2 was fitted on one kind of machine.** The fold models of A2 (and of AW, AT, ATW and AW2) were
>   fitted on GitHub's Ubuntu runners only, and gated on the Windows PC against F3's cached members. A's
>   own gate was repeated on the PC (section 3.5): ...

The rest of the bullet stays.

**F10f. Section 7, small ones.** "in AW and AW2 the SkillCorner rows" -> "in A and A2"; "F3's, AW's or
AW2's, which keep seed 0" -> "F3's, A's or A2's"; "not repeated for AW or AW2" -> "for A or A2"; "repeat
AW2's gate on a second machine" -> "repeat A2's gate".

**F10g. Section 7, "Model bytes".** Swap the roles of A and AW in the paragraph "AW (section 3.5) was trained
on all 30 matches on four machines": A's runners (macOS 15 on Apple M1, Ubuntu 24.04 on AMD EPYC 9V45,
Windows Server 2025 on AMD EPYC 7763) and the PC; LightGBM `eb1030c8ec94...` on all four (alive head
`59bfcd31dc1e...`); the PC's TCN `209beece8bf2...` is its own, a fourth set, and so is its bundle; dressed
CSV `98b2919621653c99...` on all four; then "AW, A with the weights, was checked the same way ...:
LightGBM `55de693c7ea2...` and dressed CSV `280060af276f1665...` on all four." The AW2 paragraph becomes:

> A2, the submitted model, differs from A in the TCN's steps only. Its all-30 model was checked on the
> same four kinds of machine, and they gave one LightGBM member and one dressed CSV (PENDING_A2_REPRO).
> Expected on the reference PC (`docs/repro/expected-windows.json`): LightGBM `HASH_PENDING_A2_GBDT`, TCN
> `HASH_PENDING_A2_TCN`, bundle `HASH_PENDING_A2_BUNDLE`; the dressed match's CSV
> `HASH_PENDING_A2_DRESSED_CSV`.

In the LightGBM paragraph: "The fold models of AW and AW2 were fitted on the runners only, so they have no
such comparison." -> "A2's LightGBM member is A's, read from its cache."

**F11. Section 8.** "AW and AW2 also change only the training step. AW's all-30 model gave ..." -> "A and A2
also change only the training step. A's all-30 model gave ..."; "On the reference PC `train` took 15
minutes for AW" -> "for A" (the log has 15 minutes for both); "AW2's TCN trains twice as long" -> "A2's".

**F12. Section 9 table.** "SkillCorner action labels and length weights (AW, AW2)" -> "SkillCorner action
labels (A, A2); the length weights tested with them (AW, AW2)"; "spec `AW2`" -> "spec `A2`". Gates row: "AW
and its sister candidates" -> "A and its sister candidates".

---

## Part 3. Other statements to check

1. **Who agreed to rule 2, and when.** Marked `PENDING_JUSTIN_NAME`. Two more things in those sentences: "one
   had by then seen a first score of the new labels" (Justin was told A's interim gate on Nov 12; if David
   confirms after reading the result entries, "one" may have to become "both"), and "before the cut-off of
   Nov 14, 18:00 CET" in the long document.
2. **"The provided match's ground truth comes from Metrica's corrected tracking"** (both, section 7). From
   the log's entry on providers' documentation; `CHALLENGE.md` says "derived from corrected tracking data"
   without naming Metrica in that sentence.
3. **"M2, F and F3 passed it"** (short, section 3, rule 1). My wording. The short document's results table
   now starts at F; M2's 0.586 is in its caption and in the summary, M0's 0.280 in the summary, and M1's
   0.432 only in the long document.
4. **"a prediction that is perfect but 1 s late scores 0.65"** (long, section 3.5) is the log's quotation of
   `EVALUATION.md`; not checked against the file.
5. **The path-equality check and the stress check** were not run with AW or AW2; both documents say so.
6. **No second machine for AW or AW2's gates.** Both documents say so. The Mac's own A and AW are still to
   come, by the log.
7. **Disclosures are unchanged.** Older than this change: the short document says the assistant "saw no
   image and no file", the long one that it "saw the public label file, which is in the repository". And
   David's side of rule 2 was given by the assistant under his hand-over; no disclosure sentence covers
   that either way.
8. **Version line**: both say 2026-11-13.

## Part 4. Outside these two files (not touched; no code was changed)

- `src/katena/pipeline.py`: `SUBMISSION_SPEC = "F3"`. `make submit` trains F3 until it names `AW2` (or
  `A2`). Section 8 of both documents describes the command after that change.
- `docs/repro/expected-windows.json`, `docs/real-run.md` (lines 85 to 89), `docs/repro/windows-reference-F3.json`:
  F3's hashes and `model_F3`.
- `docs/REPRODUCE.md` (lines 36, 39, 119), `docs/ci.md` (line 157), the docstring of
  `scripts/make_submission.py`: they name F3.
- The decision log needs the adoption entry, and the AW2 intervals of part 1 if the documents are to
  quote them.

## Part 5. What was cut from the short document to stay at 5 pages

Draft build: 5 pages, page 5 is 90% used (between five and six text lines free; between three and four
with a long team name and the three markers filled in). That is the slack of the first version or a
little more. Removing the three bracketed tokens frees a few more words.

Cut in the first version (all still in the long document): the F3 detail bullets and most of the bake-off
bullet in section 3; the separate bench bullet (now one sentence in section 7); the M1 row; in "Tried",
the HSMM decoder and spell rescoring, the touch columns at 1,200 steps, the label-kind column (now a
clause in section 3), the switch-cost model, the second TCN seed, smooth-only training, the team head
without SkillCorner (now the second read in section 3), PathCRF and StatsBomb; the row "Who owns a dead
ball"; "0.495 to 0.653 across the conventions"; "Short possessions"; "Without `make` ..."; the fresh-clone
sentence; the labels "Pipeline." and "Models and decoder.".

Cut in this version, to make room for the longer TCN's rule and its three table rows:

- The two results tables are one ladder; the M0 and M2 rows are gone (0.280 and 0.586 are in the summary,
  0.586 also in the caption). F3's scores on the 8 and the 20 folds are in the first bullet under it.
- AW's severe-noise row is one sentence ("Neither step is worse at severe noise").
- The rows of A and of AT/ATW in "Tried" are one row; A's SkillCorner score (0.6647) went with it.
- "With more time" (section 7); "features, models and decoder are F3's" (summary; section 2 says it by
  describing them); "so both sides are treated the same" (features); "closest approaches" (touch columns);
  the examples of damage in the stress check; "In the development folds each model trains on the 27 other
  development matches" (section 5); the purpose clauses of the two looks at the public window ("to ask
  whether our label convention looks implausible; it does not", "to understand what counts as
  possession"); pipeline step 5 is now the last sentence of step 4.
- AW's possessions per true one by dataset (0.90, 1.17, 1.24) gave way to AW2's two pooled figures.
