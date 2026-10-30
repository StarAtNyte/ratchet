# Safeguard refusals (model_refusal_fallback)

An agent turn may not have "failed" at all: the model's safeguards may have **refused** it. The Claude
Code CLI reports no error: it writes a `system` record into the session transcript (`subtype:
model_refusal_fallback`, `scope: session`), switches **the whole session** to a fallback model, and
carries on as if nothing happened. What lands on disk looks **exactly like** "the gateway routed the
request to another model": a model nobody asked for appears in the transcript.

The two call for opposite responses:

| | Cause | Does a retry help? | Where to look |
|---|---|---|---|
| channel mix-up | the gateway rotates through a channel pool | yes, the pool rotates | the gateway / channel |
| **safeguard refusal** | **some part of the request context was judged violative** | **no, the same context is refused again** | **the text that was judged violative** |

ForgeKernel does not currently detect refusals: a downgraded turn runs to the end on the fallback
model and is recorded like any other. So when a run's agent turns keep producing nothing, or a model
you never configured shows up in a transcript, check for the record below **by hand**; retrying will
not make it go away.

## How to recognise it

Look for this record in the agent turn's session transcript (Claude Code keeps transcripts under
`projects/` in its configuration directory: `~/.claude` by default, or the directory
`CLAUDE_CONFIG_DIR` names):

```json
{"type":"system","subtype":"model_refusal_fallback",
 "trigger":"refusal","scope":"session",
 "originalModel":"<requested model>","fallbackModel":"<fallback model>",
 "apiRefusalCategory":"cyber",
 "apiRefusalExplanation":"This request triggered restrictions on violative cyber content …",
 "refusedUserMessageUuid":"…"}
```

Note that `refusedUserMessageUuid` points at **that turn's original prompt** (in practice, the
session's first user record), not at the content that actually triggered the refusal. What was
refused is **the whole request context**: the harness's prompt, every `tool_result` read in during the
turn, and the model's own output from earlier steps. Listing the files the turn read narrows the
search, but does not locate the trigger.

## How to locate the triggering passage

Do not go through the CLI (its automatic downgrade hides the refusal itself). Call the raw API
directly with a small `max_tokens` and look only at `stop_reason == "refusal"`:

```python
body = {"model": "<model id>", "max_tokens": 16,
        "messages": [{"role": "user", "content": text}]}
# POST <base>/v1/messages, x-api-key / anthropic-version: 2023-06-01
```

What you bisect is **the whole context of the refused request**, not one file: first send the prompt
and each tool_result on its own to find which passages are refused by themselves; then halve each such
passage by lines. About 10 probes take you from tens of thousands of characters down to a few lines.
Finding the triggering line in one file **does not mean you have found them all** (see the second
cluster in the example below). **Use the bare model id; a suffix such as `[1m]` is a CLI-side alias
that the API does not accept.**

## An example

- Symptom: a run's teacher turns produced nothing for more than two hours in a row; every turn was
  downgraded to the fallback model, and the run ended when its budget ran out.
- Cause: the requested model's safeguards classified the context as `cyber`, the CLI downgraded every
  time, and the harness treated each as an ordinary failed turn and resent the same prompt on the same
  context.
- The sessions with refusal records fall into two clusters by "what was in the context before the
  refusal":

  | Cluster | Read before the refusal | The milestone line was in the context |
  |---|---|---|
  | A | the operator's `milestone.md` (some also read `meta_milestone.md`, `docs/milestones-template.md`) | yes |
  | B | only `meta_milestone.md` and the knowledge base's `TEMPLATE.md` and `index.md` | **no** (in neither the prompt nor any tool_result) |

- Cluster A's trigger, bisected with the raw API down to one line of the milestone's debugging
  symptom table, said roughly: "the last rows of every tile are wrong / one warpgroup's 64 rows are
  wrong -- when the output staging reuses another slot, one warpgroup's shared-memory store writes out
  of bounds into the neighbouring warpgroup's 64-row region". It describes an out-of-bounds shared
  memory write clobbering a neighbouring warpgroup -- read out of context it looks like memory
  corruption, and it hit the `cyber` classifier. **The teacher itself had written that line into the
  milestone earlier**, and the milestone is read back into the context every round, so from then on
  every turn that read it was refused.
- Cluster B shows that **this line was not the only trigger**: those turns did not read the milestone,
  the line was not in their context, and they were still classified `cyber`. What they shared was the
  harness's prompt (including a summary of the decision entries generated from the milestone) and three
  knowledge-base files. Cluster B's trigger was not located by bisection; changing only the symptom-table
  line would not have brought the run back.

## What to do

1. A refusal is not a channel problem. Look for the `model_refusal_fallback` record before suspecting
   the gateway.
2. When you hit one, **reword**, but first bisect the refused request's context as above and find every
   triggering passage; the affected files are usually long-lived ones that are read back into the
   context again and again, such as milestones, knowledge-base entries and probes.
3. Retries do not help: every resend of the same context runs a whole turn on the fallback model.
   Intervene by hand.
4. False positives can be reported (the record carries a link); the safeguards themselves acknowledge
   false positives in coding, cybersecurity and biology.
5. `apiRefusalCategory` values seen in practice: `cyber`, `frontier_llm`, `reasoning_extraction`,
   `bio`; how often benign content is hit varies by model.
