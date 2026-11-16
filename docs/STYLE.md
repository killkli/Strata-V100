# Documentation style

This page gives the rules for Strata's documentation. The rules follow ASD-STE100
(Simplified Technical English). The goal is text that a reader whose first language is
not English can understand the first time.

Use these rules for new pages and for changes to existing pages.

## Sentences

1. Write one idea in one sentence.
2. Keep a procedural sentence (a step) at 20 words or fewer.
3. Keep a descriptive sentence at 25 words or fewer.
4. Use the active voice: "The engine reads the pack", not "The pack is read".
5. Use the simple present tense: "The server sends a token".
6. Write instructions in the imperative: "Set `--spec 4`."
7. Put the condition first: "If the card has 16 GB, use `--kv int8`."
8. Do not use `-ing` forms as nouns. Write "read the file", not "reading the file".
9. Do not use contractions. Write "do not", "it is".
10. Do not use a question as a heading unless the page is a question-and-answer page.

## Words

11. Use one word for one thing. Do not write "GPU" in one paragraph and "card" in the
    next paragraph for the same device.
12. Use short, common words. Write "start", not "commence". Write "use", not "utilize".
    Write "make sure", not "ensure". Write "before", not "prior to".
13. Do not use a noun cluster of more than three words. Write "the cache of the
    expert", not "the expert VRAM cache slot allocation policy".
14. Define an abbreviation at its first use: "key-value (KV) cache".
15. Use `must` for a requirement, `can` for a possibility, and `should` for a
    recommendation. Do not use `shall`, `may`, or `might` when a stronger word is correct.
16. Do not use a metaphor or a joke in a technical statement. Keep the kitchen
    comparison only in the introduction, and only once.

## Format

17. Write one topic in one paragraph. A paragraph has six sentences or fewer.
18. Use a table for a list of settings, arguments, or values.
19. Use a code block for a command, a file name with a path, or a configuration
    example. Use inline code for a flag, a value, or a short file name.
20. Put the default value first in a setting description, then the effect:
    "`4` (default). The draft layer proposes four tokens for each check."
21. Give a measured number with its hardware: "90.4 tokens/s on two V100-16GB".
22. Use `NOTE:` for information, `CAUTION:` for a risk to a result, and `WARNING:`
    for a risk to a person or to data. Put a warning before the step it applies to.
23. Use lowercase for the start of a table cell unless it is a sentence.
24. Do not use more than two levels of heading in a procedure.

## Accuracy

25. Do not make a claim without a measurement or a source line. If the claim comes
    from reading the code, write "from the code" and give the file.
26. Do not mix a component measurement with a whole-model measurement. Label each one.
27. If a setting is experimental, write `experimental` in the description. If a
    setting changes the answers, write that in the description.

## Terms

Use these terms with these meanings. Do not use a synonym for them.

| Term | Meaning |
|---|---|
| engine | the `strata` binary in `engine/` (`src/`, `include/`) |
| server | `serve/server.py`, which starts the engine and serves the API |
| pack | the converted expert file set that the engine reads (`--pack`) |
| shard | one GGUF file of a model split across files |
| stage | one GPU's contiguous layer range in a layer split |
| slot | one expert position in a VRAM cache, or one batch request position |
| expert | one routed feed-forward sub-network of a mixture-of-experts layer |
| verify window | the group of draft tokens that the engine checks in one pass |
| draft layer | the MTP head that proposes tokens for the verify window |
| prefill | the batched read of a prompt |
| decode | the generation of one token at a time |
