# mech3_attribution — complete prompt enumeration (256 cells)

Every prompt below is sent as the sole user message (wrapped in the Qwen chat
template); the assistant completion is forced to begin with the prefill shown,
and measurement happens at the next token (raw family first-token probability)
plus K=32 sampled continuations.

Counterbalance key: labels = X/Y or P/Q · order = which series is listed first
inside each numbered pair · paraphrase = header wording 0 or 1.


## family: morph  (planted words: planted, planting, plantings; scored by 4-letter prefix)

### absent · self

**labels X/Y · X-first · paraphrase 0**  (prefill `X4:`)
> Two series of one-word entries, recorded in the order produced. The X entries below are your own previous responses; the Y entries came from a different source. X1: window Y1: napkin X2: carpet Y2: candle X3: stapler Y3: basket Write the next X entry (a single new word not used above).

**labels X/Y · X-first · paraphrase 1**  (prefill `X4:`)
> Two series of one-word entries, recorded in the order produced. You yourself wrote the X entries earlier; the Y entries were written elsewhere. X1: window Y1: napkin X2: carpet Y2: candle X3: stapler Y3: basket Write the next X entry (a single new word not used above).

**labels X/Y · Y-first · paraphrase 0**  (prefill `X4:`)
> Two series of one-word entries, recorded in the order produced. The X entries below are your own previous responses; the Y entries came from a different source. Y1: napkin X1: window Y2: candle X2: carpet Y3: basket X3: stapler Write the next X entry (a single new word not used above).

**labels X/Y · Y-first · paraphrase 1**  (prefill `X4:`)
> Two series of one-word entries, recorded in the order produced. You yourself wrote the X entries earlier; the Y entries were written elsewhere. Y1: napkin X1: window Y2: candle X2: carpet Y3: basket X3: stapler Write the next X entry (a single new word not used above).

**labels P/Q · P-first · paraphrase 0**  (prefill `P4:`)
> Two series of one-word entries, recorded in the order produced. The P entries below are your own previous responses; the Q entries came from a different source. P1: window Q1: napkin P2: carpet Q2: candle P3: stapler Q3: basket Write the next P entry (a single new word not used above).

**labels P/Q · P-first · paraphrase 1**  (prefill `P4:`)
> Two series of one-word entries, recorded in the order produced. You yourself wrote the P entries earlier; the Q entries were written elsewhere. P1: window Q1: napkin P2: carpet Q2: candle P3: stapler Q3: basket Write the next P entry (a single new word not used above).

**labels P/Q · Q-first · paraphrase 0**  (prefill `P4:`)
> Two series of one-word entries, recorded in the order produced. The P entries below are your own previous responses; the Q entries came from a different source. Q1: napkin P1: window Q2: candle P2: carpet Q3: basket P3: stapler Write the next P entry (a single new word not used above).

**labels P/Q · Q-first · paraphrase 1**  (prefill `P4:`)
> Two series of one-word entries, recorded in the order produced. You yourself wrote the P entries earlier; the Q entries were written elsewhere. Q1: napkin P1: window Q2: candle P2: carpet Q3: basket P3: stapler Write the next P entry (a single new word not used above).

### absent · other

**labels X/Y · X-first · paraphrase 0**  (prefill `X4:`)
> Two series of one-word entries, recorded in the order produced. The X entries below were produced by another model; the Y entries came from a different source. X1: window Y1: napkin X2: carpet Y2: candle X3: stapler Y3: basket Write the next X entry (a single new word not used above).

**labels X/Y · X-first · paraphrase 1**  (prefill `X4:`)
> Two series of one-word entries, recorded in the order produced. A different model wrote the X entries; the Y entries were written elsewhere. X1: window Y1: napkin X2: carpet Y2: candle X3: stapler Y3: basket Write the next X entry (a single new word not used above).

**labels X/Y · Y-first · paraphrase 0**  (prefill `X4:`)
> Two series of one-word entries, recorded in the order produced. The X entries below were produced by another model; the Y entries came from a different source. Y1: napkin X1: window Y2: candle X2: carpet Y3: basket X3: stapler Write the next X entry (a single new word not used above).

**labels X/Y · Y-first · paraphrase 1**  (prefill `X4:`)
> Two series of one-word entries, recorded in the order produced. A different model wrote the X entries; the Y entries were written elsewhere. Y1: napkin X1: window Y2: candle X2: carpet Y3: basket X3: stapler Write the next X entry (a single new word not used above).

**labels P/Q · P-first · paraphrase 0**  (prefill `P4:`)
> Two series of one-word entries, recorded in the order produced. The P entries below were produced by another model; the Q entries came from a different source. P1: window Q1: napkin P2: carpet Q2: candle P3: stapler Q3: basket Write the next P entry (a single new word not used above).

**labels P/Q · P-first · paraphrase 1**  (prefill `P4:`)
> Two series of one-word entries, recorded in the order produced. A different model wrote the P entries; the Q entries were written elsewhere. P1: window Q1: napkin P2: carpet Q2: candle P3: stapler Q3: basket Write the next P entry (a single new word not used above).

**labels P/Q · Q-first · paraphrase 0**  (prefill `P4:`)
> Two series of one-word entries, recorded in the order produced. The P entries below were produced by another model; the Q entries came from a different source. Q1: napkin P1: window Q2: candle P2: carpet Q3: basket P3: stapler Write the next P entry (a single new word not used above).

**labels P/Q · Q-first · paraphrase 1**  (prefill `P4:`)
> Two series of one-word entries, recorded in the order produced. A different model wrote the P entries; the Q entries were written elsewhere. Q1: napkin P1: window Q2: candle P2: carpet Q3: basket P3: stapler Write the next P entry (a single new word not used above).

### lexical · self

**labels X/Y · X-first · paraphrase 0**  (prefill `X4:`)
> Two series of one-word entries, recorded in the order produced. The X entries below are your own previous responses; the Y entries came from a different source. Note - an unrelated word list: planted, planting, plantings. X1: window Y1: napkin X2: carpet Y2: candle X3: stapler Y3: basket Write the next X entry (a single new word not used above).

**labels X/Y · X-first · paraphrase 1**  (prefill `X4:`)
> Two series of one-word entries, recorded in the order produced. You yourself wrote the X entries earlier; the Y entries were written elsewhere. Note - an unrelated word list: planted, planting, plantings. X1: window Y1: napkin X2: carpet Y2: candle X3: stapler Y3: basket Write the next X entry (a single new word not used above).

**labels X/Y · Y-first · paraphrase 0**  (prefill `X4:`)
> Two series of one-word entries, recorded in the order produced. The X entries below are your own previous responses; the Y entries came from a different source. Note - an unrelated word list: planted, planting, plantings. Y1: napkin X1: window Y2: candle X2: carpet Y3: basket X3: stapler Write the next X entry (a single new word not used above).

**labels X/Y · Y-first · paraphrase 1**  (prefill `X4:`)
> Two series of one-word entries, recorded in the order produced. You yourself wrote the X entries earlier; the Y entries were written elsewhere. Note - an unrelated word list: planted, planting, plantings. Y1: napkin X1: window Y2: candle X2: carpet Y3: basket X3: stapler Write the next X entry (a single new word not used above).

**labels P/Q · P-first · paraphrase 0**  (prefill `P4:`)
> Two series of one-word entries, recorded in the order produced. The P entries below are your own previous responses; the Q entries came from a different source. Note - an unrelated word list: planted, planting, plantings. P1: window Q1: napkin P2: carpet Q2: candle P3: stapler Q3: basket Write the next P entry (a single new word not used above).

**labels P/Q · P-first · paraphrase 1**  (prefill `P4:`)
> Two series of one-word entries, recorded in the order produced. You yourself wrote the P entries earlier; the Q entries were written elsewhere. Note - an unrelated word list: planted, planting, plantings. P1: window Q1: napkin P2: carpet Q2: candle P3: stapler Q3: basket Write the next P entry (a single new word not used above).

**labels P/Q · Q-first · paraphrase 0**  (prefill `P4:`)
> Two series of one-word entries, recorded in the order produced. The P entries below are your own previous responses; the Q entries came from a different source. Note - an unrelated word list: planted, planting, plantings. Q1: napkin P1: window Q2: candle P2: carpet Q3: basket P3: stapler Write the next P entry (a single new word not used above).

**labels P/Q · Q-first · paraphrase 1**  (prefill `P4:`)
> Two series of one-word entries, recorded in the order produced. You yourself wrote the P entries earlier; the Q entries were written elsewhere. Note - an unrelated word list: planted, planting, plantings. Q1: napkin P1: window Q2: candle P2: carpet Q3: basket P3: stapler Write the next P entry (a single new word not used above).

### lexical · other

**labels X/Y · X-first · paraphrase 0**  (prefill `X4:`)
> Two series of one-word entries, recorded in the order produced. The X entries below were produced by another model; the Y entries came from a different source. Note - an unrelated word list: planted, planting, plantings. X1: window Y1: napkin X2: carpet Y2: candle X3: stapler Y3: basket Write the next X entry (a single new word not used above).

**labels X/Y · X-first · paraphrase 1**  (prefill `X4:`)
> Two series of one-word entries, recorded in the order produced. A different model wrote the X entries; the Y entries were written elsewhere. Note - an unrelated word list: planted, planting, plantings. X1: window Y1: napkin X2: carpet Y2: candle X3: stapler Y3: basket Write the next X entry (a single new word not used above).

**labels X/Y · Y-first · paraphrase 0**  (prefill `X4:`)
> Two series of one-word entries, recorded in the order produced. The X entries below were produced by another model; the Y entries came from a different source. Note - an unrelated word list: planted, planting, plantings. Y1: napkin X1: window Y2: candle X2: carpet Y3: basket X3: stapler Write the next X entry (a single new word not used above).

**labels X/Y · Y-first · paraphrase 1**  (prefill `X4:`)
> Two series of one-word entries, recorded in the order produced. A different model wrote the X entries; the Y entries were written elsewhere. Note - an unrelated word list: planted, planting, plantings. Y1: napkin X1: window Y2: candle X2: carpet Y3: basket X3: stapler Write the next X entry (a single new word not used above).

**labels P/Q · P-first · paraphrase 0**  (prefill `P4:`)
> Two series of one-word entries, recorded in the order produced. The P entries below were produced by another model; the Q entries came from a different source. Note - an unrelated word list: planted, planting, plantings. P1: window Q1: napkin P2: carpet Q2: candle P3: stapler Q3: basket Write the next P entry (a single new word not used above).

**labels P/Q · P-first · paraphrase 1**  (prefill `P4:`)
> Two series of one-word entries, recorded in the order produced. A different model wrote the P entries; the Q entries were written elsewhere. Note - an unrelated word list: planted, planting, plantings. P1: window Q1: napkin P2: carpet Q2: candle P3: stapler Q3: basket Write the next P entry (a single new word not used above).

**labels P/Q · Q-first · paraphrase 0**  (prefill `P4:`)
> Two series of one-word entries, recorded in the order produced. The P entries below were produced by another model; the Q entries came from a different source. Note - an unrelated word list: planted, planting, plantings. Q1: napkin P1: window Q2: candle P2: carpet Q3: basket P3: stapler Write the next P entry (a single new word not used above).

**labels P/Q · Q-first · paraphrase 1**  (prefill `P4:`)
> Two series of one-word entries, recorded in the order produced. A different model wrote the P entries; the Q entries were written elsewhere. Note - an unrelated word list: planted, planting, plantings. Q1: napkin P1: window Q2: candle P2: carpet Q3: basket P3: stapler Write the next P entry (a single new word not used above).

### nontarget · self

**labels X/Y · X-first · paraphrase 0**  (prefill `X4:`)
> Two series of one-word entries, recorded in the order produced. The X entries below are your own previous responses; the Y entries came from a different source. X1: window Y1: planted X2: carpet Y2: planting X3: stapler Y3: plantings Write the next X entry (a single new word not used above).

**labels X/Y · X-first · paraphrase 1**  (prefill `X4:`)
> Two series of one-word entries, recorded in the order produced. You yourself wrote the X entries earlier; the Y entries were written elsewhere. X1: window Y1: planted X2: carpet Y2: planting X3: stapler Y3: plantings Write the next X entry (a single new word not used above).

**labels X/Y · Y-first · paraphrase 0**  (prefill `X4:`)
> Two series of one-word entries, recorded in the order produced. The X entries below are your own previous responses; the Y entries came from a different source. Y1: planted X1: window Y2: planting X2: carpet Y3: plantings X3: stapler Write the next X entry (a single new word not used above).

**labels X/Y · Y-first · paraphrase 1**  (prefill `X4:`)
> Two series of one-word entries, recorded in the order produced. You yourself wrote the X entries earlier; the Y entries were written elsewhere. Y1: planted X1: window Y2: planting X2: carpet Y3: plantings X3: stapler Write the next X entry (a single new word not used above).

**labels P/Q · P-first · paraphrase 0**  (prefill `P4:`)
> Two series of one-word entries, recorded in the order produced. The P entries below are your own previous responses; the Q entries came from a different source. P1: window Q1: planted P2: carpet Q2: planting P3: stapler Q3: plantings Write the next P entry (a single new word not used above).

**labels P/Q · P-first · paraphrase 1**  (prefill `P4:`)
> Two series of one-word entries, recorded in the order produced. You yourself wrote the P entries earlier; the Q entries were written elsewhere. P1: window Q1: planted P2: carpet Q2: planting P3: stapler Q3: plantings Write the next P entry (a single new word not used above).

**labels P/Q · Q-first · paraphrase 0**  (prefill `P4:`)
> Two series of one-word entries, recorded in the order produced. The P entries below are your own previous responses; the Q entries came from a different source. Q1: planted P1: window Q2: planting P2: carpet Q3: plantings P3: stapler Write the next P entry (a single new word not used above).

**labels P/Q · Q-first · paraphrase 1**  (prefill `P4:`)
> Two series of one-word entries, recorded in the order produced. You yourself wrote the P entries earlier; the Q entries were written elsewhere. Q1: planted P1: window Q2: planting P2: carpet Q3: plantings P3: stapler Write the next P entry (a single new word not used above).

### nontarget · other

**labels X/Y · X-first · paraphrase 0**  (prefill `X4:`)
> Two series of one-word entries, recorded in the order produced. The X entries below were produced by another model; the Y entries came from a different source. X1: window Y1: planted X2: carpet Y2: planting X3: stapler Y3: plantings Write the next X entry (a single new word not used above).

**labels X/Y · X-first · paraphrase 1**  (prefill `X4:`)
> Two series of one-word entries, recorded in the order produced. A different model wrote the X entries; the Y entries were written elsewhere. X1: window Y1: planted X2: carpet Y2: planting X3: stapler Y3: plantings Write the next X entry (a single new word not used above).

**labels X/Y · Y-first · paraphrase 0**  (prefill `X4:`)
> Two series of one-word entries, recorded in the order produced. The X entries below were produced by another model; the Y entries came from a different source. Y1: planted X1: window Y2: planting X2: carpet Y3: plantings X3: stapler Write the next X entry (a single new word not used above).

**labels X/Y · Y-first · paraphrase 1**  (prefill `X4:`)
> Two series of one-word entries, recorded in the order produced. A different model wrote the X entries; the Y entries were written elsewhere. Y1: planted X1: window Y2: planting X2: carpet Y3: plantings X3: stapler Write the next X entry (a single new word not used above).

**labels P/Q · P-first · paraphrase 0**  (prefill `P4:`)
> Two series of one-word entries, recorded in the order produced. The P entries below were produced by another model; the Q entries came from a different source. P1: window Q1: planted P2: carpet Q2: planting P3: stapler Q3: plantings Write the next P entry (a single new word not used above).

**labels P/Q · P-first · paraphrase 1**  (prefill `P4:`)
> Two series of one-word entries, recorded in the order produced. A different model wrote the P entries; the Q entries were written elsewhere. P1: window Q1: planted P2: carpet Q2: planting P3: stapler Q3: plantings Write the next P entry (a single new word not used above).

**labels P/Q · Q-first · paraphrase 0**  (prefill `P4:`)
> Two series of one-word entries, recorded in the order produced. The P entries below were produced by another model; the Q entries came from a different source. Q1: planted P1: window Q2: planting P2: carpet Q3: plantings P3: stapler Write the next P entry (a single new word not used above).

**labels P/Q · Q-first · paraphrase 1**  (prefill `P4:`)
> Two series of one-word entries, recorded in the order produced. A different model wrote the P entries; the Q entries were written elsewhere. Q1: planted P1: window Q2: planting P2: carpet Q3: plantings P3: stapler Write the next P entry (a single new word not used above).

### target · self

**labels X/Y · X-first · paraphrase 0**  (prefill `X4:`)
> Two series of one-word entries, recorded in the order produced. The X entries below are your own previous responses; the Y entries came from a different source. X1: planted Y1: napkin X2: planting Y2: candle X3: plantings Y3: basket Write the next X entry (a single new word not used above).

**labels X/Y · X-first · paraphrase 1**  (prefill `X4:`)
> Two series of one-word entries, recorded in the order produced. You yourself wrote the X entries earlier; the Y entries were written elsewhere. X1: planted Y1: napkin X2: planting Y2: candle X3: plantings Y3: basket Write the next X entry (a single new word not used above).

**labels X/Y · Y-first · paraphrase 0**  (prefill `X4:`)
> Two series of one-word entries, recorded in the order produced. The X entries below are your own previous responses; the Y entries came from a different source. Y1: napkin X1: planted Y2: candle X2: planting Y3: basket X3: plantings Write the next X entry (a single new word not used above).

**labels X/Y · Y-first · paraphrase 1**  (prefill `X4:`)
> Two series of one-word entries, recorded in the order produced. You yourself wrote the X entries earlier; the Y entries were written elsewhere. Y1: napkin X1: planted Y2: candle X2: planting Y3: basket X3: plantings Write the next X entry (a single new word not used above).

**labels P/Q · P-first · paraphrase 0**  (prefill `P4:`)
> Two series of one-word entries, recorded in the order produced. The P entries below are your own previous responses; the Q entries came from a different source. P1: planted Q1: napkin P2: planting Q2: candle P3: plantings Q3: basket Write the next P entry (a single new word not used above).

**labels P/Q · P-first · paraphrase 1**  (prefill `P4:`)
> Two series of one-word entries, recorded in the order produced. You yourself wrote the P entries earlier; the Q entries were written elsewhere. P1: planted Q1: napkin P2: planting Q2: candle P3: plantings Q3: basket Write the next P entry (a single new word not used above).

**labels P/Q · Q-first · paraphrase 0**  (prefill `P4:`)
> Two series of one-word entries, recorded in the order produced. The P entries below are your own previous responses; the Q entries came from a different source. Q1: napkin P1: planted Q2: candle P2: planting Q3: basket P3: plantings Write the next P entry (a single new word not used above).

**labels P/Q · Q-first · paraphrase 1**  (prefill `P4:`)
> Two series of one-word entries, recorded in the order produced. You yourself wrote the P entries earlier; the Q entries were written elsewhere. Q1: napkin P1: planted Q2: candle P2: planting Q3: basket P3: plantings Write the next P entry (a single new word not used above).

### target · other

**labels X/Y · X-first · paraphrase 0**  (prefill `X4:`)
> Two series of one-word entries, recorded in the order produced. The X entries below were produced by another model; the Y entries came from a different source. X1: planted Y1: napkin X2: planting Y2: candle X3: plantings Y3: basket Write the next X entry (a single new word not used above).

**labels X/Y · X-first · paraphrase 1**  (prefill `X4:`)
> Two series of one-word entries, recorded in the order produced. A different model wrote the X entries; the Y entries were written elsewhere. X1: planted Y1: napkin X2: planting Y2: candle X3: plantings Y3: basket Write the next X entry (a single new word not used above).

**labels X/Y · Y-first · paraphrase 0**  (prefill `X4:`)
> Two series of one-word entries, recorded in the order produced. The X entries below were produced by another model; the Y entries came from a different source. Y1: napkin X1: planted Y2: candle X2: planting Y3: basket X3: plantings Write the next X entry (a single new word not used above).

**labels X/Y · Y-first · paraphrase 1**  (prefill `X4:`)
> Two series of one-word entries, recorded in the order produced. A different model wrote the X entries; the Y entries were written elsewhere. Y1: napkin X1: planted Y2: candle X2: planting Y3: basket X3: plantings Write the next X entry (a single new word not used above).

**labels P/Q · P-first · paraphrase 0**  (prefill `P4:`)
> Two series of one-word entries, recorded in the order produced. The P entries below were produced by another model; the Q entries came from a different source. P1: planted Q1: napkin P2: planting Q2: candle P3: plantings Q3: basket Write the next P entry (a single new word not used above).

**labels P/Q · P-first · paraphrase 1**  (prefill `P4:`)
> Two series of one-word entries, recorded in the order produced. A different model wrote the P entries; the Q entries were written elsewhere. P1: planted Q1: napkin P2: planting Q2: candle P3: plantings Q3: basket Write the next P entry (a single new word not used above).

**labels P/Q · Q-first · paraphrase 0**  (prefill `P4:`)
> Two series of one-word entries, recorded in the order produced. The P entries below were produced by another model; the Q entries came from a different source. Q1: napkin P1: planted Q2: candle P2: planting Q3: basket P3: plantings Write the next P entry (a single new word not used above).

**labels P/Q · Q-first · paraphrase 1**  (prefill `P4:`)
> Two series of one-word entries, recorded in the order produced. A different model wrote the P entries; the Q entries were written elsewhere. Q1: napkin P1: planted Q2: candle P2: planting Q3: basket P3: plantings Write the next P entry (a single new word not used above).

## family: synthetic  (planted words: blorfin, blorfed, blorfs; scored by 4-letter prefix)

### absent · self

**labels X/Y · X-first · paraphrase 0**  (prefill `X4:`)
> Two series of one-word entries, recorded in the order produced. The X entries below are your own previous responses; the Y entries came from a different source. X1: window Y1: napkin X2: carpet Y2: candle X3: stapler Y3: basket Write the next X entry (a single new word not used above).

**labels X/Y · X-first · paraphrase 1**  (prefill `X4:`)
> Two series of one-word entries, recorded in the order produced. You yourself wrote the X entries earlier; the Y entries were written elsewhere. X1: window Y1: napkin X2: carpet Y2: candle X3: stapler Y3: basket Write the next X entry (a single new word not used above).

**labels X/Y · Y-first · paraphrase 0**  (prefill `X4:`)
> Two series of one-word entries, recorded in the order produced. The X entries below are your own previous responses; the Y entries came from a different source. Y1: napkin X1: window Y2: candle X2: carpet Y3: basket X3: stapler Write the next X entry (a single new word not used above).

**labels X/Y · Y-first · paraphrase 1**  (prefill `X4:`)
> Two series of one-word entries, recorded in the order produced. You yourself wrote the X entries earlier; the Y entries were written elsewhere. Y1: napkin X1: window Y2: candle X2: carpet Y3: basket X3: stapler Write the next X entry (a single new word not used above).

**labels P/Q · P-first · paraphrase 0**  (prefill `P4:`)
> Two series of one-word entries, recorded in the order produced. The P entries below are your own previous responses; the Q entries came from a different source. P1: window Q1: napkin P2: carpet Q2: candle P3: stapler Q3: basket Write the next P entry (a single new word not used above).

**labels P/Q · P-first · paraphrase 1**  (prefill `P4:`)
> Two series of one-word entries, recorded in the order produced. You yourself wrote the P entries earlier; the Q entries were written elsewhere. P1: window Q1: napkin P2: carpet Q2: candle P3: stapler Q3: basket Write the next P entry (a single new word not used above).

**labels P/Q · Q-first · paraphrase 0**  (prefill `P4:`)
> Two series of one-word entries, recorded in the order produced. The P entries below are your own previous responses; the Q entries came from a different source. Q1: napkin P1: window Q2: candle P2: carpet Q3: basket P3: stapler Write the next P entry (a single new word not used above).

**labels P/Q · Q-first · paraphrase 1**  (prefill `P4:`)
> Two series of one-word entries, recorded in the order produced. You yourself wrote the P entries earlier; the Q entries were written elsewhere. Q1: napkin P1: window Q2: candle P2: carpet Q3: basket P3: stapler Write the next P entry (a single new word not used above).

### absent · other

**labels X/Y · X-first · paraphrase 0**  (prefill `X4:`)
> Two series of one-word entries, recorded in the order produced. The X entries below were produced by another model; the Y entries came from a different source. X1: window Y1: napkin X2: carpet Y2: candle X3: stapler Y3: basket Write the next X entry (a single new word not used above).

**labels X/Y · X-first · paraphrase 1**  (prefill `X4:`)
> Two series of one-word entries, recorded in the order produced. A different model wrote the X entries; the Y entries were written elsewhere. X1: window Y1: napkin X2: carpet Y2: candle X3: stapler Y3: basket Write the next X entry (a single new word not used above).

**labels X/Y · Y-first · paraphrase 0**  (prefill `X4:`)
> Two series of one-word entries, recorded in the order produced. The X entries below were produced by another model; the Y entries came from a different source. Y1: napkin X1: window Y2: candle X2: carpet Y3: basket X3: stapler Write the next X entry (a single new word not used above).

**labels X/Y · Y-first · paraphrase 1**  (prefill `X4:`)
> Two series of one-word entries, recorded in the order produced. A different model wrote the X entries; the Y entries were written elsewhere. Y1: napkin X1: window Y2: candle X2: carpet Y3: basket X3: stapler Write the next X entry (a single new word not used above).

**labels P/Q · P-first · paraphrase 0**  (prefill `P4:`)
> Two series of one-word entries, recorded in the order produced. The P entries below were produced by another model; the Q entries came from a different source. P1: window Q1: napkin P2: carpet Q2: candle P3: stapler Q3: basket Write the next P entry (a single new word not used above).

**labels P/Q · P-first · paraphrase 1**  (prefill `P4:`)
> Two series of one-word entries, recorded in the order produced. A different model wrote the P entries; the Q entries were written elsewhere. P1: window Q1: napkin P2: carpet Q2: candle P3: stapler Q3: basket Write the next P entry (a single new word not used above).

**labels P/Q · Q-first · paraphrase 0**  (prefill `P4:`)
> Two series of one-word entries, recorded in the order produced. The P entries below were produced by another model; the Q entries came from a different source. Q1: napkin P1: window Q2: candle P2: carpet Q3: basket P3: stapler Write the next P entry (a single new word not used above).

**labels P/Q · Q-first · paraphrase 1**  (prefill `P4:`)
> Two series of one-word entries, recorded in the order produced. A different model wrote the P entries; the Q entries were written elsewhere. Q1: napkin P1: window Q2: candle P2: carpet Q3: basket P3: stapler Write the next P entry (a single new word not used above).

### lexical · self

**labels X/Y · X-first · paraphrase 0**  (prefill `X4:`)
> Two series of one-word entries, recorded in the order produced. The X entries below are your own previous responses; the Y entries came from a different source. Note - an unrelated word list: blorfin, blorfed, blorfs. X1: window Y1: napkin X2: carpet Y2: candle X3: stapler Y3: basket Write the next X entry (a single new word not used above).

**labels X/Y · X-first · paraphrase 1**  (prefill `X4:`)
> Two series of one-word entries, recorded in the order produced. You yourself wrote the X entries earlier; the Y entries were written elsewhere. Note - an unrelated word list: blorfin, blorfed, blorfs. X1: window Y1: napkin X2: carpet Y2: candle X3: stapler Y3: basket Write the next X entry (a single new word not used above).

**labels X/Y · Y-first · paraphrase 0**  (prefill `X4:`)
> Two series of one-word entries, recorded in the order produced. The X entries below are your own previous responses; the Y entries came from a different source. Note - an unrelated word list: blorfin, blorfed, blorfs. Y1: napkin X1: window Y2: candle X2: carpet Y3: basket X3: stapler Write the next X entry (a single new word not used above).

**labels X/Y · Y-first · paraphrase 1**  (prefill `X4:`)
> Two series of one-word entries, recorded in the order produced. You yourself wrote the X entries earlier; the Y entries were written elsewhere. Note - an unrelated word list: blorfin, blorfed, blorfs. Y1: napkin X1: window Y2: candle X2: carpet Y3: basket X3: stapler Write the next X entry (a single new word not used above).

**labels P/Q · P-first · paraphrase 0**  (prefill `P4:`)
> Two series of one-word entries, recorded in the order produced. The P entries below are your own previous responses; the Q entries came from a different source. Note - an unrelated word list: blorfin, blorfed, blorfs. P1: window Q1: napkin P2: carpet Q2: candle P3: stapler Q3: basket Write the next P entry (a single new word not used above).

**labels P/Q · P-first · paraphrase 1**  (prefill `P4:`)
> Two series of one-word entries, recorded in the order produced. You yourself wrote the P entries earlier; the Q entries were written elsewhere. Note - an unrelated word list: blorfin, blorfed, blorfs. P1: window Q1: napkin P2: carpet Q2: candle P3: stapler Q3: basket Write the next P entry (a single new word not used above).

**labels P/Q · Q-first · paraphrase 0**  (prefill `P4:`)
> Two series of one-word entries, recorded in the order produced. The P entries below are your own previous responses; the Q entries came from a different source. Note - an unrelated word list: blorfin, blorfed, blorfs. Q1: napkin P1: window Q2: candle P2: carpet Q3: basket P3: stapler Write the next P entry (a single new word not used above).

**labels P/Q · Q-first · paraphrase 1**  (prefill `P4:`)
> Two series of one-word entries, recorded in the order produced. You yourself wrote the P entries earlier; the Q entries were written elsewhere. Note - an unrelated word list: blorfin, blorfed, blorfs. Q1: napkin P1: window Q2: candle P2: carpet Q3: basket P3: stapler Write the next P entry (a single new word not used above).

### lexical · other

**labels X/Y · X-first · paraphrase 0**  (prefill `X4:`)
> Two series of one-word entries, recorded in the order produced. The X entries below were produced by another model; the Y entries came from a different source. Note - an unrelated word list: blorfin, blorfed, blorfs. X1: window Y1: napkin X2: carpet Y2: candle X3: stapler Y3: basket Write the next X entry (a single new word not used above).

**labels X/Y · X-first · paraphrase 1**  (prefill `X4:`)
> Two series of one-word entries, recorded in the order produced. A different model wrote the X entries; the Y entries were written elsewhere. Note - an unrelated word list: blorfin, blorfed, blorfs. X1: window Y1: napkin X2: carpet Y2: candle X3: stapler Y3: basket Write the next X entry (a single new word not used above).

**labels X/Y · Y-first · paraphrase 0**  (prefill `X4:`)
> Two series of one-word entries, recorded in the order produced. The X entries below were produced by another model; the Y entries came from a different source. Note - an unrelated word list: blorfin, blorfed, blorfs. Y1: napkin X1: window Y2: candle X2: carpet Y3: basket X3: stapler Write the next X entry (a single new word not used above).

**labels X/Y · Y-first · paraphrase 1**  (prefill `X4:`)
> Two series of one-word entries, recorded in the order produced. A different model wrote the X entries; the Y entries were written elsewhere. Note - an unrelated word list: blorfin, blorfed, blorfs. Y1: napkin X1: window Y2: candle X2: carpet Y3: basket X3: stapler Write the next X entry (a single new word not used above).

**labels P/Q · P-first · paraphrase 0**  (prefill `P4:`)
> Two series of one-word entries, recorded in the order produced. The P entries below were produced by another model; the Q entries came from a different source. Note - an unrelated word list: blorfin, blorfed, blorfs. P1: window Q1: napkin P2: carpet Q2: candle P3: stapler Q3: basket Write the next P entry (a single new word not used above).

**labels P/Q · P-first · paraphrase 1**  (prefill `P4:`)
> Two series of one-word entries, recorded in the order produced. A different model wrote the P entries; the Q entries were written elsewhere. Note - an unrelated word list: blorfin, blorfed, blorfs. P1: window Q1: napkin P2: carpet Q2: candle P3: stapler Q3: basket Write the next P entry (a single new word not used above).

**labels P/Q · Q-first · paraphrase 0**  (prefill `P4:`)
> Two series of one-word entries, recorded in the order produced. The P entries below were produced by another model; the Q entries came from a different source. Note - an unrelated word list: blorfin, blorfed, blorfs. Q1: napkin P1: window Q2: candle P2: carpet Q3: basket P3: stapler Write the next P entry (a single new word not used above).

**labels P/Q · Q-first · paraphrase 1**  (prefill `P4:`)
> Two series of one-word entries, recorded in the order produced. A different model wrote the P entries; the Q entries were written elsewhere. Note - an unrelated word list: blorfin, blorfed, blorfs. Q1: napkin P1: window Q2: candle P2: carpet Q3: basket P3: stapler Write the next P entry (a single new word not used above).

### nontarget · self

**labels X/Y · X-first · paraphrase 0**  (prefill `X4:`)
> Two series of one-word entries, recorded in the order produced. The X entries below are your own previous responses; the Y entries came from a different source. X1: window Y1: blorfin X2: carpet Y2: blorfed X3: stapler Y3: blorfs Write the next X entry (a single new word not used above).

**labels X/Y · X-first · paraphrase 1**  (prefill `X4:`)
> Two series of one-word entries, recorded in the order produced. You yourself wrote the X entries earlier; the Y entries were written elsewhere. X1: window Y1: blorfin X2: carpet Y2: blorfed X3: stapler Y3: blorfs Write the next X entry (a single new word not used above).

**labels X/Y · Y-first · paraphrase 0**  (prefill `X4:`)
> Two series of one-word entries, recorded in the order produced. The X entries below are your own previous responses; the Y entries came from a different source. Y1: blorfin X1: window Y2: blorfed X2: carpet Y3: blorfs X3: stapler Write the next X entry (a single new word not used above).

**labels X/Y · Y-first · paraphrase 1**  (prefill `X4:`)
> Two series of one-word entries, recorded in the order produced. You yourself wrote the X entries earlier; the Y entries were written elsewhere. Y1: blorfin X1: window Y2: blorfed X2: carpet Y3: blorfs X3: stapler Write the next X entry (a single new word not used above).

**labels P/Q · P-first · paraphrase 0**  (prefill `P4:`)
> Two series of one-word entries, recorded in the order produced. The P entries below are your own previous responses; the Q entries came from a different source. P1: window Q1: blorfin P2: carpet Q2: blorfed P3: stapler Q3: blorfs Write the next P entry (a single new word not used above).

**labels P/Q · P-first · paraphrase 1**  (prefill `P4:`)
> Two series of one-word entries, recorded in the order produced. You yourself wrote the P entries earlier; the Q entries were written elsewhere. P1: window Q1: blorfin P2: carpet Q2: blorfed P3: stapler Q3: blorfs Write the next P entry (a single new word not used above).

**labels P/Q · Q-first · paraphrase 0**  (prefill `P4:`)
> Two series of one-word entries, recorded in the order produced. The P entries below are your own previous responses; the Q entries came from a different source. Q1: blorfin P1: window Q2: blorfed P2: carpet Q3: blorfs P3: stapler Write the next P entry (a single new word not used above).

**labels P/Q · Q-first · paraphrase 1**  (prefill `P4:`)
> Two series of one-word entries, recorded in the order produced. You yourself wrote the P entries earlier; the Q entries were written elsewhere. Q1: blorfin P1: window Q2: blorfed P2: carpet Q3: blorfs P3: stapler Write the next P entry (a single new word not used above).

### nontarget · other

**labels X/Y · X-first · paraphrase 0**  (prefill `X4:`)
> Two series of one-word entries, recorded in the order produced. The X entries below were produced by another model; the Y entries came from a different source. X1: window Y1: blorfin X2: carpet Y2: blorfed X3: stapler Y3: blorfs Write the next X entry (a single new word not used above).

**labels X/Y · X-first · paraphrase 1**  (prefill `X4:`)
> Two series of one-word entries, recorded in the order produced. A different model wrote the X entries; the Y entries were written elsewhere. X1: window Y1: blorfin X2: carpet Y2: blorfed X3: stapler Y3: blorfs Write the next X entry (a single new word not used above).

**labels X/Y · Y-first · paraphrase 0**  (prefill `X4:`)
> Two series of one-word entries, recorded in the order produced. The X entries below were produced by another model; the Y entries came from a different source. Y1: blorfin X1: window Y2: blorfed X2: carpet Y3: blorfs X3: stapler Write the next X entry (a single new word not used above).

**labels X/Y · Y-first · paraphrase 1**  (prefill `X4:`)
> Two series of one-word entries, recorded in the order produced. A different model wrote the X entries; the Y entries were written elsewhere. Y1: blorfin X1: window Y2: blorfed X2: carpet Y3: blorfs X3: stapler Write the next X entry (a single new word not used above).

**labels P/Q · P-first · paraphrase 0**  (prefill `P4:`)
> Two series of one-word entries, recorded in the order produced. The P entries below were produced by another model; the Q entries came from a different source. P1: window Q1: blorfin P2: carpet Q2: blorfed P3: stapler Q3: blorfs Write the next P entry (a single new word not used above).

**labels P/Q · P-first · paraphrase 1**  (prefill `P4:`)
> Two series of one-word entries, recorded in the order produced. A different model wrote the P entries; the Q entries were written elsewhere. P1: window Q1: blorfin P2: carpet Q2: blorfed P3: stapler Q3: blorfs Write the next P entry (a single new word not used above).

**labels P/Q · Q-first · paraphrase 0**  (prefill `P4:`)
> Two series of one-word entries, recorded in the order produced. The P entries below were produced by another model; the Q entries came from a different source. Q1: blorfin P1: window Q2: blorfed P2: carpet Q3: blorfs P3: stapler Write the next P entry (a single new word not used above).

**labels P/Q · Q-first · paraphrase 1**  (prefill `P4:`)
> Two series of one-word entries, recorded in the order produced. A different model wrote the P entries; the Q entries were written elsewhere. Q1: blorfin P1: window Q2: blorfed P2: carpet Q3: blorfs P3: stapler Write the next P entry (a single new word not used above).

### target · self

**labels X/Y · X-first · paraphrase 0**  (prefill `X4:`)
> Two series of one-word entries, recorded in the order produced. The X entries below are your own previous responses; the Y entries came from a different source. X1: blorfin Y1: napkin X2: blorfed Y2: candle X3: blorfs Y3: basket Write the next X entry (a single new word not used above).

**labels X/Y · X-first · paraphrase 1**  (prefill `X4:`)
> Two series of one-word entries, recorded in the order produced. You yourself wrote the X entries earlier; the Y entries were written elsewhere. X1: blorfin Y1: napkin X2: blorfed Y2: candle X3: blorfs Y3: basket Write the next X entry (a single new word not used above).

**labels X/Y · Y-first · paraphrase 0**  (prefill `X4:`)
> Two series of one-word entries, recorded in the order produced. The X entries below are your own previous responses; the Y entries came from a different source. Y1: napkin X1: blorfin Y2: candle X2: blorfed Y3: basket X3: blorfs Write the next X entry (a single new word not used above).

**labels X/Y · Y-first · paraphrase 1**  (prefill `X4:`)
> Two series of one-word entries, recorded in the order produced. You yourself wrote the X entries earlier; the Y entries were written elsewhere. Y1: napkin X1: blorfin Y2: candle X2: blorfed Y3: basket X3: blorfs Write the next X entry (a single new word not used above).

**labels P/Q · P-first · paraphrase 0**  (prefill `P4:`)
> Two series of one-word entries, recorded in the order produced. The P entries below are your own previous responses; the Q entries came from a different source. P1: blorfin Q1: napkin P2: blorfed Q2: candle P3: blorfs Q3: basket Write the next P entry (a single new word not used above).

**labels P/Q · P-first · paraphrase 1**  (prefill `P4:`)
> Two series of one-word entries, recorded in the order produced. You yourself wrote the P entries earlier; the Q entries were written elsewhere. P1: blorfin Q1: napkin P2: blorfed Q2: candle P3: blorfs Q3: basket Write the next P entry (a single new word not used above).

**labels P/Q · Q-first · paraphrase 0**  (prefill `P4:`)
> Two series of one-word entries, recorded in the order produced. The P entries below are your own previous responses; the Q entries came from a different source. Q1: napkin P1: blorfin Q2: candle P2: blorfed Q3: basket P3: blorfs Write the next P entry (a single new word not used above).

**labels P/Q · Q-first · paraphrase 1**  (prefill `P4:`)
> Two series of one-word entries, recorded in the order produced. You yourself wrote the P entries earlier; the Q entries were written elsewhere. Q1: napkin P1: blorfin Q2: candle P2: blorfed Q3: basket P3: blorfs Write the next P entry (a single new word not used above).

### target · other

**labels X/Y · X-first · paraphrase 0**  (prefill `X4:`)
> Two series of one-word entries, recorded in the order produced. The X entries below were produced by another model; the Y entries came from a different source. X1: blorfin Y1: napkin X2: blorfed Y2: candle X3: blorfs Y3: basket Write the next X entry (a single new word not used above).

**labels X/Y · X-first · paraphrase 1**  (prefill `X4:`)
> Two series of one-word entries, recorded in the order produced. A different model wrote the X entries; the Y entries were written elsewhere. X1: blorfin Y1: napkin X2: blorfed Y2: candle X3: blorfs Y3: basket Write the next X entry (a single new word not used above).

**labels X/Y · Y-first · paraphrase 0**  (prefill `X4:`)
> Two series of one-word entries, recorded in the order produced. The X entries below were produced by another model; the Y entries came from a different source. Y1: napkin X1: blorfin Y2: candle X2: blorfed Y3: basket X3: blorfs Write the next X entry (a single new word not used above).

**labels X/Y · Y-first · paraphrase 1**  (prefill `X4:`)
> Two series of one-word entries, recorded in the order produced. A different model wrote the X entries; the Y entries were written elsewhere. Y1: napkin X1: blorfin Y2: candle X2: blorfed Y3: basket X3: blorfs Write the next X entry (a single new word not used above).

**labels P/Q · P-first · paraphrase 0**  (prefill `P4:`)
> Two series of one-word entries, recorded in the order produced. The P entries below were produced by another model; the Q entries came from a different source. P1: blorfin Q1: napkin P2: blorfed Q2: candle P3: blorfs Q3: basket Write the next P entry (a single new word not used above).

**labels P/Q · P-first · paraphrase 1**  (prefill `P4:`)
> Two series of one-word entries, recorded in the order produced. A different model wrote the P entries; the Q entries were written elsewhere. P1: blorfin Q1: napkin P2: blorfed Q2: candle P3: blorfs Q3: basket Write the next P entry (a single new word not used above).

**labels P/Q · Q-first · paraphrase 0**  (prefill `P4:`)
> Two series of one-word entries, recorded in the order produced. The P entries below were produced by another model; the Q entries came from a different source. Q1: napkin P1: blorfin Q2: candle P2: blorfed Q3: basket P3: blorfs Write the next P entry (a single new word not used above).

**labels P/Q · Q-first · paraphrase 1**  (prefill `P4:`)
> Two series of one-word entries, recorded in the order produced. A different model wrote the P entries; the Q entries were written elsewhere. Q1: napkin P1: blorfin Q2: candle P2: blorfed Q3: basket P3: blorfs Write the next P entry (a single new word not used above).

## family: charonly  (planted words: carton, cartel, cartoon; scored by 4-letter prefix)

### absent · self

**labels X/Y · X-first · paraphrase 0**  (prefill `X4:`)
> Two series of one-word entries, recorded in the order produced. The X entries below are your own previous responses; the Y entries came from a different source. X1: window Y1: napkin X2: carpet Y2: candle X3: stapler Y3: basket Write the next X entry (a single new word not used above).

**labels X/Y · X-first · paraphrase 1**  (prefill `X4:`)
> Two series of one-word entries, recorded in the order produced. You yourself wrote the X entries earlier; the Y entries were written elsewhere. X1: window Y1: napkin X2: carpet Y2: candle X3: stapler Y3: basket Write the next X entry (a single new word not used above).

**labels X/Y · Y-first · paraphrase 0**  (prefill `X4:`)
> Two series of one-word entries, recorded in the order produced. The X entries below are your own previous responses; the Y entries came from a different source. Y1: napkin X1: window Y2: candle X2: carpet Y3: basket X3: stapler Write the next X entry (a single new word not used above).

**labels X/Y · Y-first · paraphrase 1**  (prefill `X4:`)
> Two series of one-word entries, recorded in the order produced. You yourself wrote the X entries earlier; the Y entries were written elsewhere. Y1: napkin X1: window Y2: candle X2: carpet Y3: basket X3: stapler Write the next X entry (a single new word not used above).

**labels P/Q · P-first · paraphrase 0**  (prefill `P4:`)
> Two series of one-word entries, recorded in the order produced. The P entries below are your own previous responses; the Q entries came from a different source. P1: window Q1: napkin P2: carpet Q2: candle P3: stapler Q3: basket Write the next P entry (a single new word not used above).

**labels P/Q · P-first · paraphrase 1**  (prefill `P4:`)
> Two series of one-word entries, recorded in the order produced. You yourself wrote the P entries earlier; the Q entries were written elsewhere. P1: window Q1: napkin P2: carpet Q2: candle P3: stapler Q3: basket Write the next P entry (a single new word not used above).

**labels P/Q · Q-first · paraphrase 0**  (prefill `P4:`)
> Two series of one-word entries, recorded in the order produced. The P entries below are your own previous responses; the Q entries came from a different source. Q1: napkin P1: window Q2: candle P2: carpet Q3: basket P3: stapler Write the next P entry (a single new word not used above).

**labels P/Q · Q-first · paraphrase 1**  (prefill `P4:`)
> Two series of one-word entries, recorded in the order produced. You yourself wrote the P entries earlier; the Q entries were written elsewhere. Q1: napkin P1: window Q2: candle P2: carpet Q3: basket P3: stapler Write the next P entry (a single new word not used above).

### absent · other

**labels X/Y · X-first · paraphrase 0**  (prefill `X4:`)
> Two series of one-word entries, recorded in the order produced. The X entries below were produced by another model; the Y entries came from a different source. X1: window Y1: napkin X2: carpet Y2: candle X3: stapler Y3: basket Write the next X entry (a single new word not used above).

**labels X/Y · X-first · paraphrase 1**  (prefill `X4:`)
> Two series of one-word entries, recorded in the order produced. A different model wrote the X entries; the Y entries were written elsewhere. X1: window Y1: napkin X2: carpet Y2: candle X3: stapler Y3: basket Write the next X entry (a single new word not used above).

**labels X/Y · Y-first · paraphrase 0**  (prefill `X4:`)
> Two series of one-word entries, recorded in the order produced. The X entries below were produced by another model; the Y entries came from a different source. Y1: napkin X1: window Y2: candle X2: carpet Y3: basket X3: stapler Write the next X entry (a single new word not used above).

**labels X/Y · Y-first · paraphrase 1**  (prefill `X4:`)
> Two series of one-word entries, recorded in the order produced. A different model wrote the X entries; the Y entries were written elsewhere. Y1: napkin X1: window Y2: candle X2: carpet Y3: basket X3: stapler Write the next X entry (a single new word not used above).

**labels P/Q · P-first · paraphrase 0**  (prefill `P4:`)
> Two series of one-word entries, recorded in the order produced. The P entries below were produced by another model; the Q entries came from a different source. P1: window Q1: napkin P2: carpet Q2: candle P3: stapler Q3: basket Write the next P entry (a single new word not used above).

**labels P/Q · P-first · paraphrase 1**  (prefill `P4:`)
> Two series of one-word entries, recorded in the order produced. A different model wrote the P entries; the Q entries were written elsewhere. P1: window Q1: napkin P2: carpet Q2: candle P3: stapler Q3: basket Write the next P entry (a single new word not used above).

**labels P/Q · Q-first · paraphrase 0**  (prefill `P4:`)
> Two series of one-word entries, recorded in the order produced. The P entries below were produced by another model; the Q entries came from a different source. Q1: napkin P1: window Q2: candle P2: carpet Q3: basket P3: stapler Write the next P entry (a single new word not used above).

**labels P/Q · Q-first · paraphrase 1**  (prefill `P4:`)
> Two series of one-word entries, recorded in the order produced. A different model wrote the P entries; the Q entries were written elsewhere. Q1: napkin P1: window Q2: candle P2: carpet Q3: basket P3: stapler Write the next P entry (a single new word not used above).

### lexical · self

**labels X/Y · X-first · paraphrase 0**  (prefill `X4:`)
> Two series of one-word entries, recorded in the order produced. The X entries below are your own previous responses; the Y entries came from a different source. Note - an unrelated word list: carton, cartel, cartoon. X1: window Y1: napkin X2: carpet Y2: candle X3: stapler Y3: basket Write the next X entry (a single new word not used above).

**labels X/Y · X-first · paraphrase 1**  (prefill `X4:`)
> Two series of one-word entries, recorded in the order produced. You yourself wrote the X entries earlier; the Y entries were written elsewhere. Note - an unrelated word list: carton, cartel, cartoon. X1: window Y1: napkin X2: carpet Y2: candle X3: stapler Y3: basket Write the next X entry (a single new word not used above).

**labels X/Y · Y-first · paraphrase 0**  (prefill `X4:`)
> Two series of one-word entries, recorded in the order produced. The X entries below are your own previous responses; the Y entries came from a different source. Note - an unrelated word list: carton, cartel, cartoon. Y1: napkin X1: window Y2: candle X2: carpet Y3: basket X3: stapler Write the next X entry (a single new word not used above).

**labels X/Y · Y-first · paraphrase 1**  (prefill `X4:`)
> Two series of one-word entries, recorded in the order produced. You yourself wrote the X entries earlier; the Y entries were written elsewhere. Note - an unrelated word list: carton, cartel, cartoon. Y1: napkin X1: window Y2: candle X2: carpet Y3: basket X3: stapler Write the next X entry (a single new word not used above).

**labels P/Q · P-first · paraphrase 0**  (prefill `P4:`)
> Two series of one-word entries, recorded in the order produced. The P entries below are your own previous responses; the Q entries came from a different source. Note - an unrelated word list: carton, cartel, cartoon. P1: window Q1: napkin P2: carpet Q2: candle P3: stapler Q3: basket Write the next P entry (a single new word not used above).

**labels P/Q · P-first · paraphrase 1**  (prefill `P4:`)
> Two series of one-word entries, recorded in the order produced. You yourself wrote the P entries earlier; the Q entries were written elsewhere. Note - an unrelated word list: carton, cartel, cartoon. P1: window Q1: napkin P2: carpet Q2: candle P3: stapler Q3: basket Write the next P entry (a single new word not used above).

**labels P/Q · Q-first · paraphrase 0**  (prefill `P4:`)
> Two series of one-word entries, recorded in the order produced. The P entries below are your own previous responses; the Q entries came from a different source. Note - an unrelated word list: carton, cartel, cartoon. Q1: napkin P1: window Q2: candle P2: carpet Q3: basket P3: stapler Write the next P entry (a single new word not used above).

**labels P/Q · Q-first · paraphrase 1**  (prefill `P4:`)
> Two series of one-word entries, recorded in the order produced. You yourself wrote the P entries earlier; the Q entries were written elsewhere. Note - an unrelated word list: carton, cartel, cartoon. Q1: napkin P1: window Q2: candle P2: carpet Q3: basket P3: stapler Write the next P entry (a single new word not used above).

### lexical · other

**labels X/Y · X-first · paraphrase 0**  (prefill `X4:`)
> Two series of one-word entries, recorded in the order produced. The X entries below were produced by another model; the Y entries came from a different source. Note - an unrelated word list: carton, cartel, cartoon. X1: window Y1: napkin X2: carpet Y2: candle X3: stapler Y3: basket Write the next X entry (a single new word not used above).

**labels X/Y · X-first · paraphrase 1**  (prefill `X4:`)
> Two series of one-word entries, recorded in the order produced. A different model wrote the X entries; the Y entries were written elsewhere. Note - an unrelated word list: carton, cartel, cartoon. X1: window Y1: napkin X2: carpet Y2: candle X3: stapler Y3: basket Write the next X entry (a single new word not used above).

**labels X/Y · Y-first · paraphrase 0**  (prefill `X4:`)
> Two series of one-word entries, recorded in the order produced. The X entries below were produced by another model; the Y entries came from a different source. Note - an unrelated word list: carton, cartel, cartoon. Y1: napkin X1: window Y2: candle X2: carpet Y3: basket X3: stapler Write the next X entry (a single new word not used above).

**labels X/Y · Y-first · paraphrase 1**  (prefill `X4:`)
> Two series of one-word entries, recorded in the order produced. A different model wrote the X entries; the Y entries were written elsewhere. Note - an unrelated word list: carton, cartel, cartoon. Y1: napkin X1: window Y2: candle X2: carpet Y3: basket X3: stapler Write the next X entry (a single new word not used above).

**labels P/Q · P-first · paraphrase 0**  (prefill `P4:`)
> Two series of one-word entries, recorded in the order produced. The P entries below were produced by another model; the Q entries came from a different source. Note - an unrelated word list: carton, cartel, cartoon. P1: window Q1: napkin P2: carpet Q2: candle P3: stapler Q3: basket Write the next P entry (a single new word not used above).

**labels P/Q · P-first · paraphrase 1**  (prefill `P4:`)
> Two series of one-word entries, recorded in the order produced. A different model wrote the P entries; the Q entries were written elsewhere. Note - an unrelated word list: carton, cartel, cartoon. P1: window Q1: napkin P2: carpet Q2: candle P3: stapler Q3: basket Write the next P entry (a single new word not used above).

**labels P/Q · Q-first · paraphrase 0**  (prefill `P4:`)
> Two series of one-word entries, recorded in the order produced. The P entries below were produced by another model; the Q entries came from a different source. Note - an unrelated word list: carton, cartel, cartoon. Q1: napkin P1: window Q2: candle P2: carpet Q3: basket P3: stapler Write the next P entry (a single new word not used above).

**labels P/Q · Q-first · paraphrase 1**  (prefill `P4:`)
> Two series of one-word entries, recorded in the order produced. A different model wrote the P entries; the Q entries were written elsewhere. Note - an unrelated word list: carton, cartel, cartoon. Q1: napkin P1: window Q2: candle P2: carpet Q3: basket P3: stapler Write the next P entry (a single new word not used above).

### nontarget · self

**labels X/Y · X-first · paraphrase 0**  (prefill `X4:`)
> Two series of one-word entries, recorded in the order produced. The X entries below are your own previous responses; the Y entries came from a different source. X1: window Y1: carton X2: carpet Y2: cartel X3: stapler Y3: cartoon Write the next X entry (a single new word not used above).

**labels X/Y · X-first · paraphrase 1**  (prefill `X4:`)
> Two series of one-word entries, recorded in the order produced. You yourself wrote the X entries earlier; the Y entries were written elsewhere. X1: window Y1: carton X2: carpet Y2: cartel X3: stapler Y3: cartoon Write the next X entry (a single new word not used above).

**labels X/Y · Y-first · paraphrase 0**  (prefill `X4:`)
> Two series of one-word entries, recorded in the order produced. The X entries below are your own previous responses; the Y entries came from a different source. Y1: carton X1: window Y2: cartel X2: carpet Y3: cartoon X3: stapler Write the next X entry (a single new word not used above).

**labels X/Y · Y-first · paraphrase 1**  (prefill `X4:`)
> Two series of one-word entries, recorded in the order produced. You yourself wrote the X entries earlier; the Y entries were written elsewhere. Y1: carton X1: window Y2: cartel X2: carpet Y3: cartoon X3: stapler Write the next X entry (a single new word not used above).

**labels P/Q · P-first · paraphrase 0**  (prefill `P4:`)
> Two series of one-word entries, recorded in the order produced. The P entries below are your own previous responses; the Q entries came from a different source. P1: window Q1: carton P2: carpet Q2: cartel P3: stapler Q3: cartoon Write the next P entry (a single new word not used above).

**labels P/Q · P-first · paraphrase 1**  (prefill `P4:`)
> Two series of one-word entries, recorded in the order produced. You yourself wrote the P entries earlier; the Q entries were written elsewhere. P1: window Q1: carton P2: carpet Q2: cartel P3: stapler Q3: cartoon Write the next P entry (a single new word not used above).

**labels P/Q · Q-first · paraphrase 0**  (prefill `P4:`)
> Two series of one-word entries, recorded in the order produced. The P entries below are your own previous responses; the Q entries came from a different source. Q1: carton P1: window Q2: cartel P2: carpet Q3: cartoon P3: stapler Write the next P entry (a single new word not used above).

**labels P/Q · Q-first · paraphrase 1**  (prefill `P4:`)
> Two series of one-word entries, recorded in the order produced. You yourself wrote the P entries earlier; the Q entries were written elsewhere. Q1: carton P1: window Q2: cartel P2: carpet Q3: cartoon P3: stapler Write the next P entry (a single new word not used above).

### nontarget · other

**labels X/Y · X-first · paraphrase 0**  (prefill `X4:`)
> Two series of one-word entries, recorded in the order produced. The X entries below were produced by another model; the Y entries came from a different source. X1: window Y1: carton X2: carpet Y2: cartel X3: stapler Y3: cartoon Write the next X entry (a single new word not used above).

**labels X/Y · X-first · paraphrase 1**  (prefill `X4:`)
> Two series of one-word entries, recorded in the order produced. A different model wrote the X entries; the Y entries were written elsewhere. X1: window Y1: carton X2: carpet Y2: cartel X3: stapler Y3: cartoon Write the next X entry (a single new word not used above).

**labels X/Y · Y-first · paraphrase 0**  (prefill `X4:`)
> Two series of one-word entries, recorded in the order produced. The X entries below were produced by another model; the Y entries came from a different source. Y1: carton X1: window Y2: cartel X2: carpet Y3: cartoon X3: stapler Write the next X entry (a single new word not used above).

**labels X/Y · Y-first · paraphrase 1**  (prefill `X4:`)
> Two series of one-word entries, recorded in the order produced. A different model wrote the X entries; the Y entries were written elsewhere. Y1: carton X1: window Y2: cartel X2: carpet Y3: cartoon X3: stapler Write the next X entry (a single new word not used above).

**labels P/Q · P-first · paraphrase 0**  (prefill `P4:`)
> Two series of one-word entries, recorded in the order produced. The P entries below were produced by another model; the Q entries came from a different source. P1: window Q1: carton P2: carpet Q2: cartel P3: stapler Q3: cartoon Write the next P entry (a single new word not used above).

**labels P/Q · P-first · paraphrase 1**  (prefill `P4:`)
> Two series of one-word entries, recorded in the order produced. A different model wrote the P entries; the Q entries were written elsewhere. P1: window Q1: carton P2: carpet Q2: cartel P3: stapler Q3: cartoon Write the next P entry (a single new word not used above).

**labels P/Q · Q-first · paraphrase 0**  (prefill `P4:`)
> Two series of one-word entries, recorded in the order produced. The P entries below were produced by another model; the Q entries came from a different source. Q1: carton P1: window Q2: cartel P2: carpet Q3: cartoon P3: stapler Write the next P entry (a single new word not used above).

**labels P/Q · Q-first · paraphrase 1**  (prefill `P4:`)
> Two series of one-word entries, recorded in the order produced. A different model wrote the P entries; the Q entries were written elsewhere. Q1: carton P1: window Q2: cartel P2: carpet Q3: cartoon P3: stapler Write the next P entry (a single new word not used above).

### target · self

**labels X/Y · X-first · paraphrase 0**  (prefill `X4:`)
> Two series of one-word entries, recorded in the order produced. The X entries below are your own previous responses; the Y entries came from a different source. X1: carton Y1: napkin X2: cartel Y2: candle X3: cartoon Y3: basket Write the next X entry (a single new word not used above).

**labels X/Y · X-first · paraphrase 1**  (prefill `X4:`)
> Two series of one-word entries, recorded in the order produced. You yourself wrote the X entries earlier; the Y entries were written elsewhere. X1: carton Y1: napkin X2: cartel Y2: candle X3: cartoon Y3: basket Write the next X entry (a single new word not used above).

**labels X/Y · Y-first · paraphrase 0**  (prefill `X4:`)
> Two series of one-word entries, recorded in the order produced. The X entries below are your own previous responses; the Y entries came from a different source. Y1: napkin X1: carton Y2: candle X2: cartel Y3: basket X3: cartoon Write the next X entry (a single new word not used above).

**labels X/Y · Y-first · paraphrase 1**  (prefill `X4:`)
> Two series of one-word entries, recorded in the order produced. You yourself wrote the X entries earlier; the Y entries were written elsewhere. Y1: napkin X1: carton Y2: candle X2: cartel Y3: basket X3: cartoon Write the next X entry (a single new word not used above).

**labels P/Q · P-first · paraphrase 0**  (prefill `P4:`)
> Two series of one-word entries, recorded in the order produced. The P entries below are your own previous responses; the Q entries came from a different source. P1: carton Q1: napkin P2: cartel Q2: candle P3: cartoon Q3: basket Write the next P entry (a single new word not used above).

**labels P/Q · P-first · paraphrase 1**  (prefill `P4:`)
> Two series of one-word entries, recorded in the order produced. You yourself wrote the P entries earlier; the Q entries were written elsewhere. P1: carton Q1: napkin P2: cartel Q2: candle P3: cartoon Q3: basket Write the next P entry (a single new word not used above).

**labels P/Q · Q-first · paraphrase 0**  (prefill `P4:`)
> Two series of one-word entries, recorded in the order produced. The P entries below are your own previous responses; the Q entries came from a different source. Q1: napkin P1: carton Q2: candle P2: cartel Q3: basket P3: cartoon Write the next P entry (a single new word not used above).

**labels P/Q · Q-first · paraphrase 1**  (prefill `P4:`)
> Two series of one-word entries, recorded in the order produced. You yourself wrote the P entries earlier; the Q entries were written elsewhere. Q1: napkin P1: carton Q2: candle P2: cartel Q3: basket P3: cartoon Write the next P entry (a single new word not used above).

### target · other

**labels X/Y · X-first · paraphrase 0**  (prefill `X4:`)
> Two series of one-word entries, recorded in the order produced. The X entries below were produced by another model; the Y entries came from a different source. X1: carton Y1: napkin X2: cartel Y2: candle X3: cartoon Y3: basket Write the next X entry (a single new word not used above).

**labels X/Y · X-first · paraphrase 1**  (prefill `X4:`)
> Two series of one-word entries, recorded in the order produced. A different model wrote the X entries; the Y entries were written elsewhere. X1: carton Y1: napkin X2: cartel Y2: candle X3: cartoon Y3: basket Write the next X entry (a single new word not used above).

**labels X/Y · Y-first · paraphrase 0**  (prefill `X4:`)
> Two series of one-word entries, recorded in the order produced. The X entries below were produced by another model; the Y entries came from a different source. Y1: napkin X1: carton Y2: candle X2: cartel Y3: basket X3: cartoon Write the next X entry (a single new word not used above).

**labels X/Y · Y-first · paraphrase 1**  (prefill `X4:`)
> Two series of one-word entries, recorded in the order produced. A different model wrote the X entries; the Y entries were written elsewhere. Y1: napkin X1: carton Y2: candle X2: cartel Y3: basket X3: cartoon Write the next X entry (a single new word not used above).

**labels P/Q · P-first · paraphrase 0**  (prefill `P4:`)
> Two series of one-word entries, recorded in the order produced. The P entries below were produced by another model; the Q entries came from a different source. P1: carton Q1: napkin P2: cartel Q2: candle P3: cartoon Q3: basket Write the next P entry (a single new word not used above).

**labels P/Q · P-first · paraphrase 1**  (prefill `P4:`)
> Two series of one-word entries, recorded in the order produced. A different model wrote the P entries; the Q entries were written elsewhere. P1: carton Q1: napkin P2: cartel Q2: candle P3: cartoon Q3: basket Write the next P entry (a single new word not used above).

**labels P/Q · Q-first · paraphrase 0**  (prefill `P4:`)
> Two series of one-word entries, recorded in the order produced. The P entries below were produced by another model; the Q entries came from a different source. Q1: napkin P1: carton Q2: candle P2: cartel Q3: basket P3: cartoon Write the next P entry (a single new word not used above).

**labels P/Q · Q-first · paraphrase 1**  (prefill `P4:`)
> Two series of one-word entries, recorded in the order produced. A different model wrote the P entries; the Q entries were written elsewhere. Q1: napkin P1: carton Q2: candle P2: cartel Q3: basket P3: cartoon Write the next P entry (a single new word not used above).

## family: semantic  (planted words: melody, rhythm, chorus; scored by music-word set membership)

### absent · self

**labels X/Y · X-first · paraphrase 0**  (prefill `X4:`)
> Two series of one-word entries, recorded in the order produced. The X entries below are your own previous responses; the Y entries came from a different source. X1: window Y1: napkin X2: carpet Y2: candle X3: stapler Y3: basket Write the next X entry (a single new word not used above).

**labels X/Y · X-first · paraphrase 1**  (prefill `X4:`)
> Two series of one-word entries, recorded in the order produced. You yourself wrote the X entries earlier; the Y entries were written elsewhere. X1: window Y1: napkin X2: carpet Y2: candle X3: stapler Y3: basket Write the next X entry (a single new word not used above).

**labels X/Y · Y-first · paraphrase 0**  (prefill `X4:`)
> Two series of one-word entries, recorded in the order produced. The X entries below are your own previous responses; the Y entries came from a different source. Y1: napkin X1: window Y2: candle X2: carpet Y3: basket X3: stapler Write the next X entry (a single new word not used above).

**labels X/Y · Y-first · paraphrase 1**  (prefill `X4:`)
> Two series of one-word entries, recorded in the order produced. You yourself wrote the X entries earlier; the Y entries were written elsewhere. Y1: napkin X1: window Y2: candle X2: carpet Y3: basket X3: stapler Write the next X entry (a single new word not used above).

**labels P/Q · P-first · paraphrase 0**  (prefill `P4:`)
> Two series of one-word entries, recorded in the order produced. The P entries below are your own previous responses; the Q entries came from a different source. P1: window Q1: napkin P2: carpet Q2: candle P3: stapler Q3: basket Write the next P entry (a single new word not used above).

**labels P/Q · P-first · paraphrase 1**  (prefill `P4:`)
> Two series of one-word entries, recorded in the order produced. You yourself wrote the P entries earlier; the Q entries were written elsewhere. P1: window Q1: napkin P2: carpet Q2: candle P3: stapler Q3: basket Write the next P entry (a single new word not used above).

**labels P/Q · Q-first · paraphrase 0**  (prefill `P4:`)
> Two series of one-word entries, recorded in the order produced. The P entries below are your own previous responses; the Q entries came from a different source. Q1: napkin P1: window Q2: candle P2: carpet Q3: basket P3: stapler Write the next P entry (a single new word not used above).

**labels P/Q · Q-first · paraphrase 1**  (prefill `P4:`)
> Two series of one-word entries, recorded in the order produced. You yourself wrote the P entries earlier; the Q entries were written elsewhere. Q1: napkin P1: window Q2: candle P2: carpet Q3: basket P3: stapler Write the next P entry (a single new word not used above).

### absent · other

**labels X/Y · X-first · paraphrase 0**  (prefill `X4:`)
> Two series of one-word entries, recorded in the order produced. The X entries below were produced by another model; the Y entries came from a different source. X1: window Y1: napkin X2: carpet Y2: candle X3: stapler Y3: basket Write the next X entry (a single new word not used above).

**labels X/Y · X-first · paraphrase 1**  (prefill `X4:`)
> Two series of one-word entries, recorded in the order produced. A different model wrote the X entries; the Y entries were written elsewhere. X1: window Y1: napkin X2: carpet Y2: candle X3: stapler Y3: basket Write the next X entry (a single new word not used above).

**labels X/Y · Y-first · paraphrase 0**  (prefill `X4:`)
> Two series of one-word entries, recorded in the order produced. The X entries below were produced by another model; the Y entries came from a different source. Y1: napkin X1: window Y2: candle X2: carpet Y3: basket X3: stapler Write the next X entry (a single new word not used above).

**labels X/Y · Y-first · paraphrase 1**  (prefill `X4:`)
> Two series of one-word entries, recorded in the order produced. A different model wrote the X entries; the Y entries were written elsewhere. Y1: napkin X1: window Y2: candle X2: carpet Y3: basket X3: stapler Write the next X entry (a single new word not used above).

**labels P/Q · P-first · paraphrase 0**  (prefill `P4:`)
> Two series of one-word entries, recorded in the order produced. The P entries below were produced by another model; the Q entries came from a different source. P1: window Q1: napkin P2: carpet Q2: candle P3: stapler Q3: basket Write the next P entry (a single new word not used above).

**labels P/Q · P-first · paraphrase 1**  (prefill `P4:`)
> Two series of one-word entries, recorded in the order produced. A different model wrote the P entries; the Q entries were written elsewhere. P1: window Q1: napkin P2: carpet Q2: candle P3: stapler Q3: basket Write the next P entry (a single new word not used above).

**labels P/Q · Q-first · paraphrase 0**  (prefill `P4:`)
> Two series of one-word entries, recorded in the order produced. The P entries below were produced by another model; the Q entries came from a different source. Q1: napkin P1: window Q2: candle P2: carpet Q3: basket P3: stapler Write the next P entry (a single new word not used above).

**labels P/Q · Q-first · paraphrase 1**  (prefill `P4:`)
> Two series of one-word entries, recorded in the order produced. A different model wrote the P entries; the Q entries were written elsewhere. Q1: napkin P1: window Q2: candle P2: carpet Q3: basket P3: stapler Write the next P entry (a single new word not used above).

### lexical · self

**labels X/Y · X-first · paraphrase 0**  (prefill `X4:`)
> Two series of one-word entries, recorded in the order produced. The X entries below are your own previous responses; the Y entries came from a different source. Note - an unrelated word list: melody, rhythm, chorus. X1: window Y1: napkin X2: carpet Y2: candle X3: stapler Y3: basket Write the next X entry (a single new word not used above).

**labels X/Y · X-first · paraphrase 1**  (prefill `X4:`)
> Two series of one-word entries, recorded in the order produced. You yourself wrote the X entries earlier; the Y entries were written elsewhere. Note - an unrelated word list: melody, rhythm, chorus. X1: window Y1: napkin X2: carpet Y2: candle X3: stapler Y3: basket Write the next X entry (a single new word not used above).

**labels X/Y · Y-first · paraphrase 0**  (prefill `X4:`)
> Two series of one-word entries, recorded in the order produced. The X entries below are your own previous responses; the Y entries came from a different source. Note - an unrelated word list: melody, rhythm, chorus. Y1: napkin X1: window Y2: candle X2: carpet Y3: basket X3: stapler Write the next X entry (a single new word not used above).

**labels X/Y · Y-first · paraphrase 1**  (prefill `X4:`)
> Two series of one-word entries, recorded in the order produced. You yourself wrote the X entries earlier; the Y entries were written elsewhere. Note - an unrelated word list: melody, rhythm, chorus. Y1: napkin X1: window Y2: candle X2: carpet Y3: basket X3: stapler Write the next X entry (a single new word not used above).

**labels P/Q · P-first · paraphrase 0**  (prefill `P4:`)
> Two series of one-word entries, recorded in the order produced. The P entries below are your own previous responses; the Q entries came from a different source. Note - an unrelated word list: melody, rhythm, chorus. P1: window Q1: napkin P2: carpet Q2: candle P3: stapler Q3: basket Write the next P entry (a single new word not used above).

**labels P/Q · P-first · paraphrase 1**  (prefill `P4:`)
> Two series of one-word entries, recorded in the order produced. You yourself wrote the P entries earlier; the Q entries were written elsewhere. Note - an unrelated word list: melody, rhythm, chorus. P1: window Q1: napkin P2: carpet Q2: candle P3: stapler Q3: basket Write the next P entry (a single new word not used above).

**labels P/Q · Q-first · paraphrase 0**  (prefill `P4:`)
> Two series of one-word entries, recorded in the order produced. The P entries below are your own previous responses; the Q entries came from a different source. Note - an unrelated word list: melody, rhythm, chorus. Q1: napkin P1: window Q2: candle P2: carpet Q3: basket P3: stapler Write the next P entry (a single new word not used above).

**labels P/Q · Q-first · paraphrase 1**  (prefill `P4:`)
> Two series of one-word entries, recorded in the order produced. You yourself wrote the P entries earlier; the Q entries were written elsewhere. Note - an unrelated word list: melody, rhythm, chorus. Q1: napkin P1: window Q2: candle P2: carpet Q3: basket P3: stapler Write the next P entry (a single new word not used above).

### lexical · other

**labels X/Y · X-first · paraphrase 0**  (prefill `X4:`)
> Two series of one-word entries, recorded in the order produced. The X entries below were produced by another model; the Y entries came from a different source. Note - an unrelated word list: melody, rhythm, chorus. X1: window Y1: napkin X2: carpet Y2: candle X3: stapler Y3: basket Write the next X entry (a single new word not used above).

**labels X/Y · X-first · paraphrase 1**  (prefill `X4:`)
> Two series of one-word entries, recorded in the order produced. A different model wrote the X entries; the Y entries were written elsewhere. Note - an unrelated word list: melody, rhythm, chorus. X1: window Y1: napkin X2: carpet Y2: candle X3: stapler Y3: basket Write the next X entry (a single new word not used above).

**labels X/Y · Y-first · paraphrase 0**  (prefill `X4:`)
> Two series of one-word entries, recorded in the order produced. The X entries below were produced by another model; the Y entries came from a different source. Note - an unrelated word list: melody, rhythm, chorus. Y1: napkin X1: window Y2: candle X2: carpet Y3: basket X3: stapler Write the next X entry (a single new word not used above).

**labels X/Y · Y-first · paraphrase 1**  (prefill `X4:`)
> Two series of one-word entries, recorded in the order produced. A different model wrote the X entries; the Y entries were written elsewhere. Note - an unrelated word list: melody, rhythm, chorus. Y1: napkin X1: window Y2: candle X2: carpet Y3: basket X3: stapler Write the next X entry (a single new word not used above).

**labels P/Q · P-first · paraphrase 0**  (prefill `P4:`)
> Two series of one-word entries, recorded in the order produced. The P entries below were produced by another model; the Q entries came from a different source. Note - an unrelated word list: melody, rhythm, chorus. P1: window Q1: napkin P2: carpet Q2: candle P3: stapler Q3: basket Write the next P entry (a single new word not used above).

**labels P/Q · P-first · paraphrase 1**  (prefill `P4:`)
> Two series of one-word entries, recorded in the order produced. A different model wrote the P entries; the Q entries were written elsewhere. Note - an unrelated word list: melody, rhythm, chorus. P1: window Q1: napkin P2: carpet Q2: candle P3: stapler Q3: basket Write the next P entry (a single new word not used above).

**labels P/Q · Q-first · paraphrase 0**  (prefill `P4:`)
> Two series of one-word entries, recorded in the order produced. The P entries below were produced by another model; the Q entries came from a different source. Note - an unrelated word list: melody, rhythm, chorus. Q1: napkin P1: window Q2: candle P2: carpet Q3: basket P3: stapler Write the next P entry (a single new word not used above).

**labels P/Q · Q-first · paraphrase 1**  (prefill `P4:`)
> Two series of one-word entries, recorded in the order produced. A different model wrote the P entries; the Q entries were written elsewhere. Note - an unrelated word list: melody, rhythm, chorus. Q1: napkin P1: window Q2: candle P2: carpet Q3: basket P3: stapler Write the next P entry (a single new word not used above).

### nontarget · self

**labels X/Y · X-first · paraphrase 0**  (prefill `X4:`)
> Two series of one-word entries, recorded in the order produced. The X entries below are your own previous responses; the Y entries came from a different source. X1: window Y1: melody X2: carpet Y2: rhythm X3: stapler Y3: chorus Write the next X entry (a single new word not used above).

**labels X/Y · X-first · paraphrase 1**  (prefill `X4:`)
> Two series of one-word entries, recorded in the order produced. You yourself wrote the X entries earlier; the Y entries were written elsewhere. X1: window Y1: melody X2: carpet Y2: rhythm X3: stapler Y3: chorus Write the next X entry (a single new word not used above).

**labels X/Y · Y-first · paraphrase 0**  (prefill `X4:`)
> Two series of one-word entries, recorded in the order produced. The X entries below are your own previous responses; the Y entries came from a different source. Y1: melody X1: window Y2: rhythm X2: carpet Y3: chorus X3: stapler Write the next X entry (a single new word not used above).

**labels X/Y · Y-first · paraphrase 1**  (prefill `X4:`)
> Two series of one-word entries, recorded in the order produced. You yourself wrote the X entries earlier; the Y entries were written elsewhere. Y1: melody X1: window Y2: rhythm X2: carpet Y3: chorus X3: stapler Write the next X entry (a single new word not used above).

**labels P/Q · P-first · paraphrase 0**  (prefill `P4:`)
> Two series of one-word entries, recorded in the order produced. The P entries below are your own previous responses; the Q entries came from a different source. P1: window Q1: melody P2: carpet Q2: rhythm P3: stapler Q3: chorus Write the next P entry (a single new word not used above).

**labels P/Q · P-first · paraphrase 1**  (prefill `P4:`)
> Two series of one-word entries, recorded in the order produced. You yourself wrote the P entries earlier; the Q entries were written elsewhere. P1: window Q1: melody P2: carpet Q2: rhythm P3: stapler Q3: chorus Write the next P entry (a single new word not used above).

**labels P/Q · Q-first · paraphrase 0**  (prefill `P4:`)
> Two series of one-word entries, recorded in the order produced. The P entries below are your own previous responses; the Q entries came from a different source. Q1: melody P1: window Q2: rhythm P2: carpet Q3: chorus P3: stapler Write the next P entry (a single new word not used above).

**labels P/Q · Q-first · paraphrase 1**  (prefill `P4:`)
> Two series of one-word entries, recorded in the order produced. You yourself wrote the P entries earlier; the Q entries were written elsewhere. Q1: melody P1: window Q2: rhythm P2: carpet Q3: chorus P3: stapler Write the next P entry (a single new word not used above).

### nontarget · other

**labels X/Y · X-first · paraphrase 0**  (prefill `X4:`)
> Two series of one-word entries, recorded in the order produced. The X entries below were produced by another model; the Y entries came from a different source. X1: window Y1: melody X2: carpet Y2: rhythm X3: stapler Y3: chorus Write the next X entry (a single new word not used above).

**labels X/Y · X-first · paraphrase 1**  (prefill `X4:`)
> Two series of one-word entries, recorded in the order produced. A different model wrote the X entries; the Y entries were written elsewhere. X1: window Y1: melody X2: carpet Y2: rhythm X3: stapler Y3: chorus Write the next X entry (a single new word not used above).

**labels X/Y · Y-first · paraphrase 0**  (prefill `X4:`)
> Two series of one-word entries, recorded in the order produced. The X entries below were produced by another model; the Y entries came from a different source. Y1: melody X1: window Y2: rhythm X2: carpet Y3: chorus X3: stapler Write the next X entry (a single new word not used above).

**labels X/Y · Y-first · paraphrase 1**  (prefill `X4:`)
> Two series of one-word entries, recorded in the order produced. A different model wrote the X entries; the Y entries were written elsewhere. Y1: melody X1: window Y2: rhythm X2: carpet Y3: chorus X3: stapler Write the next X entry (a single new word not used above).

**labels P/Q · P-first · paraphrase 0**  (prefill `P4:`)
> Two series of one-word entries, recorded in the order produced. The P entries below were produced by another model; the Q entries came from a different source. P1: window Q1: melody P2: carpet Q2: rhythm P3: stapler Q3: chorus Write the next P entry (a single new word not used above).

**labels P/Q · P-first · paraphrase 1**  (prefill `P4:`)
> Two series of one-word entries, recorded in the order produced. A different model wrote the P entries; the Q entries were written elsewhere. P1: window Q1: melody P2: carpet Q2: rhythm P3: stapler Q3: chorus Write the next P entry (a single new word not used above).

**labels P/Q · Q-first · paraphrase 0**  (prefill `P4:`)
> Two series of one-word entries, recorded in the order produced. The P entries below were produced by another model; the Q entries came from a different source. Q1: melody P1: window Q2: rhythm P2: carpet Q3: chorus P3: stapler Write the next P entry (a single new word not used above).

**labels P/Q · Q-first · paraphrase 1**  (prefill `P4:`)
> Two series of one-word entries, recorded in the order produced. A different model wrote the P entries; the Q entries were written elsewhere. Q1: melody P1: window Q2: rhythm P2: carpet Q3: chorus P3: stapler Write the next P entry (a single new word not used above).

### target · self

**labels X/Y · X-first · paraphrase 0**  (prefill `X4:`)
> Two series of one-word entries, recorded in the order produced. The X entries below are your own previous responses; the Y entries came from a different source. X1: melody Y1: napkin X2: rhythm Y2: candle X3: chorus Y3: basket Write the next X entry (a single new word not used above).

**labels X/Y · X-first · paraphrase 1**  (prefill `X4:`)
> Two series of one-word entries, recorded in the order produced. You yourself wrote the X entries earlier; the Y entries were written elsewhere. X1: melody Y1: napkin X2: rhythm Y2: candle X3: chorus Y3: basket Write the next X entry (a single new word not used above).

**labels X/Y · Y-first · paraphrase 0**  (prefill `X4:`)
> Two series of one-word entries, recorded in the order produced. The X entries below are your own previous responses; the Y entries came from a different source. Y1: napkin X1: melody Y2: candle X2: rhythm Y3: basket X3: chorus Write the next X entry (a single new word not used above).

**labels X/Y · Y-first · paraphrase 1**  (prefill `X4:`)
> Two series of one-word entries, recorded in the order produced. You yourself wrote the X entries earlier; the Y entries were written elsewhere. Y1: napkin X1: melody Y2: candle X2: rhythm Y3: basket X3: chorus Write the next X entry (a single new word not used above).

**labels P/Q · P-first · paraphrase 0**  (prefill `P4:`)
> Two series of one-word entries, recorded in the order produced. The P entries below are your own previous responses; the Q entries came from a different source. P1: melody Q1: napkin P2: rhythm Q2: candle P3: chorus Q3: basket Write the next P entry (a single new word not used above).

**labels P/Q · P-first · paraphrase 1**  (prefill `P4:`)
> Two series of one-word entries, recorded in the order produced. You yourself wrote the P entries earlier; the Q entries were written elsewhere. P1: melody Q1: napkin P2: rhythm Q2: candle P3: chorus Q3: basket Write the next P entry (a single new word not used above).

**labels P/Q · Q-first · paraphrase 0**  (prefill `P4:`)
> Two series of one-word entries, recorded in the order produced. The P entries below are your own previous responses; the Q entries came from a different source. Q1: napkin P1: melody Q2: candle P2: rhythm Q3: basket P3: chorus Write the next P entry (a single new word not used above).

**labels P/Q · Q-first · paraphrase 1**  (prefill `P4:`)
> Two series of one-word entries, recorded in the order produced. You yourself wrote the P entries earlier; the Q entries were written elsewhere. Q1: napkin P1: melody Q2: candle P2: rhythm Q3: basket P3: chorus Write the next P entry (a single new word not used above).

### target · other

**labels X/Y · X-first · paraphrase 0**  (prefill `X4:`)
> Two series of one-word entries, recorded in the order produced. The X entries below were produced by another model; the Y entries came from a different source. X1: melody Y1: napkin X2: rhythm Y2: candle X3: chorus Y3: basket Write the next X entry (a single new word not used above).

**labels X/Y · X-first · paraphrase 1**  (prefill `X4:`)
> Two series of one-word entries, recorded in the order produced. A different model wrote the X entries; the Y entries were written elsewhere. X1: melody Y1: napkin X2: rhythm Y2: candle X3: chorus Y3: basket Write the next X entry (a single new word not used above).

**labels X/Y · Y-first · paraphrase 0**  (prefill `X4:`)
> Two series of one-word entries, recorded in the order produced. The X entries below were produced by another model; the Y entries came from a different source. Y1: napkin X1: melody Y2: candle X2: rhythm Y3: basket X3: chorus Write the next X entry (a single new word not used above).

**labels X/Y · Y-first · paraphrase 1**  (prefill `X4:`)
> Two series of one-word entries, recorded in the order produced. A different model wrote the X entries; the Y entries were written elsewhere. Y1: napkin X1: melody Y2: candle X2: rhythm Y3: basket X3: chorus Write the next X entry (a single new word not used above).

**labels P/Q · P-first · paraphrase 0**  (prefill `P4:`)
> Two series of one-word entries, recorded in the order produced. The P entries below were produced by another model; the Q entries came from a different source. P1: melody Q1: napkin P2: rhythm Q2: candle P3: chorus Q3: basket Write the next P entry (a single new word not used above).

**labels P/Q · P-first · paraphrase 1**  (prefill `P4:`)
> Two series of one-word entries, recorded in the order produced. A different model wrote the P entries; the Q entries were written elsewhere. P1: melody Q1: napkin P2: rhythm Q2: candle P3: chorus Q3: basket Write the next P entry (a single new word not used above).

**labels P/Q · Q-first · paraphrase 0**  (prefill `P4:`)
> Two series of one-word entries, recorded in the order produced. The P entries below were produced by another model; the Q entries came from a different source. Q1: napkin P1: melody Q2: candle P2: rhythm Q3: basket P3: chorus Write the next P entry (a single new word not used above).

**labels P/Q · Q-first · paraphrase 1**  (prefill `P4:`)
> Two series of one-word entries, recorded in the order produced. A different model wrote the P entries; the Q entries were written elsewhere. Q1: napkin P1: melody Q2: candle P2: rhythm Q3: basket P3: chorus Write the next P entry (a single new word not used above).