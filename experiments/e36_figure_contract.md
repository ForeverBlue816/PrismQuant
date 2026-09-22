# E36 figure contract

Question: Does fp16 offset rounding approach a meaningful fraction of one INT4 step, and is that tail error specific to PrismQuant?

Claim before results: the figure will bound the measured tail error across every layer and both quantized sites; its eventual caption will follow the observed data, without calling nonzero storage error lossless.

Archetype: quantitative grid. Backend: the repository's Python/matplotlib workflow (saved preference confirmed). Four comparable panels in one wide row: Qwen qkv, Qwen down, Llama qkv, Llama down. The qkv/down split locates different activation distributions; the second model checks generality. Both methods appear in every panel. No layer/site/seed is omitted.

The main line is each layer/site's exact pooled P99 offset-rounding error divided by the stored fp16 step. The light band spans the three individual seed P99 values, a descriptive seed envelope rather than a confidence interval. Include the half-step threshold, an explicit log axis if all P99/envelope values are positive, otherwise a symlog axis preserving zero. No smoothing/interpolation, no data-selected layers, no clipping or broken axes. PPL precision contrasts and rare-token strata belong in the accompanying tables, not redundant figure panels.

Export: approximately 183 mm wide by 70 mm high; white background, thin lines, one shared legend, editable PDF/SVG text, 600-dpi PNG preview. Use the paper's restrained method colors with a contrasting line style. At least 5 pt for every rendered glyph. Retain the complete figure source CSV and all three seed values.

QA: measure final axes rectangles and panel-label anchors with the required 1.5 pt alignment gate; run source validation, PDF text-size and PDF collision audits; inspect the full figure and all four panels at final physical size. Document intentional uncertainty fills, any audit warnings and the numerical axis range. Do not claim venue submission clearance; this is a paper-ready experiment figure, not an official venue-format certification.
