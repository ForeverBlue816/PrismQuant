# Reproduction and research archive

The public release separates reusable implementation and model use from the
complete experimental record. No experiment history was rewritten.

The immutable tag
[`research-archive-2026-09-25`](https://github.com/ForeverBlue816/PrismQuant/tree/research-archive-2026-09-25)
points to commit `728e6e5a7fe6395d8c38c5a286295fc87926ae8a`, containing all
experiments through E36, the report, CSV results, figure sources, pre-registrations
and execution records. The release removes those files from the current tree;
the tag and prior branches retain them.

```bash
git clone --branch research-archive-2026-09-25 --depth 1 \
  https://github.com/ForeverBlue816/PrismQuant.git prismquant-research
```

- [Full report](https://github.com/ForeverBlue816/PrismQuant/blob/research-archive-2026-09-25/report.md)
- [Calibration and quantization protocol](https://github.com/ForeverBlue816/PrismQuant/blob/research-archive-2026-09-25/nar/README.md)
- [E29–E33 reproduction](https://github.com/ForeverBlue816/PrismQuant/blob/research-archive-2026-09-25/experiments/e29_e33_README.md)
- [E34 attribution](https://github.com/ForeverBlue816/PrismQuant/blob/research-archive-2026-09-25/experiments/e34_README.md)
- [E36 offset precision](https://github.com/ForeverBlue816/PrismQuant/blob/research-archive-2026-09-25/experiments/e36_README.md)

The archived jobs use the original cluster environment and storage layout.
For downloading and using a released model on another machine, follow the
[public model guide](models.md) instead. The public package requires no Slurm,
cluster paths or private experiment caches.

The author-provided final PDFs in `Figures/` take precedence over older figure
exports. Their source hashes are recorded in `Figures/provenance.json`; these
are presentation assets and do not alter archived measurements.
