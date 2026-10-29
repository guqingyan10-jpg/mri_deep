# Pinned model source for offline AutoDL training

These are unchanged source files exported from the authors' Git blobs, with their license texts preserved beside them. No pretrained weights are included.

| Local model file | Author repository | Commit | SHA-256 of model file | License |
| --- | --- | --- | --- | --- |
| `doubleblock/DB_MaxViT.py` | [Laptq201/DoubleBlock-ViT-Unet-segment](https://github.com/Laptq201/DoubleBlock-ViT-Unet-segment) | `5b617d72608986f38e602f4dd69b95d45443f5c7` | `1b9caf37be4b0c1b4fd00d16dd9ef4dfd71610bcf2e009c1bc4124aa5176c8b2` | MIT; see `doubleblock/LICENSE` |
| `superlight/superlightnet.py` | [WTU-MIS-Laboratory/SuperLightNet](https://github.com/WTU-MIS-Laboratory/SuperLightNet) | `0d6532434586dd68750bf417b158511f46f4dd75` | `b613d7e09f5795274e04bb71c66096fc9477aec03b80d3c570cbf9a5a278b303` | GPL-3.0; see `superlight/LICENSE` |

`models/paper_baselines.py` checks these SHA-256 values before import. DoubleBlock's top-level CUDA demo statements are skipped at load time without modifying this file. The released SuperLightNet source is used directly; its repository's separate training entrypoint refers to a module absent from the public checkout, as documented in `docs/PAPER_BASELINES.md`.
