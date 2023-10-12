# METER: A Multimodal End-to-end TransformER Framework

## Install

```bash
pip install -r requirements.txt
pip install -e .
```

## Dataset Preparation

We follow [ViLT](https://github.com/dandelin/ViLT) and use `pyarrow` to serialize the datasets. See [this link](https://github.com/dandelin/ViLT/blob/master/DATA.md) for details.

## Fine-tuning on Downstream Tasks

Work on the METER:

```bash
cd METER
```

Work on the ViLT:

```bash
cd ViLT
```

### VQAv2

#### Search

```bash
sh script/vqa_search.sh
```

#### Train

Add search result to vqa_train.sh by additional parameter 'skip_module'.  

```bash
sh script/vqa_train.sh
```

#### Evaluate

Add the path of checkpoint and 'skip_module' to vqa_eval.sh.

```bash
sh script/vqa_eval.sh
```

### Flickr30k IR/TR

#### Search

```bash
sh script/F30K_search.sh
```

#### Train

Add search result to F30K_train.sh by additional parameter 'skip_module'.  

```bash
sh script/F30K_train.sh
```

#### Evaluate

Add the path of checkpoint and 'skip_module' to F30K_eval.sh.

```bash
sh script/F30K_eval.sh
```

### NLVR2

#### Search

```bash
sh script/nlvr_search.sh
```

#### Train

Add search result to F30K_train.sh by additional parameter 'skip_module'.  

```bash
sh script/nlvr_train.sh
```

#### Evaluate

Add the path of checkpoint and 'skip_module' to nlvr_eval.sh.

```bash
sh script/nlvr_eval.sh
```

## Acknowledgements

The code is based on [ViLT](https://github.com/dandelin/ViLT) licensed under [Apache 2.0](https://github.com/dandelin/ViLT/blob/master/LICENSE) and [METER](https://github.com/zdou0830/METER/tree/main) licensed under [MIT](https://github.com/zdou0830/METER/blob/main/LICENSE) and some of the code is borrowed from [CLIP](https://github.com/openai/CLIP) and [Swin-Transformer](https://github.com/microsoft/Swin-Transformer).