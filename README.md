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

### ScienceQA

We also evaluate the experiment results on SceinceQA following [LaVIN](https://github.com/luogen1996/LaVIN/tree/main)

#### Experiments results

<p>Table B: Comparison of DAS and PETL methods on ScienceQA for LLaMA.</p>
<table>
<thead>
<tr>
<th>Method</th>
<th>Update Params</th>
<th>FLOPs(G)</th>
<th>Modality Natural</th>
<th>Modality Social</th>
<th>Modality Language</th>
<th>Context Text</th>
<th>Context Image</th>
<th>Context No</th>
<th>Grade G1-6</th>
<th>Grade G7-12</th>
<th>Avg</th>
</tr>
</thead>
<tbody><tr>
<td>LLaVA-13B</td>
<td>100%</td>
<td>-</td>
<td>90.36</td>
<td>95.95</td>
<td>88.00</td>
<td>89.49</td>
<td>88.00</td>
<td>90.66</td>
<td>90.93</td>
<td>90.90</td>
<td>90.92</td>
</tr>
<tr>
<td>LaVIN-7B</td>
<td>3.8M</td>
<td>833</td>
<td>89.25</td>
<td>94.94</td>
<td>85.24</td>
<td>88.51</td>
<td>87.46</td>
<td>88.08</td>
<td>90.16</td>
<td>88.07</td>
<td>89.41</td>
</tr>
<tr>
<td>DAS4-7B</td>
<td>44.26M</td>
<td>729 (-18.61%)</td>
<td>90.54</td>
<td>94.26</td>
<td>86.82</td>
<td>89.74</td>
<td>87.65</td>
<td>89.76</td>
<td>90.97</td>
<td>89.26</td>
<td>90.36</td>
</tr>
<tr>
<td>DAS6-7B</td>
<td>44.26M</td>
<td>678 (-24.85%)</td>
<td>89.96</td>
<td>94.71</td>
<td>87.18</td>
<td>89.00</td>
<td>87.7</td>
<td>89.97</td>
<td>90.75</td>
<td>89.32</td>
<td>90.24</td>
</tr>
</tbody></table>

## Acknowledgements

The code is based on [ViLT](https://github.com/dandelin/ViLT) licensed under [Apache 2.0](https://github.com/dandelin/ViLT/blob/master/LICENSE) and [METER](https://github.com/zdou0830/METER/tree/main) licensed under [MIT](https://github.com/zdou0830/METER/blob/main/LICENSE) and some of the code is borrowed from [CLIP](https://github.com/openai/CLIP) and [Swin-Transformer](https://github.com/microsoft/Swin-Transformer).