<div align="center">
  <h1>How Do Large Language Models Acquire Factual Knowledge During Pretraining? (https://arxiv.org/abs/2406.11813)</h1>
</div>


This repository contains the code, dataset, and experimental log data  for the paper 'How Do Large Language Models Acquire Factual Knowledge During Pretraining?'. The code is based on the [OLMo](https://github.com/allenai/OLMo) project, with modifications to support knowledge injection during pre-training and additional analysis tools.

You can also find the Fictional Knowledge dataset on HuggingFace: https://huggingface.co/datasets/kaist-ai/fictional-knowledge

## Key Differences from Original OLMo Repository

1. Modified `olmo/train.py` to:
   - Apply knowledge injection during pre-training
   - Log perplexity measurement data on fictional knowledge dataset probes during pretraining
2. Added post-processing of perplexity logs and analysis utils in `analysis/` folder
3. Added post-processed perplexity logs for main experiments using three OLMo-7B intermediate checkpoints in `analysis/results/` folder (can be downloaded using Git LFS)

## Installation

```bash
git clone factual-knowledge-acquisition
cd factual-knowledge-acquisition
pip install -e .
```

## Training

1. Extract Dolma corpus starting from step 360000:
```bash
python analysis/extract_data.py \
    --input_path <path_to_original_dolma_data> \
    --output_path <path_to_extracted_data> \
    --start_step 360000
```
- Note that the path to the full Dolma corpus should be specified in `configs/official/OLMo-7B.yaml` before running this code.
- Running this will generate `dolma_extracted/360000-363000.npy` file, which contains tokenized batch sequence data at training step 360000-363000.

2. Example training script:

```bash
BASE_STEP=5000
RUN_NAME=EXP_NAME

export SCRATCH_DIR='PATH_TO_THE_REPOSITORY'

python -m torch.distributed.run --nproc_per_node=4 ${SCRATCH_DIR}/scripts/train.py configs/official/OLMo-1B-105.yaml \
    --run_name=${RUN_NAME} \
    --data.paths=[${SCRATCH_DIR}/dolma_extracted/360000-363100.npy] \
    --load_path=PATH_TO_THE_CHECKPOINT/step${BASE_STEP}-unsharded \
    --base_step=${BASE_STEP} \
    --inject_indices_map=${SCRATCH_DIR}/analysis/inject_indices_map/7b-360000.pkl \
    --save_overwrite
```

## Analysis

### 1. Post-process Perplexity Logs

```bash
python analysis/postprocess_ppl.py \
    --base_dir PATH_TO_THE_REPOSITORY \
    --exp_name EXP_NAME
```

### 2. Run Analysis

#### a. Draw Loss Figures (reproduce Fig.2 in the paper)

```bash
exp_name=EXP_NAME
save_dir=PATH_TO_SAVED_FIGURES
base_dir=PATH_TO_THE_REPOSITORY

python ppl_analysis.py \
    --mode=draw_figures \
    --base_dir=${base_dir} \
    --exp_name=${exp_name} \
    --save_dir=${save_dir} \
    --no_take_exp
```

#### b. Measure Effectivity Scores

```bash
exp_name=EXP_NAME
base_dir=PATH_TO_THE_REPOSITORY

python analysis/ppl_analysis.py \
    --mode=measure_scores \
    --base_dir=${base_dir} \
    --skip_log_forgetting \
    --absolute \
    --no_take_exp \
    --exp_name=${exp_name}
```

#### c. Measure Retainability Scores

```bash
exp_name=EXP_NAME
base_dir=PATH_TO_THE_REPOSITORY

python analysis/ppl_analysis.py \
    --mode=measure_scores \
    --base_dir=${base_dir} \
    --skip_log_effectivity \
    --absolute \
    --no_take_exp \
    --exp_name=${exp_name}
```

After running the retainability measurement, you'll find the log file in `analysis/forgetting_measurements/` folder with the same `exp_name`.

#### d. Draw Retainability Figures & Measure Decay Constants

```bash
python analysis/forgetting_plot.py \
    --exp_name PATH_TO_FORGETTING_MEASUREMENT_FILE
```

## Citation

If you use this code in your research, please cite both this repository and the original OLMo paper:

```bibtex
@article{chang2024large,
  title={How Do Large Language Models Acquire Factual Knowledge During Pretraining?},
  author={Chang, Hoyeon and Park, Jinho and Ye, Seonghyeon and Yang, Sohee and Seo, Youngkyung and Chang, Du-Seong and Seo, Minjoon},
  journal={arXiv preprint arXiv:2406.11813},
  year={2024}
}

@article{OLMo,
  title={OLMo: Accelerating the Science of Language Models},
  author={Dirk Groeneveld and Iz Beltagy and Pete Walsh and Akshita Bhagia and Rodney Kinney and Oyvind Tafjord and A. Jha and Hamish Ivison and Ian Magnusson and Yizhong Wang and Shane Arora and David Atkinson and Russell Authur and Khyathi Raghavi Chandu and Arman Cohan and Jennifer Dumas and Yanai Elazar and Yuling Gu and Jack Hessel and Tushar Khot and William Merrill and Jacob Daniel Morrison and Niklas Muennighoff and Aakanksha Naik and Crystal Nam and Matthew E. Peters and Valentina Pyatkin and Abhilasha Ravichander and Dustin Schwenk and Saurabh Shah and Will Smith and Emma Strubell and Nishant Subramani and Mitchell Wortsman and Pradeep Dasigi and Nathan Lambert and Kyle Richardson and Luke Zettlemoyer and Jesse Dodge and Kyle Lo and Luca Soldaini and Noah A. Smith and Hanna Hajishirzi},
  year={2024},
  url={https://api.semanticscholar.org/CorpusID:267365485},
  journal={arXiv preprint},
}
```

## License

Apache 2.0
