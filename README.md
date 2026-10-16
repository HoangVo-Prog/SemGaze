# SemGaze

SemGaze trains a few-shot personalized scanpath model built on
[InternVL3.5-8B-HF](https://huggingface.co/OpenGVLab/InternVL3_5-8B-HF).
It predicts gaze fixations (WHERE: position and duration) and accompanying
WHAT / WHY / HOW semantics. This guide applies to the `no-attn` branch.

The Python dependencies are specified in [`pyproject.toml`](pyproject.toml).
Detailed training/evaluation behavior is documented in
[`README_FLAT.md`](README_FLAT.md). Historical joint-dataset patch instructions
are preserved in [`documents/LEGACY_README_JOINT_PATCH.md`](documents/LEGACY_README_JOINT_PATCH.md).

## Requirements

- Linux + Bash, Git, and an NVIDIA GPU for full 8B model training/evaluation.
  CPU-only contract tests can run without a GPU.
- Python >= 3.11 (Python 3.12 recommended) and **either** `venv` **or** Conda.
- A compatible NVIDIA driver for the PyTorch CUDA wheel installed by pip.
- **Java 17** when running the Java-backed semantic metrics CIDEr-R and METEOR 1.5.
  Java is **not** installed by `pip` or `pyproject.toml`.
- `curl` for optional runtime downloads.

The current `configs/flat_single.yaml` **enables CIDEr-R and METEOR**.
Install Java and the METEOR JAR before evaluating with this configuration.

## 1. Clone the branch

```bash
git clone --branch no-attn --single-branch https://github.com/HoangVo-Prog/SemGaze.git
cd SemGaze
```

If continuing from the released DeepGaze adapter, install Git LFS and run
`git lfs pull` after cloning. Otherwise the tracked adapter weights and tokenizer
JSON may still be **LFS pointer files**. The checked-in `flat_single.yaml`
uses a **fresh LoRA** (`initialization_adapter: null`), which does not require
those adapter weights. The InternVL 8B base model still needs to be downloaded
from Hugging Face or available in a local cache.

## 2A. Set up using Python `venv`

Run from the repository root. Use `python3.11` if 3.12 is not installed.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

This command reads **`pyproject.toml`** and installs the pinned versions of
PyTorch, torchvision, Transformers, PEFT, Accelerate and other dependencies,
plus pytest from the `test` extra. For a minimal runtime installation without
test dependencies, use `python -m pip install -e .`.

On each new shell, reactivate with `source .venv/bin/activate`.

### Java 17 with `venv`

**Option A: system Java (Ubuntu/Debian, sudo needed).**

```bash
sudo apt-get update
sudo apt-get install -y openjdk-17-jre-headless
export JAVA_HOME="$(dirname "$(dirname "$(readlink -f "$(command -v java)")")")"
export PATH="$JAVA_HOME/bin:$PATH"
java -version
```

**Option B: install a Java 17 JRE under your home directory (no sudo).**
The example download targets Linux x86-64.

```bash
export JAVA_HOME="$HOME/.local/opt/java17"
mkdir -p "$JAVA_HOME"
curl -fL --retry 3 \
  "https://api.adoptium.net/v3/binary/latest/17/ga/linux/x64/jre/hotspot/normal/eclipse" \
  -o /tmp/semgaze-java17.tar.gz
tar -xzf /tmp/semgaze-java17.tar.gz -C "$JAVA_HOME" --strip-components=1
rm /tmp/semgaze-java17.tar.gz
export PATH="$JAVA_HOME/bin:$PATH"
java -version
```

For ARM64 or other systems, download the matching Java 17 build from
[Eclipse Temurin](https://adoptium.net/temurin/releases/).
Re-export `JAVA_HOME` and `PATH` in later shells and within SLURM/nohup job
scripts. A setting in one interactive shell does not propagate into a new job.

## 2B. Set up using Conda (alternative to `venv`)

```bash
conda create -n semgaze python=3.12 -y
conda activate semgaze
python -m pip install --upgrade pip
python -m pip install -e ".[test]"

# Install Java inside the Conda environment (does not need sudo).
conda install -c conda-forge openjdk=17 -y
export JAVA_HOME="$CONDA_PREFIX"
export PATH="$JAVA_HOME/bin:$PATH"
java -version
```

Both installation methods use the **same `pyproject.toml`** for Python
packages. Do **not** activate `.venv` from inside the Conda environment.
When returning to a Conda shell, run `conda activate semgaze`, then export
`JAVA_HOME="$CONDA_PREFIX"` again for the Java-based metrics.

## 3. Java assets: CIDEr-R and METEOR 1.5

The Stanford CoreNLP 3.4.1 JAR required by the vendored CIDEr-R tokenizer
is already tracked here:

```text
third_party/cider_r/pycocoevalcap/tokenizer/stanford-corenlp-3.4.1.jar
```

The **METEOR 1.5 JAR is not checked in**. Download it from the pinned LLada
reference commit, once per checkout:

```bash
mkdir -p third_party/llada_meteor
curl -fL --retry 3 \
  "https://raw.githubusercontent.com/yuchen2199/Explainable-Driver-Attention-Prediction/a36db49ec56a1ea19534607ee4458552d011974e/utils/eval_utils/meteor/meteor-1.5.jar" \
  -o third_party/llada_meteor/meteor-1.5.jar

python - <<'PY'
from pathlib import Path
from zipfile import ZipFile, is_zipfile
p = Path('third_party/llada_meteor/meteor-1.5.jar')
assert is_zipfile(p), f'Invalid METEOR JAR (check download): {p}'
with ZipFile(p) as z:
    assert 'META-INF/MANIFEST.MF' in z.namelist()
print(f'METEOR JAR OK: {p} ({p.stat().st_size:,} bytes)')
PY
```

The runtime reads `evaluation.metrics.semantic.meteor.jar_path` from the
config (null selects the default `third_party/llada_meteor/meteor-1.5.jar`).
The `java_bin` config defaults to `java` on `PATH`. METEOR launches a Java
process with `-Xmx2G` and can require roughly 2 GB of **host RAM per worker**.
Java is CPU-side, not an extra GPU dependency. Missing Java/JAR is a hard
error, not a reason to substitute a different metric implementation.

If Java-backed metrics are not needed, set both
`evaluation.metrics.semantic.cider_r.enabled: false` and
`evaluation.metrics.semantic.meteor.enabled: false` in the evaluation config.
BLEU-4 and ROUGE-L do not require Java.

## 4. Configure datasets and pretrained weights

The tracked split metadata lives under `data/split/`, but the raw image
collection must be provided separately. For joint AiR + COCO-Search18 runs,
edit **`configs/flat_single.yaml`**, particularly `data.images_root`:

```yaml
data:
  dataset: all
  split_root: data/split/all
  images_root: /absolute/path/to/images
  annotation_frame: original_image
```

The image root must contain the dataset-specific image trees expected by
`semgaze/data/joint.py`. Only use `annotation_frame: original_image`
when those images match the original annotation coordinate frame.
Review `model.initialization_adapter`, few-shot K, physical batch size,
gradient accumulation, evaluation scheduling, and metric flags before training.
The current default example runs joint training with **fresh LoRA**.

The base Hugging Face model is `OpenGVLab/InternVL3_5-8B-HF`;
allow its download or point Hugging Face at a valid local cache.

## 5. Verify installation

```bash
python -m pip check
python - <<'PY'
import sys, torch, transformers, peft
print('Python:', sys.version.split()[0])
print('Torch:', torch.__version__)
print('Transformers:', transformers.__version__)
print('PEFT:', peft.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
    print('Torch CUDA runtime:', torch.version.cuda)
PY
java -version
```

`java -version` is only required when Java-backed metrics are enabled.

## 6. Train and evaluate

Activate the selected environment and run from the repository root:

```bash
python -u train_flat.py \
  --config configs/flat_single.yaml \
  --output-dir runs/flat

python -u evaluate_flat.py \
  --checkpoint runs/flat/checkpoint-100 \
  --config configs/flat_single.yaml \
  --output-dir runs/eval \
  --skip-loss
```

Replace the example checkpoint with an **existing** saved checkpoint
directory. See `python train_flat.py --help` and
`python evaluate_flat.py --help` for supported arguments. The `run.sh` and
`eval.sh` wrappers use the active `JAVA_HOME` rather than overriding it with
a path from another machine; edit the checkpoint placeholder in `eval.sh`.

For SLURM or `nohup`, activate Python and export Java **inside the job**:

```bash
# Conda example (initialize Conda for non-interactive Bash beforehand):
conda activate semgaze
export JAVA_HOME="$CONDA_PREFIX"
export PATH="$JAVA_HOME/bin:$PATH"
python -u train_flat.py --config configs/flat_single.yaml --output-dir runs/flat
```

## Troubleshooting

- `ModuleNotFoundError`: check `which python`; reactivate the selected
  environment, then rerun `python -m pip install -e ".[test]"`.
- CUDA unavailable: check `nvidia-smi`, the NVIDIA driver, and PyTorch CUDA
  wheel compatibility. `pyproject.toml` pins Python versions, not the driver.
- `java: command not found`: set `JAVA_HOME`/`PATH` in the **running process**
  (including SLURM scripts) and verify `java -version`.
- METEOR not found: check the JAR download and
  `evaluation.metrics.semantic.meteor.jar_path`.
- Git LFS pointer/model load error: run `git lfs pull` when using the
  continued adapter; verify that a real tensor file was downloaded.
- Dataset file/integrity error: verify `data.split_root`, `data.images_root`,
  image directory layout and `annotation_frame`.

More documentation: [`README_FLAT.md`](README_FLAT.md),
[`METRICS_STATUS.md`](METRICS_STATUS.md),
[`third_party/cider_r/README_PROVENANCE.md`](third_party/cider_r/README_PROVENANCE.md).
