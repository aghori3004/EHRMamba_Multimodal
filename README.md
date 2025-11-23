# Multimodal EHRMamba: Integrating Clinical Notes with State Space Models
**Student:** Divyansh Gangwar | **Course:** CS F434: Data Science for Healthcare
**Semester:** First Semester 2025-2026 | **Instructor:** Prof. Manik Gupta

## GitHub Repository & Reproducibility
This notebook serves as the main entry point for the project. The complete source code, including data pipelines and training scripts, is modularized in the `src/` directory to ensure **full reproducibility**.

### How to Run This Project

#### 1. System Requirements:
* **OS:** Linux (Ubuntu 20.04+) or WSL2
* **GPU:** NVIDIA RTX 4060 (8GB VRAM) or better
* **Python:** 3.10 (Required for Mamba compilation)
* **CUDA:** 12.1

#### 2. Installation:

**STEP 1: Clone the repository**
<pre>
git clone https://github.com/aghori3004/EHRMamba_Multimodal.git
cd EHRMamba-Multimodal/
</pre>
<br>

**STEP 2: Create Environment (Strictly use Python 3.10)**
<pre>
python3.10 -m venv .venv
source .venv/bin/activate
</pre>
<br>

**STEP 3: Install Dependencies (PyTorch 2.1.2 + Mamba-SSM)**
<pre>
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
pip install packaging ninja transformers pandas scikit-learn tqdm matplotlib seaborn
</pre>
**Note:** Compiling Mamba takes
<pre>
TORCH_CUDA_ARCH_LIST="8.9" pip install causal-conv1d>=1.2.0 mamba-ssm --no-cache-dir
</pre>

#### 3. Execution: We provide a master script to regenerate all artifacts (Data -> Models -> Metrics):
<pre>
./run_pipeline.sh
</pre>
