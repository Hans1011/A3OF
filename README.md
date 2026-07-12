# A3OF
An Autonomous, Apprehensible, and Accelerated Optimization Framework for Nanoparticle Interfacial Dynamics

# An Autonomous, Apprehensible, and Accelerated Optimization Framework for Nanoparticle Interfacial Dynamics

This repository contains the source code, demonstration data, and documentation for an autonomous multi-agent framework designed to analyze and optimize nanoparticle interfacial dynamics.

The framework integrates machine learning, SHAP-based model interpretation, scientific literature retrieval, multimodal evidence analysis, and large language model agents. It is designed to identify experimentally meaningful nonlinear associations between formulation variables and the interfacial behavior of magnetic nanoparticles.

> This repository accompanies the manuscript  
> **“An Autonomous, Apprehensible, and Accelerated Optimization Framework for Nanoparticle Interfacial Dynamics.”**

---

## Framework overview

![Overview of the autonomous multi-agent optimization framework](docs/framework-overview.png)

**Figure 1.** Overview of the proposed autonomous, apprehensible, and accelerated optimization framework. Experimental data, microscopy-derived descriptors, machine-learning predictions, SHAP interpretations, and scientific literature are processed by specialized agents to generate evidence-supported experimental associations and optimization recommendations.

The main workflow consists of the following stages:

1. Experimental data preprocessing.
2. Machine-learning model training and prediction.
3. SHAP-based global and local interpretation.
4. Multimodal integration of experimental images and image descriptors.
5. Scientific literature extraction and retrieval.
6. Multi-agent evidence analysis.
7. Evaluation and selection of experimentally meaningful associations.
8. Generation of optimization recommendations and structured outputs.

---

## Repository structure

```text
nanoparticle-interfacial-optimization/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── .env.example
│
├── multiagent_pipeline.py
├── Extraction agent.py
├── searching agent.py
├── mining agent.py
├── query agent.py
├── judging agent.py
├── writing agent.py
│
├── demo/
│   ├── demo_data.csv
│   ├── demo_caption.csv
│   ├── demo_query.json
│   └── expected_output/
│       └── example_result.json
│
├── docs/
│   ├── framework-overview.png
│   └── reproduction.md
│
├── Papers/
└── results/
```

The names and locations of individual scripts may differ slightly between software releases.

---

# 1. System requirements

## Supported operating systems

The software has been developed and tested on:

- Windows 11, 64-bit
- Python 3.11
- Conda-based Python environment

Linux and macOS have not yet been formally validated.

## Software dependencies

The principal dependencies include:

- Python 3.11
- NumPy
- pandas
- scikit-learn
- XGBoost
- SHAP
- OpenAI Python SDK
- LangChain
- ChromaDB
- Fast-GraphRAG
- hnswlib
- python-dotenv
- pypdf
- requests
- Matplotlib

Exact package versions are provided in [`requirements.txt`](requirements.txt).

## External services

Some components require access to external services:

- OpenAI API for language-model inference and text embeddings.
- MinerU API for PDF parsing and scientific-document conversion.

Internet access is therefore required when these components are enabled.

## Tested configuration

The framework has been tested with the following configuration:

- Operating system: Windows 11, 64-bit
- Python: 3.11
- Environment manager: Conda
- CPU: Standard multi-core desktop processor
- Memory: 16 GB RAM or greater recommended
- GPU: Not required
- Storage: At least 5 GB of free disk space recommended

## Non-standard hardware

No non-standard hardware is required.

A GPU is not required for the demonstration workflow. Model training and SHAP analysis can be performed on a standard desktop CPU for the supplied demonstration dataset.

---

# 2. Installation guide

## Clone the repository

```bash
git clone https://github.com/<YOUR_GITHUB_USERNAME>/nanoparticle-interfacial-optimization.git
cd nanoparticle-interfacial-optimization
```

Replace `<YOUR_GITHUB_USERNAME>` with the owner of the GitHub repository.

## Create a Conda environment

```bash
conda create -n nanoparticle-framework python=3.11
conda activate nanoparticle-framework
```

## Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Configure API credentials

Copy the example environment file:

### Windows PowerShell

```powershell
Copy-Item ".env.example" ".env"
```

### Linux or macOS

```bash
cp .env.example .env
```

Open `.env` and provide the required credentials:

```dotenv
OPENAI_API_KEY=your_openai_api_key
MINERU_TOKEN=your_mineru_api_token
```

Do not commit the real `.env` file to GitHub.

The provided `.gitignore` excludes `.env` and other credential files.

## Typical installation time

Installation typically takes approximately **5–15 minutes** on a normal desktop computer with a stable internet connection.

Installation time may vary depending on network speed and whether Conda or pip packages are already cached.

---

# 3. Demo

## Demonstration data

A small demonstration dataset is provided in the `demo` directory.

The demonstration dataset contains representative experimental variables for nanoparticle transport across an oil–water interface, including:

| Column | Type | Description |
|---|---|---|
| `surfactant_in_water` | Categorical | Surfactant used in the aqueous phase |
| `oil_type` | Categorical | Oil-phase material |
| `surfactant_in_oil` | Categorical | Surfactant used in the oil phase |
| `ratio_of_surfactant_in_water` | Numeric | Surfactant ratio in the aqueous phase |
| `ratio_of_surfactant_in_oil` | Numeric | Surfactant ratio in the oil phase |
| `ion_concentration_in_water` | Numeric | Ionic concentration in the aqueous phase |
| `reward` | Numeric | Experimental performance target |
| `picindex` | String | Identifier of the corresponding experimental image |

The demonstration data are intended only to verify installation and illustrate the software workflow. They are not intended to reproduce every quantitative result reported in the manuscript.

## Run the demonstration

Activate the environment:

```bash
conda activate nanoparticle-framework
```

Run the main pipeline:

```bash
python multiagent_pipeline.py
```

If the demonstration uses a dedicated input argument, run:

```bash
python multiagent_pipeline.py --data demo/demo_data.csv
```

On Windows, scripts containing spaces in their filenames should be enclosed in quotation marks:

```powershell
python "mining agent.py"
```

## Expected output

A successful demonstration run generates structured outputs such as:

```text
results/
├── associations.json
├── query_results.json
├── model_metrics.json
└── final_report.md
```

A representative association output has the following structure:

```json
{
  "associations": [
    {
      "rank": 1,
      "description": "A nonlinear dependence between an experimental variable and nanoparticle interfacial transport.",
      "evidence": "Supporting experimental and literature-derived evidence."
    },
    {
      "rank": 2,
      "description": "A second evidence-supported experimental association.",
      "evidence": "Supporting evidence."
    },
    {
      "rank": 3,
      "description": "A third evidence-supported experimental association.",
      "evidence": "Supporting evidence."
    }
  ]
}
```

A reference result is provided at:

```text
demo/expected_output/example_result.json
```

Because the framework uses stochastic machine-learning procedures and external language models, the wording of generated explanations may vary between runs. The principal scientific trends should remain consistent when the same data, configuration, model versions, and random seeds are used.

## Expected runtime

Typical execution time on a normal desktop computer is approximately:

- Data preprocessing and model fitting: 1–5 minutes
- SHAP analysis: 1–5 minutes
- Small offline demonstration: 2–10 minutes
- Literature-assisted workflow: 10–60 minutes

Runtime depends on:

- Dataset size
- Number and length of scientific papers
- Network speed
- External API response times
- API rate limits
- Number of multi-agent analysis steps

## API usage notice

The complete demonstration may send requests to OpenAI and MinerU and may incur API usage charges.

Users should review the selected models, number of documents, and expected API usage before running the complete literature-assisted pipeline.

---

# 4. Instructions for use

## Prepare experimental data

Prepare a CSV file containing the required experimental variables.

Example:

```csv
surfactant_in_water,oil_type,surfactant_in_oil,ratio_of_surfactant_in_water,ratio_of_surfactant_in_oil,ion_concentration_in_water,reward,picindex
CEDA,oil_1,ODEA,0.10,0.20,0.05,0.82,image_001.png
LDEA,oil_2,CEDA,0.15,0.10,0.08,0.71,image_002.png
```

Categorical values must be represented consistently. Numeric columns must not contain units or free-form text.

## Prepare image descriptors

If multimodal analysis is enabled, prepare a caption CSV file containing image-derived descriptors.

Example:

```csv
image,phase,cluster_uniformity,residue_level,interface_cross,satellite_cluster_ratio
image_001.png,oil,yes,very little,yes,0.05
image_002.png,interface,no,extensive,no,0.32
```

The framework can use the following qualitative descriptors:

- `phase`
- `cluster_uniformity`
- `residue_level`
- `interface_cross`
- `satellite_cluster_ratio`

## Prepare scientific literature

Place authorized PDF files in:

```text
Papers/
```

The user is responsible for ensuring that all documents are used in accordance with their licenses and copyright conditions.

When MinerU conversion is enabled, the extracted text is stored in the configured converted-text directory.

## Run individual agents

The pipeline contains specialized agents that can also be executed independently.

```powershell
python "Extraction agent.py"
python "searching agent.py"
python "mining agent.py"
python "query agent.py"
python "judging agent.py"
python "writing agent.py"
```

## Run the complete pipeline

```bash
python multiagent_pipeline.py
```

The complete pipeline performs:

1. Data loading and preprocessing.
2. Model training.
3. SHAP interpretation.
4. Literature retrieval.
5. Evidence extraction.
6. Candidate association generation.
7. Association evaluation.
8. Scientific report generation.

## Use your own data

To analyze a new dataset:

1. Copy the dataset into the project directory.
2. Ensure that required column names are present.
3. Update the relevant input path in the configuration or command-line arguments.
4. Add optional image descriptors.
5. Add authorized literature files if evidence retrieval is required.
6. Run the complete pipeline.
7. Review generated outputs in the results directory.

---

# 5. Reproduction of manuscript results

Detailed reproduction instructions should be provided in:

[`docs/reproduction.md`](docs/reproduction.md)

The reproduction procedure should specify:

1. Dataset version and checksum.
2. Data preprocessing procedure.
3. Feature definitions.
4. Train/test split.
5. Random seeds.
6. Model hyperparameters.
7. SHAP configuration.
8. Image descriptor generation procedure.
9. Literature corpus.
10. Language models and embedding models.
11. Prompt templates.
12. Agent execution order.
13. Commands used to generate manuscript figures and tables.
14. Expected quantitative outputs.
15. Acceptable numerical tolerances.

A typical reproduction command is:

```bash
python multiagent_pipeline.py --config config/reproduction.json
```

Expected reproduction outputs may include:

```text
results/
├── model_metrics.json
├── associations.json
├── manuscript_table_1.csv
├── manuscript_table_2.csv
├── manuscript_figure_3.png
└── manuscript_figure_4.png
```

## Reproducibility considerations

Exact natural-language responses may vary because external language models are nondeterministic and may be updated by their service providers.

For reproducibility:

- Record all model names and versions.
- Preserve prompt templates.
- Fix random seeds for machine-learning procedures.
- Save structured intermediate results.
- Record the date of each external API run.
- Report package versions.
- Store expected outputs for the demonstration dataset.

---

# 6. Data availability

The repository includes a small demonstration dataset sufficient to verify the main software workflow.

The full experimental dataset used in the manuscript may be provided:

- Directly in this repository;
- Through a public scientific data repository; or
- Upon reasonable request, subject to institutional and publication policies.

If the complete dataset cannot be distributed, the repository should include a simulated or anonymized dataset with the same schema.

---

# 7. Security and credentials

Never commit API keys or access tokens.

The following files and directories should be excluded through `.gitignore`:

```gitignore
.env
*.key
*.pem
__pycache__/
*.pyc
.idea/
graph_db*/
rag_db*/
mineru_outputs/
downloads/
evidence_*/
sources_*/
```

Before publishing the repository, verify that source files do not contain hard-coded credentials.

Any credential accidentally committed to GitHub should be revoked immediately, even if the repository is private.

---

# 8. Limitations

- The literature-analysis workflow depends on external APIs.
- Generated explanations should be reviewed by domain experts.
- Language-model outputs may vary between runs.
- The demonstration dataset is smaller than the complete experimental dataset.
- The framework does not replace experimental validation.
- Retrieved literature evidence depends on the supplied document corpus.
- Performance on operating systems other than Windows has not yet been formally validated.

---

# 9. Citation

If you use this software, please cite:

```bibtex
@article{nanoparticle_interfacial_framework,
  title   = {An Autonomous, Apprehensible, and Accelerated Optimization Framework for Nanoparticle Interfacial Dynamics},
  author  = {Author names},
  journal = {Journal name},
  year    = {2026},
  doi     = {DOI pending}
}
```

Replace the placeholder author, journal, year, and DOI fields with the final publication information.

---

# 10. License

This project is distributed under the terms described in the [`LICENSE`](LICENSE) file.

Before selecting a license, confirm that the license is compatible with institutional policies, third-party dependencies, datasets, and journal requirements.

---

# 11. Contact

For questions, bug reports, and scientific collaboration, please contact:

```text
Name: [Corresponding author]
Institution: [Institution name]
Email: [Contact email]
```

Alternatively, open an issue in this GitHub repository.
