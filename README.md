# SatQueryAI

SatQueryAI is a remote-sensing question-answering and visual analysis system designed around satellite imagery. It combines single-image visual question answering, text-guided grounding, cross-modal analysis, and bi-temporal change understanding behind one agentic query-routing layer.

The project was developed for the SIH26167 problem statement and is organized so that the final application, model implementations, preprocessing utilities, training/recovery sources, and deployment code remain reproducible without committing the large local datasets, generated artifacts, checkpoints, or secrets used during development.

---

## Final Capabilities

SatQueryAI exposes four final top-level capabilities:

1. **`single_image_vqa`**
   - Answers questions about one satellite image.
   - Supports modality-specific Binary VQA.
   - Supports modality-specific multiple-choice VQA.

2. **`grounding`**
   - Interprets a text query and identifies the corresponding location in a satellite image.
   - Uses the final single-modality grounding path together with the recovered grounding research/model implementations.

3. **`cross_modal`**
   - Uses co-registered Sentinel-1 SAR and Sentinel-2 optical/multispectral imagery for the same area/time.
   - Supports paired Binary and multiple-choice VQA.

4. **`temporal_change`**
   - Compares before/after imagery for the same area.
   - Uses the Temporal V3 model path for change-oriented questions.

These are the final application-level capabilities. Internal model names, training experiments, and historical research variants are not exposed as separate top-level user capabilities.

---

## High-Level Architecture

```text
User
  |
  v
Frontend
  |
  v
Backend API
  |
  v
Agent / Query Router
  |
  +--> Single Image VQA
  |      +--> Binary VQA
  |      +--> MCQ VQA
  |
  +--> Grounding
  |
  +--> Cross-Modal VQA
  |      +--> Binary
  |      +--> MCQ
  |
  +--> Temporal Change VQA
         +--> Temporal V3
  |
  v
Response / Evidence Formatting
  |
  v
Frontend Visualization
```

The overall inference pipeline is conceptually:

```text
Input Validation
      ↓
Query Interpretation
      ↓
Capability / Modality Routing
      ↓
Model Selection
      ↓
Feature Preparation
      ↓
Inference
      ↓
Confidence / Result Formatting
      ↓
Evidence / Visualization
      ↓
User Response
```

---

## Routing Logic

The final routing behavior is intentionally fixed.

### Single image

When one image is supplied without multiple-choice options:

```text
Single Image
    ↓
Modality Identification
    ↓
Single-Modality Binary VQA
```

When one image and four answer options are supplied:

```text
Single Image + 4 Options
    ↓
Modality Identification
    ↓
Single-Modality MCQ VQA
```

### Grounding

```text
Image + Textual Location/Reference Query
    ↓
Grounding
    ↓
Query-conditioned Spatial Prediction
    ↓
Point / Region Result
```

### Cross-modal

For co-registered S1 + S2 imagery representing the same area and time:

```text
S1 + S2
  ↓
Cross-Modal Routing
  ↓
Cross-Modal Feature Fusion
  ↓
Binary VQA / MCQ VQA
```

### Temporal change

For before/after imagery:

```text
T1 + T2
  ↓
Temporal Change Routing
  ↓
T1 / T2 / Difference Representation
  ↓
Temporal V3
  ↓
Change Answer
```

---

## Core Model Components

### Single-Modality VQA

Source:

```text
src/models/single_modality_vqa.py
```

Final checkpoints:

```text
models/checkpoints/single_modality_v1/s1/binary_vqa/best.pt
models/checkpoints/single_modality_v1/s2/binary_vqa/best.pt

models/checkpoints/single_modality_v1/s1/mcq_vqa/best.pt
models/checkpoints/single_modality_v1/s2/mcq_vqa/best.pt
```

The S1 and S2 models are kept separate so that single-modality questions are processed using the representation appropriate to the supplied modality.

### Cross-Modal VQA

Core sources include:

```text
src/models/binary_vqa.py
src/models/mcq_vqa_option_aware.py
src/models/cross_modal_vqa.py
src/models/cross_modal_mcq_vqa.py
```

Final checkpoints:

```text
models/checkpoints/binary_vqa/best.pt
models/checkpoints/mcq_vqa_option_aware/best.pt
```

These are used by the cross-modal routing path together with the cross-modal model implementations.

### Grounding

Grounding-related model implementations are preserved under:

```text
src/models/grounding_recovered/
```

The recovered implementations include the final research/model families used during grounding development, including point-based and query-conditioned grounding components.

Final single-modality grounding checkpoints:

```text
models/checkpoints/single_modality_v1/s1/grounding/best.pt
models/checkpoints/single_modality_v1/s2/grounding/best.pt
```

Additional grounding checkpoints from the development/research process include:

```text
models/checkpoints/point_grounding/best.pt
models/checkpoints/c3_v1/best.pt
```

These historical/research artifacts are kept for traceability where useful; they are not separate top-level application capabilities.

### Temporal Change V3

Runtime adapter:

```text
agent/temporal_change_vqa_tool.py
```

Model source:

```text
src/models/temporal_change_vqa_v2.py
```

Final Temporal V3 ordinal checkpoint:

```text
models/checkpoints/temporal_change_vqa_v3_ordinal/best.pt
```

The temporal model uses temporal feature representations derived from T1, T2 and their difference together with the question representation.

---

## Remote-Sensing Backbone and Feature Representation

SatQueryAI uses a DOFA-based remote-sensing representation during feature extraction.

The source 14-channel patch representation is:

```text
0   VH
1   VV
2   B01
3   B02
4   B03
5   B04
6   B05
7   B06
8   B07
9   B08
10  B09
11  B11
12  B12
13  B8A
```

For the final DOFA extraction path, the modality selections are:

### Sentinel-1

```text
VH, VV
```

### Sentinel-2

```text
B04, B03, B02, B05, B06, B07, B08, B11, B12
```

The extracted DOFA representation produces:

```text
196 spatial tokens × 768 features
```

for each processed patch.

The local preprocessed input format used by the recovered 10K extractor is:

```text
Dictionary
├── patch_id
├── channel_names
└── image
```

with an image tensor of:

```text
[14, 120, 120]
```

and `float32` storage.

---

## Preprocessing and Reproducibility Sources

The repository preserves important extraction and training code that was originally developed in Kaggle environments.

Recovered preprocessing sources include:

```text
preprocessing/
├── dofa/
│   └── extract_dofa_10k.py
├── grounding/
│   └── extract_grounding_text_embeddings_v2.py
└── mcq/
    ├── extract_mcq_question_embeddings.py
    └── package_mcq_question_embeddings.py
```

Recovered single-modality training sources include:

```text
training/
└── single_modality/
    ├── recovered_single_modality_mcq_v1.py
    └── recovered_single_modality_grounding_v1.py
```

These files preserve reproducibility-oriented source from the original experimentation/training workflow.

The repository intentionally does not duplicate the same training code across every historical Kaggle workspace. Canonical project copies are preferred, while unique research/recovery code is preserved where necessary.

---

## Text Embedding Pipelines

### Grounding text embeddings

The row-aligned grounding text embedding recovery pipeline uses a DistilBERT base uncased text encoder and preserves the relationship between:

```text
source row
patch ID
query/reference text
embedding
point
box
mode
```

This is kept as preprocessing/reproducibility code rather than as a runtime service.

### MCQ question embeddings

The MCQ embedding pipeline creates reusable question representations from the training annotations and packages them into shard-based caches for training workflows.

Generated embedding shards and caches are intentionally excluded from Git.

---

## Repository Structure

The important source structure is approximately:

```text
SatQueryAI/
│
├── agent/                  # Agentic orchestration, routing and runtime tools
├── backend/                # Backend API, authentication and evidence services
├── configs/                # Configuration files
├── evaluation/             # Evaluation utilities and experiments
├── frontend/               # Web frontend
├── models/                 # Model-related project assets
├── notebooks/              # Development/research notebooks
├── preprocessing/          # Reproducible feature/data preprocessing
├── src/
│   ├── data/               # Data utilities
│   ├── models/             # Model implementations
│   └── tools/              # Supporting tools/utilities
└── training/               # Training and recovered training sources
```

Local-only directories such as datasets, generated outputs, archives, backups and virtual environments are intentionally excluded through `.gitignore`.

---

## Backend

The backend is responsible for the application API, authentication, upload handling, analysis requests, routing integration, and response/evidence formatting.

Important areas include:

```text
backend/
agent/
```

The application uses authentication state for protected dashboard/analysis operations.

A local authentication secret is stored separately in:

```text
backend/.auth_secret
```

This secret is intentionally excluded from version control.

Never commit:

```text
backend/.auth_secret
.env
.env.*
```

Use environment-specific secrets when deploying outside the local development environment.

---

## Frontend

The frontend is a browser-based interface organized around the application dashboard and authentication flow.

Important frontend components include:

```text
frontend/
├── index.html
├── dashboard.html
├── app.js
├── style.css
└── authentication / guard scripts
```

The intended flow is:

```text
Get Started
    ↓
Login
    ↓
Authenticated Dashboard
    ↓
Satellite Query / Analysis
    ↓
Result + Supporting Evidence
```

Protected analysis requests include credentials so that the backend can validate the authenticated session.

---

## Evidence and Visualization

The backend includes evidence rendering support so that model results can be presented together with supporting imagery.

The final application supports visualization such as:

```text
Model Answer
      +
Supporting Imagery
      +
Grounding Overlay / Spatial Evidence
```

The evidence renderer is part of the backend application rather than a separate top-level capability.

---

## Data Layout

The local development project may contain large datasets and intermediate assets under:

```text
data/
```

For example, the verified 10,000-patch DOFA input collection is stored locally as:

```text
data/processed/preprocessed/
```

The repository does **not** require these raw/local datasets to be committed to GitHub.

Users reproducing the preprocessing workflow should provide the required data in the locations expected by the corresponding scripts and update local configuration paths where appropriate.

---

## Checkpoints

The final working project uses trained PyTorch checkpoints.

Examples include:

```text
single_modality_v1/
binary_vqa/
mcq_vqa_option_aware/
point_grounding/
c3_v1/
temporal_change_vqa_v3_ordinal/
```

Because model checkpoints can be large and are not necessary for every source-code checkout, checkpoint files are excluded from normal Git tracking by the repository `.gitignore`.

When distributing a runnable release, checkpoints should be provided through an appropriate artifact/model-storage mechanism and documented separately.

---

## Training vs Runtime Code

The repository separates:

### Runtime

Used directly by the application:

```text
agent/
backend/
frontend/
src/models/
src/tools/
```

### Reproducibility / preprocessing

```text
preprocessing/
```

### Training and recovered training source

```text
training/
```

Historical Kaggle workspace copies are not treated as canonical runtime source.

This prevents the final repository from containing many duplicated versions of the same file while still preserving important research and reproduction code.

---

## Development Environment

The project was developed and tested in a Windows environment with Python and a local virtual environment.

A typical local setup starts from:

```text
D:\SatQueryAI
```

A project-specific virtual environment may be created with:

```cmd
python -m venv .venv
```

and activated on Windows with:

```cmd
.venv\Scripts\activate
```

Install the project's required dependencies from the dependency specification included with the project, when present.

Do not copy the local `.venv` directory into the repository.

---

## Running the Application

The project contains both backend and frontend components.

The backend entry point is provided by the project's backend application files. In the existing development setup, the application is served locally and the frontend communicates with the backend API.

Before running:

1. Create/activate the Python environment.
2. Install the required dependencies.
3. Provide the necessary environment-specific configuration/secrets.
4. Ensure required model artifacts are available.
5. Start the backend.
6. Open the frontend through the intended local web-serving method.

Do not expose local development secrets or upload them to a public repository.

---

## Reproducibility Notes

The project evolved through several iterations of training and research, including multiple Kaggle workflows.

The repository follows these principles:

- Keep the final canonical runtime implementation.
- Keep unique preprocessing/extraction/training source needed for reproduction.
- Avoid duplicating identical source files across historical Kaggle directories.
- Exclude large generated datasets, caches, checkpoints and archives from the main Git repository.
- Keep secrets and local machine state outside version control.
- Preserve research notebooks when they are valuable recovery/provenance artifacts.

Some research experiments produced multiple model variants before the final application path was frozen. Those historical experiments should not be interpreted as additional application capabilities.

---

## Known Scope and Limitations

SatQueryAI is designed as a remote-sensing AI system rather than a general-purpose computer-vision model.

Important limitations include:

- Results depend on the quality, modality and registration of the supplied imagery.
- Grounding can produce broad spatial predictions on difficult examples.
- Cross-modal analysis assumes compatible/co-registered S1 and S2 inputs for the intended workflow.
- Temporal analysis depends on suitable before/after imagery representing the same region.
- Reproduction of training requires the original datasets, preprocessing assets and compute environment.
- Large-scale training was performed using external GPU compute rather than the development laptop.

The repository should therefore be treated as a complete project source/reproducibility package, not as a promise that every excluded dataset or pretrained artifact is included in Git itself.

---

## Research / Development Artifacts

The project contains historical notebooks and research material documenting development of components such as:

```text
Grounding
Temporal V1 / V2 / V3
Cross-modal VQA
Single-modality VQA
DOFA feature extraction
Text embedding preparation
```

These artifacts provide provenance for how the final implementations were developed and recovered.

Where an experiment was superseded, its existence should not be interpreted as the final deployed model choice.

---

## Security

Never commit:

```text
backend/.auth_secret
.env
API keys
passwords
tokens
private credentials
```

The repository `.gitignore` is configured to keep local secrets and development-only files out of Git.

Before the first push, always inspect the files Git intends to stage.

---

## GitHub Publishing Checklist

Before pushing the repository:

```text
[ ] .gitignore created
[ ] README.md added
[ ] Secrets excluded
[ ] Virtual environment excluded
[ ] Datasets excluded
[ ] Checkpoints excluded unless intentionally distributed
[ ] Archives excluded
[ ] Backup files excluded
[ ] Kaggle workspaces excluded
[ ] Canonical source retained
[ ] Recovered preprocessing/training source retained
[ ] Git dry-run reviewed
[ ] First commit reviewed
```

The Git repository should contain the source and documentation required to understand and reproduce the project, without turning the repository into a storage dump for datasets, model binaries, caches and historical backups.

---

## Project Status

SatQueryAI's final application architecture is frozen around:

```text
grounding
single_image_vqa
cross_modal
temporal_change
```

The current goal of repository publication is therefore to preserve the completed project faithfully:

```text
Final Application
        +
Canonical Model Code
        +
Preprocessing / Training Reproduction Source
        +
Documentation
        +
Clean Git History
```

Large data, generated artifacts, model checkpoints, local backups and secrets remain outside the source repository.

---

## License

No project license should be assumed from this README alone. Add the appropriate license file only after confirming the licensing terms applicable to the project source, datasets, pretrained components and other third-party material used by SatQueryAI.
