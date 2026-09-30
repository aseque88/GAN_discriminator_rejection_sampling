# ProjetDL3 : Adversarial Examples Are Not Bugs, They Are Features

## About This Project

This repository contains our work for the Deep Learning project (ProjetDL3). The main objective of this project was to implement, test, and analyze the concepts presented in the following research paper:

> 📄 **[Adversarial Examples Are Not Bugs, They Are Features](https://arxiv.org/abs/1810.06758)** (Ilyas et al.)

### Repository Contents

*   **Source Code**: The complete Python implementation used to reproduce and test the paper's experiments.
*   **Report**: A detailed document explaining our methodology, results, and critical analysis of the paper.
*   **Presentation Slides**: The slide deck used for our project defense/presentation.

---

## Technical Instructions (Assignment Guidelines)

### `generate.py`
Use the file `generate.py` to generate 10,000 samples of MNIST in the `samples` folder. 
**Example:**
```bash
python3 generate.py --batch_size 64

```

### `requirements.txt`

As a good practice in data science, we encourage you to use `conda` or `virtualenv` to create an isolated Python environment.
To ensure your code runs correctly on our evaluation platform, please keep the `requirements.txt` updated with the libraries you used.
During the evaluation, the following command will be executed:

```bash
pip install -r requirements.txt

```

### Checkpoints
You can find our best trained model in the `checkpoints` folder. 
