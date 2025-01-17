
![Awesome ReadME](https://media.licdn.com/dms/image/v2/D5616AQGHD9BS-MxP6A/profile-displaybackgroundimage-shrink_350_1400/profile-displaybackgroundimage-shrink_350_1400/0/1719351839679?e=1742428800&v=beta&t=RsX9uLGbWrhOcwdqOyGnadSrm5IGqUItSG5sQaLOFBk)

# Performant Agentic Framework: Alignment First LLM Agentic Workflow

Arena experiment code for Performant Agentic Framework (PAF), including the dataset.

# Quick Start Demo

To use the dataset to generate Golden vs. Response Dataset, run

```bash
python -m data.generator.py
```

To use the evaluation script, run

```bash
python main.py
```

# Table of Contents

This is a table of contents for your project. It helps the reader navigate through the README quickly.
- [Project Title](#project-title)
- [Quick Start Demo](#quick-start-demo)
- [Table of Contents](#table-of-contents)
- [Installation](#installation)
- [Usage](#usage)
- [Contribute](#contribute)
- [License](#license)


# Installation
[(Back to top)](#table-of-contents)

1. To install the repo, run:

```shell
git clone https://github.com/wayne-wang-1119/performant-agentic-framework.git
```

2. Install the required packages

```bash
brew install python3.12
pip install virtualenv
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
```



# Usage
[(Back to top)](#table-of-contents)

You can leverage the PAF repo to: 1. Generate synthetic data that can be used to evaluate Agentic Performance in a simulated live environment, or 2. Run evaluation scripts to benchmark different methods of navigating through a logic tree and see the results of Agent performance. 

1.  Generate the dataset

```bash
python -m data.generator
```

This will generate a dataset in the `data` folder called dataset.csv with the following columns:

- `SystemPromp`: The system prompt used in the experiment containing the Node navigation map with a simplified version of the Agentic logic framework
- `Conversation History`: The conversation history of the user and the Agent, and the conversation turns back and forth generated by /data/generator.py
- `Golden Response`: The golden response that the Agent should respond with, generated first by the LLM, then audited by Human Evaluation one by one to ensure the quality of the golden response from us to allow for testing truthfulness and contextual awareness of Agentic System in a simplified and complex setting.

2. Run the experiment, gather the statistics:

```bash
python main.py
python stat_proof.py
```

To run individual evaluation scripts:
We leverage score / semantic similarity to the alignment of the latest generation to the golden response.

- `evaluation/eval_naive`: This script will evaluate the performance of the naive setup, which always sends the entire map and receives the entire response.

Example:

```bash
python -m evaluation.eval_naive
```

The same goes for the other evaluation scripts



# Contribute
[(Back to top)](#table-of-contents)

Feel free to contribute to our project


# License
[(Back to top)](#table-of-contents)

[MIT license](./LICENSE)


