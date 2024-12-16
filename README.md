ThreatTracer is an innovative framework for autonomous Advanced Persistent Threat (APT) analysis. It employs graph-based threat modeling and attribution techniques to provide comprehensive insights into cyber threats. This framework integrates graph-based structures with state-of-the-art machine learning methods, including Graph Attention Networks (GATs) and Local Interpretable Model-Agnostic Explanations (LIME), to enhance accuracy, explainability, and efficiency.
The datasets used in this framework are derived from Caldera, an open-source adversary emulation platform, and the DARPA Transparent Computing (TC) dataset, which offers real-world APT attack data. These datasets ensure robust evaluation across simulated and realistic environments.
Features
  •	Automated Threat Modeling: Leverages graph-based techniques to model complex APT scenarios.
  •	Graph Attention Networks (GAT): For accurate threat attribution and highlighting essential attack patterns.
  •	Two-Level Graph Summarization: Combines community detection with attention mechanisms to reduce graph complexity without compromising essential details.
  •	Explainable AI (XAI): Incorporates LIME for interpretability, making model decisions transparent and understandable.
  •	Integration with MITRE ATT&CK and Cyber Kill Chain (CKC) frameworks for alignment with industry standards.
Datasets
The framework utilizes the following datasets:
1.	Caldera Dataset:
  o	Simulated APT scenarios using the MITRE ATT&CK framework.
  o	Provides diverse predefined threats for benchmarking.
2.	DARPA Transparent Computing (TC) Dataset:
  o	Contains over 726 million audit events, including web, SSH, email, and SMB service logs.
  o	Documents real-world APT attacks, such as backdoors and phishing.
These datasets enhance the robustness of ThreatTracer’s evaluations by covering both synthetic and real-world threat scenarios.
Installation
To set up ThreatTracer on your local machine:
1.	Clone the repository:
bash
Copy code
https://github.com/CyberScienceLab/Threat-Tracer.git
cd Threat- Tracer
Dataset Preparation
Caldera Dataset
1.	Install and configure the Caldera platform as per Caldera Documentation.
2.	Simulate APT scenarios aligned with the MITRE ATT&CK framework.
3.	Export the logs to the data/raw_logs/caldera/ directory.
DARPA TC Dataset
1.	Download the dataset from the DARPA TC repository.
2.	Place the dataset files in the data/raw_logs/darpa/ directory.
3.	Use the provided annotation scripts to tag events with TTPs and CKC phases.
