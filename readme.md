# DFA Simulator

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)](https://enjoydfa.streamlit.app)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An **interactive Deterministic Finite Automata (DFA) Simulator** built with [Streamlit](https://streamlit.io).
Easily **create**, **visualize**, and **simulate** DFAs directly in your browser.

🚀 **Live Demo:** [enjoydfa.streamlit.app](https://enjoydfa.streamlit.app)

---

## ✨ Features

- **Interactive DFA Builder**
  - Add states, alphabet symbols, and transitions
  - Mark start and accept states
  - Automatic DFA validation

- **Visualization**
  - NetworkX-powered DFA graphs
  - Color-coded nodes for start, accept, and regular states
  - Path highlighting during simulations

- **Simulation Engine**
  - Step-by-step simulation or full string evaluation
  - Supports wildcards:
    - `?` → matches any single symbol
    - `*` → repeats previous character 0–3 times

- **Preloaded Examples**
  - Binary strings ending with `01`
  - Even number of `a`s
  - Strings containing substring `abc`

---

## 🛠️ Installation

Clone the repository:
```bash
git clone https://github.com/rnks2003/exp_dfa.git
cd exp_dfa
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Run locally:
```bash
streamlit run main.py
```

---

## 🌐 Online Access

Skip installation. Just visit **[enjoydfa.streamlit.app](https://enjoydfa.streamlit.app)** and start using the simulator instantly.

---

## 📂 Project Structure

```
exp_dfa/
│-- main.py              # Main Streamlit app
│-- requirements.txt     # Python dependencies
│-- .streamlit/          # Optional Streamlit config
```

---

## 🧪 Example DFAs

| DFA Name                 | Alphabet | Description                        |
|-------------------------|----------|------------------------------------|
| Binary `01` Ending      | {0, 1}   | Accepts strings ending with `01`   |
| Even `a`s               | {a, b}   | Accepts strings with even `a`s     |
| Contains `abc` Substring| {a, b, c}| Accepts strings containing `abc`   |

---

## ⚡ Tech Stack

- **Python 3.12**
- **Streamlit** – Web UI framework
- **NetworkX** + **Matplotlib** – Graph visualization
- **Pandas** – Transition table rendering

---

## 📝 Commit Guidelines

Follow this convention:
```
[TYPE] Commit message
```

**Common types**:
- **[FIX]** – Bug fixes
- **[ADD]** – New features
- **[DOCS]** – Documentation changes
- **[MNT]** – Code refactoring & Maintenance
- **[TEST]** – Tests additions or fixes

---

## 📝 License

This project is licensed under the MIT License.
You are free to use, modify, and distribute it.
