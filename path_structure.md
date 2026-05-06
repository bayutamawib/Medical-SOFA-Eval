sofa-eval/
├── src/
│   └── sofa_eval/
│       ├── __init__.py
│       ├── oracle.py       <-- Pindahkan logik SOFA di sini
│       ├── rewards.py      <-- Pindahkan fungsi reward GRPO di sini
│       └── utils.py        <-- Regex & helper ekstraksi teks
├── tests/
│   └── test_oracle.py      <-- Gunakan Module 10 sebagai basis
├── README.md
└── pyproject.toml
