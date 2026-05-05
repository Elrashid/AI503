# A1 Methods Inventory — Quantization Methods × Paper Coverage

Built from `comparative_analysis_table.csv` + `papers_md/` (50 papers). A method is counted for a paper if any of its name aliases appears anywhere in the paper's text or extracted fields. Use this to pick A2 baselines: prefer methods with the highest coverage (most comparable to prior work).

**Total papers in corpus: 50**

| Item | Papers (n) | Coverage % | RP_IDs |
|---|---:|---:|---|
| PTQ (generic) | 34 | 68% | RP07, RP09, RP10, RP11, RP13, RP14, RP15, RP16, RP17, RP18, RP19, RP20, … (+22) |
| GPTQ | 28 | 56% | RP09, RP10, RP13, RP15, RP16, RP17, RP18, RP19, RP26, RP27, RP28, RP29, … (+16) |
| SmoothQuant | 24 | 48% | RP10, RP13, RP15, RP16, RP18, RP20, RP22, RP26, RP27, RP28, RP29, RP30, … (+12) |
| AWQ | 24 | 48% | RP15, RP16, RP18, RP19, RP21, RP22, RP24, RP26, RP27, RP29, RP30, RP31, … (+12) |
| QAT (generic) | 19 | 38% | RP07, RP14, RP15, RP16, RP18, RP19, RP21, RP22, RP27, RP30, RP32, RP35, … (+7) |
| LLM.int8() | 19 | 38% | RP09, RP10, RP11, RP13, RP15, RP17, RP18, RP19, RP22, RP24, RP26, RP29, … (+7) |
| ZeroQuant | 16 | 32% | RP09, RP10, RP11, RP13, RP14, RP15, RP16, RP17, RP30, RP31, RP33, RP40, … (+4) |
| RTN | 13 | 26% | RP09, RP15, RP16, RP17, RP18, RP27, RP29, RP30, RP35, RP36, RP38, RP42, … (+1) |
| QLoRA | 13 | 26% | RP16, RP18, RP19, RP22, RP27, RP30, RP40, RP41, RP43, RP45, RP46, RP49, … (+1) |
| QuIP | 12 | 24% | RP13, RP16, RP18, RP21, RP26, RP29, RP30, RP35, RP36, RP42, RP48, RP50 |
| SqueezeLLM | 11 | 22% | RP16, RP18, RP21, RP22, RP27, RP30, RP35, RP36, RP38, RP42, RP49 |
| SpQR | 11 | 22% | RP16, RP17, RP18, RP19, RP21, RP22, RP27, RP28, RP29, RP30, RP42 |
| OmniQuant | 11 | 22% | RP16, RP18, RP21, RP22, RP29, RP35, RP36, RP37, RP42, RP48, RP49 |
| QuIP# | 9 | 18% | RP21, RP26, RP29, RP30, RP35, RP36, RP42, RP48, RP50 |
| BitsAndBytes | 7 | 14% | RP11, RP17, RP24, RP40, RP41, RP43, RP45 |
| KIVI | 7 | 14% | RP22, RP24, RP29, RP35, RP38, RP41, RP42 |
| QuaRot | 6 | 12% | RP29, RP35, RP36, RP42, RP48, RP49 |
| AQLM | 4 | 8% | RP21, RP30, RP49, RP50 |
| SpinQuant | 3 | 6% | RP35, RP36, RP48 |
| OstQuant | 2 | 4% | RP45, RP48 |
| FlatQuant | 1 | 2% | RP35 |
