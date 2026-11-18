# Lab 9 Evaluation Report
## Retrieval-Augmented Generation (RAG) API

### 1. Evaluation Objective

The purpose of this evaluation was to assess the ability of the RAG question-answering API to answer questions using the RAD Human Resource Policy 2021 and provide relevant source-page references.

### 2. Evaluation Dataset

- Source: RAD Human Resource Policy 2021
- Institution: Relief Adversity and Development Organization (RAD)
- Available document length: 16 pages
- Number of test questions: 10
- Language model: Llama 3.2 3B
- Embedding model: all-MiniLM-L6-v2
- Vector database: ChromaDB
- API framework: FastAPI

Note: The assignment specifies 20 pages. The available institutional policy document contains 16 pages. This limitation is reported transparently.

### 3. Test Questions and Manual Assessment

| No. | Evaluation topic | Result |
|---|---|---|
| 1 | Probation period | Correct |
| 2 | Organizational vision | Correct |
| 3 | Organizational mission | Correct content; page description needs correction |
| 4 | Core values | Incomplete; two values omitted |
| 5 | Nepotism during recruitment | Correct |
| 6 | Performance assessments | Correct |
| 7 | Working hours and overtime | Correct |
| 8 | Employee leave types | Incomplete; several leave types omitted |
| 9 | Confidentiality | Correct |
| 10 | Workplace harassment | Partially correct; disciplinary consequences need to be included |

### 4. Quantitative Results

- Total questions submitted: 10
- Successful HTTP responses: 10
- Answers returned with source references: 10
- Fully correct answers after manual review: 7
- Manually assessed accuracy: 70%

Calculation:

Accuracy = (Fully correct answers / Total questions) × 100

Accuracy = (7 / 10) × 100 = 70%

The 70% result is based on manual assessment of answer correctness and completeness. Successful API responses and the presence of source references alone do not establish factual correctness.

### 5. Observations

1. The API successfully processed all ten evaluation questions.
2. Every question received an answer with source-page references.
3. Questions about probation, vision, mission, nepotism, performance assessments, working hours, and confidentiality were answered correctly.
4. The core-values response omitted two values listed in the policy.
5. The leave response did not include all leave categories described in the policy.
6. The harassment response did not fully describe the policy's zero-tolerance position and possible disciplinary consequences.

### 6. Recommended Improvements

- Improve document chunking and retrieval relevance.
- Encourage the language model to include all relevant details.
- Improve the prompt so that answers remain grounded in retrieved policy content.
- Improve source-page relevance.
- Expand the document collection when additional institutional documents become available.
- Repeat manual evaluation after improvements.

### 7. Conclusion

The Lab 9 RAG API successfully retrieved information from the RAD Human Resource Policy 2021 and generated answers for all ten evaluation questions.

All ten API requests returned successful responses with source references. Manual assessment identified seven fully correct answers, resulting in an accuracy score of 70%.

The evaluation demonstrates the practical use of Retrieval-Augmented Generation for institutional document question answering while identifying areas for improvement in answer completeness and retrieval relevance.
