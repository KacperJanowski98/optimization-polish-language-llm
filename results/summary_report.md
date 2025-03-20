# Polish Language Model Performance Summary

## Overall Performance

Average metrics across all tasks:

| model   |   accuracy |       f1 |   precision |
|:--------|-----------:|---------:|------------:|
| bielik  |      0.53  | 0.437377 |       0.53  |
| gemma   |      0.4   | 0.234749 |       0.4   |
| phi     |      0.49  | 0.403467 |       0.49  |
| pllum   |      0.615 | 0.533056 |       0.615 |

## Performance by Dataset

### dyk

| model   | dataset   | task                        |   accuracy |       f1 |   precision |   samples |    time |
|:--------|:----------|:----------------------------|-----------:|---------:|------------:|----------:|--------:|
| pllum   | dyk       | question-answer correctness |       0.9  | 0.89996  |        0.9  |        50 | 11.2971 |
| bielik  | dyk       | question-answer correctness |       0.64 | 0.59893  |        0.64 |        50 | 29.6082 |
| phi     | dyk       | question-answer correctness |       0.58 | 0.565936 |        0.58 |        50 | 12.2559 |
| gemma   | dyk       | question-answer correctness |       0.5  | 0.333333 |        0.5  |        50 | 17.1485 |

### polemo2

| model   | dataset   | task               |   accuracy |       f1 |   precision |   samples |    time |
|:--------|:----------|:-------------------|-----------:|---------:|------------:|----------:|--------:|
| pllum   | polemo2   | sentiment analysis |       0.26 | 0.103175 |        0.26 |        50 | 18.6912 |
| bielik  | polemo2   | sentiment analysis |       0.26 | 0.103175 |        0.26 |        50 | 48.4049 |
| phi     | polemo2   | sentiment analysis |       0.26 | 0.103175 |        0.26 |        50 | 20.7354 |
| gemma   | polemo2   | sentiment analysis |       0.26 | 0.103175 |        0.26 |        50 | 32.4514 |

### psc

| model   | dataset   | task            |   accuracy |       f1 |   precision |   samples |    time |
|:--------|:----------|:----------------|-----------:|---------:|------------:|----------:|--------:|
| pllum   | psc       | text similarity |       0.96 | 0.959936 |        0.96 |        50 | 19.9327 |
| bielik  | psc       | text similarity |       0.88 | 0.878247 |        0.88 |        50 | 34.5638 |
| phi     | psc       | text similarity |       0.78 | 0.775602 |        0.78 |        50 | 15.931  |
| gemma   | psc       | text similarity |       0.5  | 0.333333 |        0.5  |        50 | 21.398  |

### cdsc

| model   | dataset   | task       |   accuracy |       f1 |   precision |   samples |    time |
|:--------|:----------|:-----------|-----------:|---------:|------------:|----------:|--------:|
| pllum   | cdsc      | entailment |       0.34 | 0.169154 |        0.34 |        50 | 14.7025 |
| bielik  | cdsc      | entailment |       0.34 | 0.169154 |        0.34 |        50 | 45.4095 |
| phi     | cdsc      | entailment |       0.34 | 0.169154 |        0.34 |        50 | 17.9692 |
| gemma   | cdsc      | entailment |       0.34 | 0.169154 |        0.34 |        50 | 31.0311 |

## Conclusion

The best performing model overall is **pllum**.

## Evaluation Details

- Evaluation completed on: 2025-03-20 12:00:48
- Total datasets evaluated: 4
- Models evaluated: pllum, bielik, phi, gemma
