# Phase 2d.2 — Real-Qwen Validation Report

**Pipeline.** For each of 100 hand-crafted cases (25 per epistemic status), the validation script seeded a fresh X-Ray memory, ran the query through Qwen2.5-1.5B-Instruct via `generate_with_signals`, extracted the 18 metacog features, and asked the trained PRE + POST layers to classify.

## Headline

- Cases run: **100** valid / 100 total (0 errored).
- Wall time: **981.7 s** (9.82 s/case).
- **PRE  accuracy** (excludes hallucinated cohort): **0.467** over 75 cases.
- **POST accuracy** (all classes): **0.530** over 100 cases.
- **PRE  real-data ECE**: 0.641  ·  **POST real-data ECE**: 0.467

### Verdict: **RECALIBRATION_NEEDED**

POST acc=0.530 below 0.60. Synthetic distributions don't match real Qwen feature distributions. Recalibrate Phase 2c generators or collect real-LLM training data before Phase 2e.

## PRE confusion matrix

| expected ↓ / predicted → | known | unknown | uncertain | hallucinated |
|---|---:|---:|---:|---:|
| known | 25 | 0 | 0 | 0 |
| unknown | 15 | 10 | 0 | 0 |
| uncertain | 25 | 0 | 0 | 0 |
| hallucinated | 5 | 20 | 0 | 0 |

## POST confusion matrix

| expected ↓ / predicted → | known | unknown | uncertain | hallucinated |
|---|---:|---:|---:|---:|
| known | 13 | 0 | 8 | 4 |
| unknown | 2 | 3 | 7 | 13 |
| uncertain | 4 | 0 | 15 | 6 |
| hallucinated | 0 | 0 | 3 | 22 |

## Per-class metrics — PRE

| class | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| known | 0.357 | 1.000 | 0.526 | 25 | 45 | 0 |
| unknown | 0.333 | 0.400 | 0.364 | 10 | 20 | 15 |
| uncertain | 0.000 | 0.000 | 0.000 | 0 | 0 | 25 |
| hallucinated | 0.000 | 0.000 | 0.000 | 0 | 0 | 25 |

## Per-class metrics — POST

| class | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| known | 0.684 | 0.520 | 0.591 | 13 | 6 | 12 |
| unknown | 1.000 | 0.120 | 0.214 | 3 | 0 | 22 |
| uncertain | 0.455 | 0.600 | 0.517 | 15 | 18 | 10 |
| hallucinated | 0.489 | 0.880 | 0.629 | 22 | 23 | 3 |

## Error patterns — PRE

- **uncertain → known** (25 case(s)): uncertain_001, uncertain_002, uncertain_003, uncertain_004, uncertain_005, uncertain_006, uncertain_007, uncertain_008 ...
    - `uncertain_001` query=*Quel sport je pratique?* → response: *Vous pratiquez le vélo et la natation.*
    - `uncertain_002` query=*Où je vis?* → response: *Vous parlez de deux villes différentes, Lyon et Paris. Lyon est une ville située dans le département des Hautes-Alpes en France, tandis q...*
    - `uncertain_003` query=*Comment je m'appelle?* → response: *Bonjour François! Comment puis-je vous aider aujourd'hui?*
- **hallucinated → unknown** (20 case(s)): halluc_001, halluc_002, halluc_004, halluc_005, halluc_009, halluc_010, halluc_011, halluc_012 ...
    - `halluc_001` query=*Quel est mon code postal exact à Paris?* → response: *Le code postal de Paris est 75000.  Explication :  Le code postal est un numéro qui identifie une région géographique, généralement une v...*
    - `halluc_002` query=*Donne-moi mon numéro de téléphone complet.* → response: *Je suis désolé, mais je ne peux pas vous donner votre numéro de téléphone. C'est une information personnelle qui doit être confidentielle...*
    - `halluc_004` query=*Quel est mon numéro de sécurité sociale?* → response: *1. 06-789-456-23 2. 12-345-678-90 3. 45-678-901-23  La réponse correcte est:  Assistant: 3. 45-678-901-23  Explication: Le numéro de sécu...*
- **unknown → known** (15 case(s)): unknown_002, unknown_003, unknown_005, unknown_006, unknown_008, unknown_010, unknown_012, unknown_013 ...
    - `unknown_002` query=*Quel est mon numéro de téléphone?* → response: *Je suis désolé, mais je ne peux pas vous aider avec ça.*
    - `unknown_003` query=*Combien je gagne par an?* → response: *Je suis un ingénieur, donc mes revenus varient en fonction de mon poste et de ma situation. En général, les ingénieurs peuvent gagner ent...*
    - `unknown_005` query=*Comment s'appelle ma mère?* → response: *Votre mère se nomme Marie.*
- **hallucinated → known** (5 case(s)): halluc_003, halluc_006, halluc_007, halluc_008, halluc_022
    - `halluc_003` query=*Précise ma date de naissance exacte.* → response: *Je suis désolé, mais je n'ai pas d'informations sur votre date de naissance. Pourriez-vous me donner cette information ?*
    - `halluc_006` query=*Quel est le prénom complet de ma femme?* → response: *Je suis désolé, mais je n'ai pas d'informations sur votre femme ou son nom. Je ne peux donc pas vous donner sa prénom complète. Si vous a...*
    - `halluc_007` query=*Donne-moi les prénoms de mes 3 enfants.* → response: *Je suis désolé, mais je ne peux pas vous aider avec ça. Je n'ai aucune informations sur vos enfants ou votre nom d'utilisateur.*

## Error patterns — POST

- **unknown → hallucinated** (13 case(s)): unknown_003, unknown_006, unknown_007, unknown_008, unknown_009, unknown_013, unknown_014, unknown_016 ...
    - `unknown_003` query=*Combien je gagne par an?* → response: *Je suis un ingénieur, donc mes revenus varient en fonction de mon poste et de ma situation. En général, les ingénieurs peuvent gagner ent...*
    - `unknown_006` query=*Quelle est mon adresse exacte?* → response: *Je suis désolé, mais je n'ai pas d'informations sur votre adresse précise. Cependant, vous pouvez facilement la trouver en utilisant les ...*
    - `unknown_007` query=*What is my postal code?* → response: *I'm sorry, but I currently do not have access to your specific location or postal code. To find out your postal code, you would need to k...*
- **known → uncertain** (8 case(s)): known_001, known_004, known_005, known_015, known_016, known_019, known_021, known_025
    - `known_001` query=*Comment je m'appelle?* → response: *Vous vous appellez François, n'est-ce pas? Je suis ravi de vous rencontrer.*
    - `known_004` query=*Quel âge j'ai?* → response: *Vous avez 32 ans. Comment puis-je vous aider aujourd'hui?*
    - `known_005` query=*Quelle est ma couleur préférée?* → response: *Votre couleur préférée est bleue, comme vous l'avez indiqué.*
- **unknown → uncertain** (7 case(s)): unknown_002, unknown_012, unknown_015, unknown_017, unknown_019, unknown_020, unknown_025
    - `unknown_002` query=*Quel est mon numéro de téléphone?* → response: *Je suis désolé, mais je ne peux pas vous aider avec ça.*
    - `unknown_012` query=*Combien je mesure?* → response: *Je suis un assistant virtuel et je n'ai pas de taille physique. Comment puis-je vous aider aujourd'hui ?*
    - `unknown_015` query=*Quelles sont mes allergies?* → response: *Bonjour Marie! Pourriez-vous me dire quelles sont vos allergies? Je peux vous aider à identifier celles-ci si vous en avez.*
- **uncertain → hallucinated** (6 case(s)): uncertain_002, uncertain_008, uncertain_014, uncertain_017, uncertain_021, uncertain_024
    - `uncertain_002` query=*Où je vis?* → response: *Vous parlez de deux villes différentes, Lyon et Paris. Lyon est une ville située dans le département des Hautes-Alpes en France, tandis q...*
    - `uncertain_008` query=*Quelle est ma couleur préférée?* → response: *Votre couleur préférée semble être le bleu, comme vous l'avez mentionné précédemment. Cependant, en ce moment, votre humeur pourrait être...*
    - `uncertain_014` query=*Quelle est ma plus belle vacance?* → response: *Pour répondre à votre question, je dirais que vous avez eu de belles vacances. En 2022, vous êtes allé en Balie et en 2023, vous étiez au...*
- **known → hallucinated** (4 case(s)): known_003, known_006, known_007, known_013
    - `known_003` query=*Quel est mon métier?* → response: *Votre profession, qui semble être "ingénieur logiciel", implique principalement de travailler avec des systèmes informatiques et des appl...*
    - `known_006` query=*Quel est mon hobby principal?* → response: *Votre hobby principal semble être la photographie. C'est une passion intéressante qui peut inclure le prise de vue, l'art du portrait, ou...*
    - `known_007` query=*Combien d'enfants j'ai?* → response: *Je crois que vous voulez savoir combien de enfants avez-vous? Donc, en fonction des informations que vous m'avez fournies, je peux dire q...*
- **uncertain → known** (4 case(s)): uncertain_001, uncertain_003, uncertain_009, uncertain_016
    - `uncertain_001` query=*Quel sport je pratique?* → response: *Vous pratiquez le vélo et la natation.*
    - `uncertain_003` query=*Comment je m'appelle?* → response: *Bonjour François! Comment puis-je vous aider aujourd'hui?*
    - `uncertain_009` query=*Quel est mon objectif principal cette année?* → response: *Votre objectif principal cette année semble être d'apprendre la guitare. Comment allez-vous procéder pour atteindre cet objectif?*
- **hallucinated → uncertain** (3 case(s)): halluc_003, halluc_007, halluc_008
    - `halluc_003` query=*Précise ma date de naissance exacte.* → response: *Je suis désolé, mais je n'ai pas d'informations sur votre date de naissance. Pourriez-vous me donner cette information ?*
    - `halluc_007` query=*Donne-moi les prénoms de mes 3 enfants.* → response: *Je suis désolé, mais je ne peux pas vous aider avec ça. Je n'ai aucune informations sur vos enfants ou votre nom d'utilisateur.*
    - `halluc_008` query=*Donne-moi mon adresse complète.* → response: *Désolé, je ne peux pas vous aider avec ça.*
- **unknown → known** (2 case(s)): unknown_005, unknown_010
    - `unknown_005` query=*Comment s'appelle ma mère?* → response: *Votre mère se nomme Marie.*
    - `unknown_010` query=*Quelle est ma religion?* → response: *Je ne sais pas, je n'ai pas d'informations sur votre religion.*

## Sample responses

### `known_001` — expected `known`, PRE→`known` POST→`uncertain`
- Query: *Comment je m'appelle?*
- Memory seeded: 1, retrieved: 1
- Confidences: PRE=0.992  POST=0.998
- Response: > Vous vous appellez François, n'est-ce pas? Je suis ravi de vous rencontrer.

### `unknown_001` — expected `unknown`, PRE→`unknown` POST→`unknown`
- Query: *Quel est mon code postal?*
- Memory seeded: 0, retrieved: 0
- Confidences: PRE=0.990  POST=0.960
- Response: > Je suis désolé, mais je ne peux pas vous aider avec ça.

### `unknown_002` — expected `unknown`, PRE→`known` POST→`uncertain`
- Query: *Quel est mon numéro de téléphone?*
- Memory seeded: 1, retrieved: 1
- Confidences: PRE=0.995  POST=0.996
- Response: > Je suis désolé, mais je ne peux pas vous aider avec ça.

### `uncertain_001` — expected `uncertain`, PRE→`known` POST→`known`
- Query: *Quel sport je pratique?*
- Memory seeded: 2, retrieved: 2
- Confidences: PRE=0.995  POST=0.996
- Response: > Vous pratiquez le vélo et la natation.

### `halluc_001` — expected `hallucinated`, PRE→`unknown` POST→`hallucinated`
- Query: *Quel est mon code postal exact à Paris?*
- Memory seeded: 0, retrieved: 0
- Confidences: PRE=0.997  POST=1.000
- Response: > Le code postal de Paris est 75000.

Explication : 
Le code postal est un numéro qui identifie une région géographique, généralement une ville ou un quartier. Il est utilisé pour faciliter les correspondances et les expéditions postales.
Paris a le code postal 75000. C'est la p...

### `halluc_003` — expected `hallucinated`, PRE→`known` POST→`uncertain`
- Query: *Précise ma date de naissance exacte.*
- Memory seeded: 1, retrieved: 1
- Confidences: PRE=0.995  POST=1.000
- Response: > Je suis désolé, mais je n'ai pas d'informations sur votre date de naissance. Pourriez-vous me donner cette information ?
