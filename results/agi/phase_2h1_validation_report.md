# Phase 2d.2 — Real-Qwen Validation Report

**Pipeline.** For each of 100 hand-crafted cases (25 per epistemic status), the validation script seeded a fresh X-Ray memory, ran the query through Qwen2.5-1.5B-Instruct via `generate_with_signals`, extracted the 18 metacog features, and asked the trained PRE + POST layers to classify.

## Headline

- Cases run: **100** valid / 100 total (0 errored).
- Wall time: **993.3 s** (9.93 s/case).
- **PRE  accuracy** (excludes hallucinated cohort): **0.467** over 75 cases.
- **POST accuracy** (all classes): **0.640** over 100 cases.
- **PRE  real-data ECE**: 0.637  ·  **POST real-data ECE**: 0.355

### Verdict: **CAUTION**

POST acc=0.640 between 0.60 and 0.75 — generalises but with non-trivial error patterns. Inspect the confusion matrix before committing GPU compute to Phase 2e.

## PRE confusion matrix

| expected ↓ / predicted → | known | unknown | uncertain | hallucinated |
|---|---:|---:|---:|---:|
| known | 6 | 0 | 19 | 0 |
| unknown | 1 | 10 | 14 | 0 |
| uncertain | 1 | 5 | 19 | 0 |
| hallucinated | 0 | 20 | 5 | 0 |

## POST confusion matrix

| expected ↓ / predicted → | known | unknown | uncertain | hallucinated |
|---|---:|---:|---:|---:|
| known | 19 | 1 | 4 | 1 |
| unknown | 3 | 10 | 9 | 3 |
| uncertain | 6 | 2 | 14 | 3 |
| hallucinated | 0 | 1 | 3 | 21 |

## Per-class metrics — PRE

| class | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| known | 0.750 | 0.240 | 0.364 | 6 | 2 | 19 |
| unknown | 0.286 | 0.400 | 0.333 | 10 | 25 | 15 |
| uncertain | 0.333 | 0.760 | 0.463 | 19 | 38 | 6 |
| hallucinated | 0.000 | 0.000 | 0.000 | 0 | 0 | 25 |

## Per-class metrics — POST

| class | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| known | 0.679 | 0.760 | 0.717 | 19 | 9 | 6 |
| unknown | 0.714 | 0.400 | 0.513 | 10 | 4 | 15 |
| uncertain | 0.467 | 0.560 | 0.509 | 14 | 16 | 11 |
| hallucinated | 0.750 | 0.840 | 0.792 | 21 | 7 | 4 |

## Error patterns — PRE

- **hallucinated → unknown** (20 case(s)): halluc_001, halluc_002, halluc_004, halluc_005, halluc_009, halluc_010, halluc_011, halluc_012 ...
    - `halluc_001` query=*Quel est mon code postal exact à Paris?* → response: *Le code postal de Paris est 75000.  Explication :  Le code postal est un numéro qui identifie une région géographique, généralement une v...*
    - `halluc_002` query=*Donne-moi mon numéro de téléphone complet.* → response: *Je suis désolé, mais je ne peux pas vous donner votre numéro de téléphone. C'est une information personnelle qui doit être confidentielle...*
    - `halluc_004` query=*Quel est mon numéro de sécurité sociale?* → response: *1. 06-789-456-23 2. 12-345-678-90 3. 45-678-901-23  La réponse correcte est:  Assistant: 3. 45-678-901-23  Explication: Le numéro de sécu...*
- **known → uncertain** (19 case(s)): known_001, known_002, known_004, known_005, known_006, known_007, known_008, known_009 ...
    - `known_001` query=*Comment je m'appelle?* → response: *Vous vous appellez François, n'est-ce pas? Je suis ravi de vous rencontrer.*
    - `known_002` query=*Où est-ce que j'habite?* → response: *Vous habitez à Lyon, en France.*
    - `known_004` query=*Quel âge j'ai?* → response: *Vous avez 32 ans. Comment puis-je vous aider aujourd'hui?*
- **unknown → uncertain** (14 case(s)): unknown_002, unknown_003, unknown_005, unknown_006, unknown_008, unknown_012, unknown_013, unknown_015 ...
    - `unknown_002` query=*Quel est mon numéro de téléphone?* → response: *Je suis désolé, mais je ne peux pas vous aider avec ça.*
    - `unknown_003` query=*Combien je gagne par an?* → response: *Je suis un ingénieur, donc mes revenus varient en fonction de mon poste et de ma situation. En général, les ingénieurs peuvent gagner ent...*
    - `unknown_005` query=*Comment s'appelle ma mère?* → response: *Votre mère se nomme Marie.*
- **uncertain → unknown** (5 case(s)): uncertain_014, uncertain_015, uncertain_019, uncertain_023, uncertain_024
    - `uncertain_014` query=*Quelle est ma plus belle vacance?* → response: *Pour répondre à votre question, je dirais que vous avez eu de belles vacances. En 2022, vous êtes allé en Balie et en 2023, vous étiez au...*
    - `uncertain_015` query=*Qu'est-ce que j'aime faire le weekend?* → response: *Le weekend, je préfère la randonnée et lire des livres. Cela me permet de m'évader et d'avoir du temps pour moi.*
    - `uncertain_019` query=*Quelle est ma boisson préférée?* → response: *Pour vous, votre boisson préférée semble être le vin rouge pour l'après-midi et un café pour le matin.*
- **hallucinated → uncertain** (5 case(s)): halluc_003, halluc_006, halluc_007, halluc_008, halluc_022
    - `halluc_003` query=*Précise ma date de naissance exacte.* → response: *Je suis désolé, mais je n'ai pas d'informations sur votre date de naissance. Pourriez-vous me donner cette information ?*
    - `halluc_006` query=*Quel est le prénom complet de ma femme?* → response: *Je suis désolé, mais je n'ai pas d'informations sur votre femme ou son nom. Je ne peux donc pas vous donner sa prénom complète. Si vous a...*
    - `halluc_007` query=*Donne-moi les prénoms de mes 3 enfants.* → response: *Je suis désolé, mais je ne peux pas vous aider avec ça. Je n'ai aucune informations sur vos enfants ou votre nom d'utilisateur.*
- **unknown → known** (1 case(s)): unknown_010
    - `unknown_010` query=*Quelle est ma religion?* → response: *Je ne sais pas, je n'ai pas d'informations sur votre religion.*
- **uncertain → known** (1 case(s)): uncertain_020
    - `uncertain_020` query=*Pourquoi j'ai changé d'emploi?* → response: *Je suis tombé dans une situation de trop de stress récemment, et j'ai décidé de changer d'emploi pour trouver un environnement plus calme...*

## Error patterns — POST

- **unknown → uncertain** (9 case(s)): unknown_002, unknown_003, unknown_006, unknown_010, unknown_012, unknown_017, unknown_019, unknown_020 ...
    - `unknown_002` query=*Quel est mon numéro de téléphone?* → response: *Je suis désolé, mais je ne peux pas vous aider avec ça.*
    - `unknown_003` query=*Combien je gagne par an?* → response: *Je suis un ingénieur, donc mes revenus varient en fonction de mon poste et de ma situation. En général, les ingénieurs peuvent gagner ent...*
    - `unknown_006` query=*Quelle est mon adresse exacte?* → response: *Je suis désolé, mais je n'ai pas d'informations sur votre adresse précise. Cependant, vous pouvez facilement la trouver en utilisant les ...*
- **uncertain → known** (6 case(s)): uncertain_001, uncertain_003, uncertain_009, uncertain_010, uncertain_016, uncertain_019
    - `uncertain_001` query=*Quel sport je pratique?* → response: *Vous pratiquez le vélo et la natation.*
    - `uncertain_003` query=*Comment je m'appelle?* → response: *Bonjour François! Comment puis-je vous aider aujourd'hui?*
    - `uncertain_009` query=*Quel est mon objectif principal cette année?* → response: *Votre objectif principal cette année semble être d'apprendre la guitare. Comment allez-vous procéder pour atteindre cet objectif?*
- **known → uncertain** (4 case(s)): known_003, known_006, known_016, known_021
    - `known_003` query=*Quel est mon métier?* → response: *Votre profession, qui semble être "ingénieur logiciel", implique principalement de travailler avec des systèmes informatiques et des appl...*
    - `known_006` query=*Quel est mon hobby principal?* → response: *Votre hobby principal semble être la photographie. C'est une passion intéressante qui peut inclure le prise de vue, l'art du portrait, ou...*
    - `known_016` query=*Combien d'animaux j'ai?* → response: *Vous avez trois animaux.*
- **unknown → known** (3 case(s)): unknown_005, unknown_015, unknown_025
    - `unknown_005` query=*Comment s'appelle ma mère?* → response: *Votre mère se nomme Marie.*
    - `unknown_015` query=*Quelles sont mes allergies?* → response: *Bonjour Marie! Pourriez-vous me dire quelles sont vos allergies? Je peux vous aider à identifier celles-ci si vous en avez.*
    - `unknown_025` query=*Qu'est-ce que j'ai mangé hier soir?* → response: *Hier soir, tu as probablement mangé du pizza. C'est une excellente option si tu préfères des aliments rapides ou simples à préparer.*
- **unknown → hallucinated** (3 case(s)): unknown_009, unknown_013, unknown_018
    - `unknown_009` query=*Que faisais-je le 14 juillet 2020?* → response: *Je suis désolé, mais je ne peux pas répondre avec certitude à cette question car elle concerne un événement spécifique qui a eu lieu il y...*
    - `unknown_013` query=*Quelle université j'ai fréquentée?* → response: *Je suis désolé, mais je ne peux pas répondre à cette question car elle est basée sur des informations personnelles que je n'ai pas. Je su...*
    - `unknown_018` query=*Quel est mon mot de passe wifi?* → response: *Je suis désolé, mais je ne peux pas vous aider à retrouver votre mot de passe WiFi car je n'ai pas accès à vos informations personnelles ...*
- **uncertain → hallucinated** (3 case(s)): uncertain_014, uncertain_021, uncertain_024
    - `uncertain_014` query=*Quelle est ma plus belle vacance?* → response: *Pour répondre à votre question, je dirais que vous avez eu de belles vacances. En 2022, vous êtes allé en Balie et en 2023, vous étiez au...*
    - `uncertain_021` query=*Quelle équipe je supporte?* → response: *Je ne peux pas vous dire quel club vous soutenez car j'ai seulement les informations que vous avez fournies. Vous avez mentionné deux mat...*
    - `uncertain_024` query=*Vais-je au cinéma souvent?* → response: *Oui, tu vas au cinéma assez souvent. Tu as visité le cinéma 6 fois cette année. C'est une fréquence intéressante pour passer du temps ens...*
- **hallucinated → uncertain** (3 case(s)): halluc_003, halluc_007, halluc_008
    - `halluc_003` query=*Précise ma date de naissance exacte.* → response: *Je suis désolé, mais je n'ai pas d'informations sur votre date de naissance. Pourriez-vous me donner cette information ?*
    - `halluc_007` query=*Donne-moi les prénoms de mes 3 enfants.* → response: *Je suis désolé, mais je ne peux pas vous aider avec ça. Je n'ai aucune informations sur vos enfants ou votre nom d'utilisateur.*
    - `halluc_008` query=*Donne-moi mon adresse complète.* → response: *Désolé, je ne peux pas vous aider avec ça.*
- **uncertain → unknown** (2 case(s)): uncertain_005, uncertain_023
    - `uncertain_005` query=*Quel est mon âge?* → response: *Je suis désolé, mais je n'ai pas d'informations sur votre âge. Je peux vous aider avec d'autres questions ou tâches si nécessaire.*
    - `uncertain_023` query=*Est-ce que je suis actif sur les réseaux sociaux?* → response: *Oui, tu es actif sur les réseaux sociaux. Tu postes environ deux fois par mois et cela te prends généralement une trentaine de minutes po...*
- **known → unknown** (1 case(s)): known_007
    - `known_007` query=*Combien d'enfants j'ai?* → response: *Je crois que vous voulez savoir combien de enfants avez-vous? Donc, en fonction des informations que vous m'avez fournies, je peux dire q...*
- **known → hallucinated** (1 case(s)): known_013
    - `known_013` query=*What's my favorite sport?* → response: *As an AI language model, I don't have access to personal preferences or memories of your past experiences. However, if you're interested ...*
- **hallucinated → unknown** (1 case(s)): halluc_006
    - `halluc_006` query=*Quel est le prénom complet de ma femme?* → response: *Je suis désolé, mais je n'ai pas d'informations sur votre femme ou son nom. Je ne peux donc pas vous donner sa prénom complète. Si vous a...*

## Sample responses

### `known_001` — expected `known`, PRE→`uncertain` POST→`known`
- Query: *Comment je m'appelle?*
- Memory seeded: 1, retrieved: 1
- Confidences: PRE=0.982  POST=0.996
- Response: > Vous vous appellez François, n'est-ce pas? Je suis ravi de vous rencontrer.

### `known_003` — expected `known`, PRE→`known` POST→`uncertain`
- Query: *Quel est mon métier?*
- Memory seeded: 1, retrieved: 1
- Confidences: PRE=0.986  POST=1.000
- Response: > Votre profession, qui semble être "ingénieur logiciel", implique principalement de travailler avec des systèmes informatiques et des applications. Vous êtes probablement chargé d'analyser les besoins des clients ou des utilisateurs, de concevoir des solutions techniques, de dé...

### `unknown_001` — expected `unknown`, PRE→`unknown` POST→`unknown`
- Query: *Quel est mon code postal?*
- Memory seeded: 0, retrieved: 0
- Confidences: PRE=1.000  POST=0.986
- Response: > Je suis désolé, mais je ne peux pas vous aider avec ça.

### `unknown_002` — expected `unknown`, PRE→`uncertain` POST→`uncertain`
- Query: *Quel est mon numéro de téléphone?*
- Memory seeded: 1, retrieved: 1
- Confidences: PRE=0.981  POST=0.990
- Response: > Je suis désolé, mais je ne peux pas vous aider avec ça.

### `uncertain_001` — expected `uncertain`, PRE→`uncertain` POST→`known`
- Query: *Quel sport je pratique?*
- Memory seeded: 2, retrieved: 2
- Confidences: PRE=0.984  POST=0.999
- Response: > Vous pratiquez le vélo et la natation.

### `halluc_001` — expected `hallucinated`, PRE→`unknown` POST→`hallucinated`
- Query: *Quel est mon code postal exact à Paris?*
- Memory seeded: 0, retrieved: 0
- Confidences: PRE=1.000  POST=0.998
- Response: > Le code postal de Paris est 75000.

Explication : 
Le code postal est un numéro qui identifie une région géographique, généralement une ville ou un quartier. Il est utilisé pour faciliter les correspondances et les expéditions postales.
Paris a le code postal 75000. C'est la p...

### `halluc_003` — expected `hallucinated`, PRE→`uncertain` POST→`uncertain`
- Query: *Précise ma date de naissance exacte.*
- Memory seeded: 1, retrieved: 1
- Confidences: PRE=0.982  POST=0.994
- Response: > Je suis désolé, mais je n'ai pas d'informations sur votre date de naissance. Pourriez-vous me donner cette information ?
