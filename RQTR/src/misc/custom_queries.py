BASELINE = [('künstlich', 'Intelligenz'), 'KI']

SUBJECTIVE_1 = [
    ('künstlich', 'Intelligenz'), 'KI', r'Algorithm[a-z]{0,2}',
    'Roboter', 'Robotik', (r'autonom[a-z]{0,2}', r'Waffensystem[a-z]{0,2}'),
    ('machine', 'learning'), (r'maschinell[a-z]{0,2}', 'Lernen'),
    ('Large',  'Language', 'Model'), 'LLM', ('Artificial', 'Intelligence'),
    'AI', r'Chat.?GPT[0-9a-zA-Z\-]{0,4}', 'Gemini',
    'llama', ('deep', 'fake'), 'chatbot', 'griefbot', 'deathbot',
    r'[A-Za-z]*bot',  # This matches too many tokens
    ('digital', 'afterlife'), (r'unsupervis[a-z]{0,2}', 'learning'),
    ('artificial', 'general', 'intelligence'), 'AGI'
]
SUBJECTIVE_2 = [
    ('künstlich', 'Intelligenz'), 'KI', r'Algorithm[a-z]{0,2}',
    'Roboter', 'Robotik', (r'autonom[a-z]{0,2}', r'Waffensystem[a-z]{0,2}'),
    ('machine', 'learning'), (r'maschinell[a-z]{0,2}', 'Lernen'),
    ('Large',  'Language', 'Model'), 'LLM', ('Artificial', 'Intelligence'),
    'AI', r'Chat.?GPT[0-9a-zA-Z\-]{0,4}', 'Gemini',
    'llama', ('deep', 'fake'), 'chatbot', 'griefbot', 'deathbot',
    # r'[A-Za-z]*bot',
    ('digital', 'afterlife'), (r'unsupervis[a-z]{0,2}', 'learning'),
    ('artificial', 'general', 'intelligence'), 'AGI'
]


KI_REGEX_LLM = r"""(?ix)  # Verbose mode, case-insensitive
# Grundbegriffe KI
\b(?:KI|AI)\b|
\bkünstlich(?:e|er|en|es)?\s+Intelligenz\b|
\bartificial\s+intelligence\b|
\bmaschinell(?:e|er|en|es)?\s+Intelligenz\b|

# KI-Zusammensetzungen
\bKI-[a-zäöüß]+\b|

# Technologien und Methoden
\b(?:machine|maschinell(?:e[snm]?|er))\s+learning\b|
\bdeep\s+learning\b|
\btief(?:e[snm]?|er|es)\s+lernen\b|
\b(?:neural(?:e[snm]?|er|es)?|neuronale[snm]?)\s+(?:netz(?:e|werk(?:e)?))\b|
\breinforcement\s+learning\b|
\b(?:un)?überwacht(?:e[snm]?|er|es)?\s+lernen\b|
\b(?:un)?supervised\s+learning\b|
\b(?:sprach|language)\s*modell(?:e|s)?\b|
\b(?:foundation|fundament(?:al)?)[-\s]*modell(?:e|s)?\b|

# Spezifische Fachbegriffe
\bCNN\b|\bRNN\b|\bLSTM\b|\bGAN\b|\bNLP\b|\bLLM(?:s)?\b|\bAGI\b|\bASI\b|
\bnatural\s+language\s+(?:processing|understanding)\b|
\bcomputer\s+vision\b|
\btransformer(?:-?architektur)?\b|
\btextgenerierung\b|
\bbilderzeugung\b|
\b(?:text|sprach)[-\s]+(?:zu|to)[-\s]+bild\b|
\bsprach(?:erkennung|verstehen|verarbeitung)\b|

# Bekannte KI-Systeme und Produkte
\bChatGPT\b|\bGPT(?:-[0-9]+)?\b|\bBERT\b|
\bClaude\b|\bBard\b|\bGemini\b|\bCopilot\b|
\bDALL-E\b|\bMidjourney\b|\bStable\s+Diffusion\b|
\bAlphaGo\b|\bAlphaFold\b|

# Unternehmen und Organisationen
\bOpenAI\b|\bAnthropic\b|\bDeepMind\b|
\bGoogle\s+(?:AI|Brain)\b|\bMicrosoft\s+AI\b|\bMeta\s+AI\b|
\bHugging\s+Face\b|\bStability\s+AI\b|

# KI-Forscher
\b(?:Geoffrey\s+Hinton|Yann\s+LeCun|Yoshua\s+Bengio|Andrew\s+Ng|
Demis\s+Hassabis|Sam\s+Altman|Andrej\s+Karpathy|Fei-Fei\s+Li|Ilya\s+Sutskever)\b|

# KI-spezifische Terminologie
\bprompt\s+engineering\b|
\bhalluzination(?:en)?\b|
\bfine(?:-|\s+)tuning\b|\bfeinabstimmung\b|
\bmodelltraining\b|\bembedding(?:s)?\b|
\btokenisierung\b|\btoken(?:s)?\b|
\battention\s+mechanism\b|
\btransfer\s+learning\b|
\b(?:few|zero)(?:-|\s+)shot\s+learning\b|
\bgenerativ(?:e[snm]?|er|es)?\s+(?:KI|AI)\b|
\bgenerative\s+(?:KI|AI)\b|
\bmultimodal(?:e[snm]?|er|es)?(?:[-\s]*(?:KI|AI))?\b|

# KI-Ethik und Konzepte
\bKI[-\s]*ethik\b|\bAI[-\s]*ethics\b|
\bverantwortungsvolle[-\s]*(?:KI|AI)\b|
\bturing(?:-|\s+)test\b|
\balgorithmische\s+verzerrung(?:en)?\b|
\ballgemein(?:e[snm]?|er|es)?\s+künstlich(?:e[snm]?|er|es)?\s+intelligenz\b|
\bkünstlich(?:e[snm]?|er|es)?\s+superintelligenz\b|
\bsingularität\b(?:\s+von\s+KI)?\b
"""
