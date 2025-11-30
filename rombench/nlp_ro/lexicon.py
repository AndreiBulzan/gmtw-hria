"""
Romanian Lexicon for Diacritic Analysis

Contains common Romanian words that require diacritics.
Each entry maps the stripped (ASCII) form to the correct diacritified form(s).

This lexicon focuses on high-frequency words where diacritic usage
is unambiguous and important for proper Romanian text.
"""

# Mapping: stripped_form -> set of correct diacritified forms
# Some words have multiple valid forms (e.g., regional variants)
DIACRITIC_WORDS: dict[str, set[str]] = {
    # ă words
    "aceasta": {"aceasta", "această"},
    "acestea": {"acestea"},
    "acesta": {"acesta"},
    "alta": {"alta", "altă"},
    "asemenea": {"asemenea"},
    "asa": {"așa"},
    "asadar": {"așadar"},
    "banca": {"banca", "bancă"},
    "bara": {"bara", "bară"},
    "casa": {"casa", "casă"},
    "catra": {"către"},
    "catre": {"către"},
    "cand": {"când"},
    "cateva": {"câteva"},
    "cativa": {"câțiva"},
    "daca": {"dacă"},
    "deasupra": {"deasupra"},
    "dimineata": {"dimineața", "dimineață"},
    "doua": {"două"},
    "fara": {"fără"},
    "fata": {"fata", "fată"},  # Can be "the face" or "girl"
    "grădina": {"grădina"},
    "gradina": {"grădina"},
    "inapoi": {"înapoi"},
    "insa": {"însă"},
    "intr": {"într"},
    "intra": {"intra", "intră"},
    "masa": {"masa", "masă"},
    "miercuri": {"miercuri"},
    "nevoie": {"nevoie"},
    "oara": {"oară"},
    "oras": {"oraș"},
    "orasul": {"orașul"},
    "oricare": {"oricare"},
    "pana": {"până"},
    "para": {"para", "pară"},  # Can be "pear" or other
    "pastra": {"păstra"},
    "peste": {"peste"},
    "plata": {"plata", "plată"},
    "poate": {"poate"},
    "poarta": {"poarta", "poartă"},
    "problema": {"problema", "problemă"},
    "putea": {"putea"},
    "rama": {"rama", "ramă"},
    "ramane": {"rămâne"},
    "romana": {"română"},
    "romaneasca": {"românească"},
    "romanesc": {"românesc"},
    "saraca": {"săraca", "săracă"},
    "seara": {"seara", "seară"},
    "scoala": {"școala", "școală"},
    "spata": {"spata", "spată"},
    "strada": {"strada", "stradă"},
    "tara": {"țara", "țară"},
    "treaba": {"treaba", "treabă"},
    "vara": {"vara", "vară"},  # Can be "summer" or "cousin"
    "vatra": {"vatra", "vatră"},
    "viata": {"viața", "viață"},
    "vineri": {"vineri"},
    "vreodata": {"vreodată"},
    "zambet": {"zâmbet"},
    "zapada": {"zăpadă"},

    # â words
    "cand": {"când"},
    "cat": {"cât"},
    "cati": {"câți"},
    "cate": {"câte"},
    "cateva": {"câteva"},
    "cativa": {"câțiva"},
    "cantec": {"cântec"},
    "camp": {"câmp"},
    "campul": {"câmpul"},
    "castig": {"câștig"},
    "castiga": {"câștiga", "câștigă"},
    "gand": {"gând"},
    "gandul": {"gândul"},
    "gandesc": {"gândesc"},
    "gandire": {"gândire"},
    "infrant": {"înfrânt"},
    "invatamant": {"învățământ"},
    "mantuit": {"mântuit"},
    "mana": {"mâna", "mână"},
    "maine": {"mâine"},
    "mancare": {"mâncare"},
    "paine": {"pâine"},
    "pamant": {"pământ"},
    "parau": {"pârâu"},
    "ramas": {"rămas"},
    "ramane": {"rămâne"},
    "rand": {"rând"},
    "randul": {"rândul"},
    "sangele": {"sângele"},
    "sange": {"sânge"},
    "sant": {"sfânt"},
    "sfant": {"sfânt"},
    "zambi": {"zâmbi"},
    "zambet": {"zâmbet"},
    "zambesc": {"zâmbesc"},

    # î words (initial/medial)
    "in": {"în"},
    "inainte": {"înainte"},
    "inapoi": {"înapoi"},
    "inalt": {"înalt"},
    "inalta": {"înaltă"},
    "incepe": {"începe"},
    "incep": {"încep"},
    "inceput": {"început"},
    "incerca": {"încerca"},
    "inchide": {"închide"},
    "inchis": {"închis"},
    "inca": {"încă"},
    "incotro": {"încotro"},
    "indrazni": {"îndrăzni"},
    "infrant": {"înfrânt"},
    "insa": {"însă"},
    "insasi": {"însăși"},
    "insusi": {"însuși"},
    "intelege": {"înțelege"},
    "inteles": {"înțeles"},
    "intotdeauna": {"întotdeauna"},
    "intr": {"într"},
    "intra": {"intră"},  # verb "enters"
    "intreaba": {"întreabă"},
    "intrebare": {"întrebare"},
    "intreg": {"întreg"},
    "intreaga": {"întreagă"},
    "invata": {"învăța", "învață"},
    "invatamant": {"învățământ"},

    # ș words
    "asa": {"așa"},
    "asadar": {"așadar"},
    "aseza": {"așeza"},
    "castig": {"câștig"},
    "castiga": {"câștiga", "câștigă"},
    "cunostinta": {"cunoștință"},
    "desigur": {"desigur"},
    "scoala": {"școala", "școală"},
    "stia": {"știa"},
    "stie": {"știe"},
    "stii": {"știi"},
    "stim": {"știm"},
    "stiinta": {"știință"},
    "stiut": {"știut"},
    "si": {"și"},
    "usa": {"ușa", "ușă"},
    "usura": {"ușura"},
    "usor": {"ușor"},
    "usoara": {"ușoară"},
    "sase": {"șase"},
    "sapte": {"șapte"},
    "saptamana": {"săptămână"},
    "sarpe": {"șarpe"},
    "sedinta": {"ședință"},
    "sefa": {"șefa"},
    "sef": {"șef"},
    "sosea": {"șosea"},
    "soseaua": {"șoseaua"},

    # ț words
    "aceasta": {"această"},
    "aceștia": {"aceștia"},
    "atata": {"atâta"},
    "atatia": {"atâția"},
    "atatea": {"atâtea"},
    "cativa": {"câțiva"},
    "cateva": {"câteva"},
    "cati": {"câți"},
    "cate": {"câte"},
    "cunostinta": {"cunoștință"},
    "dimineata": {"dimineața"},
    "fata": {"față"},  # "face" meaning
    "functioneaza": {"funcționează"},
    "imediat": {"imediat"},
    "intelege": {"înțelege"},
    "inteles": {"înțeles"},
    "invatamant": {"învățământ"},
    "intelepciune": {"înțelepciune"},
    "natiune": {"națiune"},
    "natie": {"nație"},
    "participanti": {"participanți"},
    "situatie": {"situație"},
    "tara": {"țară"},
    "tarile": {"țările"},
    "tarii": {"țării"},
    "taran": {"țăran"},
    "tarani": {"țărani"},
    "tel": {"țel"},
    "tinut": {"ținut"},
    "tinutul": {"ținutul"},
    "tine": {"ține"},
    "tinta": {"țintă"},
    "tot": {"tot"},  # No diacritic needed
    "viata": {"viața", "viață"},

    # Common function words
    "acestia": {"aceștia"},
    "acestea": {"acestea"},
    "aceasta": {"aceasta", "această"},
    "acela": {"acela"},
    "aceea": {"aceea"},
}

# Words that MUST have diacritics (unambiguous cases)
# These are words where the ASCII form is NEVER valid Romanian.
# stripped_form -> canonical_diacritified_form
MUST_HAVE_DIACRITICS: dict[str, str] = {
    # === Essential function words ===
    "si": "și",           # and
    "in": "în",           # in
    # "sa" excluded - can be possessive "a sa" (hers)
    "asa": "așa",         # so/thus
    "daca": "dacă",       # if
    "fara": "fără",       # without
    "cand": "când",       # when
    "insa": "însă",       # however
    "inca": "încă",       # still/yet
    "dupa": "după",       # after
    "catre": "către",     # towards

    # === Quantity words ===
    "cat": "cât",         # how much
    "cati": "câți",       # how many (masc)
    "cate": "câte",       # how many (fem)
    "cateva": "câteva",   # a few (fem)
    "cativa": "câțiva",   # a few (masc)
    "atat": "atât",       # so much
    "atata": "atâta",     # so much
    "atatia": "atâția",   # so many
    "atatea": "atâtea",   # so many

    # === Common verbs (î- prefix) ===
    "incepe": "începe",   # begins
    "incep": "încep",     # I begin
    "inceput": "început", # beginning/begun
    "incerca": "încerca", # to try
    "incerc": "încerc",   # I try
    "inchide": "închide", # closes
    "inchis": "închis",   # closed
    "invata": "învață",   # learns
    "inveti": "înveți",   # you learn
    "intelege": "înțelege", # understands
    "inteles": "înțeles", # understood
    "intreb": "întreb",   # I ask
    "intreaba": "întreabă", # asks
    "intrebare": "întrebare", # question
    "intalnesc": "întâlnesc", # I meet
    "intalnire": "întâlnire", # meeting
    "intorc": "întorc",   # I return
    "intoarce": "întoarce", # returns
    "inainte": "înainte", # before/forward
    "inapoi": "înapoi",   # back
    "impotriva": "împotriva", # against
    "impreuna": "împreună", # together
    "intotdeauna": "întotdeauna", # always

    # === Common verbs (ști-) ===
    "stie": "știe",       # knows
    "stii": "știi",       # you know
    "stiu": "știu",       # I know
    "stim": "știm",       # we know
    "stiut": "știut",     # known
    "stiinta": "știință", # science

    # === Common nouns ===
    "tara": "țară",       # country
    "tarii": "țării",     # of the country
    "tarile": "țările",   # the countries
    "oras": "oraș",       # city
    "orasul": "orașul",   # the city
    "orase": "orașe",     # cities
    "viata": "viață",     # life
    "maine": "mâine",     # tomorrow
    "paine": "pâine",     # bread
    "gand": "gând",       # thought
    "ganduri": "gânduri", # thoughts
    "mana": "mână",       # hand
    "maini": "mâini",     # hands
    "pamant": "pământ",   # earth/ground
    "scoala": "școală",   # school
    "scoli": "școli",     # schools
    "saptamana": "săptămână", # week
    "saptamani": "săptămâni", # weeks
    "raspuns": "răspuns", # answer
    "raspunsuri": "răspunsuri", # answers
    "acasa": "acasă",     # home (adverb)
    "afara": "afară",     # outside
    "parinti": "părinți", # parents
    "baiat": "băiat",     # boy
    "baieti": "băieți",   # boys
    "barbat": "bărbat",   # man
    "barbati": "bărbați", # men
    "batran": "bătrân",   # old (man)
    "batrani": "bătrâni", # old (men)

    # === Adjectives/adverbs ===
    "usor": "ușor",       # easy/easily
    "usoara": "ușoară",   # easy (fem)
    "asadar": "așadar",   # therefore

    # === Verb forms (a fi - to be) ===
    "esti": "ești",       # you are
    "fiti": "fiți",       # be! (plural imperative)

    # === Numbers ===
    "sase": "șase",       # six
    "sapte": "șapte",     # seven
}

# Common English words that indicate code-switching (should not appear in Romanian)
ENGLISH_STOPWORDS: set[str] = {
    "the", "a", "an", "and", "or", "but", "if", "then", "else",
    "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having",
    "do", "does", "did", "doing",
    "will", "would", "shall", "should", "may", "might", "must", "can", "could",
    "this", "that", "these", "those",
    "i", "you", "he", "she", "it", "we", "they",
    "my", "your", "his", "her", "its", "our", "their",
    "what", "which", "who", "whom", "whose",
    "where", "when", "why", "how",
    "all", "each", "every", "both", "few", "more", "most", "other", "some", "such",
    "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very",
    "just", "also", "now", "here", "there",
    "because", "although", "while", "unless", "until", "since",
    "about", "above", "across", "after", "against", "along", "among", "around",
    "before", "behind", "below", "beneath", "beside", "between", "beyond",
    "during", "except", "inside", "into", "near", "off", "onto", "outside",
    "over", "through", "toward", "under", "upon", "with", "within", "without",
    "however", "therefore", "furthermore", "moreover", "nevertheless",
    "for", "from", "of", "to", "by", "at", "on", "in",
    "as", "up", "out",
}

# Words that look English but are valid Romanian or proper nouns (whitelist)
ENGLISH_WHITELIST: set[str] = {
    "ok", "weekend", "online", "email", "internet", "computer", "software",
    "marketing", "management", "design", "hotel", "restaurant", "taxi",
    "metro", "video", "audio", "tv", "radio", "film", "sport", "golf",
    "tennis", "fotbal", "fitness", "yoga", "pizza", "pasta", "menu",
    # Romanian words that look English
    "nu", "de", "pe", "care", "este", "sunt", "era", "avea",
    "face", "vine", "merge", "place", "vine", "vor", "fi",
    "din", "cu", "la", "prin", "spre", "sub", "asupra",
    "mare", "mic", "bun", "nou", "alt", "tot", "ori",
    "am", "ai", "are", "au",  # Romanian verb forms
}
