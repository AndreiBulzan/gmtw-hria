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
    "doua": {"două", "doua"},  # "două" = cardinal (two), "doua" = ordinal (a doua = the second)
    "fara": {"fără"},
    "fata": {"fata", "fată", "față"},  # "fata"=the girl/face, "fată"=girl, "față"=face
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
    # NOTE: "intra" already defined above as {"intra", "intră"} - both present and imperfect
    "intreaba": {"întreabă"},
    "intrebare": {"întrebare"},
    "intreg": {"întreg"},
    "intreaga": {"întreagă", "întreaga"},  # Both indefinite and articulated forms valid
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

    # ț words (only unique entries - many already defined in ă/â sections above)
    "aceștia": {"aceștia"},
    "atata": {"atâta"},
    "atatia": {"atâția"},
    "atatea": {"atâtea"},
    # NOTE: cativa, cateva, cati, cate already defined above
    "cunostinta": {"cunoștință"},
    # NOTE: "dimineata" already defined above with both forms {"dimineața", "dimineață"}
    # NOTE: "fata" already defined above as {"fata", "fată"} - includes both meanings
    "functioneaza": {"funcționează"},
    "imediat": {"imediat"},
    "intelege": {"înțelege"},
    "inteles": {"înțeles"},
    # NOTE: "invatamant" already defined above
    "intelepciune": {"înțelepciune"},
    "natiune": {"națiune"},
    "natie": {"nație"},
    "participanti": {"participanți"},
    "situatie": {"situație"},
    # NOTE: "tara" already defined above as {"țara", "țară"} with both forms
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
    # NOTE: "viata" already defined above as {"viața", "viață"}

    # Common function words (only unique entries)
    "acestia": {"aceștia"},
    "acestea": {"acestea"},
    # NOTE: "aceasta" already defined above as {"aceasta", "această"}
    "acela": {"acela"},
    "aceea": {"aceea"},
}

# Words that MUST have diacritics (unambiguous cases)
# These are words where the ASCII form is NEVER valid Romanian.
# IMPORTANT: Do NOT include words with multiple valid diacritified forms.
# Examples excluded:
#   - "mana" - could be "mână" (hand) or "mană" (manna)
#   - "scoala" - could be "școală" (school) or "scoală" (wake up!)
#   - "invata" - could be "învață" (learns) or "învăța" (to learn)
#   - "sa" - could be "să" (subjunctive) or "a sa" (hers)
#   - "pana" - could be "până" (until) or "pană" (feather)
#
# stripped_form -> canonical_diacritified_form
MUST_HAVE_DIACRITICS: dict[str, str] = {
    # =========================================================================
    # ESSENTIAL FUNCTION WORDS
    # =========================================================================
    "si": "și",           # and - NEVER valid as "si"
    "in": "în",           # in - NEVER valid as "in"
    "asa": "așa",         # so/thus
    "daca": "dacă",       # if
    "fara": "fără",       # without
    "cand": "când",       # when
    "insa": "însă",       # however
    "inca": "încă",       # still/yet
    "dupa": "după",       # after
    "catre": "către",     # towards
    "decat": "decât",     # than/only
    "incat": "încât",     # so that
    "totusi": "totuși",   # nevertheless
    "macar": "măcar",     # at least
    "asadar": "așadar",   # therefore
    "fiindca": "fiindcă", # because

    # =========================================================================
    # QUANTITY / DEGREE WORDS
    # =========================================================================
    "cat": "cât",         # how much
    "cati": "câți",       # how many (masc)
    "cate": "câte",       # how many (fem)
    "cateva": "câteva",   # a few (fem)
    "cativa": "câțiva",   # a few (masc)
    "atat": "atât",       # so much
    "atata": "atâta",     # so much
    "atatia": "atâția",   # so many
    "atatea": "atâtea",   # so many
    "oricat": "oricât",   # however much
    "oricand": "oricând", # anytime
    "oricati": "oricâți", # however many (masc)
    "oricate": "oricâte", # however many (fem)

    # =========================================================================
    # TIME WORDS
    # =========================================================================
    "intai": "întâi",     # first
    "maine": "mâine",     # tomorrow
    "cateodata": "câteodată",   # sometimes
    "niciodata": "niciodată",   # never
    "vreodata": "vreodată",     # ever
    "intotdeauna": "întotdeauna", # always
    "inainte": "înainte", # before/forward
    "inapoi": "înapoi",   # back

    # Days of week (only unambiguous ones)
    "marti": "marți",     # Tuesday
    "sambata": "sâmbătă", # Saturday

    # =========================================================================
    # COMMON VERBS (î- prefix) - all unambiguous
    # =========================================================================
    "incepe": "începe",   # begins
    "incep": "încep",     # I begin
    "inceput": "început", # beginning/begun
    "incerca": "încerca", # to try
    "incerc": "încerc",   # I try
    "inchide": "închide", # closes
    "inchis": "închis",   # closed
    "inveti": "înveți",   # you learn (singular)
    "intelege": "înțelege", # understands
    "inteles": "înțeles", # understood
    "intreb": "întreb",   # I ask
    "intreaba": "întreabă", # asks
    "intrebare": "întrebare", # question
    "intalnesc": "întâlnesc", # I meet
    "intalnire": "întâlnire", # meeting
    "intorc": "întorc",   # I return
    "intoarce": "întoarce", # returns
    "impotriva": "împotriva", # against
    "impreuna": "împreună", # together
    "incotro": "încotro", # where to

    # =========================================================================
    # COMMON VERBS (ști- stem)
    # =========================================================================
    "stie": "știe",       # knows (3rd person)
    "stii": "știi",       # you know (singular)
    "stiu": "știu",       # I know
    "stim": "știm",       # we know
    "stiti": "știți",     # you know (plural)
    "stiut": "știut",     # known
    "stiinta": "știință", # science

    # =========================================================================
    # COMMON VERBS (-ește/-ează endings)
    # =========================================================================
    "gaseste": "găsește",     # finds
    "gandeste": "gândește",   # thinks
    "reuseste": "reușește",   # succeeds
    "traieste": "trăiește",   # lives
    "urmareste": "urmărește", # follows/watches
    "lucreaza": "lucrează",   # works
    "asteapta": "așteaptă",   # waits
    "astept": "aștept",       # I wait
    "asteptam": "așteptăm",   # we wait
    "ramane": "rămâne",       # remains
    "raman": "rămân",         # I remain

    # =========================================================================
    # VERB FORMS - 2nd person plural (-ți ending)
    # =========================================================================
    "faceti": "faceți",   # you do/make (plural)
    "vreti": "vreți",     # you want (plural)
    "puteti": "puteți",   # you can (plural)
    "aveti": "aveți",     # you have (plural)
    "sunteti": "sunteți", # you are (plural)
    "spuneti": "spuneți", # you say (plural)
    "vedeti": "vedeți",   # you see (plural)
    "luati": "luați",     # you take (plural)
    "dati": "dați",       # you give (plural)
    "stati": "stați",     # you stay (plural)
    "auziti": "auziți",   # you hear (plural)
    "mergeti": "mergeți", # you go (plural)
    "veniti": "veniți",   # you come (plural)
    "cititi": "citiți",   # you read (plural)
    "scrieti": "scrieți", # you write (plural)

    # =========================================================================
    # VERB FORMS - "a fi" (to be)
    # =========================================================================
    "esti": "ești",       # you are (singular)
    "fiti": "fiți",       # be! (plural imperative)

    # =========================================================================
    # COMMON NOUNS - strictly unambiguous
    # =========================================================================
    "tara": "țară",       # country
    "tarii": "țării",     # of the country
    "tarile": "țările",   # the countries
    "oras": "oraș",       # city
    "orasul": "orașul",   # the city
    "orase": "orașe",     # cities
    "viata": "viață",     # life
    "paine": "pâine",     # bread
    "gand": "gând",       # thought
    "ganduri": "gânduri", # thoughts
    "maini": "mâini",     # hands
    "pamant": "pământ",   # earth/ground
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

    # =========================================================================
    # ADJECTIVES / ADVERBS
    # =========================================================================
    "usor": "ușor",       # easy/easily
    "usoara": "ușoară",   # easy (fem)
    "urmatorul": "următorul",   # the next (masc)
    "urmatoarea": "următoarea", # the next (fem)
    "urmator": "următor", # next

    # =========================================================================
    # NUMBERS
    # =========================================================================
    "sase": "șase",       # six
    "sapte": "șapte",     # seven
    # "doua" removed - context-dependent: "două" (cardinal) vs "doua" in "a doua" (ordinal)

    # =========================================================================
    # ADDITIONAL COMMON WORDS (expanded coverage)
    # =========================================================================
    # Prepositions and adverbs with î-
    "intre": "între",           # between
    "inauntru": "înăuntru",     # inside
    "imprejur": "împrejur",     # around
    "inapoi": "înapoi",         # back (already present but ensuring)
    "inainte": "înainte",       # before/forward (already present)

    # Compound words
    "bineinteles": "bineînțeles",     # of course
    "niciodata": "niciodată",         # never
    # NOTE: "totdeauna" and "oriunde" removed - they're valid without diacritics

    # Nouns with diacritics
    "intelegere": "înțelegere",       # understanding
    "intelepciune": "înțelepciune",   # wisdom
    "insemnatate": "însemnătate",     # importance

    # Verbs with diacritics
    "insemna": "însemna",       # to mean
    "insoti": "însoți",         # to accompany
    "indruma": "îndruma",       # to guide
    "ingriji": "îngriji",       # to care for
    "invata": "învăța",         # to learn

    # Reflexives and pronouns
    "insusi": "însuși",         # himself
    "insasi": "însăși",         # herself
    "insesi": "înseși",         # themselves (fem)
    "insisi": "înșiși",         # themselves (masc)
    "isi": "își",               # reflexive "își" - very common!

    # Common adjectives
    "insarcinat": "însărcinat",       # pregnant/charged
    "insarcinata": "însărcinată",     # pregnant (fem)
    "intelept": "înțelept",           # wise
    "inteleapta": "înțeleaptă",       # wise (fem)
    "intreaga": "întreagă",           # whole
    "gresit": "greșit",               # wrong
    "romanesc": "românesc",           # Romanian (adj masc)
    "romaneasca": "românească",       # Romanian (adj fem)

    # Days and time
    "duminica": "duminică",     # Sunday
    "dimineata": "dimineața",   # morning - ASCII never valid (needs ț)
    "aseara": "aseară",         # last night

    # More verb forms
    "incearca": "încearcă",     # tries
    "incercam": "încercăm",     # we try
    "incercati": "încercați",   # you try (plural)
    "intelegem": "înțelegem",   # we understand
    "intelegeti": "înțelegeți", # you understand (plural)
    "gresesc": "greșesc",       # I'm wrong
    "greseste": "greșește",     # is wrong
    # NOTE: "exista" removed - can be valid past tense "existed"
    "prezinta": "prezintă",     # presents
    "pastreaza": "păstrează",   # keeps/preserves
    "hotaraste": "hotărăște",   # decides

    # =========================================================================
    # GERUNDS (-ând/-ind forms) - ASCII never valid
    # =========================================================================
    "facand": "făcând",         # doing
    "stiind": "știind",         # knowing
    "gandind": "gândind",       # thinking
    "ramanand": "rămânând",     # remaining
    "cantand": "cântând",       # singing
    "parand": "părând",         # seeming
    "avand": "având",           # having
    "vazand": "văzând",         # seeing
    "cautand": "căutând",       # searching
    "incepand": "începând",     # beginning
    "incercand": "încercând",   # trying
    "intelegand": "înțelegând", # understanding
    "asteptand": "așteptând",   # waiting

    # =========================================================================
    # NOUNS WITH -ție/-țiune (very common, ASCII never valid)
    # =========================================================================
    "functie": "funcție",       # function
    "conditie": "condiție",     # condition
    "atentie": "atenție",       # attention
    "traditie": "tradiție",     # tradition
    "pozitie": "poziție",       # position
    "sectie": "secție",         # section
    "directie": "direcție",     # direction
    "actiune": "acțiune",       # action
    "mentiune": "mențiune",     # mention
    "exceptie": "excepție",     # exception
    "propozitie": "propoziție", # sentence/proposition
    "emotie": "emoție",         # emotion
    "promotie": "promoție",     # promotion
    "relatia": "relația",       # the relationship
    "relatie": "relație",       # relationship
    "statia": "stația",         # the station
    "statie": "stație",         # station
    "operatie": "operație",     # operation
    "situatia": "situația",     # the situation
    "informatia": "informația", # the information
    "informatie": "informație", # information

    # =========================================================================
    # MORE COMMON NOUNS (ASCII never valid)
    # =========================================================================
    "cuvant": "cuvânt",         # word
    "cuvantul": "cuvântul",     # the word
    "masina": "mașină",         # car
    "masinile": "mașinile",     # the cars
    "greseala": "greșeală",     # mistake
    "incercare": "încercare",   # attempt
    "intamplare": "întâmplare", # happening/event
    "stiinta": "știință",       # science (already present but ensuring)
    "cunostinta": "cunoștință", # knowledge/acquaintance
    "fiinta": "ființă",         # being/creature
    "privinta": "privință",     # regard (în privința = regarding)
    "tacere": "tăcere",         # silence
    "razboi": "război",         # war
    "cantec": "cântec",         # song
    "cantece": "cântece",       # songs

    # =========================================================================
    # NATIONALITIES AND LANGUAGES
    # =========================================================================
    "romani": "români",         # Romanians
    "romanca": "româncă",       # Romanian woman
    "romana": "română",         # Romanian (language/adj fem)
    "franceza": "franceză",     # French (language/adj fem)
    "engleza": "engleză",       # English (language/adj fem)
    "germana": "germană",       # German (language/adj fem)

    # =========================================================================
    # DETERMINERS AND PRONOUNS
    # =========================================================================
    "niste": "niște",           # some (very common!)
    "acestia": "aceștia",       # these (masc)
    "carui": "cărui",           # whose (masc gen)
    "carei": "cărei",           # whose (fem gen)
    "carora": "cărora",         # whose (plural gen)

    # =========================================================================
    # AUXILIARY/MODAL VERBS
    # =========================================================================
    "as": "aș",                 # conditional 1st person (I would)
    "ati": "ați",               # 2nd person plural auxiliary (you have)
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
    # JSON format tokens (expected in structured output)
    "day", "day1", "day2", "day3", "day4", "day5",
    "null", "true", "false",  # JSON literals
    # NOTE: "high", "medium", "low" removed - prompts now use Romanian
    # priority names (înaltă, medie, scăzută). If model outputs English, penalize.
}
