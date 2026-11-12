"""Pattern set of review_7_natural.py: residue of an assistant's reply in civic contributions, nine languages.

Each entry is (family, pattern id, scope, regex). The regexes are applied case-insensitively to a normalised text
(typographic apostrophes and quotes made ASCII, non-breaking spaces made spaces). Scope: "any" = whole text,
"head" = first HEAD characters, "tail" = last TAIL characters.

DROPPED lists the pattern ids (or whole families) excluded from the final set after the specificity check on the
pre-ChatGPT texts and the manual reading; they stay in the file so that the script keeps reporting their rates.
"""
HEAD = 300
TAIL = 500

TOOL = r"(?:chat\s?-?gpt|gpt-?[345]\w*|copilot|gemini|claude|bard|mistral ai|perplexity|deepseek|grok)"
EDIT_EN = (r"(?:adjust|adapt|tailor|refine|tweak|expand|shorten|condense|rewrite|rephrase|reword|revise|draft|translate|polish|"
           r"simplify|format|make (?:it|this|any|the)|turn (?:it|this)|add (?:any|more|a |specific|further)|help (?:you )?(?:draft|"
           r"refine|tailor|format|with (?:anything|any)|further)|create a|prepare a|provide (?:a|more|an)|include (?:any|more|specific)|"
           r"modify|customi[sz]e|elaborate|go deeper|also)")

P = []


def add(family, pid, scope, rx):
    P.append((family, pid, scope, rx))


# ------------------------------------------------------------------ self-reference of the model
add("self_reference", "self_en_1", "any", r"\bas an? (?:ai|artificial intelligence)(?: language| text-based| text)? (?:model|assistant|chatbot)\b[^.\n]{0,40}\bi\b")
add("self_reference", "self_en_2", "any", r"\bas a (?:large )?language model\b,? (?:i\b|trained|developed|created)")
add("self_reference", "self_en_3", "any", r"\bi(?: am|'m) (?:chat\s?gpt|(?:just |only )?an ai (?:language model|assistant|model|chatbot)|a (?:large )?language model)\b")
add("self_reference", "self_en_4", "any", r"\bas an ai,? i (?:do not|don't|cannot|can't|am|have no|lack)\b")
add("self_reference", "self_en_5", "any", r"\b(?:my (?:knowledge|training)(?: data)? cut-?off|as of my (?:last|latest|most recent) (?:knowledge |training )?(?:update|cut-?off)|"
    r"my (?:last|latest) (?:knowledge )?update in)\b")
add("self_reference", "self_en_6", "any", r"\bi (?:do not|don't) have (?:personal (?:opinions|beliefs|experiences|views)|access to real-time|real-time access|the ability to browse)")
add("self_reference", "self_es_1", "any", r"\bcomo (?:un |una )?(?:modelo de lenguaje|modelo de ia|inteligencia artificial|ia|asistente (?:de ia|virtual))\b,? "
    r"(?:no (?:puedo|tengo|dispongo)|yo no|desarrollad|entrenad|creado por)")
add("self_reference", "self_es_2", "any", r"\bsoy (?:chat\s?gpt|un modelo de lenguaje|una inteligencia artificial|una ia\b|un asistente (?:de ia|virtual))")
add("self_reference", "self_fr_1", "any", r"\ben tant qu(?:e |')(?:mod[eè]le de langage|mod[eè]le linguistique|ia\b|intelligence artificielle|assistant (?:ia|virtuel))"
    r"[^.\n]{0,30}\bje\b")
add("self_reference", "self_fr_2", "any", r"\bje suis (?:chat\s?gpt|un mod[eè]le de langage|une ia\b|une intelligence artificielle|un assistant (?:ia|virtuel))")
add("self_reference", "self_de_1", "any", r"\bals (?:ki|k[uü]nstliche intelligenz|ki-?(?:sprach)?modell|sprachmodell|ki-assistent)\b,? (?:kann|habe|bin|verf[uü]ge|besitze) ich\b")
add("self_reference", "self_de_2", "any", r"\bich bin (?:chat\s?gpt|ein (?:ki-?)?sprachmodell|ein ki-?modell|eine ki\b|eine k[uü]nstliche intelligenz|ein ki-assistent)")
add("self_reference", "self_it_1", "any", r"\b(?:come|in quanto) (?:modello linguistico|modello di linguaggio|intelligenza artificiale|ia|assistente (?:ai|ia|virtuale))\b,? "
    r"(?:non (?:posso|ho|dispongo)|io non|sviluppat|addestrat)")
add("self_reference", "self_it_2", "any", r"\bsono (?:chat\s?gpt|un modello linguistico|un modello di linguaggio|un'intelligenza artificiale|un assistente (?:ai|ia|virtuale))")
add("self_reference", "self_ca_1", "any", r"\bcom a (?:model de llenguatge|intel·lig[eè]ncia artificial|ia|assistent (?:d'ia|virtual))\b,? (?:no (?:puc|tinc|disposo)|jo no|desenvolupat|entrenat)")
add("self_reference", "self_ca_2", "any", r"\bs[oó]c (?:(?:el |en )?chat\s?gpt|un model de llenguatge|una intel·lig[eè]ncia artificial|un assistent (?:d'ia|virtual))")
add("self_reference", "self_pt_1", "any", r"\bcomo (?:um |uma )?(?:modelo de linguagem|intelig[eê]ncia artificial|ia|assistente (?:de ia|virtual))\b,? "
    r"(?:n[aã]o (?:posso|tenho|disponho)|eu n[aã]o|desenvolvid|treinad)")
add("self_reference", "self_pt_2", "any", r"\bsou (?:o chat\s?gpt|chat\s?gpt|um modelo de linguagem|uma intelig[eê]ncia artificial|um assistente (?:de ia|virtual))")
add("self_reference", "self_sv_1", "any", r"\bsom (?:en )?(?:ai-?(?:spr[aå]k)?modell|spr[aå]kmodell|ai-assistent|ai)\b,? (?:kan|har|[aä]r) jag\b")
add("self_reference", "self_sv_2", "any", r"\bjag [aä]r (?:chat\s?gpt|en (?:ai-?)?spr[aå]kmodell|en ai-assistent|en ai\b)")
add("self_reference", "self_fi_1", "any", r"\b(?:teko[aä]lyn[aä]|kielimallina|teko[aä]lymallina|teko[aä]lyavustajana)\b,? (?:en\b|minulla|olen|min[aä]\b)")
add("self_reference", "self_fi_2", "any", r"\bolen (?:chat\s?gpt|teko[aä]ly\b|kielimalli\b|teko[aä]lyavustaja)")

# ------------------------------------------------------------------ reply openers: affirmative interjection + presentation
add("opener_exclaim", "opex_en_1", "any", r"\b(?:certainly|sure|sure thing|of course|absolutely|great|got it|understood)\s?[!,.:]\s*(?:here(?:'s| is| are)|below is|"
    r"i(?:'ve| have) (?:revised|rewritten|drafted|prepared))\b")
add("opener_exclaim", "opex_es_1", "any", r"(?:\bclaro|\bpor supuesto|\bdesde luego|\bperfecto|\bentendido|\bcon gusto)\s?[!,.:]\s*¿?(?:aqu[ií] (?:tienes|te (?:dejo|presento|propongo)|"
    r"est[aá]|va)\b|a continuaci[oó]n,? te\b|te (?:dejo|presento|propongo)\b)")
add("opener_exclaim", "opex_fr_1", "any", r"\b(?:bien s[uû]r|certainement|absolument|d'accord|tr[eè]s bien|parfait|avec plaisir)\s?[!,.:]\s*(?:voici|voil[aà]|je vous propose|je te propose)\b")
add("opener_exclaim", "opex_de_1", "any", r"\b(?:gerne|nat[uü]rlich|selbstverst[aä]ndlich|klar|sicher|sehr gerne|verstanden)\s?[!,.:]\s*hier (?:ist|sind|kommt)\b")
add("opener_exclaim", "opex_it_1", "any", r"\b(?:certo|certamente|ovviamente|volentieri|perfetto|va bene)\s?[!,.:]\s*ecco\b")
add("opener_exclaim", "opex_ca_1", "any", r"(?:\b[eé]s clar|\bi tant|\bper descomptat|\bclar|\bperfecte|\bentesos)\s?[!,.:]\s*aqu[ií] (?:tens|teniu|et deixo|us deixo)\b")
add("opener_exclaim", "opex_pt_1", "any", r"\b(?:claro|com certeza|certamente|perfeito|entendido)\s?[!,.:]\s*(?:aqui est[aá]|aqui vai|segue)\b")
add("opener_exclaim", "opex_sv_1", "any", r"\b(?:sj[aä]lvklart|absolut|visst|javisst)\s?[!,.:]\s*h[aä]r (?:[aä]r|kommer)\b")
add("opener_exclaim", "opex_fi_1", "any", r"\b(?:totta kai|tietysti|toki|selv[aä]|tottakai)\s?[!,.:]\s*t[aä]ss[aä] (?:on|ovat|tulee)\b")

# ------------------------------------------------------------------ reply openers: "here is a revised version / a comment you can submit"
add("opener_version", "opve_en_1", "any", r"\bhere(?:'s| is) (?:a|an|the|your) (?:revised|rewritten|reworded|rephrased|improved|polished|refined|more (?:formal|concise|polished|"
    r"professional|detailed|natural)|shorter|shortened|condensed|sample|suggested|possible|cleaned[- ]up|translated)\b[^.\n]{0,60}?"
    r"\b(?:version|draft|comment|letter|response|feedback|text|statement|submission|proposal|summary|translation|message|email|reply)\b")
add("opener_version", "opve_en_2", "any", r"\b(?:here(?:'s| is)|below is) (?:a|an|the|your) [^.\n]{0,70}?\b(?:you (?:can|could|may|might) (?:use|submit|send|post|adapt|copy|"
    r"customi[sz]e|consider)|for you to (?:use|submit|send)|tailored to your|based on your (?:request|input|notes|points))")
add("opener_version", "opve_en_3", "head", r"^\W{0,6}here(?:'s| is) (?:a|an|the|your) (?:draft|formal|professional|concise|detailed|comprehensive|structured|well-structured|"
    r"public|thoughtful|strong|compelling|persuasive|clear)\b[^.\n]{0,60}?\b(?:comment|letter|response|feedback|statement|submission|reply)\b")
add("opener_version", "opve_es_1", "any", r"\baqu[ií] (?:tienes|te (?:dejo|presento|propongo)|est[aá]) (?:una?|la|el|tu) (?:versi[oó]n|propuesta de (?:texto|redacci[oó]n|"
    r"comentario|respuesta)|borrador|texto (?:revisado|mejorado|corregido|reformulado)|redacci[oó]n|ejemplo de|posible (?:texto|redacci[oó]n|respuesta|comentario))")
add("opener_version", "opve_fr_1", "any", r"\bvoici (?:une|la|un|le|votre|ta) (?:version (?:r[eé]vis[eé]e|reformul[eé]e|am[eé]lior[eé]e|corrig[eé]e|plus|courte|concise|"
    r"synth[eé]tique|condens[eé]e|raccourcie|structur[eé]e|finale|retravaill[eé]e)|proposition de (?:texte|r[eé]daction|contribution|r[eé]ponse|commentaire|reformulation)|"
    r"reformulation|r[eé][eé]criture|texte (?:r[eé]vis[eé]|reformul[eé]|am[eé]lior[eé]|corrig[eé])|exemple de (?:texte|contribution|r[eé]ponse|commentaire|lettre)|brouillon)")
add("opener_version", "opve_de_1", "any", r"\bhier (?:ist|sind) (?:eine?|der|die|das|dein|ihr|deine|ihre) (?:[uü]berarbeitete|verbesserte|m[oö]gliche|formellere|umformulierte|"
    r"gek[uü]rzte|k[uü]rzere|korrigierte|optimierte|pr[aä]gnante|sachliche)[rsn]?\b")
add("opener_version", "opve_de_2", "any", r"\bhier (?:ist|sind) (?:eine?|der|die|das|dein|ihr) (?:entwurf|vorschlag|formulierungsvorschlag|textvorschlag|beispieltext|mustertext)\b"
    r"[^.\n]{0,60}(?:den (?:du|sie)|die (?:du|sie)|f[uü]r (?:deine|ihre|dein|ihr))")
add("opener_version", "opve_it_1", "any", r"\becco (?:una|la|un|il|a te una|qui una) (?:versione (?:rivista|riformulata|migliorata|corretta|pi[uù]|sintetica|breve|formale|"
    r"concisa|riscritta|ottimizzata)|bozza|proposta di (?:testo|commento|contributo|risposta)|riscrittura|riformulazione|"
    r"possibile (?:testo|versione|commento|risposta|formulazione)|testo (?:rivisto|riformulato|migliorato|corretto))")
add("opener_version", "opve_ca_1", "any", r"\baqu[ií] (?:tens|teniu|et deixo|us deixo) (?:una?|la|el) (?:versi[oó]|proposta de (?:text|redacci[oó]|resposta|comentari)|esborrany|"
    r"text (?:revisat|millorat|reformulat|corregit)|redacci[oó]|possible (?:text|redacci[oó]|resposta))")
add("opener_version", "opve_pt_1", "any", r"\baqui est[aá] (?:uma|a|o|um) (?:vers[aã]o|proposta de (?:texto|reda[cç][aã]o|resposta|coment[aá]rio)|rascunho|"
    r"texto (?:revisado|revisto|melhorado|corrigido|reformulado)|sugest[aã]o de (?:texto|resposta|coment[aá]rio))")
add("opener_version", "opve_sv_1", "any", r"\bh[aä]r [aä]r (?:en|ett) (?:reviderad|f[oö]rb[aä]ttrad|omskriv\w*|omformulerad|kortare|mer formell|f[oö]rslag p[aå] (?:text|svar|"
    r"formulering|kommentar)|utkast)")
add("opener_version", "opve_fi_1", "any", r"\bt[aä]ss[aä] on (?:(?:paranneltu|muokattu|tiivistetty|lyhyempi|virallisempi|uudelleenmuotoiltu|selke[aä]mpi) (?:versio|teksti|ehdotus)|"
    r"ehdotus tekstiksi|ehdotus kommentiksi|luonnos (?:tekstiksi|kommentiksi|palautteeksi|vastaukseksi))\b")
# "... version of your text:" - the reply addresses the person who asked
add("opener_version", "opve_your_1", "any", r"\b(?:version|rewrite|revision|draft|summary|translation) of your (?:text|statement|draft|comment|message|letter|feedback|submission|"
    r"description|proposal|response|points|notes)\b[^.\n]{0,40}:|"
    r"\b(?:version|reformulation|r[eé][eé]criture|synth[eè]se|traduction) (?:\w+ ){0,4}de (?:votre|ton|ta) (?:texte|description|message|commentaire|contribution|proposition|"
    r"r[eé]ponse|avis)\b[^.\n]{0,40}:|"
    r"\b(?:version|fassung|zusammenfassung|[uü]bersetzung) (?:deines|ihres|deiner|ihrer) (?:textes|statements|kommentars|beitrags|stellungnahme|nachricht|entwurfs)\b[^.\n]{0,40}:|"
    r"\b(?:versi[oó]n|redacci[oó]n|reformulaci[oó]n|traducci[oó]n) (?:\w+ ){0,4}de tu (?:texto|comentario|mensaje|propuesta|respuesta|descripci[oó]n|escrito)\b[^.\n]{0,40}:|"
    r"\b(?:versione|riformulazione|riscrittura|traduzione) (?:\w+ ){0,4}del tuo (?:testo|commento|messaggio|contributo)\b[^.\n]{0,40}:|"
    r"\b(?:versi[oó]|redacci[oó]|reformulaci[oó]|traducci[oó]) (?:\w+ ){0,4}del teu (?:text|comentari|missatge|escrit)\b[^.\n]{0,40}:|"
    r"\b(?:vers[aã]o|reformula[cç][aã]o|tradu[cç][aã]o) (?:\w+ ){0,4}do (?:seu|teu) (?:texto|coment[aá]rio)\b[^.\n]{0,40}:|"
    r"\b(?:version|omformulering|[oö]vers[aä]ttning) av din (?:text|kommentar)\b[^.\n]{0,40}:")
# note of the assistant on what its new version changes
add("opener_version", "opve_meta_1", "any", r"\bwas diese (?:version|fassung) (?:[aä]ndert|verbessert)|\bwhat (?:this version|i(?:'ve| have)?) (?:changes|changed|improved)\s?:|"
    r"\bce que (?:cette version|j'ai) (?:change|modifi[eé]|am[eé]lior[eé])\s?:|\blo que (?:cambia|he cambiado|he mejorado) (?:en )?esta versi[oó]n|"
    r"\bcosa (?:cambia in questa versione|ho modificato)\s?:")
# "here is a comment you can submit" in the other languages
add("opener_version", "opve_use_1", "any",
    r"\b(?:voici|voil[aà]) [^.\n]{3,90}\bque (?:vous (?:pouvez|pourriez)|tu (?:peux|pourrais)) (?:envoyer|soumettre|utiliser|d[eé]poser|adapter|publier|copier|transmettre)\b|"
    r"\baqu[ií] (?:tienes|te dejo|est[aá]) [^.\n]{3,90}\bque puedes (?:enviar|usar|presentar|adaptar|utilizar|copiar|publicar)\b|"
    r"\bhier (?:ist|sind) [^.\n]{3,90}\b(?:den|die|das) (?:du|sie) [^.\n]{0,30}(?:verwenden|einreichen|anpassen|nutzen|abschicken|kopieren) (?:kannst|k[oö]nnen|k[oö]nntest)\b|"
    r"\becco [^.\n]{3,90}\bche puoi (?:inviare|usare|utilizzare|adattare|presentare|copiare)\b|"
    r"\baqu[ií] (?:tens|teniu) [^.\n]{3,90}\bque pots (?:enviar|fer servir|utilitzar|adaptar|presentar|copiar)\b|"
    r"\baqui est[aá] [^.\n]{3,90}\bque (?:voc[eê] pode|podes) (?:enviar|usar|utilizar|adaptar|apresentar|copiar)\b|"
    r"\bh[aä]r [aä]r [^.\n]{3,90}\bsom du kan (?:anv[aä]nda|skicka|anpassa|kopiera)\b|"
    r"\bt[aä]ss[aä] on [^.\n]{3,90}\bjo(?:ta|nka) voit (?:k[aä]ytt[aä][aä]|l[aä]hett[aä][aä]|muokata)\b")

# ------------------------------------------------------------------ reply openers at the very start of the text: "Here is a <kind of text> ...:"
NOUN_EN = r"(?:comment|draft|response|letter|version|text|statement|message|feedback|summary|submission|email|reply|proposal|rewrite|translation|template)"
add("opener_head", "ophe_en_1", "head", r"^\W{0,10}(?:here is|here's|below is) (?:a|an|your) [^.!?\n]{0,120}\b" + NOUN_EN + r"\b[^.!?\n]{0,120}:\s*(?:\n|-{3}|\*|\"|#|$)")
add("opener_head", "ophe_fr_1", "head", r"^\W{0,10}voici (?:une?|votre|ton|ta) [^.!?\n]{0,120}\b(?:version|proposition|texte|contribution|r[eé]ponse|commentaire|lettre|"
    r"reformulation|synth[eè]se|brouillon|r[eé]sum[eé]|traduction)\b[^.!?\n]{0,120}:\s*(?:\n|-{3}|\*|\"|#|$)")
add("opener_head", "ophe_es_1", "head", r"^\W{0,10}(?:aqu[ií] (?:tienes|te dejo|te presento)|a continuaci[oó]n,? te (?:dejo|presento|propongo)) [^.!?\n]{3,200}:\s*(?:\n|-{3}|\*|\"|#|$)")
add("opener_head", "ophe_de_1", "head", r"^\W{0,10}hier (?:ist|sind) (?:eine?|der|die|das|dein|ihr|deine|ihre) [^.!?\n]{0,120}\b(?:version|fassung|text|entwurf|vorschlag|kommentar|"
    r"stellungnahme|antwort|zusammenfassung|[uü]bersetzung|formulierung|nachricht|beitrag)\b[^.!?\n]{0,120}:\s*(?:\n|-{3}|\*|\"|#|$)")
add("opener_head", "ophe_it_1", "head", r"^\W{0,10}ecco (?:una?|la|il|lo) [^.!?\n]{0,120}\b(?:versione|bozza|proposta|testo|commento|risposta|lettera|riformulazione|sintesi|"
    r"messaggio|traduzione|contributo)\b[^.!?\n]{0,120}:\s*(?:\n|-{3}|\*|\"|#|$)")
add("opener_head", "ophe_ca_1", "head", r"^\W{0,10}aqu[ií] (?:tens|teniu|et deixo|us deixo) [^.!?\n]{3,200}:\s*(?:\n|-{3}|\*|\"|#|$)")
add("opener_head", "ophe_pt_1", "head", r"^\W{0,10}(?:aqui est[aá]|aqui vai|segue) (?:uma?|a|o) [^.!?\n]{0,120}\b(?:vers[aã]o|proposta|texto|coment[aá]rio|resposta|carta|rascunho|"
    r"resumo|mensagem|tradu[cç][aã]o)\b[^.!?\n]{0,120}:\s*(?:\n|-{3}|\*|\"|#|$)")
add("opener_head", "ophe_sv_1", "head", r"^\W{0,10}h[aä]r (?:[aä]r|kommer) (?:en|ett) [^.!?\n]{0,120}\b(?:version|text|f[oö]rslag|kommentar|svar|sammanfattning|utkast|[oö]vers[aä]ttning)"
    r"\b[^.!?\n]{0,120}:\s*(?:\n|-{3}|\*|\"|#|$)")
add("opener_head", "ophe_fi_1", "head", r"^\W{0,10}t[aä]ss[aä] (?:on|tulee) [^.!?\n]{0,120}\b(?:versio|teksti|ehdotus|kommentti|vastaus|tiivistelm[aä]|luonnos|k[aä][aä]nn[oö]s)\b"
    r"[^.!?\n]{0,120}:\s*(?:\n|-{3}|\*|\"|#|$)")

# exploratory: a list announced at the very start ("Here are some suggestions ...:") and a "Title:" label opening the text
add("opener_list", "opli_1", "head", r"^\W{0,10}(?:(?:here are|below are) (?:some|a few|several|\d+|three|four|five)|voici (?:quelques|plusieurs|\d+|trois|quatre|cinq)|"
    r"hier sind (?:einige|ein paar|mehrere|\d+|drei|vier|f[uü]nf)|aqu[ií] (?:tienes|te dejo) (?:algun[oa]s|vari[oa]s|\d+|tres|cuatro|cinco)|"
    r"ecco (?:alcun[ei]|qualche|divers[ei]|\d+|tre|quattro|cinque)|aqu[ií] tens (?:algun(?:e)?s|diverses|diversos|\d+)|aqui est[aã]o (?:algumas|alguns|v[aá]rias|\d+)|"
    r"h[aä]r [aä]r (?:n[aå]gra|flera|\d+)|t[aä]ss[aä] on (?:muutamia|joitakin|useita)) [^.!?\n]{3,200}:\s*(?:\n|-{3}|\*|\"|#|\d|$)")
add("label_head", "labh_1", "head", r"^\W{0,6}(?:title|titre|t[ií]tulo|titel|titolo|t[ií]tol|otsikko|rubrik|proposal title|titre de la proposition|t[ií]tulo de la propuesta)\s?\*{0,2}\s?:")

# ------------------------------------------------------------------ reply closers: offers to continue editing
add("closer_offer", "clof_en_1", "any", r"\blet me know if you(?:'d| would) like (?:me to|any|a |an |to " + EDIT_EN + r"|it |this |help|further|more)")
add("closer_offer", "clof_en_2", "any", r"\blet me know if you (?:need|want) (?:any (?:further|more|other|additional|changes|adjustments|modifications|edits)|further (?:changes|adjustments|refinements|edits|modifications)|"
    r"me to|it (?:shorter|longer|adjusted|more)|a (?:shorter|longer|different))")
add("closer_offer", "clof_en_6", "tail", r"\b(?:would you like|do you want) me to [^?\n]{3,160}\?\W*$")
add("closer_offer", "clof_en_3", "any", r"\b(?:would you like|do you want) me to " + EDIT_EN + r"\b")
add("closer_offer", "clof_en_4", "any", r"\bif you(?:'d| would) like,? i (?:can|could) (?:also )?" + EDIT_EN + r"\b")
add("closer_offer", "clof_en_5", "any", r"\bwould you like (?:a (?:shorter|longer|more (?:formal|concise|detailed))|it (?:shorter|longer|to be|in a)|this (?:shortened|expanded|adapted|"
    r"formatted|in a)|any (?:changes|adjustments|modifications|refinements))")
add("closer_offer", "clof_es_1", "tail", r"(?:\b|¿)(?:quieres|deseas|te gustar[ií]a|necesitas|prefieres|quiere|desea) que (?:te |se )?(?:lo |la |le |los |las )?(?:ajuste|adapte|redacte|haga|"
    r"prepare|resuma|acorte|ampl[ií]e|traduzca|a[nñ]ada|modifique|reformule|pase|convierta|genere|elabore|ayude|desarrolle|incluya|revise|mejore|"
    r"sintetice|estructure|d[eé])\b[^?\n]{0,160}\?\W*$")
add("closer_offer", "clof_es_2", "any", r"\bsi (?:quieres|deseas|lo deseas|lo prefieres|te parece|gustas),? (?:tambi[eé]n )?(?:puedo|te puedo|puedo ayudarte) "
    r"(?:ajustar|adaptar|redactar|preparar|resumir|acortar|ampliar|traducir|modificar|reformular|ayudar|convertir|generar|elaborar|desarrollar|"
    r"revisar|mejorar|sintetizar|a (?:redactar|preparar|organizar|estructurar|elaborar|crear|convertir|adaptar))")
add("closer_offer", "clof_es_3", "any", r"\b(?:av[ií]same|dime|d[ií]melo|ind[ií]came|h[aá]zmelo saber|d[eé]jame saber) si (?:quieres|deseas|necesitas|prefieres)\b")
add("closer_offer", "clof_fr_1", "tail", r"(?<!comment )\b(?:souhaitez|voulez|souhaites|veux|aimeriez|aimerais|d[eé]sirez)[- ](?:vous|tu) que je "
    r"(?:vous |te |t'|l'|la |le |les )*(?:adapte|reformule|r[eé]dige|raccourcisse|d[eé]veloppe|traduise|ajoute|modifie|propose|pr[eé]pare|aide|ajuste|compl[eè]te|"
    r"synth[eé]tise|fournisse|transforme|mette|fasse|g[eé]n[eè]re|cr[eé]e|structure)\b[^?\n]{0,160}\?\W*$")
add("closer_offer", "clof_fr_2", "any", r"\bsi (?:vous le souhaitez|tu le souhaites|vous voulez|tu veux|vous le d[eé]sirez|besoin),? je peux (?:aussi |[eé]galement )?"
    r"(?:vous |te |l'|la |le |les )?(?:adapter|reformuler|r[eé]diger|raccourcir|d[eé]velopper|traduire|ajouter|modifier|proposer|pr[eé]parer|aider|"
    r"ajuster|compl[eé]ter|synth[eé]tiser|fournir|transformer|mettre)")
add("closer_offer", "clof_fr_3", "any", r"\b(?:dites-moi|dis-moi|faites-moi savoir|fais-moi savoir|faites-le moi savoir) si (?:vous (?:souhaitez|voulez|d[eé]sirez|avez besoin)|"
    r"tu (?:souhaites|veux|as besoin))")
add("closer_offer", "clof_fr_4", "any", r"\bn'h[eé]site[zs]? pas [aà] (?:me (?:dire|demander|le faire savoir|pr[eé]ciser|solliciter)|l'adapter|le modifier|la modifier|l'ajuster|"
    r"le personnaliser|la personnaliser|personnaliser|adapter ce|modifier ce|ajuster)")
add("closer_offer", "clof_de_1", "any", r"\b(?:m[oö]chten sie|m[oö]chtest du|soll ich|wollen sie|willst du),? (?:dass ich )?[^.?!\n]{0,70}\b(?:anpasse|k[uü]rze|umformuliere|erstelle|"
    r"[uü]bersetze|erg[aä]nze|formuliere|ausarbeite|anpassen|k[uü]rzen|umformulieren|erstellen|[uü]bersetzen|erg[aä]nzen|formulieren|ausarbeiten|"
    r"[uü]berarbeiten|[uü]berarbeite)\b[^.?!\n]{0,40}\?")
add("closer_offer", "clof_de_2", "any", r"\b(?:lass|lassen sie) (?:es )?mich (?:es )?(?:gerne )?wissen,? (?:wenn|falls|ob) (?:du|sie)\b")
add("closer_offer", "clof_de_3", "any", r"\bwenn (?:du m[oö]chtest|sie m[oö]chten|du willst|sie wollen|gew[uü]nscht),? kann ich\b")
add("closer_offer", "clof_de_4", "tail", r"\bsoll ich (?:dir|ihnen|das|den|die|es|noch|auch|daraus|daf[uü]r)\b[^?\n]{3,160}\?\W*$")
add("closer_offer", "clof_en_7", "tail", r"\b(?:shall|should) i (?:also )?(?:draft|make|adapt|shorten|expand|translate|rewrite|prepare|tailor|format|turn|create|refine|add)\b[^?\n]{0,160}\?\W*$")
add("closer_offer", "clof_fr_5", "tail", r"\b(?:veux-tu|voulez-vous|souhaitez-vous|souhaites-tu) (?:une version|que j'|aussi que|[eé]galement que)[^?\n]{3,160}\?\W*$")
add("closer_offer", "clof_it_1", "any", r"\b(?:vuoi|desideri|vuole|preferisci|desidera) che (?:ti |le |te )?(?:lo |la |ne )?(?:adatti|riformuli|prepari|renda|accorci|ampli|traduca|"
    r"scriva|aggiunga|modifichi|sintetizzi|faccia|trasformi|aiuti|riveda|sviluppi|integri|elabori)\b")
add("closer_offer", "clof_it_2", "any", r"\bse (?:vuoi|desideri|preferisci|lo desideri|ti va|vuole|desidera),? posso (?:anche )?(?:adattar|riformular|preparar|render|accorciar|"
    r"ampliar|tradur|scriver|aggiunger|modificar|sintetizzar|far|trasformar|aiutar|riveder|sviluppar|integrar|elaborar|fornir)")
add("closer_offer", "clof_it_3", "any", r"\bfammi sapere se (?:vuoi|desideri|hai bisogno|preferisci)\b")
add("closer_offer", "clof_ca_1", "any", r"\b(?:vols|voleu|vol|necessites|prefereixes) que (?:et |us |li |t')?(?:ho |el |la |ne |en )?(?:adapti|redacti|faci|prepari|resumeixi|escurci|"
    r"ampli[iï]|tradueixi|afegeixi|modifiqui|reformuli|ajudi|converteixi|desenvolupi|revisi|millori|inclogui|generi)\b")
add("closer_offer", "clof_ca_2", "any", r"\bsi (?:vols|voleu|ho vols|ho prefereixes|et sembla b[eé]),? (?:tamb[eé] )?(?:puc|et puc|us puc|t'ho puc|ho puc) (?:adaptar|redactar|fer|"
    r"preparar|resumir|escur[cç]ar|ampliar|traduir|afegir|modificar|reformular|ajudar|convertir|desenvolupar|revisar|millorar|incloure|generar)")
add("closer_offer", "clof_ca_3", "any", r"\b(?:digues-me|fes-m'ho saber|avisa'm|digue'm) si (?:vols|necessites|prefereixes)\b")
add("closer_offer", "clof_pt_1", "any", r"\b(?:quer|queres|gostaria|deseja|gostarias|prefere) que eu (?:adapte|redija|fa[cç]a|prepare|resuma|encurte|amplie|traduza|adicione|"
    r"modifique|reformule|ajude|converta|desenvolva|revise|melhore|inclua|gere|ajuste|elabore|torne|crie)\b")
add("closer_offer", "clof_pt_2", "any", r"\bse (?:quiser|quiseres|preferir|desejar|precisar),? (?:tamb[eé]m )?posso (?:adaptar|redigir|fazer|preparar|resumir|encurtar|ampliar|traduzir|"
    r"adicionar|modificar|reformular|ajudar|converter|desenvolver|revisar|melhorar|incluir|gerar|ajustar|elaborar|tornar|criar)")
add("closer_offer", "clof_pt_3", "any", r"\b(?:me avise|avise-me|diga-me|me diga) se (?:quiser|precisar|deseja|desejar)\b")
add("closer_offer", "clof_sv_1", "any", r"\bvill du att jag (?:ska )?(?:anpassa|korta|f[oö]rkorta|utveckla|[oö]vers[aä]tt|l[aä]gg|skriv|formulera|g[oö]r|justera|sammanfatta|"
    r"hj[aä]lp|[aä]ndra|ta fram|skapa)")
add("closer_offer", "clof_sv_2", "any", r"\bom du vill,? kan jag (?:ocks[aå] )?(?:anpassa|korta|f[oö]rkorta|utveckla|[oö]vers[aä]tta|l[aä]gga|skriva|formulera|g[oö]ra|justera|"
    r"sammanfatta|hj[aä]lpa|[aä]ndra|ta fram|skapa)")
add("closer_offer", "clof_sv_3", "any", r"\b(?:s[aä]g till|l[aå]t mig veta) om du (?:vill|beh[oö]ver|[oö]nskar)\b")
add("closer_offer", "clof_fi_1", "any", r"\bhaluatko,? ett[aä] (?:min[aä] )?(?:muokkaan|lyhenn[aä]n|tiivist[aä]n|laajennan|k[aä][aä]nn[aä]n|lis[aä][aä]n|kirjoitan|muotoilen|teen|"
    r"laadin|autan|tarkennan)")
add("closer_offer", "clof_fi_2", "any", r"\bjos haluat,? voin (?:my[oö]s )?(?:muokata|lyhent[aä][aä]|tiivist[aä][aä]|laajentaa|k[aä][aä]nt[aä][aä]|lis[aä]t[aä]|kirjoittaa|muotoilla|"
    r"tehd[aä]|laatia|auttaa|tarkentaa)")
add("closer_offer", "clof_fi_3", "any", r"\bkerro,? jos haluat\b")

# ------------------------------------------------------------------ reply closers: invitation to customise the draft, "I hope this helps"
add("closer_customize", "clcu_en_1", "any", r"\bfeel free to (?:adjust|modify|customi[sz]e|edit|tailor|adapt|personali[sz]e|tweak)\b")
add("closer_customize", "clcu_en_2", "tail", r"\bi hope this (?:helps|draft|version|works for you|meets your)\b")
add("closer_customize", "clcu_en_3", "any", r"\b(?:make sure|be sure|remember|don't forget) to (?:replace|fill in|customi[sz]e|personali[sz]e|update) (?:the |any )?"
    r"(?:placeholders?|bracketed|brackets|details in brackets|information in brackets)")
add("closer_customize", "clcu_es_1", "any", r"\b(?:si[eé]ntete libre de|no dudes en|puedes) (?:ajustar|modificar|adaptar|personalizar|editar)(?:lo|la|los|las)? "
    r"(?:seg[uú]n|a tu|de acuerdo|el texto|este texto|como)")
add("closer_customize", "clcu_es_2", "tail", r"\bespero que (?:esto|esta versi[oó]n|este texto|este borrador) te (?:sea|ayude|sirva|resulte)\b")
add("closer_customize", "clcu_fr_1", "tail", r"\bj'esp[eè]re que (?:cela|ceci|cette version|ce texte|cette proposition) (?:vous|te|t') (?:aide|aidera|convient|conviendra|sera utile|est utile)")
add("closer_customize", "clcu_de_1", "tail", r"\bich hoffe,? (?:das|dies|diese version|dieser text|dieser entwurf) (?:hilft|passt|entspricht)\b")
add("closer_customize", "clcu_de_2", "any", r"\b(?:du kannst|sie k[oö]nnen) (?:den|diesen) (?:text|entwurf) (?:gerne |nat[uü]rlich )?(?:noch )?(?:anpassen|[aä]ndern|personalisieren|erg[aä]nzen)")
add("closer_customize", "clcu_it_1", "any", r"\bsentiti liber[oa] di (?:adattar|modificar|personalizzar|cambiar|aggiunger)")
add("closer_customize", "clcu_it_2", "tail", r"\bspero (?:che (?:questo|questa versione|questo testo|questa bozza) (?:ti|le|vi) (?:sia|possa)|di esserti stat[oa])")
add("closer_customize", "clcu_ca_1", "tail", r"\bespero que (?:aix[oò]|aquesta versi[oó]|aquest text) (?:et|us|t') ?(?:sigui|ajudi|serveixi|vagi b[eé])")
add("closer_customize", "clcu_ca_2", "any", r"\b(?:sent-te lliure de|no dubtis a|no dubtis en) (?:ajustar|modificar|adaptar|personalitzar)")
add("closer_customize", "clcu_pt_1", "tail", r"\bespero que (?:isso|isto|esta vers[aã]o|este texto) (?:ajude|te ajude|seja [uú]til|lhe seja)")
add("closer_customize", "clcu_pt_2", "any", r"\b(?:sinta-se [aà] vontade para|fique [aà] vontade para) (?:ajustar|modificar|adaptar|personalizar|editar)")
add("closer_customize", "clcu_sv_1", "tail", r"\bhoppas (?:att )?(?:det h[aä]r|detta|texten) (?:hj[aä]lper|[aä]r till hj[aä]lp|passar)")
add("closer_customize", "clcu_fi_1", "tail", r"\btoivottavasti t[aä]m[aä] (?:auttaa|on avuksi|sopii)")

# ------------------------------------------------------------------ unfilled placeholders
POSS = (r"(?:your|tu|tus|su|vuestro|vuestra|il tuo|la tua|il suo|il vostro|votre|ton|ta|vos|ihr|ihre|ihren|dein|deine|deinen|el teu|la teva|el seu|el vostre|"
        r"o seu|a sua|seu|sua|teu|tua|ditt|din|dina|sinun|oma)")
YNOUN = (r"(?:full |company |organi[sz]ation |business |first |last )?(?:name|nombre|nombre completo|nome|nom|pr[eé]nom|namn|nimi|"
         r"title|position|job title|role|address|adresse|direcci[oó]n|indirizzo|adre[cç]a|endere[cç]o|city|ciudad|citt[aà]|ville|stadt|ort|wohnort|ciutat|cidade|"
         r"organi[sz]ation|organizaci[oó]n|organizzazione|organisation|company|empresa|entreprise|unternehmen|azienda|affiliation|"
         r"email|e-mail|correo|phone|phone number|contact information|contact details|contact info|coordonn[eé]es|kontaktdaten|"
         r"signature|firma|unterschrift|state|country|location|town|profession|occupation|institution|credentials)")
add("placeholder_your", "phyo_1", "any", r"\[" + POSS + r" " + YNOUN + r"(?:[,/ ]+(?:and |or |y |et |und |e )?" + YNOUN + r"){0,3}\]")
add("placeholder_your", "phyo_2", "any", r"\[" + POSS + r" [^\]\[\n]{2,40}\]")
SLOT = (r"(?:name|full name|first name|last name|nombre|nombre completo|nombre y apellidos?|nome|nome completo|nome e cognome|nom|nom complet|pr[eé]nom|nom et pr[eé]nom|"
        r"pr[eé]nom nom|namn|nimi|vorname nachname|vor- und nachname|city|ciudad|citt[aà]|ville|stadt|ort|ciutat|cidade|date|data|fecha|datum|p[aä]iv[aä]m[aä][aä]r[aä]|"
        r"address|adresse|direcci[oó]n|indirizzo|adre[cç]a|endere[cç]o|organi[sz]ation|organi[sz]ation name|organizaci[oó]n|organizzazione|company name|company|"
        r"title|position|job title|contact information|contact details|email|e-mail|email address|phone number|signature|firma|unterschrift|"
        r"recipient'?s? name|nom du destinataire|destinataire|destinatario|empf[aä]nger|city,? state|state|country|pa[ií]s|paese|pays|land|"
        r"lieu|luogo|lugar|lloc|ort,? datum|lieu,? date|luogo,? data|lugar y fecha)")
add("placeholder_bare", "phba_1", "any", r"\[" + SLOT + r"\]")
add("placeholder_bare", "phba_2", "any", r"\[(?:x{3,}|_{3,})\]")
add("placeholder_insert", "phin_0", "any", r"\[(?:insert|name of|number of|specific|relevant|inserire|inserisci|insertar|inserta|indicar|ins[eé]rer|ins[eé]rez|"
    r"indiquer|indiquez|inserir|insira|indique) [^\]\[\n]{3,90}\]")
add("placeholder_insert", "phin_1", "any", r"\[(?:insert|add|specify|mention|describe|include|enter|fill in|provide|state|name of|number of|list|explain|briefly|"
    r"inserire|inserisci|indicare|indica|specificare|specifica|aggiungere|aggiungi|insertar|inserta|indicar|a[nñ]adir|a[nñ]ade|especificar|especifica|"
    r"ins[eé]rer|ins[eé]rez|indiquer|indiquez|pr[eé]ciser|pr[eé]cisez|ajouter|ajoutez|mentionner|mentionnez|einf[uü]gen|inserir|insira|indique|"
    r"especifique|afegir|afegeix|indiqueu|indiqueu|especificar|ange|infoga|l[aä]gg till|lis[aä][aä]|mainitse) [^\]\[\n]{3,90}\]")
add("placeholder_insert", "phin_2", "any", r"\[[^\]\[\n]{3,60} (?:einf[uü]gen|eintragen|angeben|erg[aä]nzen|hier einf[uü]gen)\]")
add("placeholder_insert", "phin_3", "any", r"\[(?:e\.g\.,?|i\.e\.,?|example:|p\. ?ej\.,?|par exemple,?|z\. ?b\.,?|ad esempio,?|es\.,?|ex\. ?:) [^\]\[\n]{3,90}\]")

# ------------------------------------------------------------------ markdown residue of a chat reply together with letter structure
LABEL = r"(?:subject|asunto|objet|betreff|oggetto|assumpte|assunto|[aä]mne|aihe|title|t[ií]tulo|titre|titel|titolo|t[ií]tol|otsikko|rubrik)"
SALUT = (r"(?:dear\b|to whom it may concern|estimad[oa]s?\b|apreciad[oa]s?\b|madame|monsieur|mesdames|ch[eè]re?s?\b|sehr geehrte|gentile\b|gentili\b|egregi[oa]?\b|"
         r"spett\.|spettabile|benvolgu|distingit|prezad[oa]s?\b|exm[oa]s?\b|hej\b|b[aä]sta\b|hyv[aä]\b|arvoisa)")
add("markdown_letter", "mdle_1", "any", r"\*\*" + LABEL + r"\s?:\s?\*\*|\*\*" + LABEL + r"\*\*\s?:|"
    r"\*\*(?:subject|asunto|objet|betreff|oggetto|assumpte|assunto|[aä]mne|aihe)\s?:[^*\n]{3,200}\*\*")
add("markdown_letter", "mdle_2", "any", r"(?:^|\n)\s*#{1,4} [^\n]{3,120}\n[\s\S]{0,400}?(?:^|\n)\s*\**" + SALUT)
add("markdown_letter", "mdle_3", "any", r"(?:^|\n)\s*" + SALUT + r"[^\n]{0,80}\n[\s\S]{0,1500}?(?:^|\n)\s*(?:#{2,4} \S|\*\*[^*\n]{3,80}:?\*\*:?\s*\n|\d\. \*\*[^*\n]{3,80}\*\*)")
add("markdown_letter", "mdle_4", "any", r"(?:^|\n)\s*(?:subject|asunto|objet|betreff|oggetto|assumpte|assunto|[aä]mne|aihe)\s?:[^\n]{5,200}\n+\s*" + SALUT)

# ------------------------------------------------------------------ unrendered markdown of a chat reply, without letter structure
# Counted only on the platforms that do not render markdown (EU portal, Regulations.gov, Decide Madrid): on Decidim markdown is native
# (FAMILY_EXCLUDED_GROUPS below). Not an explicit residue of a dialogue: retained or not according to the pre-ChatGPT rate and the reading.
BOLD = r"\*\*[^*\n]{2,80}\*\*"
add("markdown_chat", "mdch_item", "any", r"(?:^|\s)(?:[-•*]|\d{1,2}[.)])\s+\*\*[^*\n]{2,80}\*\*\s?[:.–-]|(?:^|\s)(?:[-•*]|\d{1,2}[.)])\s+\*\*[^*\n]{2,80}[:.]\*\*")
add("markdown_chat", "mdch_head", "any", r"(?:^|\s)#{2,4} \*{0,2}[A-ZÀ-Ý0-9]\w")
add("markdown_chat", "mdch_bold3", "any", BOLD + r"[\s\S]*?" + BOLD + r"[\s\S]*?" + BOLD)
add("markdown_chat", "mdch_bold2", "any", BOLD + r"[\s\S]*?" + BOLD)
add("markdown_chat", "mdch_bold1", "any", BOLD)

# ------------------------------------------------------------------ meta notes of a reply (length of the produced text)
add("meta_note", "meta_1", "any", r"[\(\[]\s?(?:word count|character count|nombre de mots|nombre de caract[eè]res|n[uú]mero de palabras|recuento de palabras|"
    r"n[uú]mero de caracteres|wortanzahl|zeichenanzahl|zeichen|conteggio parole|numero di parole|numero di caratteri|antal ord|sanam[aä][aä]r[aä]|"
    r"nombre de paraules|n[uú]mero de palavras)\s?:\s?(?:approximately |approx\.? |environ |ca\.? |circa |~)?\d+")
add("meta_note", "meta_2", "tail", r"\((?:approximately |approx\.? |environ |ca\.? |circa |~)?\d{2,4} (?:words|characters|mots|caract[eè]res|palabras|caracteres|"
    r"w[oö]rter|zeichen|parole|caratteri|ord|tecken|sanaa|merkki[aä]|paraules|car[aà]cters|palavras)\)\s*$")

# ------------------------------------------------------------------ traces of the chat interface or of its citation system
add("tool_trace", "trace_1", "any", r"contentreference\[oaicite|\boaicite\b|\bturn\d+(?:search|view|news|file)\d+|【\d+[†:][^】]{0,60}】")
add("tool_trace", "trace_2", "any", r"utm_source=(?:chatgpt\.com|openai|copilot\.com|perplexity)")
add("tool_trace", "trace_3", "any", r"(?:^|\n)\s*chat\s?gpt (?:said|a dit|dijo|ha detto|sagte|disse|va dir|sa)\s?:")
add("tool_trace", "trace_4", "any", r"(?:^|\n)\s*(?:you said|vous avez dit|t[uú] dijiste|dijiste|hai detto|du hast gesagt|has dit|voc[eê] disse)\s?:\s*\n")
add("tool_trace", "trace_5", "any", r"\bregenerate response\b|(?:^|\n)\s*copy code\s*\n")

# ------------------------------------------------------------------ disclosed use (separate category)
# Three strict forms per language: (1) participle of writing + a NAMED tool; (2) first person + verb of use/asking + tool + purpose of
# writing; (3) the author's own text ("this comment", "ce texte") + participle of writing + AI in general. "Generated by AI" alone is
# not matched: it is the ordinary wording of contributions ABOUT AI.
AI_EN = r"(?:" + TOOL + r"|an? ai(?: tool| assistant| language model| chatbot| model)?|ai|artificial intelligence|a (?:large )?language model|an llm)"
VERB_EN = r"(?:written|drafted|generated|composed|created|produced|prepared|polished|edited|refined|translated|summari[sz]ed|formulated|rewritten|proofread|improved)"
OWN_EN = (r"(?:this|the above|the following|my|our|above|present) (?:comment|text|response|feedback|letter|submission|message|contribution|reply|answer|post|proposal|"
          r"summary|translation|statement|document|opinion|input|note)s?")
add("disclosed_use", "disc_en_1", "any", r"\b" + OWN_EN + r"\b[^.\n]{0,40}\b(?:was|is|has been|were|have been|are)\b[^.\n]{0,30}\b" + VERB_EN +
    r" (?:by|with|using|via|through|with the (?:help|assistance|aid|support|use) of|with (?:help|assistance|support) from) " + AI_EN + r"\b")
add("disclosed_use", "disc_en_2", "any", r"\b(?:i|we)(?: have|'ve)? (?:used|asked|had|prompted|let|consulted|employed) " + AI_EN + r"\b[^.\n]{0,40}\b(?:to (?:help )?(?:me |us )?(?:write|"
    r"draft|summari[sz]e|translate|generate|polish|formulate|rewrite|rephrase|compose|create|improve|edit|prepare|word|structure)|for (?:writing|drafting|translating|preparing) (?:this|the|my|our))\b")
add("disclosed_use", "disc_en_5", "any", r"\b(?:i|we)\b[^.\n]{0,60}\b(?:translated|wrote|written|drafted|prepared|formulated|summari[sz]ed)\b[^.\n]{0,40}\b(?:with|using) "
    r"(?:the (?:help|assistance|aid|support) of )?" + TOOL + r"\b")
add("disclosed_use", "disc_en_3", "any", r"(?:^|\n|\(|\[|\*|note: |disclaimer: |disclosure: |ps:? |p\.s\.:? )\s*(?:text |comment |this was |this is |response )?" + VERB_EN +
    r" (?:by|with|using|with the (?:help|assistance|aid) of|with (?:help|assistance) from) " + AI_EN + r"\b")
add("disclosed_use", "disc_en_4", "any", r"\b(?:with|thanks to|using) the (?:help|assistance|aid|support) of " + TOOL + r"\b")

AI_ES = r"(?:" + TOOL + r"|(?:la |una )?(?:ia|inteligencia artificial|herramienta de ia))"
PART_ES = r"(?:redactad|escrit|generad|elaborad|cread|traducid|resumid|preparad|hech|mejorad|corregid|revisad)[oa]s?"
add("disclosed_use", "disc_es_1", "any", r"\b" + PART_ES + r" (?:con|por|mediante|con (?:la )?ayuda de|usando|utilizando|a trav[eé]s de) " + TOOL + r"\b|\bcon (?:la )?ayuda de " + TOOL + r"\b")
add("disclosed_use", "disc_es_2", "any", r"\b(?:he|hemos) (?:usado|utilizado|pedido a|preguntado a|consultado(?: a| con)?|recurrido a) " + AI_ES + r"\b[^.\n]{0,50}\b(?:para|a) (?:que |ayudarme a |"
    r"ayudarnos a )?(?:redact|escrib|gener|traduc|resum|elabor|prepar|mejor|correg|me |nos )|\ble (?:he|hemos) (?:pedido|preguntado) a " + AI_ES + r"\b")
add("disclosed_use", "disc_es_3", "any", r"\b(?:este|esta|el presente|la presente|mi|nuestro|nuestra) (?:texto|comentario|escrito|mensaje|documento|respuesta|aportaci[oó]n|propuesta|"
    r"contribuci[oó]n|opini[oó]n|carta)\b[^.\n]{0,60}\b" + PART_ES + r" (?:con|por|mediante|con (?:la )?ayuda de|usando|utilizando) " + AI_ES + r"\b")

AI_FR = r"(?:" + TOOL + r"|l'ia|une ia|l'intelligence artificielle|une intelligence artificielle|un outil d'ia)"
PART_FR = r"(?:r[eé]dig[eé]|[eé]crit|g[eé]n[eé]r[eé]|[eé]labor[eé]|traduit|r[eé]sum[eé]|reformul[eé]|produit|cr[eé][eé]|corrig[eé]|am[eé]lior[eé])e?s?"
PREP_FR = r"(?:par|avec|[aà] l'aide d[e']|gr[aâ]ce [aà]|via|avec l'aide d[e']|avec l'assistance d[e']) ?"
add("disclosed_use", "disc_fr_1", "any", r"\b" + PART_FR + r" " + PREP_FR + TOOL + r"\b|\b(?:avec|gr[aâ]ce [aà]) l'aide de " + TOOL + r"\b")
add("disclosed_use", "disc_fr_2", "any", r"\b(?:j'ai|nous avons) (?:demand[eé] [aà]|utilis[eé]|interrog[eé]|sollicit[eé]|pos[eé] la question [aà]|eu recours [aà]|consult[eé]) " + AI_FR +
    r"\b[^.\n]{0,50}\b(?:pour|de|d') ?(?:m'aider [aà] |nous aider [aà] )?(?:r[eé]dig|[eé]cri|g[eé]n[eé]r|tradui|r[eé]sum|reformul|formul|corrig|am[eé]lior|me |nous )")
add("disclosed_use", "disc_fr_3", "any", r"\b(?:ce|cette|cet|le pr[eé]sent|la pr[eé]sente|mon|ma|notre) (?:texte|commentaire|message|contribution|r[eé]ponse|avis|proposition|"
    r"courrier|lettre|document)\b[^.\n]{0,60}\b" + PART_FR + r" " + PREP_FR + AI_FR + r"\b")

AI_DE = r"(?:" + TOOL + r"|(?:einer |der |eine |die )?ki|k[uü]nstlicher intelligenz|(?:einem |eines )?ki-tools?)"
PART_DE = r"(?:geschrieben|verfasst|erstellt|generiert|formuliert|[uü]bersetzt|zusammengefasst|erzeugt|[uü]berarbeitet|korrigiert)"
add("disclosed_use", "disc_de_1", "any", r"\b(?:mit|von|durch|mithilfe von|mit hilfe von|mit unterst[uü]tzung von|mittels) " + TOOL + r"\b[^.\n]{0,30}\b" + PART_DE + r"\b")
add("disclosed_use", "disc_de_2", "any", r"\b(?:ich habe|wir haben|habe ich|haben wir) (?:" + TOOL + r"|die ki|eine ki)\b[^.\n]{0,40}\b(?:gefragt|gebeten|benutzt|genutzt|verwendet|befragt)\b")
add("disclosed_use", "disc_de_3", "any", r"\b(?:dieser|diese|dieses|mein|meine|unser|unsere|vorliegende[rs]?) (?:text|kommentar|beitrag|stellungnahme|antwort|nachricht|schreiben|"
    r"r[uü]ckmeldung|feedback)\b[^.\n]{0,60}\b(?:mit|von|durch|mithilfe von|mit hilfe von|mit unterst[uü]tzung von|mittels) " + AI_DE + r"\b[^.\n]{0,30}\b" + PART_DE + r"\b")

AI_IT = r"(?:" + TOOL + r"|l'ia|l'ai|l'intelligenza artificiale|un'intelligenza artificiale|un'ia)"
PART_IT = r"(?:scritt|redatt|generat|elaborat|tradott|riassunt|creat|prodott|rivist|corrett)[oaie]"
PREP_IT = r"(?:con|da|tramite|mediante|con l'aiuto di|con l'aiuto del|con l'ausilio di|con il supporto di|grazie a|utilizzando|usando|attraverso|dall'|dall'aiuto di) ?"
add("disclosed_use", "disc_it_1", "any", r"\b" + PART_IT + r" " + PREP_IT + TOOL + r"\b|\bcon l'aiuto di " + TOOL + r"\b")
add("disclosed_use", "disc_it_2", "any", r"\b(?:ho|abbiamo) (?:chiesto a|usato|utilizzato|interrogato|domandato a|consultato|fatto ricorso a) " + AI_IT + r"\b[^.\n]{0,50}\b(?:per|di) "
    r"(?:aiutarmi a |aiutarci a )?(?:scriver|redig|generar|tradur|riassum|elaborar|formular|corregger|miglior)")
add("disclosed_use", "disc_it_3", "any", r"\b(?:questo|questa|il presente|la presente|il mio|la mia|il nostro|la nostra) (?:testo|commento|messaggio|contributo|risposta|proposta|"
    r"lettera|documento|osservazione)\b[^.\n]{0,60}\b" + PART_IT + r" " + PREP_IT + AI_IT + r"\b")

AI_CA = r"(?:(?:el |la |l')?(?:" + TOOL + r"|ia|intel·lig[eè]ncia artificial))"
PART_CA = r"(?:redactat|escrit|generat|elaborat|tradu[iï]t|resumit|creat|fet|millorat|corregit|revisat)(?:da|s|des)?"
PREP_CA = r"(?:amb|per|mitjan[cç]ant|amb (?:l')?ajuda de|amb l'ajut de|utilitzant|fent servir) "
add("disclosed_use", "disc_ca_1", "any", r"\b" + PART_CA + r" " + PREP_CA + r"(?:el |l')?" + TOOL + r"\b|\bamb l'aju[dt]a? de(?:l)? " + TOOL + r"\b")
add("disclosed_use", "disc_ca_2", "any", r"\b(?:he|hem) (?:demanat a|preguntat a|utilitzat|fet servir|usat|consultat) (?:al |a la |a l')?" + AI_CA + r"\b[^.\n]{0,50}\b(?:per|que) "
    r"(?:a )?(?:ajudar-me a |ajudar-nos a )?(?:redact|escriu|gener|tradu|resum|elabor|millor|corregi|em |ens )")
add("disclosed_use", "disc_ca_3", "any", r"\b(?:aquest|aquesta|el present|la present|el meu|la meva|el nostre|la nostra) (?:text|comentari|missatge|document|resposta|aportaci[oó]|"
    r"proposta|contribuci[oó]|opini[oó]|carta)\b[^.\n]{0,60}\b" + PART_CA + r" " + PREP_CA + AI_CA + r"\b")

AI_PT = r"(?:(?:o |a )?(?:" + TOOL + r"|ia|intelig[eê]ncia artificial))"
PART_PT = r"(?:escrit|redigid|gerad|elaborad|traduzid|resumid|criad|feit|melhorad|corrigid|revist|revisad)[oa]s?"
PREP_PT = r"(?:com|por|pelo|pela|atrav[eé]s d[oea]|com (?:a )?ajuda d[oea]|usando|utilizando) "
add("disclosed_use", "disc_pt_1", "any", r"\b" + PART_PT + r" " + PREP_PT + r"(?:o )?" + TOOL + r"\b|\bcom (?:a )?ajuda do " + TOOL + r"\b")
add("disclosed_use", "disc_pt_2", "any", r"\b(?:usei|utilizei|pedi ao|perguntei ao|pedi para o|consultei|recorri ao) (?:o )?" + TOOL + r"\b")
add("disclosed_use", "disc_pt_3", "any", r"\b(?:este|esta|o presente|a presente|o meu|a minha|meu|minha|o nosso|a nossa) (?:texto|coment[aá]rio|mensagem|documento|resposta|"
    r"contribui[cç][aã]o|proposta|opini[aã]o|carta)\b[^.\n]{0,60}\b" + PART_PT + r" " + PREP_PT + AI_PT + r"\b")

PART_SV = r"(?:skriven|skrivet|skrivna|genererad|genererat|framtagen|framtaget|[oö]versatt|sammanfattad|sammanfattat|formulerad|formulerat)"
add("disclosed_use", "disc_sv_1", "any", r"\b" + PART_SV + r" (?:av|med|med hj[aä]lp av|genom) " + TOOL + r"\b|\bmed hj[aä]lp av " + TOOL + r"\b")
add("disclosed_use", "disc_sv_2", "any", r"\b(?:jag|vi) (?:har )?(?:bad|bett|fr[aå]gade|fr[aå]gat|anv[aä]nde|anv[aä]nt) " + TOOL + r"\b")
add("disclosed_use", "disc_sv_3", "any", r"\b(?:den h[aä]r|denna|detta|det h[aä]r|min|mitt|v[aå]r|v[aå]rt) (?:texten?|kommentar(?:en)?|svar(?:et)?|inl[aä]gg(?:et)?|yttrande(?:t)?)\b"
    r"[^.\n]{0,60}\b" + PART_SV + r" (?:av|med|med hj[aä]lp av|genom) (?:" + TOOL + r"|ai|en ai)\b")
add("disclosed_use", "disc_fi_1", "any", r"\b(?:chat\s?gpt|copilot|gemini)\w* (?:kirjoittama|tuottama|laatima|luoma|generoima|k[aä][aä]nt[aä]m[aä]|muotoilema)|"
    r"\b(?:chat\s?gpt|copilot|gemini)\w* avulla (?:kirjoitettu|tehty|laadittu|tuotettu|muotoiltu|k[aä][aä]nnetty)")
add("disclosed_use", "disc_fi_2", "any", r"\b(?:kysyin|pyysin|k[aä]ytin|kysyimme|pyysimme|k[aä]ytimme) (?:chat\s?gpt|copilot|gemini|teko[aä]ly)")
add("disclosed_use", "disc_fi_3", "any", r"\bt[aä]m[aä] (?:teksti|kommentti|palaute|vastaus|ehdotus|viesti) on [^.\n]{0,40}teko[aä]ly\w* (?:avulla |kanssa )?(?:kirjoit|tuot|laadit|luo|teh|muotoil)")

# ------------------------------------------------------------------ exclusions decided after the specificity check and the reading
# Whole families dropped:
#   opener_list, label_head   form prompts and ordinary human openings ("Voici quelques ...", "Title: ..."); matched before ChatGPT
#   closer_customize          every match read was a human formula ("I hope this helps", "feel free to modify" in campaign templates)
#   placeholder_bare/_insert  as frequent before ChatGPT as after: slots of campaign form letters, of legal drafting and of bug-report forms
#   meta_note                 no match after ChatGPT
DROPPED_FAMILIES = {"opener_list", "label_head", "closer_customize", "placeholder_bare", "placeholder_insert", "meta_note"}
# Single patterns dropped:
#   phyo_2                    any "[your ...]" slot: the form of one platform prints "[your answer here]"
#   mdle_4                    "Subject: ... Dear ..." without markdown: an ordinary letter (as frequent before ChatGPT)
#   mdch_bold2, mdch_bold1    one or two bold spans: emphasis typed by hand, matched before ChatGPT
#   disc_xx_1, disc_en_3/4    participle + tool without the author's own text as subject: texts ABOUT AI ("a poster made with ChatGPT")
DROPPED_PATTERNS = {"phyo_2", "mdle_4", "mdch_bold2", "mdch_bold1", "disc_en_3", "disc_en_4", "disc_es_1", "disc_fr_1", "disc_de_1", "disc_it_1",
                    "disc_ca_1", "disc_pt_1", "disc_sv_1", "disc_fi_1"}
# families counted only outside some source groups (see the note on markdown_chat)
FAMILY_EXCLUDED_GROUPS = {"markdown_chat": {"decidim_instances"}}
# Tiers of the final set, reported apart: "dialogue" = explicit residue of the exchange with the assistant; "format" = unrendered chat
# markdown; "disclosed" = the author states the use of an assistant.
FAMILY_TIER = {"markdown_chat": "format", "disclosed_use": "disclosed"}      # every other family: "dialogue"
DISCLOSED_FAMILIES = {"disclosed_use"}

FAMILY_ORDER = []
for _f, _pid, _s, _rx in P:
    if _f not in FAMILY_ORDER:
        FAMILY_ORDER.append(_f)
