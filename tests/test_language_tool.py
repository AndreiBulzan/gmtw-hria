import json
import re
import language_tool_python

json_input = """
{"instance_id": "travel_000000", "output": "Înainte de a crea un plan pentru excursia ta în Timișoara, am considerat câteva aspecte importante. Am ales să includem cel puțin un monument istoric în plan, așa cum este cerut. De asemenea, am dorit să nu depășim două activități în aer liber pe zi, pentru a nu obosea copiii.\\n\\nPentru prima zi, am ales să vizităm Parcul Rozelor, un parc potrivit pentru copii, unde pot să se joace și să se relaxeze. Apoi, am ales să vizităm Bastionul Theresia, un monument istoric care oferă o priveliște impresionantă asupra orașului. A doua zi, am ales să vizităm Muzeul de Artă, care este potrivit pentru copii și oferă o experiență culturală interesantă. A treia zi, am ales să vizităm Catedrala Mitropolitană, un monument istoric care oferă o priveliște impresionantă asupra orașului.\\n\\nPlanul nostru de excursie este unul variat și interesant, care include activități în aer liber și vizite la monumente istorice. Sperăm că veți avea o experiență plăcută în Timișoara!\\n\\n{\\n  \\"day1\\": [\\"Parcul Rozelor\\", \\"Bastionul Theresia\\"],\\n  \\"day2\\": [\\"Muzeul de Artă\\"],\\n  \\"day3\\": [\\"Catedrala Mitropolitană\\"]\\n}", "model": "llama-3.1-8b-instant", "language": "ro", "latency": 1.0179879665374756}
"""

data = json.loads(json_input)
full_text = data['output']
lang_code = data['language']

clean_text = re.split(r'\n\s*\{', full_text)[0].strip()

print(f"--- Analyzing Text ({lang_code}) ---")

with language_tool_python.LanguageTool(lang_code) as tool:
    matches = tool.check(clean_text)

    if not matches:
        print("No errors found!")
    else:
        print(f"Found {len(matches)} potential issues:\n")
        
        for i, match in enumerate(matches, 1):
            print(f"{i}. Rule ID: {match.rule_id}") 
            print(f"   Message: {match.message}")
            print(f"   Context: \"...{match.context}...\"")
            print(f"   Suggestion: {match.replacements}")
            print("-" * 30)

    word_count = len(clean_text.split())
    if word_count > 0:
        error_density = (len(matches) / word_count) * 100
        print(f"\nError Density: {error_density:.2f}% (Lower is better)")