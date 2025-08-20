# First, let's extract and understand the key components from the notebook
# The notebook shows they implemented:
# 1. Language classification using Multinomial Naive Bayes (27 languages)
# 2. Continent classification using LDA/QDA (4 continents)

# Key details from the notebook:
languages = [
    'af-ZA', 'da-DK', 'de-DE', 'en-US', 'es-ES', 'fr-FR', 'fi-FI', 'hu-HU', 'is-IS', 'it-IT',
    'jv-ID', 'lv-LV', 'ms-MY', 'nb-NO', 'nl-NL', 'pl-PL', 'pt-PT', 'ro-RO', 'ru-RU', 'sl-SL',
    'sv-SE', 'sq-AL', 'sw-KE', 'tl-PH', 'tr-TR', 'vi-VN', 'cy-GB'
]

continent_lookup = {
    'ZA': 'Africa', 'KE': 'Africa', 'AL': 'Europe', 'GB': 'Europe', 'DK': 'Europe', 'DE': 'Europe',
    'ES': 'Europe', 'FR': 'Europe', 'FI': 'Europe', 'HU': 'Europe', 'IS': 'Europe', 'IT': 'Europe',
    'ID': 'Asia', 'LV': 'Europe', 'MY': 'Asia', 'NO': 'Europe', 'NL': 'Europe', 'PL': 'Europe',
    'PT': 'Europe', 'RO': 'Europe', 'RU': 'Europe', 'SL': 'Europe', 'SE': 'Europe', 'PH': 'Asia',
    'TR': 'Asia', 'VN': 'Asia', 'US': 'North America'
}

print("Languages to process:", len(languages))
print("Continent mapping:", continent_lookup)

# Performance from their notebook:
print("\nPerformance achieved:")
print("Language Classification (Multinomial Naive Bayes):")
print("- Validation Accuracy: ~98.4%")
print("- Test Accuracy: ~98.3%")
print("\nContinent Classification:")
print("- LDA: ~89.9% validation accuracy")
print("- QDA: ~79.1% test accuracy")
