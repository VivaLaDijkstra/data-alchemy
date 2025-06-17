LANG_CODE_TO_NAME = {
    "en": "English",
    "zh": "Chinese",
    "fr": "French",
    "de": "German",
    "jp": "Japanese",
    "kr": "Korean",
    "ru": "Russian",
}


def detect_language(text: str) -> str:
    """Detect text language by checking unicode range of characters.
    For Chinese (zh), Japanese (jp), Korean (kr), German (de), French (fr), and Russian (ru),
    if there exists one character in the text, return the corresponding language.
    For English (en), if all characters are English, return 'en'.
    """

    def is_chinese(char):
        return "\u4e00" <= char <= "\u9fff"

    def is_japanese(char):
        # Japanese characters include Hiragana, Katakana, and Kanji
        return (
            "\u3040" <= char <= "\u309f"  # Hiragana
            or "\u30a0" <= char <= "\u30ff"  # Katakana
            or "\u4e00" <= char <= "\u9fff"
        )  # Kanji (also used in Chinese)

    def is_korean(char):
        # Korean characters include Hangul syllables
        return "\uac00" <= char <= "\ud7af"

    def is_russian(char):
        # Russian characters are within the Cyrillic block
        return "\u0400" <= char <= "\u04ff"

    def is_german(char):
        german_chars = "äöüßÄÖÜ"
        return char in german_chars

    def is_french(char):
        french_chars = "éèêëàâîïôûùçÉÈÊËÀÂÎÏÔÛÙÇ"
        return char in french_chars

    def is_english(char):
        return "a" <= char.lower() <= "z"

    found_languages = set()

    for char in text:
        if is_chinese(char):
            return "zh"
        if is_japanese(char):
            return "jp"
        if is_korean(char):
            return "kr"
        if is_russian(char):
            return "ru"
        if is_german(char):
            found_languages.add("de")
        if is_french(char):
            found_languages.add("fr")
        if not is_english(char) and char.isalpha():
            found_languages.discard("en")

    if "de" in found_languages:
        return "de"
    if "fr" in found_languages:
        return "fr"
    if len(text) > 0 and all(is_english(char) for char in text):
        return "en"

    return "unknown"
