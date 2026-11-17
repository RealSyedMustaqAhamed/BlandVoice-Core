"""
Conversational Text & Prosody Markup Parser for Phone Agents.
Extracts phoneme tokens, prosodic inflections, conversational fillers, and emotional cues.
Handles telephony-specific phenomena: hesitation markers, breathing artifacts, and question inflections.
"""

import re
from typing import List, Dict, Any, Tuple


# Telephony phonetic alphabet + conversational token symbols
VOCAB = [
    "<pad>", "<unk>", "<bos>", "<eos>", "<sil>",
    # English Phonetic approximation symbols (ARPAbet subset)
    "AA", "AE", "AH", "AO", "AW", "AY", "B", "CH", "D", "DH", "EH", "ER", "EY",
    "F", "G", "HH", "IH", "IY", "JH", "K", "L", "M", "N", "NG", "OW", "OY",
    "P", "R", "S", "SH", "T", "TH", "UH", "UW", "V", "W", "Y", "Z", "ZH",
    # Punctuation & Boundary markers
    ",", ".", "?", "!", "-",
    # Conversational speech artifacts
    "<filler_um>", "<filler_uh>", "<filler_uh_huh>", "<filler_mhm>",
    "<breath_in>", "<breath_out>", "<sigh>", "<cough>"
]

TOKEN_TO_ID = {tok: idx for idx, tok in enumerate(VOCAB)}
ID_TO_TOKEN = {idx: tok for idx, tok in enumerate(VOCAB)}


class ConversationalPhonemizer:
    """
    Parses conversational text enriched with inline acoustic tags.
    Example:
    "[emotion:empathetic] [breath_in] Hi John, [pause:250ms] I see your payment is pending. [pitch:rising] Would you like me to process that now?"
    """
    def __init__(self):
        self.vocab = VOCAB
        self.token_to_id = TOKEN_TO_ID
        
        # Simple letter-to-sound dictionary for conversational vocabulary
        self.basic_dict = {
            "hi": ["HH", "AY"],
            "hello": ["HH", "AH", "L", "OW"],
            "yes": ["Y", "EH", "S"],
            "no": ["N", "OW"],
            "okay": ["OW", "K", "EY"],
            "john": ["JH", "AA", "N"],
            "bland": ["B", "L", "AE", "N", "D"],
            "voice": ["V", "OY", "S"],
            "agent": ["EY", "JH", "AH", "N", "T"],
            "call": ["K", "AO", "L"],
            "phone": ["F", "OW", "N"],
            "is": ["IH", "Z"],
            "this": ["DH", "IH", "S"],
            "the": ["DH", "AH"],
            "best": ["B", "EH", "S", "T"],
            "time": ["T", "AY", "M"],
            "to": ["T", "UW"],
            "speak": ["S", "P", "IY", "K"],
            "i": ["AY"],
            "am": ["AE", "M"],
            "calling": ["K", "AO", "L", "IH", "NG"],
            "from": ["F", "R", "AH", "M"],
            "support": ["S", "AH", "P", "AO", "R", "T"],
            "can": ["K", "AE", "N"],
            "you": ["Y", "UW"],
            "hear": ["HH", "IH", "R"],
            "me": ["M", "IY"],
            "how": ["HH", "AW"],
            "are": ["AA", "R"],
            "today": ["T", "AH", "D", "EY"],
            "would": ["W", "UH", "D"],
            "like": ["L", "AY", "K"],
            "help": ["HH", "EH", "L", "P"],
            "with": ["W", "IH", "DH"],
            "that": ["DH", "AE", "T"]
        }

    def _word_to_phonemes(self, word: str) -> List[str]:
        w = word.lower().strip(",.?!:;\"'")
        if w in self.basic_dict:
            return self.basic_dict[w]
            
        # Fallback pseudo-grapheme-to-phoneme mapper
        phonemes = []
        mapping = {
            'a': 'AE', 'b': 'B', 'c': 'K', 'd': 'D', 'e': 'EH',
            'f': 'F', 'g': 'G', 'h': 'HH', 'i': 'IH', 'j': 'JH',
            'k': 'K', 'l': 'L', 'm': 'M', 'n': 'N', 'o': 'AA',
            'p': 'P', 'q': 'K', 'r': 'R', 's': 'S', 't': 'T',
            'u': 'AH', 'v': 'V', 'w': 'W', 'x': 'K', 'y': 'Y', 'z': 'Z'
        }
        for char in w:
            if char in mapping:
                phonemes.append(mapping[char])
        return phonemes or ["AH"]

    def parse(self, text: str) -> Dict[str, Any]:
        """
        Parses text and extracts:
        - phoneme_ids: List[int]
        - is_question: bool
        - emotion: str (neutral, empathetic, urgent, sales)
        - tags: list of explicit conversational triggers
        """
        emotion = "neutral"
        if "[emotion:empathetic]" in text:
            emotion = "empathetic"
        elif "[emotion:urgent]" in text:
            emotion = "urgent"
        elif "[emotion:sales]" in text:
            emotion = "sales"

        has_question_tag = "[pitch:rising]" in text or text.strip().endswith("?")

        # Clean tags while extracting fillers
        cleaned_text = text
        for emo_tag in ["[emotion:neutral]", "[emotion:empathetic]", "[emotion:urgent]", "[emotion:sales]", "[pitch:rising]", "[pitch:falling]"]:
            cleaned_text = cleaned_text.replace(emo_tag, "")

        tokens: List[str] = ["<bos>"]
        words_and_punct = re.findall(r"\[filler:\w+\]|\[breath_\w+\]|<[^>]+>|[\w']+|[.,?!]", cleaned_text)

        for item in words_and_punct:
            if item == "[filler:um]":
                tokens.append("<filler_um>")
            elif item == "[filler:uh]":
                tokens.append("<filler_uh>")
            elif item == "[filler:uh_huh]":
                tokens.append("<filler_uh_huh>")
            elif item == "[breath_in]":
                tokens.append("<breath_in>")
            elif item in [",", ".", "?", "!"]:
                tokens.append(item)
            else:
                ph_list = self._word_to_phonemes(item)
                tokens.extend(ph_list)

        tokens.append("<eos>")

        token_ids = [self.token_to_id.get(tok, self.token_to_id["<unk>"]) for tok in tokens]

        return {
            "raw_text": text,
            "tokens": tokens,
            "token_ids": token_ids,
            "emotion": emotion,
            "is_question": has_question_tag
        }
