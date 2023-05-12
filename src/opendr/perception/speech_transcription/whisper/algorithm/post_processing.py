# Copyright 2020-2023 OpenDR European Project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import re
import string
from typing import Optional, List

import Levenshtein
from abydos.phonetic import RefinedSoundex


def process_string(s: str) -> List[str]:
    # lowwercase.
    s = s.lower()

    # Remove leading and trailing whitespaces
    s = s.strip()
    
    # Replace multiple spaces with a single space
    s = re.sub(r'\s+', ' ', s)
    
    # Remove punctuations
    s = s.translate(str.maketrans('', '', string.punctuation))
    
    # Split the cleaned string into words
    words = s.split()
    
    return words


def closest_word(word: str, keywords_list: Optional[List[str]], keywords_normalized = None, normalizer = None) -> str:
    if keywords_list is None or len(keywords_list) == 0:
        return word

    if normalizer:
        if normalizer(word) in keywords_normalized:
            # print("found normalized word")
            return word

    rs = RefinedSoundex(retain_vowels=True)

    min_distance = float("inf")
    closest = None

    # print(f"word: {word}")
    for keyword in keywords_list:
        distance = Levenshtein.distance(rs.encode(word), rs.encode(keyword))
        # print(f"keyword: {keyword}, distance: {distance}")
        if distance < min_distance:
            min_distance = distance
            closest = keyword

    print(f"closest: {closest}")
    return closest


if __name__ == "__main__":
    # Example usage
    input_string = " Hello,  World! This  is an example: string, with  punctuation. "
    cleaned_string = process_string(input_string)
    keyword_list = ["hello", "strong"]

    results = [closest_word(word, keyword_list) for word in cleaned_string]
    print(f"Original string: {input_string}")
    print(f"Cleaned string: {' '.join(cleaned_string)}")
    print(f"Closest words: {' '.join(results)}")
