from tokenizers import Tokenizer
import os

tokenizer_path = "models/amb_tokenizer_v1/tokenizer.json"
if os.path.exists(tokenizer_path):
    tokenizer = Tokenizer.from_file(tokenizer_path)
    
    test_words = [
        "வீடுகளிலிருந்து",
        "போகவேண்டியிருந்தது",
        "அரசியலமைப்பு"
    ]
    
    print(f"--- AMB Pipeline Boundary Test ---")
    for word in test_words:
        encoded = tokenizer.encode(word)
        print(f"\nWord: {word}")
        
        # Use offsets to see exact substrings
        tokens_with_text = []
        for i, (start, end) in enumerate(encoded.offsets):
            token_text = word[start:end]
            tokens_with_text.append(token_text)
        
        print(f"AMB Tokens: {' | '.join(tokens_with_text)}")
else:
    print(f"Error: {tokenizer_path} not found")
