INSTRUCTION = "Complete the sentence with only the missing entity. Do not explain.\n"


def chat_prompt(tok, cloze):
    return tok.apply_chat_template(
        [{"role": "user", "content": INSTRUCTION + cloze}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
