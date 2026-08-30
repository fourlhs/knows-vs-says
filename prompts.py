def chat_prompt(tok, user_text):
    return tok.apply_chat_template(
        [{"role": "user", "content": user_text}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
