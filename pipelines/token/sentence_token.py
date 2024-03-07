class SentencePieceTokenProcessor:
    def __init__(self, sp_model):
        self.sp_model = sp_model
        self.post_process_remove_list = {
            self.sp_model.unk_id(),
            self.sp_model.eos_id(),
            self.sp_model.pad_id(),
        }

    def __call__(self, tokens, lstrip: bool = True) -> str:
        filtered_hypo_tokens = [
            token_index
            for token_index in tokens[1:]
            if token_index not in self.post_process_remove_list
        ]
        output_string = "".join(
            self.sp_model.id_to_piece(filtered_hypo_tokens)
        ).replace("\u2581", " ")

        if lstrip:
            return output_string.lstrip()
        else:
            return output_string
