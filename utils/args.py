

global_tokenizer = None


def register_tokenizer(tokenizer):
    global global_tokenizer
    global_tokenizer = tokenizer
    

def get_global_tokenizer():
    return global_tokenizer
