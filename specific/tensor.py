from utils.tensor import convert_to_tensor


def make_dataloader(task, *args, **kwargs):
    assert task in ['exp', 'exp_0', 'exp_1', 'exp_2', 'exp_3', 'exp_4', 'exp_5', 'neo4j', 'conceptnet']
    
    if task == 'exp':
        return _make_dataloader(*args, **kwargs)
    elif task == 'exp_0':
        return _make_dataloader_0(*args, **kwargs)
    elif task == 'exp_1':
        return _make_dataloader_1(*args, **kwargs)
    elif task == 'exp_2':
        return _make_dataloader_2(*args, **kwargs)
    elif task == 'exp_3':
        return _make_dataloader_3(*args, **kwargs)
    elif task == 'exp_4':
        return _make_dataloader_4(*args, **kwargs)
    elif task == 'exp_5':
        return _make_dataloader_5(*args, **kwargs)
    elif task == 'neo4j':
        return _make_dataloader_neo4j(*args, **kwargs)
    elif task == 'conceptnet':
        return _make_dataloader_conceptnet(*args, **kwargs)

    
def _make_dataloader(examples, tokenizer, batch_size, drop_last, max_seq_length, shuffle=True):
    F = []
    L = []

    for example in examples:
        # fe, la = example.fl(tokenizer, max_seq_length)
        f1, f2, f3, f4, f5, la = example.fl(tokenizer, max_seq_length)

        F.append((f1, f2, f3, f4, f5))
        L.append(la)

    return convert_to_tensor((F, L), batch_size, drop_last, shuffle=shuffle)

def _make_dataloader_0(examples, tokenizer, batch_size, drop_last, max_seq_length, shuffle=True):
    F = []
    L = []

    for example in examples:
        # fe, la = example.fl(tokenizer, max_seq_length)
        f1, f2, f3, f4, f5, la = example.fl(tokenizer, max_seq_length)

        F.append((f1, f2, f3, f4, f5))
        L.append(la)

    return convert_to_tensor((F, L), batch_size, drop_last, shuffle=shuffle)
    
def _make_dataloader_1(examples, tokenizer, batch_size, drop_last, max_seq_length, shuffle=True):
    F = []
    L = []

    for example in examples:
        # fe, la = example.fl(tokenizer, max_seq_length)
        f1, f2, f3, f4, f5, la = example.fl(tokenizer, max_seq_length)

        F.append((f1, f2, f3, f4, f5))
        L.append(la)

    return convert_to_tensor((F, L), batch_size, drop_last, shuffle=shuffle)
    
def _make_dataloader_2(examples, tokenizer, batch_size, drop_last, max_seq_length, shuffle=True):
    F = []
    L = []

    for example in examples:
        # fe, la = example.fl(tokenizer, max_seq_length)
        f1, f2, f3, f4, f5, la = example.fl(tokenizer, max_seq_length)

        F.append((f1, f2, f3, f4, f5))
        L.append(la)

    return convert_to_tensor((F, L), batch_size, drop_last, shuffle=shuffle)
 
def _make_dataloader_3(examples, tokenizer, batch_size, drop_last, max_seq_length, shuffle=True):
    F = []
    L = []

    for example in examples:
        # fe, la = example.fl(tokenizer, max_seq_length)
        f1, f2, f3, f4, f5, la = example.fl(tokenizer, max_seq_length)

        F.append((f1, f2, f3, f4, f5))
        L.append(la)

    return convert_to_tensor((F, L), batch_size, drop_last, shuffle=shuffle)

def _make_dataloader_4(examples, tokenizer, batch_size, drop_last, max_seq_length, shuffle=True):
    F = []
    L = []

    for example in examples:
        # fe, la = example.fl(tokenizer, max_seq_length)
        f1, f2, f3, f4, f5, la = example.fl(tokenizer, max_seq_length)

        F.append((f1, f2, f3, f4, f5))
        L.append(la)

    return convert_to_tensor((F, L), batch_size, drop_last, shuffle=shuffle)
 
def _make_dataloader_5(examples, tokenizer, batch_size, drop_last, max_seq_length, shuffle=True):
    F = []
    L = []

    for example in examples:
        # fe, la = example.fl(tokenizer, max_seq_length)
        f1, f2, f3, f4, f5, la = example.fl(tokenizer, max_seq_length)

        F.append((f1, f2, f3, f4, f5))
        L.append(la)

    return convert_to_tensor((F, L), batch_size, drop_last, shuffle=shuffle)

def _make_dataloader_neo4j(examples, tokenizer, batch_size, drop_last, max_seq_length, shuffle=True):
    F = []
    L = []

    for example in examples:
        # fe, la = example.fl(tokenizer, max_seq_length)
        f1, f2, f3, f4, f5, la = example.fl(tokenizer, max_seq_length)

        F.append((f1, f2, f3, f4, f5))
        L.append(la)

    return convert_to_tensor((F, L), batch_size, drop_last, shuffle=shuffle)

def _make_dataloader_conceptnet(examples, tokenizer, batch_size, drop_last, max_seq_length, shuffle=True):
    F = []
    L = []

    for example in examples:
        # fe, la = example.fl(tokenizer, max_seq_length)
        f1, f2, f3, f4, f5, la = example.fl(tokenizer, max_seq_length)

        F.append((f1, f2, f3, f4, f5))
        L.append(la)

    return convert_to_tensor((F, L), batch_size, drop_last, shuffle=shuffle)
 