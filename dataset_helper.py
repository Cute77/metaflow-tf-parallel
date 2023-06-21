import tensorflow as tf
from transformers import T5Tokenizer
import datasets

TOKENIZER = T5Tokenizer.from_pretrained('t5-small')

def encode(example, encode_max_len=250, decode_max_len = 54):
    context = example['context']
    question = example['question']
    answer = example['answers']['text']

    encoder_input = f"answer_me: {str(question)}" + f" context: {str(context)} </s>"
    decoder_input = ','.join([i for i in list(answer)]) + "</s>"

    encoder_inputs = TOKENIZER(encoder_input, truncation=True, return_tensors='tf', max_length=encode_max_len, pad_to_max_length=True)
    decoder_inputs = TOKENIZER(decoder_input, truncation=True, return_tensors='tf', max_length=decode_max_len, pad_to_max_length=True)

    input_ids = encoder_inputs['input_ids'][0]
    input_attention = encoder_inputs['attention_mask'][0]
    target_ids = decoder_inputs['input_ids'][0]
    target_attention = decoder_inputs['attention_mask'][0]

    outputs = {
        'input_ids': input_ids, 
        'attention_mask': input_attention,
        'labels': target_ids,
        'decoder_attention_mask': target_attention,
    }

    return outputs

def to_tf_dataset(dataset):
    columns = ['input_ids', 'attention_mask', 'labels', 'decoder_attention_mask']
    # ds = dataset.to_tf_dataset(columns=columns)
    # ex = next(iter(ds))
    # print("ds: ", ds)
    # print("Example data from the dataset: \n")
    # for e in ex:
    #     print(e, ex[e])
    #     print(type(ex[e]))
    dataset.set_format(type='tf', columns=columns)
    return_types = {'input_ids': tf.int32, 'attention_mask': tf.int32, 
                'labels': tf.int32, 'decoder_attention_mask': tf.int32,}
    return_shapes = {'input_ids': tf.TensorShape([None]), 'attention_mask': tf.TensorShape([None]), 
                    'labels': tf.TensorShape([None]), 'decoder_attention_mask': tf.TensorShape([None])}
    ds = tf.data.Dataset.from_generator(lambda : dataset, return_types, return_shapes)

    ex = next(iter(ds))
    print("ds: ", ds)
    print("Example data from the dataset: \n")
    for e in ex:
        print(e, ex[e])
        print(type(ex[e]))

    return ds
  

def create_dataset(dataset, cache_path=None, batch_size=4, 
                   buffer_size= 1000, shuffling=True):    
    if cache_path is not None:
        dataset = dataset.cache(cache_path)        
    if shuffling:
        dataset = dataset.shuffle(buffer_size)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset
