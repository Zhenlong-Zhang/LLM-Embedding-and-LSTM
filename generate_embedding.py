import openai
import pandas as pd

def get_embeddings(think_aloud, text_model='text-embedding-ada-002'):
    question_prompt = (
        "This is a transcription of a participant's think-aloud session during a decision-making task. "
        "just try to find out the strategy they use"
    )
    instruction = question_prompt + '\n' + think_aloud + ' The participant is using strategy.'

    try:
        response = openai.Embedding.create(
            model=text_model,
            input=instruction
        )
        return response['data'][0]['embedding']
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None
    
def generate_all_embeddings_from_df(text_df, api_key):
    openai.api_key = api_key

    embeddings = []

    for idx, row in text_df.iterrows():
        text = row.get('text', '').strip()
        if not text:
            embeddings.append(None)
            continue

        emb = get_embeddings(text)
        embeddings.append(emb)

   
    text_df = text_df.copy()
    text_df['embedding'] = embeddings
    text_df = text_df.drop(columns=['text'])

   
    text_df = text_df[text_df['embedding'].notnull()]

    return text_df

