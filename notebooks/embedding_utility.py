'''
Utility to create embeddings using openAI embeddings models
'''


import os
os.environ['OPENAI_API_KEY'] = "API KEY HERE"
from openai import OpenAI
client = OpenAI()
from numpy.linalg import norm
import numpy as np


def cosine_similarity(a,b):
    return np.dot(a,b)/(norm(a)*norm(b))


def create_embeddings(sentence) -> list:
    response = client.embeddings.create(
        input=[sentence],
        model="text-embedding-ada-002"
    )
    return response.data[0].embedding


if __name__=="__main__":
    emb=create_embeddings("my text goes here")
    pass