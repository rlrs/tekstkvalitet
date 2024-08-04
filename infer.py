from transformers import AutoModel, AutoTokenizer
from model import XLMRobertaForRegression

import sqlite3

model = XLMRobertaForRegression.from_pretrained("./model-final")
tokenizer = AutoTokenizer.from_pretrained("./model-final")

conn = sqlite3.connect("evaluated_texts.db")
cursor = conn.cursor()
data = cursor.execute("SELECT text,quality_score FROM evaluated_texts where quality_score<2 ORDER BY random() LIMIT 1").fetchone()

input = tokenizer(data[0], return_tensors="pt", add_special_tokens=True, truncation=True)
# input = tokenizer("""
# Hans Christian Andersen, mest kendt som H.C. Andersen (født 2. april 1805 i Odense, død 4. august 1875 i København), var en dansk forfatter og digter, der er verdensberømt for sine eventyr. Han var opfinderen af tingseventyr, der er en undergenre af kunsteventyr, og var en af den danske guldalders vigtigste personer. Han var desuden kendt for sine papirklip, og han menes at have opfundet det flettede danske julehjerte.[2] 
# """, return_tensors="pt", add_special_tokens=True, truncation=True)

output = model(**input)
print(data[0][:300])
print(data[1])
print(output*5)
