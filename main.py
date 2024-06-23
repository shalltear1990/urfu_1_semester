from fastapi import FastAPI
from transformers import AutoTokenizer, T5ForConditionalGeneration
from pydantic import BaseModel
import re


# Данная функция считает примерное количество слов в тексте
def get_word_count(text_str):
    text_str_mod = re.sub(r'(?:(?<=\d)[,]{1}(?=\d)|[-_@.])', '', text_str)
    res = re.findall(r'[0-9a-zA-Zа-яА-ЯёЁ]+', text_str_mod)
    return len(res)


def load_model(name_of_model):
    tokenizer = AutoTokenizer.from_pretrained(name_of_model)
    model = T5ForConditionalGeneration.from_pretrained(name_of_model)
    return model, tokenizer


def get_sum(model, article_text):
    input_ids = model[1](
        [article_text],
        max_length=600,
        add_special_tokens=True,
        padding="max_length",
        truncation=True,
        return_tensors="pt")["input_ids"]

    output_ids = model[0].generate(
        input_ids=input_ids,
        no_repeat_ngram_size=4)[0]

    summary = model[1].decode(output_ids, skip_special_tokens=True)

    return summary


class Item(BaseModel):
    text: str


model_name = "IlyaGusev/rut5_base_sum_gazeta"
summarizer = load_model(model_name)

# get_sum(summarizer, source_text) #Способ применения данной модели МО

app = FastAPI()


@app.get("/")
def root():
    """Данный метод возвращает данные создателя этого API"""
    return {
        "Студент": {
            "Имя": "Сергей",
            "Фамилия": "Никульшин",
            "Группа": "РИМ-130963"
        }
    }


@app.post("/predict/")
def predict(item: Item):
    """Данный метод принимает в качестве параметра \"text\" в теле запроса
    текст, который будет использован для создания аннотации с помощью
    предобученной модели МО и возвращает json документ, в котором содержатся:
    принятый к обработке текст, количество слов в этом тексте и
    сгенерированная аннотация"""
    return {
        "source_text": item.text,
        "word_in_text": get_word_count(item.text),
        "target_text": get_sum(summarizer, item.text)
    }
