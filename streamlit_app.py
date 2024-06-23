import streamlit as st
from transformers import AutoTokenizer, T5ForConditionalGeneration
import re


# Данная функция считает примерное количество "слов" в тексте (словом будет
# считаться непрерывная последовательность букв и цифр (с небольшими
# исключениями)). Т.к. данное веб-приложение создано для работы с простыми
# русскими текстами (преимущественно, с новостными и научно-популярными
# статьями), логика подсчета слов примитивная и не гарантируется корректная
# работа, если текст будет содержать большое количество сложных конструкций
# (например, формул или нестандартных email-адресов и т.д.), будет перегружен
# сложными символьными небуквенными конструкциями или будет иметь большое
# количество синтаксических ошибок (например, пропусков пробелов между словами)
# Т.к. модель МО все равно не рассчитана на подобные тексты, нет смысла сильно
# усложнять данную функцию.
def get_word_count(text_str):
    text_str_mod = re.sub(r'(?:(?<=\d)[,]{1}(?=\d)|[-_@.])', '', text_str)
    res = re.findall(r'[0-9a-zA-Zа-яА-ЯёЁ]+', text_str_mod)
    return len(res)


@st.cache_resource
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


model_name = "IlyaGusev/rut5_base_sum_gazeta"
summarizer = load_model(model_name)

st.markdown("<h1 style='text-align: center;'>Создание аннотации текста</h1>",
            unsafe_allow_html=True)

description_text = "Данное streamlit веб-приложение умеет автоматически \
генерировать аннотацию текста на русском языке с помощью предобученной \
модели машинного обучения [IlyaGusev/rut5_base_sum_gazeta]\
(https://huggingface.co/IlyaGusev/rut5_base_sum_gazeta). Для корректной \
работы исходный текст должен быть:\n\
- на русском языке\n\
- объемом не менее 100 слов и не более 600 слов\n\
- содержать минимум грамматических и синтаксических ошибок\n\
- быть простым (не содержать слишком много формул, сложных небуквенных \
конструкций, слов на английском языке и т.д.)\n\n\
Для генерации аннотации вставьте текст в поле ввода и нажмите кнопку \
**Создать аннотацию**, либо нажмите сочетание клавиш **Ctrl+Enter** при \
активном курсоре в поле ввода, либо просто **уберите курсор** из поля \
ввода (кликнув мышкой вне области ввода)."

with st.expander("Краткое описание данного веб-приложения"):
    st.write(description_text)

source_text = st.text_area("Введите текст для обработки")

# В данном конкретном случае нет необходимости как-то отдельно
# обрабатывать событие нажатия кнопки, т.к. сам факт нажатия
# запустит перевыполнение всего кода
st.button("Создать аннотацию")

if source_text != "":
    word_in_text = get_word_count(source_text)
    if 100 <= word_in_text <= 600:
        st.success("Слов в тексте - "
                   + str(word_in_text)
                   + ", текст подходит по объему!")
    elif word_in_text != 0:
        st.warning("Слов в тексте - "
                   + str(word_in_text)
                   + ", текст не подходит по объему! Приложение попытается \
                    создать аннотацию, но корректный результат не \
                    гарантируется!")
    elif word_in_text == 0:
        st.error("В поле ввода не обнаружено ни одного слова. \
                 Генерация аннотации отменена.")

    if word_in_text != 0:
        st.write(get_sum(summarizer, source_text))
