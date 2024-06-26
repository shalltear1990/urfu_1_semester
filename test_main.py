from fastapi.testclient import TestClient
from main import app
from main import get_word_count

client = TestClient(app)

source_text = "Высота башни составляет 324 метра (1063 фута), примерно такая \
же высота, как у 81-этажного здания, и самое высокое сооружение в Париже. \
Его основание квадратно, размером 125 метров (410 футов) с любой стороны. \
Во время строительства Эйфелева башня превзошла монумент Вашингтона, став \
самым высоким искусственным сооружением в мире, и этот титул она \
удерживала в течение 41 года до завершения строительство здания Крайслер \
в Нью-Йорке в 1930 году. Это первое сооружение которое достигло высоты \
300 метров. Из-за добавления вещательной антенны на вершине башни в 1957 \
году она сейчас выше здания Крайслер на 5,2 метра (17 футов). За \
исключением передатчиков, Эйфелева башня является второй самой высокой \
отдельно стоящей структурой во Франции после виадука Мийо."
target_text = "Эйфелева башня является второй самой высокой отдельно стоящей \
структурой во Франции после виадука Мийо. Ее высота составляет 324 метра \
(1063 фута), примерно такая же высота, как у 81-этажного здания \
в Нью-Йорке."


def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == \
        {"Студент": {"Имя": "Сергей",
                     "Фамилия": "Никульшин",
                     "Группа": "РИМ-130963"}}


def test_read_predict():
    response = client.post("/predict/", json={"text": source_text})
    json_data = response.json()
    assert response.status_code == 200
    assert json_data['word_in_text'] == 111
    assert json_data['target_text'] == target_text


def test_get_word_count():
    assert get_word_count("Высота от земли до лба до 5,8 м (самое высокое \
                          наземное животное).") == 12
    assert get_word_count(" abc@gmail.com ++") == 1
    assert get_word_count("С Новым 2024 Годом!") == 4
