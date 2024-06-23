from transformers import pipeline

summarizer = pipeline("summarization", "IlyaGusev/rut5_base_sum_gazeta")

text_for_example = "Высота башни составляет 324 метра (1063 фута), примерно \
такая же высота, как у 81-этажного здания, и самое высокое сооружение \
в Париже. Его основание квадратно, размером 125 метров (410 футов) \
с любой стороны. Во время строительства Эйфелева башня превзошла \
монумент Вашингтона, став самым высоким искусственным сооружением в мире, \
и этот титул она удерживала в течение 41 года до завершения строительство \
здания Крайслер в Нью-Йорке в 1930 году. Это первое сооружение которое \
достигло высоты 300 метров. Из-за добавления вещательной антенны на \
вершине башни в 1957 году она сейчас выше здания Крайслер на 5,2 метра \
(17 футов). За исключением передатчиков, Эйфелева башня является второй \
самой высокой отдельно стоящей структурой во Франции после виадука Мийо."

print("ТЕКСТ ИСТОЧНИК:\n")
print("\""+text_for_example+"\"")
print("\nСГЕНЕРИРОВАННАЯ АННОТАЦИЯ:\n")

print("\""+summarizer(text_for_example)[0]['summary_text']+"\"\n")
