# Система обнаружения дефектов на ноутбуках
## <div align="center">Описание</div>
Мы представляем веб-приложение для автоматического выявления и классификации дефектов на ноутбуках, основанное на передовых методах машинного обучения. Это решение позволяет быстро и эффективно оценивать качество оборудования, минимизируя время визуального осмотра и риск человеческой ошибки. Инженеры по качеству смогут легко загружать фотографии ноутбуков и получать подробные отчеты о найденных дефектах. Одной из ключевых особенностей нашего приложения является каскадное обучение, которое обеспечивает высокую точность детекции и возможность дообучения моделей без ухудшения результатов. 

Технологический стек: Python, YOLOv11, Gradio. Благодаря этому решению, специалисты смогут не только быстро находить дефекты, но и улучшать модели детекции на основе своих корректировок, что сделает процесс контроля качества более надежным и эффективным.

## <div align="center">Структура системы обнаружения дефектов</div>
  
Раздельное обучение позволяет гибко дообучать детекцию тех дефектов которые на данный момент система определяет хуже и хочется повысить качество без риска ухудшить показатели детекции других дефектов.Это достигается благодаря тому что нет смещения по классам при увеличении размера датасета.

Обучена модель для сегментации изображений. Данная модель выделяет области на изображении с ноутбуком, области с клавиатурой и монитором. Обучены три нейронных сети для детектирования: битых пикселей, царапин и сколов, проблем с винтами.

[comment]: <> (https://mermaid.js.org/syntax/flowchart.html)
```mermaid
---
title: Схема взаимосвязи нейросетевых компонентов
---
graph LR
I(Изображение) --> A
A[Модель сегментации] -->|Фрагмент с матрицей| B[Модель детекции]
A -->|Фрагмент с корпусом ноутбука| C[Модель детекции]
A -->|Фрагмент с корпусом ноутбука| D[Модель детекции]
B -->|Битые пиксели| E[Список дефектов]
C -->|Царапины, сколы| E
D -->|Замок, отсутствие болта| E

subgraph 0
A1[(Ноутбук, клавиатура, матрица, монитор)] --> A
end
subgraph 1
B1[(Битый пиксель)] --> B
end
subgraph 2
C1[(Царапина, скол)] --> C
end
subgraph 3
D1[(Нормальное положение болта, отсутствие болта, смещение болта)] --> D
end
```
Система имеет основной модуль 'main.py' и набор модулей, содержащих дополнительный функционал.
```mermaid
---
title: Схема взаимосвязи программных компонентов
---
graph TD
id1[[main.py]]
id2[[detector.py]] --> id1
id3[[fumctions.py]] --> id1

A[Основная программа, соддержащаая в себе запуск всей системы, запуск интерфейса и инференс нейронных сетей] --> id1
A1[Набор функций и методов для работы с отрисовкойи редактированием аннотаций] --> id2
A2[Набор функций для сохранения данных о ноутбуках, формирования статистики и отчетов] --> id3
```

## <div align="center">Документация</div>
### Необходимые библиотеки
```
gradio                   5.0.2
fpdf                     1.7.2
matplotlib               3.8.4
numpy                    1.26.4
pillow                   10.3.0
ultralytics              8.2.90
opencv-contrib-python    4.9.0.80
opencv-python            4.10.0.84

```

### Запуск 
```
python3 main.py
```
